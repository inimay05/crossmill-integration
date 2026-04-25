"""
GRPO + QLoRA fine-tuning of Qwen2.5-3B-Instruct on the SafeNutri environment.

Trains the LLM "Plant Manager" to emit strategy text whose keyword content,
when mapped to a fixed control action, maximises SafeNutri episode reward.
After training, generates one strategy and saves a 4-dim strategy_bias.npy
that downstream RecurrentPPO consumes as observation augmentation.

Run from the integration repo with the venv active:
    cd ~/crossmill-integration && source venv/bin/activate
    python scripts/train_llm_grpo.py

Requires GPU + the following packages installed in the venv:
    transformers peft trl datasets bitsandbytes accelerate
"""

# ── SECTION 1: imports & path setup ─────────────────────────────────────────
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Script is at <integration_root>/scripts/train_llm_grpo.py
# Sibling environment repos live one level above the integration root.
INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
SIBLINGS_ROOT    = INTEGRATION_ROOT.parent

sys.path.insert(0, str(SIBLINGS_ROOT / "crossmill-safenutri"))
sys.path.insert(0, str(SIBLINGS_ROOT / "crossmill-megaforge"))
sys.path.insert(0, str(INTEGRATION_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from crossmill.platform import CrossMillPlatform
from crossmill.memory import CrossIndustryMemory
from crossmill.config import ENVIRONMENTS

LLM_SAVE_PATH = str(INTEGRATION_ROOT / "runs" / "grpo_llm")
os.makedirs(LLM_SAVE_PATH, exist_ok=True)


# ── SECTION 2: platform + reward function ───────────────────────────────────
_platform = CrossMillPlatform(
    memory=CrossIndustryMemory(),
    seed=42,
    safenutri_task="easy",
    megaforge_task="easy",
)
ACTION_DIM = ENVIRONMENTS["safenutri"]["action_dim"]


def crossmill_reward_fn(completions, prompts=None, **kwargs):
    """Map each LLM completion to a fixed control action via keyword extraction,
    roll out one SafeNutri episode with that action, return normalised reward."""
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        t = text.lower()

        action = np.full(ACTION_DIM, 0.5, dtype=np.float32)

        if "increase temperature" in t:
            action[0] = 0.8
        elif "decrease temperature" in t:
            action[0] = 0.2

        if "increase ph" in t:
            action[1] = 0.8
        elif "decrease ph" in t:
            action[1] = 0.2

        if "safety" in t or "careful" in t or "gradual" in t:
            action[2] = 0.3
            action[4] = 0.2

        if "speed" in t or "fast" in t:
            action[3] = 0.8

        action = np.clip(action, 0.0, 1.0)

        try:
            _platform.reset("safenutri", seed=42)
            episode_reward = 0.0
            done = truncated = False
            while not (done or truncated):
                result = _platform.step("safenutri", action.tolist())
                episode_reward += result["reward"]
                done = result["done"]
                truncated = result["truncated"]
        except Exception as e:
            print(f"Reward error: {e}")
            episode_reward = -1.0

        normalised = float(np.clip(episode_reward / 10.0, -1.0, 1.0))
        rewards.append(normalised)
    return rewards


def build_dataset(n=200):
    """Generate n prompts conditioned on freshly reset SafeNutri states."""
    platform = CrossMillPlatform(
        memory=CrossIndustryMemory(),
        seed=0,
        safenutri_task="easy",
        megaforge_task="easy",
    )
    data = []
    for i in range(n):
        obs = platform.reset("safenutri", seed=i)
        prompt = (
            "You are controlling a pharmaceutical pasteuriser.\n"
            f"State: temperature={obs[0]:.2f}, pH={obs[1]:.2f}, "
            f"vitamin_c={obs[2]:.2f}, safety_margin={obs[3]:.2f}.\n"
            "Choose strategy: increase temperature / decrease temperature / "
            "increase pH / decrease pH / prioritise safety / gradual ramp / "
            "increase speed.\n"
            "Respond with one sentence starting with Strategy:"
        )
        data.append({"prompt": prompt})
    return data


dataset = build_dataset(200)

# Reward-variance smoke test — GRPO learns nothing if rewards are constant.
test_rewards = crossmill_reward_fn([
    "Strategy: decrease temperature and prioritise safety with gradual ramp.",
    "Strategy: increase speed and increase temperature rapidly.",
])
print(f"Reward test 1 (safe):  {test_rewards[0]:.4f}")
print(f"Reward test 2 (fast):  {test_rewards[1]:.4f}")
if abs(test_rewards[0] - test_rewards[1]) < 0.01:
    print("WARNING: Rewards are identical. GRPO will not learn. Fix reward function.")
    sys.exit(1)
else:
    print("Reward variance OK — proceeding to training")


# ── SECTION 3: model + QLoRA ────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()       # required for gradient checkpointing with PEFT
model.gradient_checkpointing_enable()    # recompute activations to save ~0.5 GB VRAM
model.print_trainable_parameters()
print("Model loaded with QLoRA")


# ── SECTION 4: GRPO training ────────────────────────────────────────────────
hf_dataset = Dataset.from_list(dataset)

grpo_config = GRPOConfig(
    output_dir=LLM_SAVE_PATH,
    num_train_epochs=1,
    per_device_train_batch_size=1,   # 4 GB GPU: keep to 1
    gradient_accumulation_steps=8,   # same effective batch (1×8=8 vs old 2×4=8)
    logging_steps=10,
    report_to="none",
    save_strategy="no",
    remove_unused_columns=False,
    max_completion_length=32,        # halved to cut KV-cache pressure
    temperature=0.9,
    top_p=0.95,
    num_generations=2,               # 4 → 2: biggest VRAM saver for GRPO
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=crossmill_reward_fn,
    args=grpo_config,
    train_dataset=hf_dataset,
)

print("Starting GRPO training...")
trainer.train()

model.save_pretrained(LLM_SAVE_PATH)
tokenizer.save_pretrained(LLM_SAVE_PATH)
print("Model saved")


# ── SECTION 5: generate strategy + save 4-dim bias ──────────────────────────
print("Generating strategy from trained LLM...")

test_prompt = dataset[0]["prompt"]
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.3,
    )

strategy_text = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print(f"LLM strategy: {strategy_text}")

t = strategy_text.lower()
temp_dir = (
    0.3 if "increase temperature" in t
    else (-0.3 if "decrease temperature" in t else 0.0)
)
safety_w = 0.8 if any(w in t for w in ["safety", "careful", "gradual"]) else 0.4
speed_w  = 0.7 if any(w in t for w in ["speed", "fast"]) else 0.3

strategy_bias = np.array(
    [temp_dir, abs(temp_dir), safety_w, speed_w],
    dtype=np.float32,
)
np.save(f"{LLM_SAVE_PATH}/strategy_bias.npy", strategy_bias)


# ── SECTION 6: verify ───────────────────────────────────────────────────────
loaded = np.load(f"{LLM_SAVE_PATH}/strategy_bias.npy")
assert loaded.shape == (4,), f"Wrong shape: {loaded.shape}"
assert all(-1 <= v <= 1 for v in loaded), "Values out of range"
print("=" * 60)
print("GRPO TRAINING COMPLETE")
print(f"Strategy text: {strategy_text}")
print(f"Strategy bias: {loaded}")
print(f"Saved to: {LLM_SAVE_PATH}/strategy_bias.npy")
print("LLM integration complete. Strategy bias ready for RecurrentPPO.")
print("=" * 60)
