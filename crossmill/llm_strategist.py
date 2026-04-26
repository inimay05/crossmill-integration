"""
LLM Strategist: SFT + GRPO pipeline for CrossMill strategy bias extraction.

Bias vector convention (STRATEGY_DIM=4):
  bias[0]: thermal/intensity axis   (negative = cool down)
  bias[1]: throughput axis          (positive = push more)
  bias[2]: safety axis              (positive = conservative)
  bias[3]: efficiency/carbon axis   (positive = reduce emissions)
"""

from __future__ import annotations

import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────────────
_INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
_SIBLINGS_ROOT    = _INTEGRATION_ROOT.parent
for _p in [
    str(_SIBLINGS_ROOT / "crossmill-safenutri"),
    str(_SIBLINGS_ROOT / "crossmill-megaforge"),
    str(_INTEGRATION_ROOT),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crossmill.training_config import STRATEGY_DIM
from crossmill.config import ENVIRONMENTS

# ── Bias parsing ─────────────────────────────────────────────────────────────

_BIAS_RE = re.compile(r'Bias:\s*\[([^\]]+)\]')


def _parse_bias(text: str):
    """Return (bias_array shape (STRATEGY_DIM,), format_ok bool)."""
    m = _BIAS_RE.search(text)
    if not m:
        return np.zeros(STRATEGY_DIM, dtype=np.float32), False
    try:
        vals = [float(x.strip()) for x in m.group(1).split(',')]
        if len(vals) != STRATEGY_DIM:
            return np.zeros(STRATEGY_DIM, dtype=np.float32), False
        return np.clip(np.array(vals, dtype=np.float32), -1.0, 1.0), True
    except (ValueError, AttributeError):
        return np.zeros(STRATEGY_DIM, dtype=np.float32), False


def _render_prompt(env_name: str, state: dict) -> str:
    """Render an observation state dict as a structured LLM prompt."""
    lines = [
        f"You are a process-control strategist for {env_name}.",
        "Current state:",
    ]
    for k, v in state.items():
        ann = ""
        if isinstance(v, float):
            if v > 0.75:
                ann = " (high)"
            elif v < 0.25:
                ann = " (low)"
        lines.append(f"- {k}: {v:.2f}{ann}")
    lines.append("Recommend a strategy.")
    return "\n".join(lines)


# ── SFT scenario banks ────────────────────────────────────────────────────────

_SAFENUTRI_SCENARIOS = [
    # High temp / thermal stress → cool down
    ({"temperature": 0.85, "e_coli": 0.30, "vitamin_c": 0.60,
      "contamination_risk": 0.50, "safety_margin": 0.30},
     "Reduce thermal load to prevent nutrient degradation; maintain safety margins.",
     [-0.5, 0.0, 0.3, 0.0]),
    ({"temperature": 0.90, "e_coli": 0.20, "vitamin_c": 0.55,
      "contamination_risk": 0.40, "safety_margin": 0.25},
     "Decrease temperature urgently; protect vitamin C retention.",
     [-0.5, 0.0, 0.3, 0.0]),
    ({"temperature": 0.80, "e_coli": 0.40, "vitamin_c": 0.50,
      "contamination_risk": 0.60, "safety_margin": 0.20},
     "Reduce thermal intensity and increase safety buffer.",
     [-0.4, -0.1, 0.4, 0.0]),
    ({"temperature": 0.78, "e_coli": 0.35, "vitamin_c": 0.65,
      "contamination_risk": 0.45, "safety_margin": 0.35},
     "Moderate cooling; maintain throughput; monitor contamination.",
     [-0.3, 0.0, 0.2, 0.0]),
    # Low throughput, healthy state → push throughput
    ({"temperature": 0.45, "e_coli": 0.10, "vitamin_c": 0.80,
      "contamination_risk": 0.15, "safety_margin": 0.70},
     "Increase throughput; system is healthy and can handle more load.",
     [0.6, 0.3, 0.0, 0.0]),
    ({"temperature": 0.40, "e_coli": 0.08, "vitamin_c": 0.85,
      "contamination_risk": 0.10, "safety_margin": 0.75},
     "Ramp up production; excellent safety margins permit throughput increase.",
     [0.7, 0.4, 0.0, 0.0]),
    ({"temperature": 0.50, "e_coli": 0.12, "vitamin_c": 0.75,
      "contamination_risk": 0.20, "safety_margin": 0.65},
     "Gradually increase throughput while monitoring key indicators.",
     [0.5, 0.3, 0.0, 0.0]),
    ({"temperature": 0.42, "e_coli": 0.09, "vitamin_c": 0.82,
      "contamination_risk": 0.12, "safety_margin": 0.72},
     "Optimal conditions for throughput increase; proceed with gradual ramp.",
     [0.6, 0.4, 0.0, 0.0]),
    # Safety risk imminent
    ({"temperature": 0.60, "e_coli": 0.65, "vitamin_c": 0.45,
      "contamination_risk": 0.80, "safety_margin": 0.10},
     "Emergency safety mode; reduce throughput and address contamination risk.",
     [0.0, -0.2, 0.8, -0.1]),
    ({"temperature": 0.55, "e_coli": 0.70, "vitamin_c": 0.40,
      "contamination_risk": 0.85, "safety_margin": 0.05},
     "Critical safety alert; halt throughput increase and sanitize system.",
     [0.0, -0.3, 0.9, -0.2]),
    ({"temperature": 0.65, "e_coli": 0.60, "vitamin_c": 0.50,
      "contamination_risk": 0.75, "safety_margin": 0.15},
     "High contamination risk; prioritize safety over throughput.",
     [0.0, -0.2, 0.7, -0.1]),
    ({"temperature": 0.58, "e_coli": 0.55, "vitamin_c": 0.48,
      "contamination_risk": 0.70, "safety_margin": 0.18},
     "Reduce contamination risk immediately; safety margin is dangerously low.",
     [0.0, -0.1, 0.7, 0.0]),
    # Vitamin C retention
    ({"temperature": 0.70, "e_coli": 0.20, "vitamin_c": 0.30,
      "contamination_risk": 0.30, "safety_margin": 0.50},
     "Reduce temperature to protect vitamin C; optimize retention over throughput.",
     [-0.4, 0.5, 0.2, 0.0]),
    ({"temperature": 0.72, "e_coli": 0.18, "vitamin_c": 0.25,
      "contamination_risk": 0.28, "safety_margin": 0.55},
     "Critical vitamin C depletion; lower thermal intensity immediately.",
     [-0.5, 0.5, 0.2, 0.0]),
    ({"temperature": 0.68, "e_coli": 0.22, "vitamin_c": 0.35,
      "contamination_risk": 0.32, "safety_margin": 0.48},
     "Moderate temperature reduction to improve nutrient preservation.",
     [-0.3, 0.4, 0.1, 0.0]),
    ({"temperature": 0.65, "e_coli": 0.25, "vitamin_c": 0.38,
      "contamination_risk": 0.35, "safety_margin": 0.52},
     "Balance thermal load with vitamin C retention; mild cooling recommended.",
     [-0.3, 0.4, 0.2, 0.0]),
    # Balanced / normal operation
    ({"temperature": 0.55, "e_coli": 0.20, "vitamin_c": 0.65,
      "contamination_risk": 0.25, "safety_margin": 0.60},
     "Maintain current operational parameters; system is performing well.",
     [0.0, 0.2, 0.1, 0.0]),
    ({"temperature": 0.50, "e_coli": 0.25, "vitamin_c": 0.70,
      "contamination_risk": 0.30, "safety_margin": 0.55},
     "Steady state operation; slight throughput increase is safe.",
     [0.1, 0.2, 0.1, 0.0]),
    ({"temperature": 0.60, "e_coli": 0.30, "vitamin_c": 0.60,
      "contamination_risk": 0.35, "safety_margin": 0.50},
     "Monitor contamination closely; maintain safety while optimizing throughput.",
     [0.0, 0.1, 0.2, 0.0]),
    ({"temperature": 0.48, "e_coli": 0.15, "vitamin_c": 0.75,
      "contamination_risk": 0.20, "safety_margin": 0.65},
     "Good conditions; gentle throughput increase recommended.",
     [0.2, 0.3, 0.0, 0.0]),
]

_MEGAFORGE_SCENARIOS = [
    # High carbon → reduce emissions
    ({"hot_metal_temp": 0.80, "carbon": 0.85, "emissions_co2": 0.80,
      "thermal_stress": 0.70, "equip_health": 0.40},
     "Reduce carbon injection; lower thermal intensity to cut emissions.",
     [-0.3, 0.0, 0.0, 0.6]),
    ({"hot_metal_temp": 0.85, "carbon": 0.90, "emissions_co2": 0.85,
      "thermal_stress": 0.75, "equip_health": 0.35},
     "Critical carbon levels; reduce coke rate and blast temperature.",
     [-0.4, 0.0, 0.0, 0.7]),
    ({"hot_metal_temp": 0.75, "carbon": 0.78, "emissions_co2": 0.72,
      "thermal_stress": 0.65, "equip_health": 0.45},
     "Moderate carbon reduction; optimize ore-to-coke ratio for efficiency.",
     [-0.3, 0.0, 0.0, 0.5]),
    ({"hot_metal_temp": 0.70, "carbon": 0.72, "emissions_co2": 0.68,
      "thermal_stress": 0.60, "equip_health": 0.50},
     "Reduce carbon emissions gradually; maintain production rate.",
     [-0.2, 0.1, 0.0, 0.5]),
    # Low throughput, healthy state → push throughput
    ({"hot_metal_temp": 0.50, "carbon": 0.30, "emissions_co2": 0.25,
      "thermal_stress": 0.30, "equip_health": 0.85},
     "Equipment in excellent condition; increase blast rate for higher production.",
     [0.3, 0.5, 0.0, 0.2]),
    ({"hot_metal_temp": 0.45, "carbon": 0.25, "emissions_co2": 0.20,
      "thermal_stress": 0.25, "equip_health": 0.90},
     "Optimal conditions; ramp up production rate aggressively.",
     [0.4, 0.6, 0.0, 0.1]),
    ({"hot_metal_temp": 0.55, "carbon": 0.28, "emissions_co2": 0.22,
      "thermal_stress": 0.28, "equip_health": 0.82},
     "Healthy system; push production while maintaining emission controls.",
     [0.3, 0.5, 0.0, 0.2]),
    ({"hot_metal_temp": 0.48, "carbon": 0.32, "emissions_co2": 0.30,
      "thermal_stress": 0.35, "equip_health": 0.85},
     "Ideal production conditions; push output while maintaining efficiency.",
     [0.4, 0.5, 0.0, 0.1]),
    # Safety / thermal stress
    ({"hot_metal_temp": 0.88, "carbon": 0.50, "emissions_co2": 0.55,
      "thermal_stress": 0.90, "equip_health": 0.20},
     "Critical thermal stress; reduce blast temperature and protect equipment.",
     [-0.5, -0.3, 0.7, 0.0]),
    ({"hot_metal_temp": 0.82, "carbon": 0.45, "emissions_co2": 0.50,
      "thermal_stress": 0.85, "equip_health": 0.25},
     "High thermal stress and low equipment health; prioritize safety.",
     [-0.4, -0.2, 0.8, 0.0]),
    ({"hot_metal_temp": 0.78, "carbon": 0.48, "emissions_co2": 0.52,
      "thermal_stress": 0.82, "equip_health": 0.30},
     "Equipment health degraded; reduce thermal stress and production rate.",
     [-0.4, -0.2, 0.6, 0.0]),
    ({"hot_metal_temp": 0.85, "carbon": 0.55, "emissions_co2": 0.60,
      "thermal_stress": 0.88, "equip_health": 0.22},
     "Emergency protocol; thermal runaway risk detected.",
     [-0.5, -0.4, 0.9, 0.0]),
    # Coke rate optimization
    ({"hot_metal_temp": 0.60, "carbon": 0.40, "emissions_co2": 0.42,
      "thermal_stress": 0.45, "equip_health": 0.70},
     "Optimize coke rate for better efficiency; moderate throughput increase.",
     [0.2, 0.4, 0.0, 0.5]),
    ({"hot_metal_temp": 0.55, "carbon": 0.35, "emissions_co2": 0.38,
      "thermal_stress": 0.40, "equip_health": 0.75},
     "Favorable conditions for coke optimization; push throughput with carbon control.",
     [0.2, 0.4, 0.0, 0.4]),
    ({"hot_metal_temp": 0.58, "carbon": 0.38, "emissions_co2": 0.35,
      "thermal_stress": 0.42, "equip_health": 0.72},
     "Low carbon with stable conditions; increase production and optimize coke usage.",
     [0.3, 0.4, 0.0, 0.3]),
    ({"hot_metal_temp": 0.62, "carbon": 0.42, "emissions_co2": 0.38,
      "thermal_stress": 0.45, "equip_health": 0.68},
     "Moderate state; maintain coke efficiency while gently pushing throughput.",
     [0.1, 0.3, 0.1, 0.3]),
    # Balanced / normal operation
    ({"hot_metal_temp": 0.65, "carbon": 0.48, "emissions_co2": 0.45,
      "thermal_stress": 0.50, "equip_health": 0.60},
     "Balanced operation; slight throughput increase with carbon monitoring.",
     [0.0, 0.2, 0.1, 0.2]),
    ({"hot_metal_temp": 0.68, "carbon": 0.52, "emissions_co2": 0.48,
      "thermal_stress": 0.52, "equip_health": 0.58},
     "Monitor carbon levels; maintain current blast parameters.",
     [0.0, 0.1, 0.1, 0.2]),
    ({"hot_metal_temp": 0.60, "carbon": 0.55, "emissions_co2": 0.52,
      "thermal_stress": 0.48, "equip_health": 0.62},
     "Slightly elevated carbon; reduce coke rate while maintaining production.",
     [-0.1, 0.1, 0.1, 0.3]),
    ({"hot_metal_temp": 0.55, "carbon": 0.60, "emissions_co2": 0.58,
      "thermal_stress": 0.40, "equip_health": 0.72},
     "Carbon slightly elevated; reduce coke rate while maintaining blast.",
     [-0.1, 0.1, 0.0, 0.4]),
]


# ── Step 1: SFT data generation ───────────────────────────────────────────────

def generate_sft_examples(env_name: str, n: int = 20) -> list:
    """
    Generate n synthetic SFT training examples for the given environment.
    Returns list of dicts with 'prompt' and 'completion' keys.

    The completion always follows the format:
        Strategy: <text>
        Bias: [b0, b1, b2, b3]
    where the four bias dimensions are:
        [thermal, throughput, safety, efficiency/carbon]
    """
    if env_name == 'safenutri':
        scenarios = _SAFENUTRI_SCENARIOS
    elif env_name == 'megaforge':
        scenarios = _MEGAFORGE_SCENARIOS
    else:
        raise ValueError(f"Unknown env_name: {env_name!r}. "
                         f"Must be 'safenutri' or 'megaforge'.")

    selected = [scenarios[i % len(scenarios)] for i in range(n)]

    examples = []
    for state, strategy, bias in selected:
        prompt = _render_prompt(env_name, state)
        bias_str = "[" + ", ".join(f"{b:.1f}" for b in bias) + "]"
        completion = f"Strategy: {strategy}\nBias: {bias_str}"
        examples.append({"prompt": prompt, "completion": completion})

    return examples


# ── Step 1: SFT training ──────────────────────────────────────────────────────

def train_sft(
    env_name: str = 'safenutri',
    n_examples: int = 20,
    output_dir: Optional[str] = None,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_epochs: int = 3,
):
    """
    SFT warmup: fine-tune model on n_examples with QLoRA for format compliance.
    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    output_dir = output_dir or str(_INTEGRATION_ROOT / "runs" / "sft_warmup")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[SFT] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Primary env + cross-cutting examples for both envs
    examples = generate_sft_examples(env_name, n_examples)
    other = 'megaforge' if env_name == 'safenutri' else 'safenutri'
    examples += generate_sft_examples(other, max(4, n_examples // 5))

    def _fmt(ex):
        messages = [
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list([_fmt(e) for e in examples])

    # TRL >= 0.12 uses SFTConfig; older versions pass args to SFTTrainer directly
    try:
        from trl import SFTTrainer, SFTConfig
        sft_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=5,
            report_to="none",
            save_strategy="no",
            max_seq_length=256,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_args,
        )
    except (ImportError, TypeError):
        from trl import SFTTrainer
        from transformers import TrainingArguments
        tr_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=5,
            report_to="none",
            save_strategy="no",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            max_seq_length=256,
            dataset_text_field="text",
            args=tr_args,
        )

    print("[SFT] Starting warmup training ...")
    trainer.train()
    print("[SFT] Training complete.")
    return model, tokenizer


def save_sft_model(model, tokenizer, path: str) -> None:
    """Save SFT-trained LoRA adapter and tokenizer to path."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[SFT] Model saved to {path}")


# ── Step 2: GRPO dataset ──────────────────────────────────────────────────────

def build_grpo_dataset(
    env_name: str,
    task_id: str = 'easy',
    n: int = 50,
) -> list:
    """
    Build GRPO training dataset by sampling n fresh env observations.
    Returns list of dicts with 'prompt' key (HuggingFace Dataset compatible).
    """
    from crossmill.platform import CrossMillPlatform
    from crossmill.memory import CrossIndustryMemory
    from crossmill.augmentation import OBS_FIELDS

    kwargs = {
        'memory': CrossIndustryMemory(),
        'seed': 0,
        'safenutri_task': task_id if env_name == 'safenutri' else 'easy',
        'megaforge_task': task_id if env_name == 'megaforge' else 'easy',
    }
    platform = CrossMillPlatform(**kwargs)
    fields = OBS_FIELDS[env_name]

    data = []
    for i in range(n):
        obs = platform.reset(env_name, seed=i)
        # obs is augmented (raw + bias); take only raw fields
        state = {f: float(obs[j]) for j, f in enumerate(fields)}
        data.append({"prompt": _render_prompt(env_name, state)})

    return data


# ── Step 2: GRPO reward function ──────────────────────────────────────────────
# Module-level state avoids closure/pickle issues with TRL multiprocessing.

_reward_platform   = None
_reward_env_name   = 'safenutri'
_reward_task_id    = 'easy'


def _ensure_reward_platform(env_name: str = 'safenutri', task_id: str = 'easy') -> None:
    global _reward_platform, _reward_env_name, _reward_task_id
    _reward_env_name = env_name
    _reward_task_id  = task_id
    if _reward_platform is None:
        from crossmill.platform import CrossMillPlatform
        from crossmill.memory import CrossIndustryMemory
        kwargs = {
            'memory': CrossIndustryMemory(),
            'seed': 42,
            'safenutri_task': task_id if env_name == 'safenutri' else 'easy',
            'megaforge_task': task_id if env_name == 'megaforge' else 'easy',
        }
        _reward_platform = CrossMillPlatform(**kwargs)


def _bias_to_action(bias: np.ndarray, env_name: str) -> np.ndarray:
    """Map 4-dim strategy bias to an environment action vector."""
    action_dim = ENVIRONMENTS[env_name]['action_dim']
    action = np.full(action_dim, 0.5, dtype=np.float32)
    # bias[0]: thermal/intensity (negative = cool down → lower action)
    action[0] = float(np.clip(0.5 + 0.25 * bias[0], 0.0, 1.0))
    if action_dim > 1:
        # bias[1]: throughput (positive = push more → higher action)
        action[1] = float(np.clip(0.5 + 0.25 * bias[1], 0.0, 1.0))
    if action_dim > 2:
        # bias[2]: safety (positive = conservative → lower action)
        action[2] = float(np.clip(0.5 - 0.2 * bias[2], 0.0, 1.0))
    if action_dim > 3:
        # bias[3]: efficiency (positive = reduce emissions → lower fuel action)
        action[3] = float(np.clip(0.5 - 0.2 * bias[3], 0.0, 1.0))
    return action


def multi_component_reward_fn(completions, prompts=None, **kwargs) -> list:
    """
    3-component GRPO reward:
        R = 0.5 * r_env  +  0.3 * r_fmt  +  0.2 * r_safe

    All components clipped to [0, 1].
    If r_fmt == 0 (malformed output), r_env and r_safe are masked to 0
    so the total reward is 0 — making format violation strictly dominated.

    Args:
        completions: list[str] — LLM-generated texts for one GRPO group.
        prompts: list[str] | None — corresponding prompts (unused directly).
        **kwargs: additional TRL keyword arguments (ignored).

    Returns:
        list[float] of per-completion rewards.
    """
    global _reward_platform, _reward_env_name

    _ensure_reward_platform(_reward_env_name, _reward_task_id)

    K = 8   # rollout steps per candidate (design doc §2.3)
    rewards = []

    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        bias, fmt_ok = _parse_bias(text)
        r_fmt = 1.0 if fmt_ok else 0.0

        if not fmt_ok:
            rewards.append(0.0)
            continue

        try:
            _reward_platform.reset(_reward_env_name, seed=42)
            action = _bias_to_action(bias, _reward_env_name)

            total_reward   = 0.0
            safety_violations = 0
            steps_taken    = 0

            for _ in range(K):
                result = _reward_platform.step(_reward_env_name, action.tolist())
                total_reward += result['reward']
                steps_taken  += 1

                # Safety violation detection: explicit flags OR large negative reward
                info = result.get('info', {})
                if any(info.get(flag, False) for flag in (
                    'safety_violation', 'contamination_event',
                    'thermal_violation', 'catastrophic',
                )):
                    safety_violations += 1
                elif result['reward'] < -0.5:
                    safety_violations += 1

                if result['done'] or result['truncated']:
                    break

            steps_taken = max(steps_taken, 1)
            r_env  = float(np.clip(total_reward / steps_taken, 0.0, 1.0))
            r_safe = 1.0 - min(safety_violations / steps_taken, 1.0)
            R = 0.5 * r_env + 0.3 * r_fmt + 0.2 * r_safe

        except Exception:
            # Format-only reward if env rollout fails
            R = 0.3 * r_fmt

        rewards.append(float(R))

    return rewards


# ── Step 2: GRPO training ─────────────────────────────────────────────────────

def grpo_train(
    model,
    tokenizer,
    dataset: list,
    reward_fn=None,
    output_dir: Optional[str] = None,
    num_steps: int = 50,
    env_name: str = 'safenutri',
    task_id: str = 'easy',
) -> None:
    """
    Run GRPO fine-tuning (50 steps, group size G=4, K=8 rollout).

    Args:
        model:      LoRA-wrapped causal LM (from train_sft).
        tokenizer:  matching tokenizer.
        dataset:    list of {'prompt': str} dicts (from build_grpo_dataset).
        reward_fn:  reward callable; defaults to multi_component_reward_fn.
        output_dir: where to write GRPO artifacts.
        num_steps:  number of gradient update steps (default 50).
        env_name:   environment for rollout rewards.
        task_id:    difficulty tier for the reward environment.
    """
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset as HFDataset

    if reward_fn is None:
        reward_fn = multi_component_reward_fn

    _ensure_reward_platform(env_name, task_id)

    output_dir = output_dir or str(_INTEGRATION_ROOT / "runs" / "grpo_llm")
    os.makedirs(output_dir, exist_ok=True)

    hf_dataset = HFDataset.from_list(dataset)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=num_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        max_completion_length=64,
        temperature=0.9,
        top_p=0.95,
        num_generations=4,   # group size G=4
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=hf_dataset,
    )

    print("[GRPO] Starting training ...")
    trainer.train()
    print("[GRPO] Training complete.")


# ── Step 3: Strategy bias extraction ─────────────────────────────────────────

# 10 diverse probe states per environment (design doc §3)
_SAFENUTRI_PROBES = [
    {"temperature": 0.45, "e_coli": 0.10, "vitamin_c": 0.80,
     "contamination_risk": 0.15, "safety_margin": 0.70},
    {"temperature": 0.60, "e_coli": 0.30, "vitamin_c": 0.60,
     "contamination_risk": 0.35, "safety_margin": 0.50},
    {"temperature": 0.82, "e_coli": 0.35, "vitamin_c": 0.45,
     "contamination_risk": 0.55, "safety_margin": 0.25},
    {"temperature": 0.65, "e_coli": 0.70, "vitamin_c": 0.40,
     "contamination_risk": 0.85, "safety_margin": 0.08},
    {"temperature": 0.92, "e_coli": 0.20, "vitamin_c": 0.30,
     "contamination_risk": 0.40, "safety_margin": 0.18},
    {"temperature": 0.38, "e_coli": 0.05, "vitamin_c": 0.90,
     "contamination_risk": 0.08, "safety_margin": 0.80},
    {"temperature": 0.72, "e_coli": 0.18, "vitamin_c": 0.22,
     "contamination_risk": 0.28, "safety_margin": 0.55},
    {"temperature": 0.55, "e_coli": 0.55, "vitamin_c": 0.50,
     "contamination_risk": 0.70, "safety_margin": 0.12},
    {"temperature": 0.52, "e_coli": 0.22, "vitamin_c": 0.68,
     "contamination_risk": 0.28, "safety_margin": 0.58},
    {"temperature": 0.68, "e_coli": 0.40, "vitamin_c": 0.55,
     "contamination_risk": 0.50, "safety_margin": 0.38},
]

_MEGAFORGE_PROBES = [
    {"hot_metal_temp": 0.50, "carbon": 0.30, "emissions_co2": 0.28,
     "thermal_stress": 0.30, "equip_health": 0.85},
    {"hot_metal_temp": 0.65, "carbon": 0.50, "emissions_co2": 0.48,
     "thermal_stress": 0.50, "equip_health": 0.65},
    {"hot_metal_temp": 0.82, "carbon": 0.78, "emissions_co2": 0.75,
     "thermal_stress": 0.80, "equip_health": 0.35},
    {"hot_metal_temp": 0.88, "carbon": 0.85, "emissions_co2": 0.88,
     "thermal_stress": 0.90, "equip_health": 0.18},
    {"hot_metal_temp": 0.45, "carbon": 0.25, "emissions_co2": 0.20,
     "thermal_stress": 0.25, "equip_health": 0.92},
    {"hot_metal_temp": 0.60, "carbon": 0.60, "emissions_co2": 0.58,
     "thermal_stress": 0.45, "equip_health": 0.72},
    {"hot_metal_temp": 0.78, "carbon": 0.42, "emissions_co2": 0.45,
     "thermal_stress": 0.82, "equip_health": 0.28},
    {"hot_metal_temp": 0.55, "carbon": 0.70, "emissions_co2": 0.68,
     "thermal_stress": 0.40, "equip_health": 0.78},
    {"hot_metal_temp": 0.62, "carbon": 0.45, "emissions_co2": 0.42,
     "thermal_stress": 0.48, "equip_health": 0.62},
    {"hot_metal_temp": 0.72, "carbon": 0.55, "emissions_co2": 0.52,
     "thermal_stress": 0.65, "equip_health": 0.50},
]


def extract_strategy_bias(
    model,
    tokenizer,
    env_name: str = 'safenutri',
    n_queries: int = 10,
) -> np.ndarray:
    """
    Query the trained LLM on n_queries diverse observations, parse each
    output to a 4-dim bias vector, average across all → stable strategy_bias.

    Returns:
        np.ndarray of shape (STRATEGY_DIM,) = (4,), clipped to [-1, 1].
    """
    import torch

    probes = _SAFENUTRI_PROBES if env_name == 'safenutri' else _MEGAFORGE_PROBES
    states = [probes[i % len(probes)] for i in range(n_queries)]

    bias_vectors   = []
    parse_failures = 0

    for state in states:
        prompt = _render_prompt(env_name, state)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,        # greedy for determinism
                temperature=1.0,
                repetition_penalty=1.1,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

        bias, fmt_ok = _parse_bias(response)
        if not fmt_ok:
            parse_failures += 1
        bias_vectors.append(bias)

    if parse_failures > 0:
        print(f"[Extract] Parse failures: {parse_failures}/{n_queries} "
              f"(zeros used for failed parses)")

    stacked   = np.stack(bias_vectors, axis=0)   # (n_queries, STRATEGY_DIM)
    mean_bias = stacked.mean(axis=0)
    std_bias  = stacked.std(axis=0)

    # Stability guard: if any axis has std > 0.5, hold it at 0 (design doc §3)
    for ax in range(STRATEGY_DIM):
        if std_bias[ax] > 0.5:
            print(f"[Extract] WARNING: axis {ax} unstable "
                  f"(std={std_bias[ax]:.3f}); resetting to 0")
            mean_bias[ax] = 0.0

    mean_bias = np.clip(mean_bias, -1.0, 1.0).astype(np.float32)
    print(f"[Extract] strategy_bias = {mean_bias}  "
          f"(std per axis = {std_bias.round(3)})")
    return mean_bias


# ── Step 3: Save / load helpers ───────────────────────────────────────────────

def save_strategy_bias(bias: np.ndarray, path: str) -> None:
    """Persist 4-dim strategy bias vector as a .npy file."""
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    np.save(path, bias.astype(np.float32))
    print(f"[Bias] strategy_bias saved → {path}")
