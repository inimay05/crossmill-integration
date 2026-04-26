import os, sys, json, time, argparse
# Make `crossmill` importable when this script is launched as
# `python scripts/train.py` (Python only adds scripts/ to sys.path by default,
# not the project root).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')   # MUST come before pyplot import
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from crossmill.config import ENVIRONMENTS
from crossmill.training_config import (
    GAMMA, GAE_LAMBDA, LEARNING_RATE, N_STEPS, BATCH_SIZE, N_ENVS, VERBOSE,
    DEFAULT_TIMESTEPS, PRE_GRADER_EPISODES, POST_GRADER_EPISODES,
    LOG_DIR_TEMPLATE, ROLLING_WINDOW, PLOT_DPI, STRATEGY_DIM,
)
from crossmill.gym_shim import CrossMillGymShim
from crossmill.plotting import plot_reward_curve
from crossmill.hub_push import push_artifacts_to_hub


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch for reproducibility."""
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_module_from_env(env_name: str, submodule: str):
    """
    Load a Python module from an environment repo by file path.
    Uses importlib.util to avoid the app.* module name collision.

    Args:
      env_name: 'safenutri' or 'megaforge'
      submodule: e.g. 'app.grader' or 'app.baseline_agent'
                 (dot-separated path relative to repo root)

    Returns:
      The loaded module object.
    """
    env_cfg  = ENVIRONMENTS[env_name]
    env_file = env_cfg['env_file']                  # .../app/environment.py
    repo_root = os.path.dirname(os.path.dirname(env_file))  # env repo root
    # Convert 'app.grader' -> 'app/grader.py'
    rel_path  = submodule.replace('.', os.sep) + '.py'
    file_path = os.path.join(repo_root, rel_path)
    module_name = f'{env_name}_{submodule.replace(".", "_")}_module'

    if module_name in sys.modules:
        return sys.modules[module_name]

    # Temporarily add repo root to sys.path so internal imports inside
    # the module (e.g. 'from app.config import ...') resolve correctly.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_baseline(env_name: str, task_id: str, seed: int) -> dict:
    """
    Run the environment's heuristic baseline agent through the grader.
    Returns the grader result dict (pre-training floor score).
    """
    grader_mod   = _load_module_from_env(env_name, 'app.grader')
    baseline_mod = _load_module_from_env(env_name, 'app.baseline_agent')

    # SafeNutri uses HTSTBaselineAgent; MegaForge uses PIDBaselineAgent
    if env_name == 'safenutri':
        agent = baseline_mod.HTSTBaselineAgent()
    else:
        agent = baseline_mod.PIDBaselineAgent()

    result = grader_mod.grader(
        agent, task_id=task_id,
        num_eval_episodes=PRE_GRADER_EPISODES,
        base_seed=seed + 5000,
    )
    return result


def run_grader(env_name: str, task_id: str, policy_fn, seed: int,
               num_episodes: int = POST_GRADER_EPISODES) -> dict:
    """
    Run the trained policy through the environment's built-in grader.
    policy_fn: callable that takes an Observation pydantic model and
               returns an action list.
    Returns the grader result dict.
    """
    grader_mod = _load_module_from_env(env_name, 'app.grader')
    result = grader_mod.grader(
        policy_fn, task_id=task_id,
        num_eval_episodes=num_episodes,
        base_seed=seed + 6000,
    )
    return result


class LSTMPolicyAdapter:
    """
    Adapts a trained RecurrentPPO model to the callable(Observation) -> list
    interface expected by each environment's built-in grader.

    The grader passes a Pydantic Observation model. This adapter converts it
    to the augmented numpy vector the policy expects, runs the LSTM predict,
    and returns a raw action list.

    The full observation seen by the policy during training is:
        [raw_obs (15/18) || memory_bias (8) || strategy_bias (4)] = 27/30 dims.
    During grader evaluation we use zero bias for both memory and strategy.
    """
    def __init__(self, model: RecurrentPPO, shim: CrossMillGymShim):
        self.model = model
        self.shim  = shim
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def __call__(self, obs) -> list:
        # obs is a Pydantic Observation from the env's own grader loop.
        # Convert to augmented vector via the augmentation layer.
        from crossmill.augmentation import obs_to_vector, augment_observation
        from crossmill.augmentation import zero_bias
        raw_vec = obs_to_vector(self.shim.env_name, obs)
        # Use zero memory bias for deterministic grader evaluation
        aug_vec = augment_observation(raw_vec, zero_bias(), self.shim.env_name)
        # Append zero strategy_bias to match training observation size (AUGMENTED_OBS_DIM)
        strategy_zeros = np.zeros(STRATEGY_DIM, dtype=np.float32)
        full_vec = np.concatenate([aug_vec, strategy_zeros]).reshape(1, -1)

        action, self.lstm_states = self.model.predict(
            full_vec,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True,
        )
        self.episode_starts = np.zeros((1,), dtype=bool)
        return action[0].tolist()

    def reset_state(self):
        """Call between episodes during grader evaluation."""
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)


def main():
    ap = argparse.ArgumentParser(
        description='CrossMill unified training script'
    )
    ap.add_argument('--env', required=True,
                    choices=['safenutri', 'megaforge'],
                    help='Which environment to train on')
    ap.add_argument('--task', default='easy',
                    choices=['easy', 'medium', 'hard'],
                    help='Task difficulty tier')
    ap.add_argument('--memory_mode', default='cross',
                    choices=['none', 'local', 'cross'],
                    help='Memory mode: none=no transfer, local=same-env replay, '
                         'cross=cross-industry transfer (default)')
    ap.add_argument('--timesteps', type=int, default=None,
                    help='Training timesteps (default: task-specific from config)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_dir', default=None,
                    help='Directory for logs and artifacts '
                         '(default: ./runs/<env_name>/)')
    ap.add_argument('--push_to_hub', action='store_true',
                    help='Push model + reward curve + summary to HF Hub')
    ap.add_argument('--hf_repo_id', default=None,
                    help='HF repo ID, e.g. username/crossmill-safenutri-easy '
                         '(required if --push_to_hub)')
    # ---- LLM Strategist flags ----
    ap.add_argument('--llm_strategist', action='store_true', default=False,
                    help='Run LLM strategist (SFT + GRPO) to derive strategy_bias '
                         'before RecurrentPPO training')
    ap.add_argument('--sft_model_path', type=str, default=None,
                    help='Path to a pre-trained SFT LoRA adapter. If --llm_strategist '
                         'is set and this is not provided, SFT will be run first.')
    args = ap.parse_args()

    # Resolve timesteps
    timesteps = args.timesteps or DEFAULT_TIMESTEPS[args.task]

    # Resolve log dir
    log_dir = args.log_dir or LOG_DIR_TEMPLATE.format(env_name=args.env)
    os.makedirs(log_dir, exist_ok=True)

    # Validate push_to_hub
    if args.push_to_hub and not args.hf_repo_id:
        print('ERROR: --push_to_hub requires --hf_repo_id username/repo-name')
        sys.exit(1)

    # seed_everything MUST be called before constructing the model
    seed_everything(args.seed)

    print(f'\n{"="*60}')
    print(f'CrossMill Training')
    print(f'  env:            {args.env}')
    print(f'  task:           {args.task}')
    print(f'  memory_mode:    {args.memory_mode}')
    print(f'  timesteps:      {timesteps:,}')
    print(f'  seed:           {args.seed}')
    print(f'  log_dir:        {log_dir}')
    print(f'  llm_strategist: {args.llm_strategist}')
    print(f'{"="*60}\n')

    # ---- LLM STRATEGIST (optional) ----
    strategy_bias_vec = None   # 4-dim array or None

    if args.llm_strategist:
        from crossmill.llm_strategist import (
            train_sft, save_sft_model,
            build_grpo_dataset, multi_component_reward_fn, grpo_train,
            extract_strategy_bias, save_strategy_bias,
            _ensure_reward_platform,
        )

        llm_dir = os.path.join(log_dir, 'grpo_llm')
        bias_path = os.path.join(llm_dir, 'strategy_bias.npy')

        # -- Load or train SFT model --
        if args.sft_model_path:
            print(f'\n=== LLM STRATEGIST: Loading SFT model from {args.sft_model_path} ===')
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
            base = AutoModelForCausalLM.from_pretrained(
                args.sft_model_path, device_map='auto'
            )
            llm_model = base
            llm_tokenizer = tokenizer
        else:
            print('\n=== LLM STRATEGIST: Running SFT warmup ===')
            sft_dir = os.path.join(llm_dir, 'sft')
            llm_model, llm_tokenizer = train_sft(
                env_name=args.env,
                n_examples=20,
                output_dir=sft_dir,
            )
            save_sft_model(llm_model, llm_tokenizer, sft_dir)

        # -- GRPO training --
        print('\n=== LLM STRATEGIST: Running GRPO training (50 steps) ===')
        _ensure_reward_platform(args.env, args.task)
        grpo_dataset = build_grpo_dataset(args.env, args.task, n=50)
        grpo_train(
            model=llm_model,
            tokenizer=llm_tokenizer,
            dataset=grpo_dataset,
            reward_fn=multi_component_reward_fn,
            output_dir=llm_dir,
            num_steps=50,
            env_name=args.env,
            task_id=args.task,
        )

        # -- Extract and save strategy bias --
        print('\n=== LLM STRATEGIST: Extracting strategy bias ===')
        strategy_bias_vec = extract_strategy_bias(
            llm_model, llm_tokenizer, args.env, n_queries=10
        )
        save_strategy_bias(strategy_bias_vec, bias_path)
        print(f'  strategy_bias = {strategy_bias_vec}')

        # Clean up GPU memory before PPO training
        try:
            import torch
            del llm_model
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- PRE-TRAINING BASELINE ----
    print('=== PRE-TRAINING BASELINE (heuristic agent) ===')
    pre = run_baseline(args.env, args.task, args.seed)
    print(f'  grader_score: {pre["grader_score"]:.3f}')
    print(f'  mean_reward:  {pre["mean_reward"]:.3f}')

    # ---- BUILD TRAINING ENVIRONMENT ----
    def make_env(bias=None):
        def _init():
            shim = CrossMillGymShim(
                env_name=args.env,
                task_id=args.task,
                memory_mode=args.memory_mode,
                seed=args.seed,
            )
            if bias is not None:
                shim.update_strategy(bias)
            return shim
        return _init

    monitor_path = os.path.join(log_dir, 'monitor')
    vec = DummyVecEnv([make_env(bias=strategy_bias_vec) for _ in range(N_ENVS)])
    vec = VecMonitor(vec, filename=monitor_path)

    # ---- TRAIN RecurrentPPO ----
    print('\n=== TRAINING (RecurrentPPO / MlpLstmPolicy) ===')
    model = RecurrentPPO(
        'MlpLstmPolicy', vec,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        verbose=VERBOSE,
        seed=args.seed,
        tensorboard_log=log_dir,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        tb_log_name=f'{args.env}-{args.task}-{args.memory_mode}',
    )
    train_time = time.time() - t0
    print(f'\nTraining finished in {train_time / 60:.1f} min')

    # ---- SAVE MODEL ----
    model_zip_path = os.path.join(
        log_dir, f'{args.env}-{args.task}-{args.memory_mode}-ppo.zip'
    )
    model.save(model_zip_path)
    print(f'Model saved: {model_zip_path}')

    # ---- POST-TRAINING GRADER ----
    print('\n=== POST-TRAINING GRADER ===')
    shim = CrossMillGymShim(
        env_name=args.env,
        task_id=args.task,
        memory_mode='none',   # deterministic eval — no memory noise
        seed=args.seed,
    )
    adapter = LSTMPolicyAdapter(model, shim)

    # Wrap adapter to reset LSTM state between episodes
    class EpisodeResettingAdapter:
        def __init__(self, inner):
            self.inner = inner
            self._step = 0
        def __call__(self, obs):
            # Detect episode boundaries via step_idx on the Observation.
            # Both environments reset step_idx to 0 at the start of each episode.
            step_idx = getattr(obs, 'step_idx', self._step)
            if step_idx == 0:
                self.inner.reset_state()
            self._step = step_idx
            return self.inner(obs)

    post = run_grader(
        args.env, args.task,
        policy_fn=EpisodeResettingAdapter(adapter),
        seed=args.seed,
        num_episodes=POST_GRADER_EPISODES,
    )
    print(f'  grader_score: {post["grader_score"]:.3f}')
    print(f'  mean_reward:  {post["mean_reward"]:.3f}')
    print(f'  std_reward:   {post["std_reward"]:.3f}')
    print(f'  safety_viol:  {post["safety_violation_rate"]:.3f}')
    print(f'  catastrophic: {post["catastrophic_rate"]:.3f}')

    # Print env-specific metrics
    if args.env == 'safenutri':
        print(f'  vit_c_retain: {post.get("mean_vit_c_retention", "N/A")}')
    else:
        print(f'  carbon_err:   {post.get("mean_carbon_error_pct", "N/A")}')
        print(f'  coke_rate:    {post.get("mean_coke_rate_kgpt", "N/A")}')
        print(f'  co2_kgpt:     {post.get("mean_co2_emissions_kgpt", "N/A")}')

    # ---- REWARD CURVE ----
    print('\n=== PLOTTING REWARD CURVE ===')
    curve_png_path = os.path.join(
        log_dir, f'reward_curve_{args.task}_{args.memory_mode}.png'
    )
    csv_path = monitor_path + '.monitor.csv'
    if os.path.exists(csv_path):
        plot_reward_curve(
            csv_path=csv_path,
            out_png_path=curve_png_path,
            env_name=args.env,
            task_id=args.task,
            memory_mode=args.memory_mode,
            baseline_score=pre['mean_reward'],
            final_score=post['mean_reward'],
        )
        print(f'Saved: {curve_png_path}')
    else:
        print(f'WARNING: monitor CSV not found at {csv_path} — curve skipped')

    # ---- SUMMARY JSON ----
    # IMPORTANT: This dict is the primary input to the unified grader.
    # All fields listed here are required. Do NOT remove any field.
    # mean_reward and std_reward are needed for stability computation.
    # mean_vit_c_retention / mean_carbon_error_pct are needed for QUALITY_LOW flag.
    summary = {
        # ---- Run identity ----
        'env':          args.env,
        'task_id':      args.task,
        'memory_mode':  args.memory_mode,
        'timesteps':    timesteps,
        'seed':         args.seed,
        # ---- Scores ----
        'pre_score':    pre['grader_score'],
        'post_score':   post['grader_score'],
        'delta':        post['grader_score'] - pre['grader_score'],
        # ---- Reward statistics (required for stability computation) ----
        'mean_reward':  post['mean_reward'],
        'std_reward':   post['std_reward'],
        # ---- Safety (required for anti-hacking flags) ----
        'safety_violation_rate': post['safety_violation_rate'],
        'catastrophic_rate':     post['catastrophic_rate'],
        # ---- Environment-specific quality metrics ----
        # SafeNutri: mean_vit_c_retention required for QUALITY_LOW flag
        # MegaForge: mean_carbon_error_pct required for QUALITY_LOW flag
        # Use None for fields that don't apply to this environment
        'mean_vit_c_retention':  post.get('mean_vit_c_retention'),
        'mean_carbon_error_pct': post.get('mean_carbon_error_pct'),
        'mean_coke_rate_kgpt':   post.get('mean_coke_rate_kgpt'),
        'mean_co2_emissions_kgpt': post.get('mean_co2_emissions_kgpt'),
        # ---- Artifact paths (for grader to locate reward CSV) ----
        'monitor_csv':  csv_path if os.path.exists(csv_path) else None,
        'model_zip':    model_zip_path,
        'curve_png':    curve_png_path if os.path.exists(curve_png_path) else None,
        # ---- LLM strategist (if used) ----
        'llm_strategist':  args.llm_strategist,
        'strategy_bias':   strategy_bias_vec.tolist() if strategy_bias_vec is not None else None,
    }
    summary_path = os.path.join(
        log_dir, f'summary_{args.task}_{args.memory_mode}.json'
    )
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== IMPROVEMENT ===')
    print(f'  grader_score: {pre["grader_score"]:.3f} -> '
          f'{post["grader_score"]:.3f}  '
          f'(delta {summary["delta"]:+.3f})')
    print(f'  Summary saved: {summary_path}')

    # ---- HF HUB PUSH (OPTIONAL) ----
    if args.push_to_hub:
        print(f'\n=== PUSHING TO HF HUB ({args.hf_repo_id}) ===')
        push_artifacts_to_hub(
            repo_id=args.hf_repo_id,
            model_zip_path=model_zip_path,
            curve_png_path=curve_png_path,
            summary=summary,
        )


if __name__ == '__main__':
    main()
