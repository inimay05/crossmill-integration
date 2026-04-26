from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import numpy as np

from crossmill.config import (
    AUGMENTED_OBS_DIM,
    BASELINE_WINDOW,
    BIAS_VECTOR_DIM,
    ENVIRONMENTS,
)
from crossmill.models import MemoryConfig, TransferResult
from crossmill.memory_interface import MemoryInterface, NoOpMemory
from crossmill.augmentation import (
    OBS_FIELDS,
    augment_observation,
    obs_to_vector,
    zero_bias,
)


def _load_env_class(env_name: str) -> type:
    """
    Load an environment class from its absolute file path.
    Uses importlib.util.spec_from_file_location so both environments can be
    loaded even though they share the module name 'app.environment'.
    Each is given a unique module name: 'safenutri_env_module' or
    'megaforge_env_module' to avoid collision in sys.modules.
    """
    env_cfg = ENVIRONMENTS[env_name]
    file_path = env_cfg['env_file']
    class_name = env_cfg['class_name']
    module_name = f'{env_name}_env_module'

    if module_name in sys.modules:
        return getattr(sys.modules[module_name], class_name)

    # environment.py does `from app.config import ...` — so the env's repo
    # root must be on sys.path for the load. Both repos use the package name
    # `app`, so we purge any prior `app.*` entries before and after loading
    # this env, so the other env can load its own `app.*` cleanly.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))

    app_keys_before = [k for k in sys.modules if k == 'app' or k.startswith('app.')]
    for k in app_keys_before:
        del sys.modules[k]

    sys.path.insert(0, repo_root)
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        env_class = getattr(module, class_name)
    finally:
        if sys.path and sys.path[0] == repo_root:
            sys.path.pop(0)
        for k in [k for k in sys.modules if k == 'app' or k.startswith('app.')]:
            del sys.modules[k]

    return env_class


class CrossMillPlatform:
    """
    CrossMill — unified entry point for cross-industry RL.

    Holds both environments and the memory layer.
    All observations are augmented to a fixed size (raw + bias vector)
    regardless of memory mode.

    By default, a fully configured CrossIndustryMemory is instantiated
    with mode='cross' and bidirectional transfer active. Pass memory=None
    to use NoOpMemory (zero bias, no transfer) for experimental baselines,
    or pass a custom MemoryInterface implementation.
    """

    def __init__(
        self,
        memory: MemoryInterface | None | str = 'default',
        safenutri_task: str = 'easy',
        megaforge_task: str = 'easy',
        seed: int | None = None,
    ):
        """
        Args:
          memory: a MemoryInterface implementation, 'default' to instantiate
                  CrossIndustryMemory with default config (cross-industry
                  transfer active), or None to use NoOpMemory (zero bias,
                  no transfer — for experimental baselines only).
          safenutri_task: task difficulty for SafeNutri ('easy'/'medium'/'hard')
          megaforge_task: task difficulty for MegaForge ('easy'/'medium'/'hard')
          seed: random seed passed to both environments
        """
        if memory == 'default':
            try:
                from crossmill.memory import CrossIndustryMemory
                self.memory = CrossIndustryMemory()
            except ImportError:
                self.memory = NoOpMemory()
        elif memory is None:
            self.memory = NoOpMemory()
        else:
            self.memory = memory

        self.envs = {}
        for env_name, env_cfg in ENVIRONMENTS.items():
            cls = _load_env_class(env_name)
            task = safenutri_task if env_name == 'safenutri' else megaforge_task
            self.envs[env_name] = cls(task_id=task, seed=seed)

        self._episode_rewards = {name: 0.0 for name in ENVIRONMENTS}
        self._episode_steps = {name: 0 for name in ENVIRONMENTS}
        self._episode_counts = {name: 0 for name in ENVIRONMENTS}
        self._global_step = 0

        self._task_ids = {
            'safenutri': safenutri_task,
            'megaforge': megaforge_task,
        }

    def reset(self, env_name: str, seed: int | None = None) -> np.ndarray:
        """
        Reset a specific environment and return the augmented observation.
        Returns np.ndarray of shape (AUGMENTED_OBS_DIM[env_name],)
        """
        assert env_name in self.envs, f'Unknown env: {env_name}'
        obs = self.envs[env_name].reset(seed=seed)
        self._episode_rewards[env_name] = 0.0
        self._episode_steps[env_name] = 0
        raw_vec = obs_to_vector(env_name, obs)
        augmented = augment_observation(raw_vec, zero_bias(), env_name)
        return augmented

    def step(self, env_name: str, action: Any) -> dict:
        """
        Step a specific environment, apply memory augmentation, return result.

        Returns dict with keys:
          'observation': np.ndarray of shape (AUGMENTED_OBS_DIM[env_name],)
          'reward':      float
          'done':        bool
          'truncated':   bool
          'info':        dict (from environment, plus 'transfer_result' key)
          'raw_obs':     the original Pydantic Observation from the env
        """
        assert env_name in self.envs, f'Unknown env: {env_name}'
        env = self.envs[env_name]

        response = env.step(action)
        # Environment may return reward as a plain float or as an object with .value
        _raw_reward = response.reward
        reward_value = _raw_reward.value if hasattr(_raw_reward, 'value') else float(_raw_reward)

        self._episode_rewards[env_name] += reward_value
        self._episode_steps[env_name] += 1
        self._global_step += 1

        obs_dict = response.observation.model_dump()

        transfer = self.memory.on_step(
            env_name=env_name,
            obs_dict=obs_dict,
            action=action,
            reward=reward_value,
            done=response.done,
            info=response.info,
            task_id=self._task_ids[env_name],
            step_idx=self._episode_steps[env_name],
        )

        raw_vec = obs_to_vector(env_name, response.observation)
        augmented = augment_observation(raw_vec, transfer.bias_vector, env_name)

        info = dict(response.info) if isinstance(response.info, dict) else {}
        info['transfer_result'] = transfer

        result = {
            'observation': augmented,
            'reward': reward_value,
            'done': response.done,
            'truncated': response.truncated,
            'info': info,
            'raw_obs': response.observation,
        }

        if response.done:
            episode_id = (
                f'{env_name}_{self._task_ids[env_name]}'
                f'_ep{self._episode_counts[env_name]}'
                f'_step{self._global_step}'
            )
            self.memory.on_episode_end(
                env_name=env_name,
                episode_reward=self._episode_rewards[env_name],
                episode_id=episode_id,
            )
            self._episode_counts[env_name] += 1

        return result

    def get_env(self, env_name: str):
        """Direct access to an environment instance (for grader use)."""
        return self.envs[env_name]

    def get_memory_stats(self) -> dict:
        return self.memory.get_stats()

    def get_augmented_obs_dim(self, env_name: str) -> int:
        return AUGMENTED_OBS_DIM[env_name]

    @property
    def global_step(self) -> int:
        return self._global_step


if __name__ == '__main__':
    platform = CrossMillPlatform(memory=None, seed=42)

    obs = platform.reset('safenutri', seed=42)
    print(f'SafeNutri reset: obs shape = {obs.shape}')
    assert obs.shape == (23,), f'Expected (23,), got {obs.shape}'

    rng = np.random.default_rng(0)
    total_r = 0.0
    steps = 0
    while True:
        action = rng.random(8).tolist()
        result = platform.step('safenutri', action)
        total_r += result['reward']
        steps += 1
        assert result['observation'].shape == (23,)
        if result['done']:
            break
    print(f'SafeNutri episode: steps={steps}, reward={total_r:.3f}')

    obs = platform.reset('megaforge', seed=42)
    print(f'MegaForge reset: obs shape = {obs.shape}')
    assert obs.shape == (26,), f'Expected (26,), got {obs.shape}'

    total_r = 0.0
    steps = 0
    while True:
        action = rng.random(10).tolist()
        result = platform.step('megaforge', action)
        total_r += result['reward']
        steps += 1
        assert result['observation'].shape == (26,)
        if result['done']:
            break
    print(f'MegaForge episode: steps={steps}, reward={total_r:.3f}')

    stats = platform.get_memory_stats()
    print(f'Memory stats: {stats}')
    print(f'Global step count: {platform.global_step}')
    print('CrossMillPlatform OK')
