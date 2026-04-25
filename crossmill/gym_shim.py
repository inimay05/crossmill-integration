import numpy as np
import gymnasium as gym
from gymnasium import spaces
from crossmill.platform import CrossMillPlatform
from crossmill.memory_interface import NoOpMemory
from crossmill.config import ENVIRONMENTS, AUGMENTED_OBS_DIM
from crossmill.training_config import STRATEGY_DIM

class CrossMillGymShim(gym.Env):
    """
    Gymnasium-compatible wrapper around CrossMillPlatform.

    Wraps one environment (safenutri or megaforge) at a time.
    Exposes the augmented observation space (23-dim or 26-dim) and the
    correct action space for the selected environment.

    Used by stable-baselines3 RecurrentPPO for training.
    """

    metadata = {'render_modes': []}

    def __init__(self, env_name: str, task_id: str = 'easy',
                 memory_mode: str = 'cross', seed: int = 42):
        """
        Args:
          env_name: 'safenutri' or 'megaforge'
          task_id: 'easy', 'medium', or 'hard'
          memory_mode: 'none', 'local', or 'cross'
          seed: random seed
        """
        super().__init__()
        assert env_name in ENVIRONMENTS, f"Unknown env: {env_name}"
        self.env_name = env_name
        self.task_id  = task_id
        self.seed_val = seed

        # Build platform with appropriate memory
        if memory_mode == 'none':
            memory = NoOpMemory()
        elif memory_mode == 'local':
            from crossmill.memory import CrossIndustryMemory
            from crossmill.models import MemoryConfig
            memory = CrossIndustryMemory(MemoryConfig(mode='local'))
        else:  # 'cross' — default
            from crossmill.memory import CrossIndustryMemory
            memory = CrossIndustryMemory()

        kwargs = {
            'memory': memory,
            'seed': seed,
        }
        if env_name == 'safenutri':
            kwargs['safenutri_task'] = task_id
        else:
            kwargs['megaforge_task'] = task_id

        self.platform = CrossMillPlatform(**kwargs)

        # ---- Observation space ----
        obs_dim = AUGMENTED_OBS_DIM[env_name]
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,   # raw obs in [0,1], bias in [-0.5,+0.5]
            shape=(obs_dim,), dtype=np.float32,
        )

        # ---- Action space ----
        action_dim = ENVIRONMENTS[env_name]['action_dim']
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(action_dim,), dtype=np.float32,
        )

        self._last_obs = None
        self.strategy_bias = np.zeros(STRATEGY_DIM, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return (obs, info).
        Gymnasium API: reset returns (observation, info_dict).
        """
        if seed is not None:
            self.seed_val = seed
        obs = self.platform.reset(self.env_name, seed=self.seed_val)
        self._last_obs = obs
        augmented = np.concatenate([obs, self.strategy_bias]).astype(np.float32)
        self._last_obs = augmented
        return augmented, {}

    def step(self, action):
        """
        Step the environment.
        action: numpy array from SB3 policy, shape (action_dim,)
        Returns: (obs, reward, terminated, truncated, info)
        Gymnasium API: step returns 5-tuple with terminated and truncated separate.
        """
        result = self.platform.step(self.env_name, action.tolist())
        obs        = result['observation'].astype(np.float32)
        reward     = float(result['reward'])
        terminated = bool(result['done'] and not result['truncated'])
        truncated  = bool(result['truncated'])
        info       = result['info']
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def update_strategy(self, strategy_bias):
        self.strategy_bias = np.clip(
            strategy_bias.astype(np.float32), -1.0, 1.0
        )



    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    rng = np.random.default_rng(42)

    for env_name, exp_obs, exp_act in [('safenutri', (23,), (8,)),
                                        ('megaforge',  (26,), (10,))]:
        env = CrossMillGymShim(env_name, task_id='easy', memory_mode='none', seed=42)

        assert env.observation_space.shape == exp_obs, (
            f"{env_name}: obs space {env.observation_space.shape} != {exp_obs}"
        )
        assert env.action_space.shape == exp_act, (
            f"{env_name}: act space {env.action_space.shape} != {exp_act}"
        )

        obs, info = env.reset()
        assert obs.shape == exp_obs, f"{env_name} reset obs shape {obs.shape} != {exp_obs}"
        assert obs.dtype == np.float32, f"{env_name} reset obs dtype {obs.dtype} != float32"

        for _ in range(5):
            action = rng.random(exp_act[0]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == exp_obs, f"{env_name} step obs shape {obs.shape} != {exp_obs}"

        # Run a full episode to confirm done fires within 300 steps
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 300:
            action = rng.random(exp_act[0]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == exp_obs
            done = terminated or truncated
            steps += 1
        assert done, f"{env_name}: episode did not terminate within 300 steps"

        env.close()

    print('CrossMillGymShim OK: safenutri (23,8) and megaforge (26,10)')
