from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from crossmill.config import BIAS_VECTOR_DIM
from crossmill.models import MemoryConfig, TransferResult


class MemoryInterface(ABC):
    """
    Contract that any memory layer must implement.
    The integration layer (CrossMillPlatform) calls only these methods.
    """

    @abstractmethod
    def on_step(
        self,
        env_name: str,
        obs_dict: dict,
        action: Any,
        reward: float,
        done: bool,
        info: dict,
        task_id: str,
        step_idx: int,
    ) -> TransferResult:
        """
        Called after every env.step().

        Responsibilities:
          1. Convert obs_dict to abstract state (using the appropriate
             per-environment abstraction function).
          2. Classify the action into an action_pattern label.
          3. Store the experience in the episodic buffer if it meets
             the reward threshold.
          4. Check if promotion from episodic to semantic should fire.
          5. Query the retriever for relevant cross-industry memories.
          6. Build the bias vector from retrieved memories.

        Args:
          env_name: 'safenutri' or 'megaforge'
          obs_dict: the normalised observation dict from the environment
                    (15 or 18 float fields, all in [0,1])
          action: the raw action taken (dict, pydantic, or numpy array)
          reward: the scalar reward received this step
          done: whether the episode ended
          info: the info dict from StepResponse
          task_id: 'easy', 'medium', or 'hard'
          step_idx: current step within the episode

        Returns:
          TransferResult with bias_vector, retrieved_memories,
          retrieval_scores, and gate_active.
        """
        pass

    @abstractmethod
    def on_episode_end(
        self,
        env_name: str,
        episode_reward: float,
        episode_id: str,
    ) -> None:
        """
        Called when an episode finishes (done=True).

        Responsibilities:
          1. Compute reward delta vs. baseline (rolling mean of last
             BASELINE_WINDOW episodes).
          2. Update confidence on all memories retrieved during this episode
             via the EMA.
          3. Log the episode_id for provenance tracking.

        Args:
          env_name: which environment just finished
          episode_reward: total reward accumulated over the episode
          episode_id: unique string identifying this episode (for provenance)
        """
        pass

    @abstractmethod
    def get_config(self) -> MemoryConfig:
        """Return the current memory configuration."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return diagnostic stats for logging / grader:
          - episodic_count: {safenutri: int, megaforge: int}
          - semantic_count: {safenutri: int, megaforge: int}
          - total_retrievals: int
          - total_suppressions: int  (confidence gate blocked)
          - confidence_distribution: list[float]  (all semantic confidences)
        """
        pass


class NoOpMemory(MemoryInterface):
    """
    Pass-through memory that does nothing. Used when memory is explicitly
    disabled (mode='none') or when testing the integration layer before the
    real memory layer is built.

    on_step always returns a TransferResult with:
      - bias_vector: [0.0] * BIAS_VECTOR_DIM
      - retrieved_memories: []
      - retrieval_scores: []
      - gate_active: False

    on_episode_end does nothing.
    get_config returns a MemoryConfig with mode='none'.
    get_stats returns all-zero counts.
    """

    def __init__(self) -> None:
        self._config = MemoryConfig(mode='none')

    def on_step(
        self,
        env_name: str,
        obs_dict: dict,
        action: Any,
        reward: float,
        done: bool,
        info: dict,
        task_id: str,
        step_idx: int,
    ) -> TransferResult:
        return TransferResult(
            bias_vector=[0.0] * BIAS_VECTOR_DIM,
            retrieved_memories=[],
            retrieval_scores=[],
            gate_active=False,
        )

    def on_episode_end(
        self,
        env_name: str,
        episode_reward: float,
        episode_id: str,
    ) -> None:
        pass

    def get_config(self) -> MemoryConfig:
        return self._config

    def get_stats(self) -> dict:
        return {
            'episodic_count': {'safenutri': 0, 'megaforge': 0},
            'semantic_count': {'safenutri': 0, 'megaforge': 0},
            'total_retrievals': 0,
            'total_suppressions': 0,
            'confidence_distribution': [],
        }


if __name__ == '__main__':
    mem = NoOpMemory()

    obs_dict = {f'field_{i}': 0.0 for i in range(15)}
    result = mem.on_step(
        env_name='safenutri',
        obs_dict=obs_dict,
        action={},
        reward=0.5,
        done=False,
        info={},
        task_id='easy',
        step_idx=0,
    )
    assert len(result.bias_vector) == 8, 'bias_vector length mismatch'
    assert all(v == 0.0 for v in result.bias_vector), 'bias_vector not all zeros'
    assert result.retrieved_memories == [], 'expected no retrieved memories'
    assert result.retrieval_scores == [], 'expected no retrieval scores'
    assert result.gate_active is False, 'gate_active should be False'

    mem.on_episode_end(env_name='safenutri', episode_reward=5.0, episode_id='ep_001')

    stats = mem.get_stats()
    print('get_stats():', stats)
    assert stats['episodic_count'] == {'safenutri': 0, 'megaforge': 0}
    assert stats['semantic_count'] == {'safenutri': 0, 'megaforge': 0}
    assert stats['total_retrievals'] == 0
    assert stats['total_suppressions'] == 0
    assert stats['confidence_distribution'] == []

    print('NoOpMemory contract OK')
