from typing import Any, Optional
from crossmill.memory_interface import MemoryInterface
from crossmill.models import MemoryRecord, MemoryConfig, TransferResult
from crossmill.memory.abstraction import abstract_observation
from crossmill.memory.classifier import classify_action
from crossmill.memory.store import MemoryStore
from crossmill.memory.retriever import Retriever
from crossmill.memory.adapter import TransferAdapter
from crossmill.memory.config import (
    SAFETY_REGIME_BY_DIFFICULTY,
    BIAS_VECTOR_DIM,
    DEFAULT_CONFIG_DICT,
)


class CrossIndustryMemory(MemoryInterface):
    """
    The complete cross-industry memory layer.

    Implements MemoryInterface. Wired into CrossMillPlatform as the default
    memory argument. Can be instantiated with no arguments to get a fully
    configured memory with mode='cross', bidirectional transfer, and
    difficulty-aware promotion.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        if config is None:
            config = MemoryConfig(**DEFAULT_CONFIG_DICT)
        self._config = config
        self._store = MemoryStore(config)
        self._retriever = Retriever(self._store, config)
        self._adapter = TransferAdapter(config)
        self._global_step = 0
        self._total_retrievals = 0
        self._total_suppressions = 0

    def on_step(self, env_name: str, obs_dict: dict, action: Any,
                reward: float, done: bool, info: dict,
                task_id: str, step_idx: int) -> TransferResult:
        """
        The main per-step hook. Runs in this order:
          1. Abstract the observation.
          2. Classify the action.
          3. Store the experience in the episodic buffer (if reward meaningful).
          4. Try to promote episodic -> semantic.
          5. Retrieve top-k cross-industry memories.
          6. Build the bias vector and return a TransferResult.
        """
        self._global_step += 1

        abstract_state = abstract_observation(env_name, obs_dict, task_id)

        action_pattern = classify_action(env_name, action)

        safety_regime = SAFETY_REGIME_BY_DIFFICULTY.get(task_id, 'moderate')
        record = MemoryRecord(
            env_id=env_name,
            abstract_state=abstract_state,
            action_pattern=action_pattern,
            outcome_reward=float(reward),
            is_negative=False,
            safety_regime=safety_regime,
            confidence=0.5,
            access_count=0,
            task_difficulty=task_id,
            provenance=[f"{env_name}_{task_id}_step{self._global_step}"],
            timestamp=self._global_step,
        )
        self._store.store(record)

        self._store.try_promote(env_name)

        retrieved, scores, gate_active = self._retriever.retrieve(
            current_env=env_name,
            abstract_state=abstract_state,
            global_step=self._global_step,
        )

        if gate_active:
            self._total_suppressions += 1
        else:
            self._total_retrievals += 1

        self._adapter.track_retrieval(env_name, retrieved)

        return self._adapter.to_transfer_result(
            records=retrieved, scores=scores, gate_active=gate_active,
        )

    def on_episode_end(self, env_name: str, episode_reward: float,
                       episode_id: str) -> None:
        """
        Called by CrossMillPlatform when an episode terminates.
        Updates confidence on all memories retrieved during this episode.
        """
        self._adapter.update_confidence(env_name, episode_reward)

    def get_config(self) -> MemoryConfig:
        return self._config

    def get_stats(self) -> dict:
        store_stats = self._store.stats()
        return {
            'episodic_count': store_stats['episodic_count'],
            'semantic_count': store_stats['semantic_count'],
            'total_retrievals': self._total_retrievals,
            'total_suppressions': self._total_suppressions,
            'confidence_distribution': store_stats['confidence_distribution'],
            'global_step': self._global_step,
        }


__all__ = ['CrossIndustryMemory']


if __name__ == '__main__':
    # ---- Instantiate with defaults ----
    mem = CrossIndustryMemory()
    cfg = mem.get_config()
    assert cfg.mode == 'cross', f"expected mode='cross', got {cfg.mode}"
    assert cfg.transfer_direction == 'bidirectional', \
        f"expected bidirectional, got {cfg.transfer_direction}"
    print(f"Instantiated: mode={cfg.mode}, direction={cfg.transfer_direction} — OK")

    # ---- SafeNutri obs template (15 fields) ----
    SN_FIELDS = [
        'temperature', 'temp_gradient', 'e_coli', 'salmonella', 'listeria',
        'vitamin_c', 'folate', 'thiamine', 'pH', 'brix', 'flow_rate',
        'energy', 'equip_efficiency', 'time_in_process', 'contamination_risk',
    ]
    def sn_obs():
        return {k: 0.5 for k in SN_FIELDS}

    # ---- 10 SafeNutri steps with varied actions and rewards ----
    sn_actions = [
        {'emergency_stop': 1.0},                               # emergency
        {'heating_rate': 0.9},                                 # rapid
        {'heating_rate': 0.2, 'hold_time': 0.5},               # gradual
        {'heating_rate': 0.45, 'flow_adjust': 0.5},            # hold
        {'heating_rate': 0.60, 'flow_adjust': 0.62},           # cautious
        {'heating_rate': 0.2, 'hold_time': 0.5},               # gradual
        {'heating_rate': 0.9},                                 # rapid
        {'heating_rate': 0.2, 'hold_time': 0.5},               # gradual
        {'heating_rate': 0.45, 'flow_adjust': 0.5},            # hold
        {'emergency_stop': 1.0},                               # emergency
    ]
    sn_rewards = [0.5, 0.0, 0.3, 0.001, -1.0, 0.4, -0.8, 0.2, 0.005, 0.6]

    for i, (act, rew) in enumerate(zip(sn_actions, sn_rewards)):
        result = mem.on_step(
            env_name='safenutri', obs_dict=sn_obs(), action=act,
            reward=rew, done=False, info={}, task_id='easy', step_idx=i,
        )
        assert len(result.bias_vector) == BIAS_VECTOR_DIM

    stats = mem.get_stats()
    total = stats['total_retrievals'] + stats['total_suppressions']
    assert total == 10, f"expected 10 retrieval calls, got {total}"
    assert stats['episodic_count']['safenutri'] > 0, \
        f"expected some episodic entries, got {stats['episodic_count']}"
    print(f"After 10 SafeNutri steps: episodic={stats['episodic_count']}, "
          f"retrievals={stats['total_retrievals']}, "
          f"suppressions={stats['total_suppressions']}")

    # ---- MegaForge 18 fields ----
    MF_FIELDS = [
        'hot_metal_temp', 'hearth_temp', 'blast_temp', 'oxygen_flow',
        'carbon', 'silicon', 'sulfur', 'top_pressure', 'co_co2_ratio',
        'coke_rate', 'ore_coke_ratio', 'energy', 'production_rate',
        'wall_temp', 'thermal_stress', 'slag_basicity', 'emissions_co2',
        'equip_health',
    ]
    def mf_obs():
        d = {k: 0.5 for k in MF_FIELDS}
        d['step_idx'] = 0
        return d

    mf_actions = [
        {'emergency_cooling': 1.0},
        {'temp_ramp_rate': 0.8},
        {'temp_ramp_rate': 0.2, 'coke_feed_delta': 0.5},
        {'oxygen_flow_delta': 0.5, 'ore_feed_delta': 0.5},
        {'temp_ramp_rate': 0.50, 'oxygen_flow_delta': 0.70,
         'coke_feed_delta': 0.65, 'ore_feed_delta': 0.65},
        {'temp_ramp_rate': 0.2, 'coke_feed_delta': 0.5},
        {'temp_ramp_rate': 0.8},
        {'temp_ramp_rate': 0.2, 'coke_feed_delta': 0.5},
        {'oxygen_flow_delta': 0.5, 'ore_feed_delta': 0.5},
        {'emergency_cooling': 1.0},
    ]
    mf_rewards = [0.3, -0.7, 0.5, 0.002, -0.6, 0.4, 0.3, 0.25, 0.001, 0.8]

    for i, (act, rew) in enumerate(zip(mf_actions, mf_rewards)):
        result = mem.on_step(
            env_name='megaforge', obs_dict=mf_obs(), action=act,
            reward=rew, done=False, info={}, task_id='easy', step_idx=i,
        )

    stats = mem.get_stats()
    assert stats['episodic_count']['megaforge'] > 0, \
        f"expected megaforge entries, got {stats['episodic_count']}"
    print(f"After 10 MegaForge steps: episodic={stats['episodic_count']}")

    # ---- Force promotion: 5 identical gradual_ramp positives on MegaForge ----
    prev_semantic = stats['semantic_count']['megaforge']
    for i in range(5):
        mem.on_step(
            env_name='megaforge', obs_dict=mf_obs(),
            action={'temp_ramp_rate': 0.2, 'coke_feed_delta': 0.5},
            reward=0.5, done=False, info={}, task_id='easy', step_idx=i,
        )
    stats = mem.get_stats()
    assert stats['semantic_count']['megaforge'] > prev_semantic, \
        f"expected semantic increase; was {prev_semantic}, now {stats['semantic_count']['megaforge']}"
    print(f"Forced promotion: megaforge semantic {prev_semantic} -> "
          f"{stats['semantic_count']['megaforge']}")

    # ---- Retrieve from SafeNutri — should (after warmup) see MegaForge memory ----
    # Warmup is 50000 steps by default, so at ~25 global steps the floor is
    # still ~0.4. The promoted record's confidence = 0.5+bonus = 0.6, which
    # passes floor=0.4. Bidirectional + SafeNutri queries MegaForge semantic.
    result = mem.on_step(
        env_name='safenutri', obs_dict=sn_obs(),
        action={'heating_rate': 0.2, 'hold_time': 0.5},
        reward=0.3, done=False, info={}, task_id='easy', step_idx=0,
    )
    print(f"Cross-retrieval into SafeNutri: "
          f"retrieved={len(result.retrieved_memories)}, "
          f"gate_active={result.gate_active}, "
          f"bias_nonzero={any(v != 0 for v in result.bias_vector)}")

    # ---- on_episode_end ----
    mem.on_episode_end('safenutri', episode_reward=5.0, episode_id='ep_sn_001')
    print(f"on_episode_end('safenutri', reward=5.0): no error — OK")

    # ---- Final stats ----
    final = mem.get_stats()
    print(f"\nFinal stats:")
    print(f"  global_step:        {final['global_step']}")
    print(f"  episodic_count:     {final['episodic_count']}")
    print(f"  semantic_count:     {final['semantic_count']}")
    print(f"  total_retrievals:   {final['total_retrievals']}")
    print(f"  total_suppressions: {final['total_suppressions']}")
    print(f"  confidence_dist:    {[f'{c:.3f}' for c in final['confidence_distribution']]}")

    print("\nCrossIndustryMemory OK: full pipeline wired")
