"""
Tests for the CrossIndustryMemory layer.

Run with:
    python -m tests.test_memory

No pytest/unittest — plain asserts only. No env instantiation — all data synthetic.
"""
from __future__ import annotations

from crossmill.memory import CrossIndustryMemory
from crossmill.memory.abstraction import abstract_observation
from crossmill.memory.classifier import classify_action
from crossmill.memory.store import MemoryStore
from crossmill.memory.retriever import Retriever
from crossmill.memory.adapter import TransferAdapter
from crossmill.memory.config import (
    PATTERN_BIAS,
    SAFETY_REGIME_BY_DIFFICULTY,
    EMA_ALPHA,
)
from crossmill.memory_interface import MemoryInterface
from crossmill.models import MemoryRecord, MemoryConfig


# ---- Helpers ----

SN_FIELDS = [
    'temperature', 'temp_gradient', 'e_coli', 'salmonella', 'listeria',
    'vitamin_c', 'folate', 'thiamine', 'pH', 'brix', 'flow_rate',
    'energy', 'equip_efficiency', 'time_in_process', 'contamination_risk',
]

MF_FIELDS = [
    'hot_metal_temp', 'hearth_temp', 'blast_temp', 'oxygen_flow',
    'carbon', 'silicon', 'sulfur', 'top_pressure', 'co_co2_ratio',
    'coke_rate', 'ore_coke_ratio', 'energy', 'production_rate',
    'wall_temp', 'thermal_stress', 'slag_basicity', 'emissions_co2',
    'equip_health',
]


def sn_obs(overrides: dict | None = None) -> dict:
    d = {k: 0.5 for k in SN_FIELDS}
    if overrides:
        d.update(overrides)
    return d


def mf_obs(overrides: dict | None = None) -> dict:
    d = {k: 0.5 for k in MF_FIELDS}
    d['step_idx'] = 0
    if overrides:
        d.update(overrides)
    return d


def make_rec(env='safenutri', pattern='gradual_ramp', conf=0.5,
             is_neg=False, difficulty='easy', reward=0.2,
             state=None) -> MemoryRecord:
    if state is None:
        state = [0.5] * 8
    return MemoryRecord(
        env_id=env,
        abstract_state=state,
        action_pattern=pattern,
        outcome_reward=reward if not is_neg else -1.0,
        is_negative=is_neg,
        safety_regime=SAFETY_REGIME_BY_DIFFICULTY[difficulty],
        confidence=conf,
        task_difficulty=difficulty,
        provenance=['test'],
    )


def fresh_store() -> MemoryStore:
    return MemoryStore(MemoryConfig())


# ---- Tests ----

def test_abstraction_safenutri():
    result = abstract_observation('safenutri', sn_obs(), 'easy')
    assert len(result) == 8, f"expected 8-dim, got {len(result)}"
    assert all(0.0 <= v <= 1.0 for v in result), f"out of [0,1]: {result}"

    extreme = abstract_observation('safenutri',
                                   sn_obs({'temperature': 1.0,
                                           'contamination_risk': 1.0}),
                                   'easy')
    assert extreme[3] == 0.0, \
        f"safety_margin at saturation should be 0.0, got {extreme[3]}"


def test_abstraction_megaforge():
    result = abstract_observation('megaforge', mf_obs(), 'easy')
    assert len(result) == 8, f"expected 8-dim, got {len(result)}"
    assert all(0.0 <= v <= 1.0 for v in result), f"out of [0,1]: {result}"

    extreme = abstract_observation('megaforge',
                                   mf_obs({'wall_temp': 1.0,
                                           'thermal_stress': 1.0}),
                                   'easy')
    assert extreme[3] == 0.0, \
        f"safety_margin at saturation should be 0.0, got {extreme[3]}"


def test_classifier_all_patterns_safenutri():
    cases = [
        ({'emergency_stop': 1.0}, 'emergency_response'),
        ({'heating_rate': 0.9}, 'rapid_correction'),
        ({'heating_rate': 0.2, 'hold_time': 0.5}, 'gradual_ramp'),
        ({'heating_rate': 0.45, 'flow_adjust': 0.5}, 'hold_steady'),
        ({'heating_rate': 0.60, 'flow_adjust': 0.62, 'cooling_rate': 0.55},
         'cautious_explore'),
    ]
    for action, expected in cases:
        got = classify_action('safenutri', action)
        assert got == expected, f"safenutri action {action}: expected {expected!r}, got {got!r}"


def test_classifier_all_patterns_megaforge():
    cases = [
        ({'emergency_cooling': 1.0}, 'emergency_response'),
        ({'temp_ramp_rate': 0.8}, 'rapid_correction'),
        ({'temp_ramp_rate': 0.2, 'coke_feed_delta': 0.5}, 'gradual_ramp'),
        ({'oxygen_flow_delta': 0.5, 'ore_feed_delta': 0.5}, 'hold_steady'),
        ({'temp_ramp_rate': 0.50, 'oxygen_flow_delta': 0.70,
          'coke_feed_delta': 0.65, 'ore_feed_delta': 0.65},
         'cautious_explore'),
    ]
    for action, expected in cases:
        got = classify_action('megaforge', action)
        assert got == expected, f"megaforge action {action}: expected {expected!r}, got {got!r}"


def test_store_promotion_easy():
    store = fresh_store()
    # 3 easy entries → fires at threshold=3
    for i in range(3):
        store.store(make_rec(pattern='gradual_ramp', difficulty='easy',
                             reward=0.3))
    assert store.stats()['episodic_count']['safenutri'] == 3
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"expected 1 promotion, got {len(promoted)}"
    assert store.stats()['semantic_count']['safenutri'] == 1
    assert store.stats()['episodic_count']['safenutri'] == 0

    # 2 more → below threshold, no promotion
    for i in range(2):
        store.store(make_rec(pattern='hold_steady', difficulty='easy',
                             reward=0.3))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 0, "2 entries should not promote"
    assert store.stats()['episodic_count']['safenutri'] == 2


def test_store_promotion_hard():
    store = fresh_store()
    # 3 hard entries → below threshold=5
    for _ in range(3):
        store.store(make_rec(pattern='cautious_explore', difficulty='hard',
                             reward=0.3))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 0, "3 hard entries should not promote"

    # 2 more → total 5, promotion fires
    for _ in range(2):
        store.store(make_rec(pattern='cautious_explore', difficulty='hard',
                             reward=0.3))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"expected 1 hard promotion, got {len(promoted)}"
    assert store.stats()['semantic_count']['safenutri'] == 1


def test_store_mixed_difficulty():
    store = fresh_store()
    # 2 easy + 2 hard = 4 entries, hardest threshold = 5 → no promotion
    for _ in range(2):
        store.store(make_rec(pattern='emergency_response', difficulty='easy',
                             reward=0.3))
    for _ in range(2):
        store.store(make_rec(pattern='emergency_response', difficulty='hard',
                             reward=0.3))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 0, "2E+2H=4 should not promote (need 5)"

    # 1 more hard → total 5 → fires
    store.store(make_rec(pattern='emergency_response', difficulty='hard',
                         reward=0.3))
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"5 mixed entries should promote, got {len(promoted)}"


def test_store_reward_threshold_filter():
    store = fresh_store()

    ok = store.store(make_rec(reward=0.005))
    assert ok is False, "reward=0.005 should be filtered"
    assert store.stats()['episodic_count']['safenutri'] == 0

    ok = store.store(make_rec(reward=0.5))
    assert ok is True, "reward=0.5 should be stored"
    assert store.stats()['episodic_count']['safenutri'] == 1

    rec = make_rec(reward=-0.6)
    ok = store.store(rec)
    assert ok is True, "reward=-0.6 should be stored as negative"
    assert rec.is_negative is True


def test_store_negative_promotion():
    store = fresh_store()
    for _ in range(3):
        r = make_rec(pattern='rapid_correction', difficulty='easy',
                     reward=-1.0, is_neg=True)
        store.store(r)
    promoted = store.try_promote('safenutri')
    assert len(promoted) == 1, f"expected 1 negative promotion, got {len(promoted)}"
    assert promoted[0].is_negative is True


def test_retriever_mode_none():
    store = fresh_store()
    store.semantic['megaforge'].append(make_rec('megaforge'))
    cfg = MemoryConfig(mode='none')
    r = Retriever(store, cfg)
    recs, scores, gate = r.retrieve('safenutri', [0.5] * 8)
    assert recs == []
    assert scores == []
    assert gate is True


def test_retriever_mode_local():
    store = fresh_store()
    store.semantic['safenutri'].append(make_rec('safenutri', conf=0.8))
    store.semantic['megaforge'].append(make_rec('megaforge', conf=0.8))
    cfg = MemoryConfig(mode='local')
    r = Retriever(store, cfg)
    recs, _, gate = r.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert len(recs) >= 1
    assert all(rec.env_id == 'safenutri' for rec in recs), \
        f"local mode leaked non-safenutri: {[r.env_id for r in recs]}"
    assert gate is False


def test_retriever_mode_cross_bidirectional():
    store = fresh_store()
    store.semantic['safenutri'].append(make_rec('safenutri', conf=0.8))
    store.semantic['megaforge'].append(make_rec('megaforge', conf=0.8))
    cfg = MemoryConfig(mode='cross', transfer_direction='bidirectional')
    r = Retriever(store, cfg)

    recs, _, _ = r.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert all(rec.env_id == 'megaforge' for rec in recs), \
        f"safenutri cross-query should return megaforge only: {[r.env_id for r in recs]}"

    recs, _, _ = r.retrieve('megaforge', [0.5] * 8, global_step=100000)
    assert all(rec.env_id == 'safenutri' for rec in recs), \
        f"megaforge cross-query should return safenutri only: {[r.env_id for r in recs]}"


def test_retriever_transfer_direction():
    store = fresh_store()
    store.semantic['megaforge'].append(make_rec('megaforge', conf=0.8))
    cfg = MemoryConfig(mode='cross', transfer_direction='steel_to_food')
    r = Retriever(store, cfg)

    recs, _, gate = r.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert len(recs) >= 1
    assert all(rec.env_id == 'megaforge' for rec in recs)

    recs, _, gate = r.retrieve('megaforge', [0.5] * 8, global_step=100000)
    assert recs == [] and gate is True


def test_retriever_confidence_warmup():
    store = fresh_store()
    rec_low  = make_rec('megaforge', conf=0.3)
    rec_high = make_rec('megaforge', conf=0.9)
    store.semantic['megaforge'].extend([rec_low, rec_high])

    cfg = MemoryConfig(
        mode='cross', transfer_direction='bidirectional',
        min_confidence=0.1, initial_min_confidence=0.5,
        confidence_warmup_steps=1000,
    )
    r = Retriever(store, cfg)

    # step=0: floor = initial_min=0.5 → only rec_high (0.9) passes
    recs0, _, _ = r.retrieve('safenutri', [0.5] * 8, global_step=0)
    assert len(recs0) == 1 and recs0[0].confidence == 0.9, \
        f"at step=0 only conf=0.9 should pass; got {[r.confidence for r in recs0]}"

    # step=1000: floor = min=0.1 → both pass
    recs1k, _, _ = r.retrieve('safenutri', [0.5] * 8, global_step=1000)
    assert len(recs1k) == 2, \
        f"at step=1000 both should pass; got {len(recs1k)}"


def test_retriever_topk():
    store = fresh_store()
    for i in range(10):
        store.semantic['megaforge'].append(
            make_rec('megaforge', conf=0.5 + i * 0.01,
                     state=[0.5 + i * 0.01] * 8)
        )

    cfg3 = MemoryConfig(mode='cross', transfer_direction='bidirectional', top_k=3)
    r3 = Retriever(store, cfg3)
    recs, _, _ = r3.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert len(recs) == 3, f"top_k=3: expected 3, got {len(recs)}"

    cfg1 = MemoryConfig(mode='cross', transfer_direction='bidirectional', top_k=1)
    r1 = Retriever(store, cfg1)
    recs, _, _ = r1.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert len(recs) == 1, f"top_k=1: expected 1, got {len(recs)}"


def test_retriever_access_count_increments():
    store = fresh_store()
    rec = make_rec('megaforge', conf=0.8)
    store.semantic['megaforge'].append(rec)
    cfg = MemoryConfig(mode='cross', transfer_direction='bidirectional')
    r = Retriever(store, cfg)

    assert rec.access_count == 0

    r.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert rec.access_count == 1

    r.retrieve('safenutri', [0.5] * 8, global_step=100000)
    assert rec.access_count == 2


def test_adapter_bias_empty():
    adapter = TransferAdapter(MemoryConfig())
    result = adapter.build_bias_vector([], [])
    assert result == [0.0] * 8, f"expected [0]*8, got {result}"


def test_adapter_bias_positive_memory():
    adapter = TransferAdapter(MemoryConfig())
    rec = make_rec(pattern='gradual_ramp', conf=0.8)
    bias = adapter.build_bias_vector([rec], [0.9])
    assert len(bias) == 8
    assert all(-0.5 <= v <= 0.5 for v in bias), f"out of clip range: {bias}"

    # Verify element-wise: weight = 0.9 * 0.8 * 1.0 = 0.72
    weight = 0.9 * 0.8 * 1.0
    pattern_vec = PATTERN_BIAS['gradual_ramp']
    for i, (b, p) in enumerate(zip(bias, pattern_vec)):
        expected_unclipped = weight * p
        expected = max(-0.5, min(0.5, expected_unclipped))
        assert abs(b - expected) < 1e-9, \
            f"dim {i}: expected {expected:.6f}, got {b:.6f}"


def test_adapter_bias_negative_inverts_sign():
    adapter = TransferAdapter(MemoryConfig())
    rec_pos = make_rec(pattern='rapid_correction', conf=0.6, is_neg=False)
    rec_neg = make_rec(pattern='rapid_correction', conf=0.6, is_neg=True)

    bias_pos = adapter.build_bias_vector([rec_pos], [0.8])
    bias_neg = adapter.build_bias_vector([rec_neg], [0.8])

    pattern_vec = PATTERN_BIAS['rapid_correction']
    for i, p in enumerate(pattern_vec):
        if p != 0.0:
            # Signs must differ on non-zero dims
            assert (bias_pos[i] > 0) != (bias_neg[i] > 0), \
                f"dim {i}: sign should invert; pos={bias_pos[i]:.4f} neg={bias_neg[i]:.4f}"


def test_adapter_ema_update_positive_signal():
    adapter = TransferAdapter(MemoryConfig())
    rec = make_rec(conf=0.6)

    adapter.track_retrieval('safenutri', [rec])
    before = rec.confidence
    info = adapter.update_confidence('safenutri', episode_reward=2.0)

    # baseline=0 → signal=(2-0)/0.01=200, clipped to 1.0
    assert info['signal'] == 1.0
    expected = (1.0 - EMA_ALPHA) * before + EMA_ALPHA * 1.0
    assert abs(rec.confidence - expected) < 1e-9, \
        f"expected {expected:.4f}, got {rec.confidence:.4f}"
    assert rec.confidence > before


def test_adapter_ema_update_negative_signal():
    adapter = TransferAdapter(MemoryConfig())
    rec = make_rec(conf=0.6)

    # Prime the baseline with a high reward
    adapter.track_retrieval('safenutri', [rec])
    adapter.update_confidence('safenutri', episode_reward=5.0)

    # Now a low reward → negative signal → confidence drops
    adapter.track_retrieval('safenutri', [rec])
    before = rec.confidence
    info = adapter.update_confidence('safenutri', episode_reward=0.1)
    assert info['signal'] < 0
    assert rec.confidence < before, \
        f"confidence should drop: before={before:.4f}, after={rec.confidence:.4f}"


def test_adapter_dedup_same_record_twice():
    adapter = TransferAdapter(MemoryConfig())
    rec = make_rec(conf=0.5)

    adapter.track_retrieval('safenutri', [rec, rec])
    info = adapter.update_confidence('safenutri', episode_reward=2.0)
    assert info['updated_count'] == 1, \
        f"dedup failed: expected updated_count=1, got {info['updated_count']}"


def test_end_to_end_synthetic():
    mem = CrossIndustryMemory()

    # Single step
    result = mem.on_step(
        env_name='safenutri',
        obs_dict=sn_obs(),
        action={'heating_rate': 0.2, 'hold_time': 0.5},
        reward=0.3,
        done=False,
        info={},
        task_id='easy',
        step_idx=0,
    )
    assert len(result.bias_vector) == 8
    assert all(-0.5 <= v <= 0.5 for v in result.bias_vector)

    # 5 identical steps with positive reward → promotion fires (easy threshold=3)
    for i in range(5):
        mem.on_step(
            env_name='safenutri',
            obs_dict=sn_obs(),
            action={'heating_rate': 0.2, 'hold_time': 0.5},
            reward=0.5,
            done=False,
            info={},
            task_id='easy',
            step_idx=i + 1,
        )
    stats = mem.get_stats()
    assert stats['semantic_count']['safenutri'] >= 1, \
        f"expected at least 1 promotion after 6 positive steps; got {stats}"

    mem.on_episode_end('safenutri', episode_reward=3.0, episode_id='ep_001')

    stats = mem.get_stats()
    required_keys = {
        'episodic_count', 'semantic_count', 'total_retrievals',
        'total_suppressions', 'confidence_distribution', 'global_step',
    }
    assert required_keys <= stats.keys(), \
        f"missing keys: {required_keys - stats.keys()}"
    assert stats['global_step'] == 6


def test_memory_interface_compliance():
    mem = CrossIndustryMemory()
    assert isinstance(mem, MemoryInterface), \
        "CrossIndustryMemory must be a MemoryInterface subclass"
    # Verify all abstract methods are concretely overridden (not abstract)
    import inspect
    for method_name in ('on_step', 'on_episode_end', 'get_config', 'get_stats'):
        method = getattr(mem, method_name)
        assert callable(method), f"{method_name} must be callable"
        assert not getattr(method, '__isabstractmethod__', False), \
            f"{method_name} must be concretely implemented"


# ---- Runner ----
if __name__ == '__main__':
    tests = [
        test_abstraction_safenutri,
        test_abstraction_megaforge,
        test_classifier_all_patterns_safenutri,
        test_classifier_all_patterns_megaforge,
        test_store_promotion_easy,
        test_store_promotion_hard,
        test_store_mixed_difficulty,
        test_store_reward_threshold_filter,
        test_store_negative_promotion,
        test_retriever_mode_none,
        test_retriever_mode_local,
        test_retriever_mode_cross_bidirectional,
        test_retriever_transfer_direction,
        test_retriever_confidence_warmup,
        test_retriever_topk,
        test_retriever_access_count_increments,
        test_adapter_bias_empty,
        test_adapter_bias_positive_memory,
        test_adapter_bias_negative_inverts_sign,
        test_adapter_ema_update_positive_signal,
        test_adapter_ema_update_negative_signal,
        test_adapter_dedup_same_record_twice,
        test_end_to_end_synthetic,
        test_memory_interface_compliance,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            failed += 1

    print(f'\n{passed} passed, {failed} failed, {passed + failed} total')
    if failed == 0:
        print('ALL TESTS PASSED')
    else:
        print('SOME TESTS FAILED')
        exit(1)
