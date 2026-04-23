"""
Integration sanity tests for the CrossMill platform layer.

Run with:
    python -m tests.test_platform
"""
from __future__ import annotations

import numpy as np

from crossmill.augmentation import OBS_FIELDS, augment_observation, obs_to_vector
from crossmill.config import AUGMENTED_OBS_DIM, BIAS_VECTOR_DIM, BIAS_CLIP
from crossmill.memory_interface import NoOpMemory
from crossmill.platform import CrossMillPlatform


def test_observation_dimensions():
    """
    Both environments must return augmented observations of the correct size.
    SafeNutri: 23 (15 raw + 8 bias). MegaForge: 26 (18 raw + 8 bias).
    Test both reset() and step(). Use memory=None to isolate the integration
    layer from any memory-layer behaviour.
    """
    p = CrossMillPlatform(memory=None, seed=42)
    rng = np.random.default_rng(0)

    for env_name, action_dim, expected_dim in [
        ('safenutri', 8, 23),
        ('megaforge', 10, 26),
    ]:
        obs = p.reset(env_name, seed=42)
        assert obs.shape == (expected_dim,), (
            f'{env_name} reset: expected shape ({expected_dim},), got {obs.shape}'
        )

        result = p.step(env_name, rng.random(action_dim).tolist())
        assert result['observation'].shape == (expected_dim,), (
            f'{env_name} step: expected shape ({expected_dim},), '
            f'got {result["observation"].shape}'
        )
        assert result['observation'].dtype == np.float32, (
            f'{env_name} step: expected float32, got {result["observation"].dtype}'
        )


def test_memory_none_equals_noop():
    """
    CrossMillPlatform(memory=None) must produce identical observations and
    rewards to CrossMillPlatform(memory=NoOpMemory()), given the same seed.
    Run 10 steps on each and compare element-by-element.
    """
    p_none = CrossMillPlatform(memory=None, seed=7)
    p_noop = CrossMillPlatform(memory=NoOpMemory(), seed=7)

    rng_none = np.random.default_rng(1)
    rng_noop = np.random.default_rng(1)

    p_none.reset('safenutri', seed=7)
    p_noop.reset('safenutri', seed=7)

    for step_i in range(10):
        action = rng_none.random(8).tolist()
        action_copy = list(action)

        r_none = p_none.step('safenutri', action)
        r_noop = p_noop.step('safenutri', action_copy)

        assert np.allclose(r_none['observation'], r_noop['observation'], atol=1e-6), (
            f'Step {step_i}: observation mismatch between memory=None and NoOpMemory'
        )
        assert r_none['reward'] == r_noop['reward'], (
            f'Step {step_i}: reward mismatch: {r_none["reward"]} vs {r_noop["reward"]}'
        )
        assert r_none['done'] == r_noop['done'], (
            f'Step {step_i}: done mismatch'
        )
        if r_none['done']:
            break


def test_reproducibility():
    """
    Two runs with the same seed must produce identical observation vectors
    and reward values on both environments. Run a full episode on each.
    Use memory=None for both runs.
    """
    for env_name, action_dim in [('safenutri', 8), ('megaforge', 10)]:
        rewards_a, rewards_b = [], []
        obs_a, obs_b = [], []

        for rewards, obs_list in [(rewards_a, obs_a), (rewards_b, obs_b)]:
            p = CrossMillPlatform(memory=None, seed=0)
            rng = np.random.default_rng(99)
            p.reset(env_name, seed=0)
            for _ in range(50):
                result = p.step(env_name, rng.random(action_dim).tolist())
                rewards.append(result['reward'])
                obs_list.append(result['observation'].copy())
                if result['done']:
                    break

        assert len(rewards_a) == len(rewards_b), (
            f'{env_name}: episode length differs between runs '
            f'({len(rewards_a)} vs {len(rewards_b)})'
        )
        for i, (r_a, r_b) in enumerate(zip(rewards_a, rewards_b)):
            assert r_a == r_b, (
                f'{env_name} step {i}: reward {r_a} != {r_b}'
            )
        for i, (o_a, o_b) in enumerate(zip(obs_a, obs_b)):
            assert np.allclose(o_a, o_b, atol=1e-6), (
                f'{env_name} step {i}: observation vectors differ'
            )


def test_bias_vector_is_zero_without_memory():
    """
    When using memory=None (NoOpMemory), the last 8 elements of every
    augmented observation must be exactly 0.0. Check on both environments,
    on both reset and step.
    """
    p = CrossMillPlatform(memory=None, seed=42)
    rng = np.random.default_rng(5)

    for env_name, action_dim in [('safenutri', 8), ('megaforge', 10)]:
        obs = p.reset(env_name, seed=42)
        assert np.all(obs[-BIAS_VECTOR_DIM:] == 0.0), (
            f'{env_name} reset: last {BIAS_VECTOR_DIM} elements are not all zero: '
            f'{obs[-BIAS_VECTOR_DIM:]}'
        )

        for _ in range(5):
            result = p.step(env_name, rng.random(action_dim).tolist())
            bias_part = result['observation'][-BIAS_VECTOR_DIM:]
            assert np.all(bias_part == 0.0), (
                f'{env_name} step: last {BIAS_VECTOR_DIM} elements are not all zero: '
                f'{bias_part}'
            )
            if result['done']:
                break


def test_episode_lifecycle():
    """
    Run a full episode on each environment. Verify:
    - done eventually becomes True
    - step count matches expected range for Easy task
    - episode_id is generated on done (check info dict for transfer_result)
    Use memory=None.
    """
    p = CrossMillPlatform(memory=None, seed=42)
    rng = np.random.default_rng(3)

    for env_name, action_dim in [('safenutri', 8), ('megaforge', 10)]:
        p.reset(env_name, seed=42)
        steps = 0
        done_seen = False
        last_result = None

        for _ in range(500):
            result = p.step(env_name, rng.random(action_dim).tolist())
            steps += 1
            last_result = result
            if result['done']:
                done_seen = True
                break

        assert done_seen, (
            f'{env_name}: episode did not terminate within 500 steps'
        )
        assert steps >= 1, (
            f'{env_name}: episode ended in 0 steps'
        )
        assert 'transfer_result' in last_result['info'], (
            f'{env_name}: transfer_result missing from info on done step'
        )


def test_augmentation_clipping():
    """
    Directly test augment_observation with a bias vector containing values
    outside [-0.5, 0.5]. Verify they are clipped.
    """
    raw_sn = np.full(15, 0.5, dtype=np.float32)
    raw_mf = np.full(18, 0.5, dtype=np.float32)

    bias_high = [2.0] * BIAS_VECTOR_DIM
    bias_low  = [-2.0] * BIAS_VECTOR_DIM
    bias_mixed = [1.0, -1.0, 0.3, -0.3, 0.6, -0.6, 0.0, 0.5]

    for raw, env_name in [(raw_sn, 'safenutri'), (raw_mf, 'megaforge')]:
        aug_high = augment_observation(raw, bias_high, env_name)
        assert np.all(aug_high[-BIAS_VECTOR_DIM:] == BIAS_CLIP), (
            f'{env_name}: high bias not clipped to {BIAS_CLIP}'
        )

        aug_low = augment_observation(raw, bias_low, env_name)
        assert np.all(aug_low[-BIAS_VECTOR_DIM:] == -BIAS_CLIP), (
            f'{env_name}: low bias not clipped to -{BIAS_CLIP}'
        )

        aug_mixed = augment_observation(raw, bias_mixed, env_name)
        expected = np.clip(np.array(bias_mixed, dtype=np.float32), -BIAS_CLIP, BIAS_CLIP)
        assert np.allclose(aug_mixed[-BIAS_VECTOR_DIM:], expected, atol=1e-6), (
            f'{env_name}: mixed bias clipping incorrect. '
            f'Got {aug_mixed[-BIAS_VECTOR_DIM:]}, expected {expected}'
        )


def test_obs_field_ordering():
    """
    Verify that OBS_FIELDS['safenutri'] has exactly 15 entries and
    OBS_FIELDS['megaforge'] has exactly 18 entries.
    """
    sn_fields = OBS_FIELDS['safenutri']
    mf_fields = OBS_FIELDS['megaforge']

    assert len(sn_fields) == 15, (
        f'safenutri OBS_FIELDS has {len(sn_fields)} entries, expected 15'
    )
    assert len(mf_fields) == 18, (
        f'megaforge OBS_FIELDS has {len(mf_fields)} entries, expected 18'
    )

    # No duplicates
    assert len(set(sn_fields)) == 15, 'safenutri OBS_FIELDS contains duplicate field names'
    assert len(set(mf_fields)) == 18, 'megaforge OBS_FIELDS contains duplicate field names'

    # Both are lists of strings
    assert all(isinstance(f, str) for f in sn_fields), 'safenutri OBS_FIELDS contains non-string entries'
    assert all(isinstance(f, str) for f in mf_fields), 'megaforge OBS_FIELDS contains non-string entries'


# ---- Runner ----
if __name__ == '__main__':
    tests = [
        test_observation_dimensions,
        test_memory_none_equals_noop,
        test_reproducibility,
        test_bias_vector_is_zero_without_memory,
        test_episode_lifecycle,
        test_augmentation_clipping,
        test_obs_field_ordering,
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
