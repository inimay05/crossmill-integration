"""
Unit tests for the CrossMill Unified Grader.

Run with:
    python -m tests.test_grader

No test framework required. All data is synthetic — no training runs needed.
"""
from __future__ import annotations

import json
import os
import tempfile

from crossmill.grader_config import GRADER_TARGETS, REQUIRED_SUMMARY_FIELDS
from crossmill.grader_validation import validate_summary, ValidationResult
from crossmill.grader import load_summary, compute_deltas


# ---- Shared fixture ----

def _clean_summary(**overrides) -> dict:
    """Return a valid SafeNutri summary dict with all required fields present."""
    base = {
        'env': 'safenutri',
        'task_id': 'easy',
        'memory_mode': 'none',
        'timesteps': 50000,
        'seed': 42,
        'pre_score': 0.50,
        'post_score': 0.74,
        'delta': 0.24,
        'grader_score': 0.74,
        'mean_reward': 0.42,
        'std_reward': 0.05,
        'safety_violation_rate': 0.0,
        'catastrophic_rate': 0.0,
        'mean_vit_c_retention': 0.82,
        'mean_carbon_error_pct': None,
    }
    base.update(overrides)
    return base


# ---- Tests ----

def test_validation_clean_pass():
    """
    A summary dict with zero violations, good vitamin retention,
    low variance, and non-zero catastrophic rate of 0.0
    should produce: PASS verdict, empty flags, adjusted == grader_score.
    """
    r = validate_summary(_clean_summary())
    assert r.verdict == 'PASS', f'Expected PASS, got {r.verdict}'
    assert r.flags == [], f'Expected no flags, got {r.flags}'
    assert r.adjusted_score == r.grader_score, (
        f'adjusted_score {r.adjusted_score} != grader_score {r.grader_score}'
    )
    assert not r.has_flags, 'has_flags should be False for clean run'
    assert not r.score_was_modified, 'score_was_modified should be False for clean run'


def test_validation_catastrophic_zeros_score():
    """
    catastrophic_rate > 0 must set adjusted_score to 0.0 regardless of
    grader_score. Flags must contain CATASTROPHIC_FAIL.
    Verdict must be 'CATASTROPHIC'.
    """
    r = validate_summary(_clean_summary(catastrophic_rate=0.02))
    assert 'CATASTROPHIC_FAIL' in r.flags, (
        f'CATASTROPHIC_FAIL not in flags: {r.flags}'
    )
    assert r.adjusted_score == 0.0, (
        f'Expected adjusted_score 0.0, got {r.adjusted_score}'
    )
    assert r.verdict == 'CATASTROPHIC', f'Expected CATASTROPHIC, got {r.verdict}'
    assert r.score_was_modified, 'score_was_modified should be True'


def test_validation_safety_warning_only():
    """
    safety_violation_rate > 0 but catastrophic_rate == 0 and quality is fine.
    Must raise SAFETY_WARNING flag but NOT modify adjusted_score.
    COMPOUND_FAIL must NOT be present.
    """
    r = validate_summary(_clean_summary(safety_violation_rate=0.05))
    assert 'SAFETY_WARNING' in r.flags, (
        f'SAFETY_WARNING not in flags: {r.flags}'
    )
    assert r.adjusted_score == r.grader_score, (
        f'Score should not be modified: {r.adjusted_score} != {r.grader_score}'
    )
    assert 'COMPOUND_FAIL' not in r.flags, (
        f'COMPOUND_FAIL should not be present: {r.flags}'
    )
    assert r.verdict == 'WARN', f'Expected WARN, got {r.verdict}'


def test_validation_quality_low_only():
    """
    SafeNutri: mean_vit_c_retention below threshold (0.5).
    Must raise QUALITY_LOW flag but NOT modify adjusted_score.
    COMPOUND_FAIL must NOT be present.
    """
    r = validate_summary(_clean_summary(mean_vit_c_retention=0.35))
    assert 'QUALITY_LOW' in r.flags, (
        f'QUALITY_LOW not in flags: {r.flags}'
    )
    assert r.adjusted_score == r.grader_score, (
        f'Score should not be modified: {r.adjusted_score} != {r.grader_score}'
    )
    assert 'COMPOUND_FAIL' not in r.flags, (
        f'COMPOUND_FAIL should not be present: {r.flags}'
    )
    assert r.verdict == 'WARN', f'Expected WARN, got {r.verdict}'


def test_validation_compound_fail_halves_score():
    """
    Both SAFETY_WARNING and QUALITY_LOW conditions active simultaneously.
    Must raise COMPOUND_FAIL and set adjusted_score = grader_score * 0.5.
    Verdict must be 'WARN_COMPOUND'.
    """
    r = validate_summary(_clean_summary(
        safety_violation_rate=0.05,
        mean_vit_c_retention=0.35,
    ))
    assert 'COMPOUND_FAIL' in r.flags, (
        f'COMPOUND_FAIL not in flags: {r.flags}'
    )
    expected = 0.74 * 0.5
    assert abs(r.adjusted_score - expected) < 1e-6, (
        f'Expected adjusted_score {expected}, got {r.adjusted_score}'
    )
    assert r.verdict == 'WARN_COMPOUND', f'Expected WARN_COMPOUND, got {r.verdict}'
    assert r.score_was_modified, 'score_was_modified should be True'


def test_validation_compound_requires_both():
    """
    COMPOUND_FAIL must NOT fire when only one of the two conditions is active.
    Test safety alone (no COMPOUND), test quality alone (no COMPOUND),
    test both together (COMPOUND fires).
    """
    safety_only = validate_summary(_clean_summary(safety_violation_rate=0.05))
    assert 'COMPOUND_FAIL' not in safety_only.flags, (
        f'COMPOUND_FAIL should not fire with safety alone: {safety_only.flags}'
    )

    quality_only = validate_summary(_clean_summary(mean_vit_c_retention=0.35))
    assert 'COMPOUND_FAIL' not in quality_only.flags, (
        f'COMPOUND_FAIL should not fire with quality alone: {quality_only.flags}'
    )

    both = validate_summary(_clean_summary(
        safety_violation_rate=0.05,
        mean_vit_c_retention=0.35,
    ))
    assert 'COMPOUND_FAIL' in both.flags, (
        f'COMPOUND_FAIL must fire when both conditions are active: {both.flags}'
    )


def test_validation_high_variance():
    """
    std_reward / mean_reward > 0.8 must raise HIGH_VARIANCE.
    Score must NOT be modified by HIGH_VARIANCE alone.
    """
    # cv = 0.45 / 0.42 ≈ 1.07 > 0.8
    r = validate_summary(_clean_summary(std_reward=0.45))
    assert 'HIGH_VARIANCE' in r.flags, (
        f'HIGH_VARIANCE not in flags: {r.flags}'
    )
    assert r.adjusted_score == r.grader_score, (
        f'HIGH_VARIANCE alone must not modify score: '
        f'{r.adjusted_score} != {r.grader_score}'
    )
    assert r.verdict == 'WARN', f'Expected WARN, got {r.verdict}'

    # cv = 0.33 / 0.42 ≈ 0.79 <= 0.8 — should NOT trigger
    r_ok = validate_summary(_clean_summary(std_reward=0.33))
    assert 'HIGH_VARIANCE' not in r_ok.flags, (
        f'HIGH_VARIANCE should not fire at cv ~0.79: {r_ok.flags}'
    )


def test_validation_megaforge_quality():
    """
    MegaForge: mean_carbon_error_pct above threshold (0.15) raises QUALITY_LOW.
    MegaForge: mean_vit_c_retention=None must NOT trigger QUALITY_LOW.
    """
    mf_bad = _clean_summary(
        env='megaforge',
        mean_vit_c_retention=None,
        mean_carbon_error_pct=0.22,
    )
    r = validate_summary(mf_bad)
    assert 'QUALITY_LOW' in r.flags, (
        f'QUALITY_LOW should fire for megaforge carbon_error=0.22: {r.flags}'
    )

    mf_good = _clean_summary(
        env='megaforge',
        mean_vit_c_retention=None,
        mean_carbon_error_pct=0.10,
    )
    r_good = validate_summary(mf_good)
    assert 'QUALITY_LOW' not in r_good.flags, (
        f'QUALITY_LOW should NOT fire for megaforge carbon_error=0.10: {r_good.flags}'
    )

    mf_none_vit = _clean_summary(
        env='megaforge',
        mean_vit_c_retention=None,
        mean_carbon_error_pct=None,
    )
    r_none = validate_summary(mf_none_vit)
    assert 'QUALITY_LOW' not in r_none.flags, (
        f'QUALITY_LOW must NOT fire when carbon_error_pct is None: {r_none.flags}'
    )


def test_validation_none_quality_field_skipped():
    """
    When mean_vit_c_retention is None for SafeNutri, QUALITY_LOW must NOT fire.
    When mean_carbon_error_pct is None for MegaForge, QUALITY_LOW must NOT fire.
    """
    sn_none = validate_summary(_clean_summary(mean_vit_c_retention=None))
    assert 'QUALITY_LOW' not in sn_none.flags, (
        f'SafeNutri: QUALITY_LOW must not fire when vit_c=None: {sn_none.flags}'
    )

    mf_none = validate_summary(_clean_summary(
        env='megaforge',
        mean_vit_c_retention=None,
        mean_carbon_error_pct=None,
    ))
    assert 'QUALITY_LOW' not in mf_none.flags, (
        f'MegaForge: QUALITY_LOW must not fire when carbon_error=None: {mf_none.flags}'
    )


def test_deltas_computed_on_raw_score():
    """
    Build three summaries where the cross condition has COMPOUND_FAIL
    (adjusted_score halved). Verify that transfer_gain is computed using the
    raw grader_score (not adjusted_score) from the summaries dict.
    transfer_gain should reflect actual policy performance, not the penalty.
    """
    none_s  = _clean_summary(memory_mode='none',  grader_score=0.60)
    local_s = _clean_summary(memory_mode='local', grader_score=0.70)
    # cross has compound fail — adjusted would be 0.40, but raw is 0.80
    cross_s = _clean_summary(
        memory_mode='cross',
        grader_score=0.80,
        safety_violation_rate=0.05,
        mean_vit_c_retention=0.35,
    )

    summaries = {'none': none_s, 'local': local_s, 'cross': cross_s}
    validations = {mode: validate_summary(s) for mode, s in summaries.items()}

    assert 'COMPOUND_FAIL' in validations['cross'].flags, (
        'Test setup: cross condition should have COMPOUND_FAIL'
    )
    assert abs(validations['cross'].adjusted_score - 0.40) < 1e-6, (
        f'Test setup: cross adjusted_score should be 0.40, got {validations["cross"].adjusted_score}'
    )

    deltas = compute_deltas(validations, summaries)

    # cross_gain: 0.80 - 0.60 = 0.20 (raw scores)
    assert abs(deltas['cross_gain'] - 0.20) < 1e-4, (
        f'cross_gain should be 0.20 (raw scores), got {deltas["cross_gain"]}'
    )
    # transfer_gain: 0.80 - 0.70 = 0.10 (raw scores, not 0.40 - 0.70)
    assert abs(deltas['transfer_gain'] - 0.10) < 1e-4, (
        f'transfer_gain should be 0.10 (raw scores), got {deltas["transfer_gain"]}'
    )
    # local_gain: 0.70 - 0.60 = 0.10
    assert abs(deltas['local_gain'] - 0.10) < 1e-4, (
        f'local_gain should be 0.10, got {deltas["local_gain"]}'
    )


def test_load_summary_missing_field_raises():
    """
    load_summary should raise ValueError if a required field is missing.
    Write a temporary JSON file with mean_reward omitted and verify the
    error message mentions the missing field.
    """
    incomplete = _clean_summary()
    del incomplete['mean_reward']

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'summary_easy_none.json')
        with open(path, 'w') as f:
            json.dump(incomplete, f)

        try:
            load_summary(tmpdir, 'easy', 'none')
            assert False, 'Expected ValueError to be raised for missing field'
        except ValueError as e:
            assert 'mean_reward' in str(e), (
                f'Error message should mention the missing field "mean_reward": {e}'
            )


def test_load_summary_missing_file_returns_none():
    """
    load_summary should return None (not raise) when the file does not exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_summary(tmpdir, 'easy', 'none')
        assert result is None, (
            f'Expected None for missing file, got {result}'
        )


def test_grader_targets_correct():
    """
    Verify GRADER_TARGETS values match published task targets:
      safenutri easy >= 0.85
      megaforge easy >= 0.88
    """
    assert GRADER_TARGETS['safenutri']['easy'] >= 0.85, (
        f'safenutri easy target should be >= 0.85, got {GRADER_TARGETS["safenutri"]["easy"]}'
    )
    assert GRADER_TARGETS['megaforge']['easy'] >= 0.88, (
        f'megaforge easy target should be >= 0.88, got {GRADER_TARGETS["megaforge"]["easy"]}'
    )
    # Spot-check that harder tasks have lower targets (sanity)
    assert GRADER_TARGETS['safenutri']['hard'] < GRADER_TARGETS['safenutri']['easy'], (
        'safenutri hard target should be lower than easy target'
    )
    assert GRADER_TARGETS['megaforge']['hard'] < GRADER_TARGETS['megaforge']['easy'], (
        'megaforge hard target should be lower than easy target'
    )


# ---- Runner ----
if __name__ == '__main__':
    tests = [
        test_validation_clean_pass,
        test_validation_catastrophic_zeros_score,
        test_validation_safety_warning_only,
        test_validation_quality_low_only,
        test_validation_compound_fail_halves_score,
        test_validation_compound_requires_both,
        test_validation_high_variance,
        test_validation_megaforge_quality,
        test_validation_none_quality_field_skipped,
        test_deltas_computed_on_raw_score,
        test_load_summary_missing_field_raises,
        test_load_summary_missing_file_returns_none,
        test_grader_targets_correct,
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
