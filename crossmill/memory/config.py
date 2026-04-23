# ---- IMPORTS FROM INTEGRATION CONFIG ----
# Re-export these so the memory layer has one import path for all constants.
from crossmill.config import (
    ABSTRACT_DIM,               # 8
    BIAS_VECTOR_DIM,            # 8
    ACTION_PATTERNS,            # list of 5 pattern labels
    EPISODIC_BUFFER_SIZE,       # 500
    POSITIVE_REWARD_THRESHOLD,  # 0.01
    NEGATIVE_REWARD_THRESHOLD,  # -0.5
    SIMILARITY_WEIGHT,          # 0.7
    CONFIDENCE_WEIGHT,          # 0.3
    BIAS_CLIP,                  # 0.5
    EMA_ALPHA,                  # 0.1
    CONFIDENCE_FLOOR,           # 0.05
    CONFIDENCE_CEILING,         # 1.0
    BASELINE_WINDOW,            # 10
    PROMOTION_CONFIDENCE_BONUS, # 0.1
    NEGATIVE_PROMOTION_THRESHOLD,  # -0.3
    DEFAULT_MEMORY_CONFIG,
    ENVIRONMENTS,
)

# ---- PATTERN-TO-BIAS MAPPING ----
# Each pattern maps to an 8-element bias vector that nudges the agent in the
# abstract state dimensions.
# Abstract state dimension order (from memory design spec):
#   0: process_temperature
#   1: temperature_rate
#   2: quality_risk
#   3: safety_margin
#   4: energy_intensity
#   5: equipment_condition
#   6: process_progress
#   7: throughput_pressure
#
# Values are deliberately small — agent is nudged, not controlled. The final
# bias vector from the TransferAdapter is clipped to [-0.5, +0.5] anyway.

PATTERN_BIAS = {
    'gradual_ramp':       [+0.20, -0.15,  0.0,  +0.10,  0.0,  0.0, 0.0,  0.0],
    # favour slow, penalise fast change, slight positive safety margin
    'hold_steady':        [ 0.0,   0.0,   0.0,   0.0,   0.0,  0.0, 0.0,  0.0],
    # neutral bias — maintain current trajectory
    'rapid_correction':   [-0.10, +0.25,  0.0,  -0.10,  0.0,  0.0, 0.0,  0.0],
    # favour faster change, acknowledge reduced safety margin
    'cautious_explore':   [+0.05, +0.05, -0.05, +0.05,  0.0,  0.0, 0.0,  0.0],
    # small exploratory nudge in multiple directions
    'emergency_response': [ 0.0,  +0.30,  0.0,  +0.30,  0.0,  0.0, 0.0, +0.25],
    # aggressive in rate, strongly favour safety margin, flag throughput urgency
}

# Validation: every pattern label in ACTION_PATTERNS has a bias entry
assert set(PATTERN_BIAS.keys()) == set(ACTION_PATTERNS), (
    f"PATTERN_BIAS keys {set(PATTERN_BIAS.keys())} do not match "
    f"ACTION_PATTERNS {set(ACTION_PATTERNS)}"
)
# Validation: every bias vector has correct length
for label, vec in PATTERN_BIAS.items():
    assert len(vec) == BIAS_VECTOR_DIM, (
        f"PATTERN_BIAS[{label!r}] has length {len(vec)}, "
        f"expected {BIAS_VECTOR_DIM}"
    )

# ---- SAFETY REGIME CLASSIFICATION ----
# Each task difficulty maps to a default safety regime. Memories stored during
# a Hard task run are tagged 'strict' because Hard has tighter partial obs and
# stochasticity. Easy is 'moderate' because the agent has full information.
SAFETY_REGIME_BY_DIFFICULTY = {
    'easy':   'moderate',
    'medium': 'moderate',
    'hard':   'strict',
}

# ---- PROMOTION DIFFICULTY ORDER ----
# When a promotion candidate has mixed-difficulty contributors, the threshold
# of the hardest difficulty applies. This lookup gives the order.
DIFFICULTY_RANK = {'easy': 0, 'medium': 1, 'hard': 2}

# ---- EPISODE ID BASELINE BUFFER ----
# The TransferAdapter needs a rolling window of recent episode rewards per env
# to compute the reward delta signal for the confidence EMA.
BASELINE_BUFFER_MAXLEN = BASELINE_WINDOW  # = 10

# ---- DEFAULT MEMORYCONFIG ----
# Re-exported so the memory layer top-level class has a single import path.
# This is just a reference to the integration layer's dict.
DEFAULT_CONFIG_DICT = DEFAULT_MEMORY_CONFIG


if __name__ == '__main__':
    print("=" * 60)
    print("PATTERN_BIAS entries")
    print("=" * 60)
    for label, vec in PATTERN_BIAS.items():
        print(f"  {label:<20s}  len={len(vec)}  {vec}")

    print()
    print("All validations passed.")

    print()
    print("=" * 60)
    print("DEFAULT_CONFIG_DICT")
    print("=" * 60)
    import json
    print(json.dumps(DEFAULT_CONFIG_DICT, indent=2))
