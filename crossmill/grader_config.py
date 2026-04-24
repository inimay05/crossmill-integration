from crossmill.training_config import (
    QUALITY_THRESHOLDS,
    HIGH_VARIANCE_CV_THRESHOLD,
)

# ---- QUALITY THRESHOLDS (imported from training_config) ----
# Re-exported here so grader code has one import path.
# These are the per-environment floor values for the QUALITY_LOW flag.
QUALITY_THRESHOLDS = QUALITY_THRESHOLDS
HIGH_VARIANCE_CV_THRESHOLD = HIGH_VARIANCE_CV_THRESHOLD

# ---- GRADER TARGETS PER TASK ----
# These are the published grader_target values from each environment's
# task config. Used as the green target line on the three-curve plot.
GRADER_TARGETS = {
    'safenutri': {'easy': 0.85, 'medium': 0.78, 'hard': 0.72},
    'megaforge':  {'easy': 0.88, 'medium': 0.80, 'hard': 0.74},
}

# ---- REQUIRED SUMMARY JSON FIELDS ----
# The grader will check these fields exist before processing.
# If any are missing it raises a clear error rather than silently failing.
REQUIRED_SUMMARY_FIELDS = [
    'env', 'task_id', 'memory_mode', 'timesteps', 'seed',
    'pre_score', 'post_score', 'delta',
    'grader_score',
    'mean_reward', 'std_reward',
    'safety_violation_rate', 'catastrophic_rate',
]

# ---- OPTIONAL SUMMARY FIELDS (used for anti-hacking if present) ----
# These are None for the non-applicable environment.
# SafeNutri: mean_vit_c_retention should be a float, mean_carbon_error_pct = None
# MegaForge: mean_carbon_error_pct should be a float, mean_vit_c_retention = None
OPTIONAL_QUALITY_FIELDS = [
    'mean_vit_c_retention',
    'mean_carbon_error_pct',
    'mean_coke_rate_kgpt',
    'monitor_csv',
]

# ---- CONDITIONS ----
# The three memory modes to compare. local is optional.
ALL_CONDITIONS      = ['none', 'local', 'cross']
REQUIRED_CONDITIONS = ['none', 'cross']   # minimum to compute cross_gain
OPTIONAL_CONDITIONS = ['local']           # needed to isolate transfer_gain

# ---- OUTPUT TEMPLATES ----
# File names written by the grader.
REPORT_JSON_TEMPLATE    = 'comparison_report_{env}_{task}.json'
COMPARISON_PNG_TEMPLATE = 'comparison_plot_{env}_{task}.png'

# ---- DELTA LABELS ----
# Human-readable names for the three delta metrics.
DELTA_LABELS = {
    'local_gain':    'Replay benefit  (local − none)',
    'cross_gain':    'Transfer benefit  (cross − none)',
    'transfer_gain': 'Cross-industry benefit  (cross − local)',
}


if __name__ == '__main__':
    print('GRADER_TARGETS[safenutri]:', GRADER_TARGETS['safenutri'])
    print('GRADER_TARGETS[megaforge]:', GRADER_TARGETS['megaforge'])
    print('REQUIRED_SUMMARY_FIELDS:', REQUIRED_SUMMARY_FIELDS)
    print('QUALITY_THRESHOLDS:', QUALITY_THRESHOLDS)
    assert GRADER_TARGETS['safenutri']['easy'] == 0.85
    assert GRADER_TARGETS['megaforge']['easy'] == 0.88
    print('grader_config OK')
