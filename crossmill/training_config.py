# ---- OBSERVATION DIMENSIONS FOR POLICY NETWORK ----
# The policy network sees augmented observations (raw + bias vector).
# These must match AUGMENTED_OBS_DIM from crossmill.config exactly.
# Imported here so the Gym shim has one import path.
from crossmill.config import AUGMENTED_OBS_DIM, ENVIRONMENTS

POLICY_OBS_DIM = AUGMENTED_OBS_DIM   # {'safenutri': 27, 'megaforge': 30}

# ---- RECURRENTPPO HYPERPARAMETERS ----
# gamma=0.995 is non-negotiable for long-horizon POMDP credit assignment.
# Terminal rewards arrive 200-2000 steps after the actions that earned them.
# A standard gamma=0.99 makes those rewards near-invisible to the policy.
GAMMA           = 0.995
GAE_LAMBDA      = 0.95
LEARNING_RATE   = 3e-4
N_STEPS         = 256      # steps per env per rollout (LSTM unroll length)
BATCH_SIZE      = 64
N_ENVS          = 1        # parallel envs; keep at 1 for LSTM correctness
                           # (vectorised LSTM requires careful hidden-state
                           # management — keep simple for hackathon)
VERBOSE         = 1        # SB3 training log verbosity

# ---- DEFAULT TIMESTEPS BY TASK ----
# These are sane defaults. Override with --timesteps at the CLI.
DEFAULT_TIMESTEPS = {
    'easy':   100_000,
    'medium': 250_000,
    'hard':   500_000,
}

# ---- SANITY CHECK TIMESTEPS ----
# Use this for fast smoke tests — finishes in ~2 minutes.
SANITY_TIMESTEPS = 5_000

# ---- GRADER EVALUATION EPISODES ----
# Pre-training baseline uses fewer episodes (faster); post-training uses full.
PRE_GRADER_EPISODES  = 10
POST_GRADER_EPISODES = 50

# ---- LOG DIRECTORIES ----
LOG_DIR_TEMPLATE = './runs/{env_name}'   # e.g. runs/safenutri/

# ---- REWARD CURVE PLOTTING ----
ROLLING_WINDOW = 50   # rolling mean window for reward curve smoothing
PLOT_DPI       = 120

# ---- HUGGINGFACE ----
HF_REPO_TEMPLATE = '{username}/crossmill-{env_name}-{task}'
# e.g. inimay05/crossmill-safenutri-easy

# ---- ANTI-HACKING QUALITY THRESHOLDS ----
# These are read by the unified grader to validate summary JSONs.
# Stored here so they are configurable without editing the grader.
QUALITY_THRESHOLDS = {
    'safenutri': {
        'min_vit_c_retention':    0.5,   # below = QUALITY_LOW flag
    },
    'megaforge': {
        'max_carbon_error_pct':   0.15,  # above = QUALITY_LOW flag
    },
}
HIGH_VARIANCE_CV_THRESHOLD = 0.8  # std/mean above this = HIGH_VARIANCE flag


if __name__ == '__main__':
    print('POLICY_OBS_DIM:', POLICY_OBS_DIM)
    assert POLICY_OBS_DIM['safenutri'] == 27, f"Expected 27, got {POLICY_OBS_DIM['safenutri']}"
    assert POLICY_OBS_DIM['megaforge'] == 30, f"Expected 30, got {POLICY_OBS_DIM['megaforge']}"
    print('  safenutri =', POLICY_OBS_DIM['safenutri'], '(expected 27) OK')
    print('  megaforge =', POLICY_OBS_DIM['megaforge'], '(expected 30) OK')
    print()
    print('GAMMA:', GAMMA)
    print('LEARNING_RATE:', LEARNING_RATE)
    print('DEFAULT_TIMESTEPS:', DEFAULT_TIMESTEPS)
    print()
    print('QUALITY_THRESHOLDS:', QUALITY_THRESHOLDS)
    print()
    print('training_config OK')
STRATEGY_DIM = 4
STRATEGY_UPDATE_INTERVAL = 256
