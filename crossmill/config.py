import os

# ---- REGISTERED ENVIRONMENTS ----
# Both repos use the module name 'app.environment', which causes a PYTHONPATH
# collision if both are imported at once. We resolve this by loading each env
# from its absolute file path using importlib.util.spec_from_file_location.
# _PARENT is computed relative to this config.py file's location so the paths
# work regardless of where the user's terminal is when they run commands.

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # crossmill-integration/crossmill/
_PARENT   = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))  # parent/ folder

ENVIRONMENTS = {
    'safenutri': {
        'env_file':   os.path.join(_PARENT, 'crossmill-safenutri', 'app', 'environment.py'),
        'class_name': 'SafeNutriEnv',
        'state_dim':  15,
        'action_dim': 8,
    },
    'megaforge': {
        'env_file':   os.path.join(_PARENT, 'crossmill-megaforge', 'app', 'environment.py'),
        'class_name': 'MegaForgeEnv',
        'state_dim':  18,
        'action_dim': 10,
    },
}

# ---- ABSTRACT STATE SPACE ----
ABSTRACT_DIM = 8
BIAS_VECTOR_DIM = 8

# ---- AUGMENTED OBSERVATION SIZES ----
# Original obs dim + bias vector dim. The policy network ALWAYS sees this size,
# even when memory is explicitly set to None (bias vector is zeros in that case).
AUGMENTED_OBS_DIM = {
    'safenutri': 27,
    'megaforge': 30,
}

# ---- DEFAULT MEMORY CONFIGURATION ----
DEFAULT_MEMORY_CONFIG = {
    'mode': 'cross',
    'transfer_direction': 'bidirectional',
    'top_k': 3,
    'min_confidence': 0.1,
    'promotion_thresholds': {
        'easy': 3,
        'medium': 4,
        'hard': 5,
    },
    'initial_min_confidence': 0.4,
    'confidence_warmup_steps': 50000,
}

# ---- EPISODIC BUFFER ----
EPISODIC_BUFFER_SIZE = 500
POSITIVE_REWARD_THRESHOLD = 0.01
NEGATIVE_REWARD_THRESHOLD = -0.5

# ---- RETRIEVAL ----
SIMILARITY_WEIGHT = 0.7
CONFIDENCE_WEIGHT = 0.3
BIAS_CLIP = 0.5

# ---- CONFIDENCE EMA ----
EMA_ALPHA = 0.1
CONFIDENCE_FLOOR = 0.05
CONFIDENCE_CEILING = 1.0
BASELINE_WINDOW = 10

# ---- PROMOTION ----
PROMOTION_CONFIDENCE_BONUS = 0.1
NEGATIVE_PROMOTION_THRESHOLD = -0.3

# ---- ACTION PATTERN LABELS ----
ACTION_PATTERNS = [
    'gradual_ramp',
    'hold_steady',
    'rapid_correction',
    'cautious_explore',
    'emergency_response',
]

# ---- GRADER ----
GRADER_EVAL_EPISODES = 50
GRADER_CONDITIONS = ['none', 'local', 'cross']
