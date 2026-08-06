"""
Global model and training configuration.
"""

# ==========================================================
# Model
# ==========================================================

IMAGE_SIZE = (224, 224)

INPUT_SHAPE = (224, 224, 3)

NUM_CLASSES = 4

DROPOUT_RATE = 0.3

PRETRAINED_WEIGHTS = "imagenet"

# ==========================================================
# Model Names
# ==========================================================

MODEL_NAME = "efficientnetv2b0_baseline"
BASELINE_MODEL_NAME = MODEL_NAME

CBAM_MODEL_NAME = "efficientnetv2b0_cbam"

# ==========================================================
# Training
# ==========================================================

EPOCHS = 20

BATCH_SIZE = 32

# ==========================================================
# Callback Settings
# ==========================================================

EARLY_STOPPING_PATIENCE = 5

LR_PATIENCE = 3

LR_FACTOR = 0.1

MONITOR_METRIC = "val_loss"

# ==========================================================
# Baseline Output Paths
# ==========================================================

BASELINE_MODEL_SAVE_PATH = (
    "saved_models/best_model.keras"
)

BASELINE_TENSORBOARD_LOG_DIR = (
    "outputs/logs/baseline"
)

BASELINE_CSV_LOG_PATH = (
    "outputs/baseline_training_log.csv"
)

BASELINE_TRAINING_HISTORY_PATH = (
    "outputs/baseline_history.csv"
)

# ==========================================================
# CBAM Output Paths
# ==========================================================

CBAM_MODEL_SAVE_PATH = (
    "saved_models/efficientnet_cbam.keras"
)

CBAM_TENSORBOARD_LOG_DIR = (
    "outputs/logs/cbam"
)

CBAM_CSV_LOG_PATH = (
    "outputs/cbam_training_log.csv"
)

CBAM_TRAINING_HISTORY_PATH = (
    "outputs/cbam_history.csv"
)