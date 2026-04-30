# config.py
import os
from datetime import datetime

# ── model ─────────────────────────────────────────────────────────────
MODEL      = "distilbert-base-uncased"
NUM_LABELS = 2

# ── dataset ───────────────────────────────────────────────────────────
DATASET        = "dmilush/shieldlm-prompt-injection"
DATASET_CONFIG = None        # nessun sottoconfig
TEXT_COL       = "text"
LABEL_COL      = "label_binary"
MAX_LENGTH     = 128

# ── training ──────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 32
EPOCHS           = 3
LEARNING_RATE    = 2e-5
SAVE_STRATEGY    = "epoch"

# ── campionamento per la demo ─────────────────────────────────────────
MAX_TRAIN_SAMPLES = 2000
MAX_EVAL_SAMPLES  = 500

# ── output ────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    "checkpoints",
    f"{MODEL}_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{EPOCHS}_{timestamp}"
)

# ── evaluation ────────────────────────────────────────────────────────
EVAL_STRATEGY = "epoch"
EVAL_STEPS    = 100
LOGGING_STEPS = 10

# ── reproducibility ───────────────────────────────────────────────────
SEED = 42