"""
config.py
============================================================
Configuration parameters for the Contrastive CLIP Damage Classifier.
Uses positive vs. negative prompt evidence scoring and classical CV preprocessing.
============================================================
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent if BASE_DIR.parent.name == "vinit-damage-module" else BASE_DIR.parent

DEFAULT_ZIP_PATH = (
    PROJECT_ROOT
    / "clip_damage_classifier"
    / "data"
    / "Spoilt Indian Banknotes.zip"
)

DEFAULT_EXTRACT_DIR = (
    PROJECT_ROOT / "clip_damage_classifier" / "data" / "extracted"
    if (PROJECT_ROOT / "clip_damage_classifier" / "data" / "extracted").exists()
    else BASE_DIR / "data" / "extracted"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE_PREFERENCE = "auto"
BATCH_SIZE = 32

DAMAGE_CATEGORIES = ["Torn", "Folded", "Burnt", "Stain"]
FINAL_CATEGORIES = ["Torn", "Folded", "Burnt", "Stain", "Normal"]

PROMPT_AGGREGATION_METHOD = "mean"

# Baseline Contrastive Evidence Thresholds (v1)
INITIAL_EVIDENCE_THRESHOLDS = {
    "Torn": 0.0005,
    "Folded": 0.0005,
    "Burnt": 0.0015,
    "Stain": 0.0005,
}

# Calibrated Contrastive Evidence Thresholds (v2 - calibrated on 90 verified images)
# Specifically addresses Burnt false positives (+0.0038) and improves Macro F1
CALIBRATED_EVIDENCE_THRESHOLDS = {
    "Torn": -0.0005,
    "Folded": 0.0000,
    "Burnt": 0.0038,
    "Stain": 0.0000,
}

# Borderline ambiguity band (within 0.0004 of decision boundary)
AMBIGUITY_BAND = 0.0004

# Margin required above threshold for strong prediction
STRONG_PREDICTION_MARGIN = 0.0020

# Classification statuses
STATUS_STRONG = "strong_prediction"
STATUS_MODERATE = "moderate_prediction"
STATUS_UNCERTAIN = "uncertain"
STATUS_ERROR = "error"

ENABLE_MULTI_VIEW = False

# Image Preprocessing & Note Detection Parameters
GAUSSIAN_BLUR_KERNEL = (5, 5)
MORPH_KERNEL_SIZE = (5, 5)
MIN_NOTE_CONTOUR_AREA_RATIO = 0.20
MAX_NOTE_CONTOUR_AREA_RATIO = 0.97
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
DEFAULT_RESIZE_TARGET = (512, 512)
