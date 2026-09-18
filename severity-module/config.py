"""
config.py
============================================================
Configuration parameters and thresholds for Sakshi's Damage
Severity Assessment & Visualization Module.
============================================================
"""

from pathlib import Path
from typing import Dict, Any

# Root directory of severity module
SEVERITY_MODULE_DIR = Path(__file__).resolve().parent

# Output and Debug directories
OUTPUT_DIR = SEVERITY_MODULE_DIR / "output"
DEBUG_DIR = SEVERITY_MODULE_DIR / "debug"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Minimum region contour area in pixels to filter out noise
MIN_REGION_AREA: int = 150

# Minimum region percentage of total note area (0.2%)
MIN_REGION_PCT: float = 0.20

# Conservative double-check thresholds (% of expected note area)
MIN_VERIFICATION_PCT: Dict[str, float] = {
    "Torn": 2.0,      # Minimum missing material to confirm Vinit Torn label (protects Normal)
    "Folded": 0.5,    # Minimum crease evidence to confirm Vinit Folded label
    "Stain": 0.5,     # Minimum stain evidence to confirm Vinit Stain label
    "Burnt": 0.5,     # Minimum burnt evidence to confirm Vinit Burnt label
}
MIN_RECOVERY_PCT: Dict[str, float] = {
    "Torn": 2.0,      # Strong missing note material needed to recover Torn
    "Folded": 1.0,    # Strong crease evidence needed to recover Folded
    "Stain": 1.0,     # Strong discoloration needed to recover Stain
    "Burnt": 1.0,     # Strong charring needed to recover Burnt
}

# Morphological kernel sizes
DEFAULT_MORPH_KERNEL_SIZE: int = 7

# Experimental project severity thresholds (Percentage of note area)
SEVERITY_THRESHOLDS: Dict[str, float] = {
    "Minor": 5.0,
    "Moderate": 15.0,
}

# Conservative project recommendations
RECOMMENDATIONS: Dict[str, str] = {
    "None": "No significant physical damage detected.",
    "Minor": "Minor physical damage detected.",
    "Moderate": "Moderate physical damage detected. Manual inspection recommended.",
    "Severe": "Severe physical damage detected. Manual inspection recommended.",
}

# Stain segmentation parameters - highly conservative to prevent false positives on printed details
STAIN_COLOR_DIFF_THRESHOLD: float = 35.0
STAIN_MIN_AREA: int = 300
STAIN_MIN_ASPECT_RATIO: float = 0.3
STAIN_MAX_ASPECT_RATIO: float = 3.0

# Burnt segmentation parameters
BURNT_INTENSITY_THRESHOLD: int = 60
BURNT_MIN_AREA: int = 30

# Folded segmentation parameters
FOLD_CANNY_LOW: int = 50
FOLD_CANNY_HIGH: int = 150
FOLD_MIN_LINE_LENGTH: int = 30
FOLD_MAX_LINE_GAP: int = 10

# Bounding box grid locations (3x3 grid)
LOCATION_GRID_NAMES = [
    ["top-left", "top-center", "top-right"],
    ["middle-left", "center", "middle-right"],
    ["bottom-left", "bottom-center", "bottom-right"],
]
