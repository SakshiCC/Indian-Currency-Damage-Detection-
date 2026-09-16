"""
Vinit V3 Damage Classifier - Calibration Module.

Provides context feature extraction and dynamic threshold calibration
grounded in human-verified ground truth annotations.
"""

from clip_damage_classifier.calibration.context_features import (
    extract_context_features,
    extract_features_from_image,
)
from clip_damage_classifier.calibration.dynamic_thresholds import (
    DynamicThresholdCalculator,
    calculate_dynamic_threshold,
    calculate_all_dynamic_thresholds,
    apply_dynamic_thresholds,
    DEFAULT_V2_BASE_THRESHOLDS,
)

__all__ = [
    "extract_context_features",
    "extract_features_from_image",
    "DynamicThresholdCalculator",
    "calculate_dynamic_threshold",
    "calculate_all_dynamic_thresholds",
    "apply_dynamic_thresholds",
    "DEFAULT_V2_BASE_THRESHOLDS",
]
