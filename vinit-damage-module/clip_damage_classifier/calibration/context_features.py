"""
Context feature extraction for Vinit V3 dynamic threshold calibration.

Extracts standardized visual and geometric features from banknote images
alongside cross-class CLIP evidence scores to inform dynamic decision boundaries.
"""

import logging
from typing import Dict, Any, Optional
import cv2
import numpy as np

from clip_damage_classifier.preprocessing import image_preprocessing as ip
from clip_damage_classifier.preprocessing import note_detection as nd

logger = logging.getLogger("clip_damage_classifier.calibration.context_features")

DAMAGE_CATEGORIES = ["Torn", "Folded", "Burnt", "Stain"]


def extract_features_from_image(
    image: np.ndarray,
    preprocessing_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Extract image quality and visual context features from an OpenCV BGR image.

    Features:
      - brightness: Normalized mean grayscale value in [0.0, 1.0]
      - contrast: Normalized standard deviation of grayscale in [0.0, 1.0]
      - dark_ratio: Proportion of dark pixels (intensity < 50)
      - edge_density: Proportion of Canny edge pixels
      - note_detected: 1.0 if note contour was found, 0.0 otherwise
      - area_ratio: Ratio of detected note area to total image area
    """
    if image is None or image.size == 0:
        return {
            "brightness": 0.5,
            "contrast": 0.25,
            "dark_ratio": 0.0,
            "edge_density": 0.0,
            "note_detected": 0.0,
            "area_ratio": 1.0,
        }

    h, w = image.shape[:2]
    total_pixels = float(h * w) if (h * w) > 0 else 1.0

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Image statistics
    brightness = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 128.0
    dark_pixels = float(np.mean(gray < 50))

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Preprocessing / geometric context
    if preprocessing_meta is not None:
        note_detected = 1.0 if preprocessing_meta.get("note_detected", False) else 0.0
        area_ratio = float(preprocessing_meta.get("area_ratio", 1.0))
    else:
        try:
            contour = nd.find_note_contour(image)
            if contour is not None:
                note_detected = 1.0
                area_ratio = float(cv2.contourArea(contour)) / total_pixels
            else:
                note_detected = 0.0
                area_ratio = 1.0
        except Exception as e:
            logger.debug(f"Contour detection failed during feature extraction: {e}")
            note_detected = 0.0
            area_ratio = 1.0

    return {
        "brightness": round(brightness, 5),
        "contrast": round(contrast, 5),
        "dark_ratio": round(dark_pixels, 5),
        "edge_density": round(edge_density, 5),
        "note_detected": round(note_detected, 1),
        "area_ratio": round(area_ratio, 5),
    }


def extract_context_features(
    image: Optional[np.ndarray] = None,
    evidence_scores: Optional[Dict[str, float]] = None,
    preprocessing_meta: Optional[Dict[str, Any]] = None,
    original_evidence_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a unified dictionary of context features combining image stats,
    cross-evidence scores, and preprocessing metadata.
    """
    features: Dict[str, float] = {}

    # Image visual features
    if image is not None:
        img_feats = extract_features_from_image(image, preprocessing_meta)
        features.update(img_feats)
    else:
        features.update({
            "brightness": 0.5,
            "contrast": 0.25,
            "dark_ratio": 0.0,
            "edge_density": 0.0,
            "note_detected": 0.0,
            "area_ratio": 1.0,
        })

    # Evidence scores
    if evidence_scores is not None:
        for cat in DAMAGE_CATEGORIES:
            key = f"{cat.lower()}_evidence"
            features[key] = float(evidence_scores.get(cat, 0.0))

    # Original vs preprocessed evidence differences if available
    if original_evidence_scores is not None and evidence_scores is not None:
        for cat in DAMAGE_CATEGORIES:
            orig_ev = float(original_evidence_scores.get(cat, 0.0))
            proc_ev = float(evidence_scores.get(cat, 0.0))
            features[f"orig_{cat.lower()}_evidence"] = orig_ev
            features[f"diff_{cat.lower()}_evidence"] = round(orig_ev - proc_ev, 6)

    return features
