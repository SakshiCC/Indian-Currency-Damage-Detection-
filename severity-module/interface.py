"""
interface.py
============================================================
Public Facade & Interface for Sakshi's Severity Module.
Connects external callers and integration adapters to SeverityAssessor.
============================================================
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import cv2
import numpy as np

from severity_assessor import SeverityAssessor, get_severity_assessor


def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Helper to safely load an image as a numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    image = cv2.imread(str(image_input))
    if image is None:
        raise ValueError(f"Unable to load image input: {image_input}")

    return image


def assess_damage(
    image_input: Union[str, Path, np.ndarray],
    damage_labels: List[str],
    currency: Optional[str] = None,
    denomination_confidence: Optional[float] = None,
    evidence_scores: Optional[Dict[str, Any]] = None,
    dynamic_thresholds: Optional[Dict[str, Any]] = None,
    preprocessing_metadata: Optional[Dict[str, Any]] = None,
    classification_status: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Public entry point for Sakshi's damage severity assessment module.

    Maintains backward compatibility with pipeline handoffs while delegating
    to SeverityAssessor.
    """
    assessor = SeverityAssessor(debug=debug)
    assessment = assessor.assess(
        image_input=image_input,
        damage_types=damage_labels,
        denomination=currency,
        denomination_confidence=denomination_confidence,
        preprocessing_metadata=preprocessing_metadata,
        debug=debug,
    )

    image = load_image(image_input)

    return {
        "result": assessment,
        "image": image,
        "visible_note_mask": assessment.get("debug_paths", {}).get("note_mask"),
        "damage_masks": assessment.get("per_damage", {}),
        "combined_damage_mask": assessment.get("visualization", {}).get("mask_path"),
    }