"""
severity.py
============================================================
Severity Calculation Module for Sakshi's Severity Module.

Calculates damaged pixel counts, total valid note pixels,
damage percentage (rounded to 2 decimal places), and project-level
severity levels using configurable thresholds.
============================================================
"""

from typing import Dict, Any
import cv2
import numpy as np

from config import SEVERITY_THRESHOLDS


def percentage_from_mask(
    damage_mask: np.ndarray,
    note_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates damaged pixels, total note pixels, and damage percentage.
    Prevents division by zero safely.
    """
    if note_mask is None or cv2.countNonZero(note_mask) == 0:
        return {
            "damaged_pixels": 0,
            "total_note_pixels": 0,
            "damage_percentage": 0.0,
        }

    total_note_pixels = int(cv2.countNonZero(note_mask))

    if damage_mask is not None:
        damage_inside_note = cv2.bitwise_and(damage_mask, note_mask)
        damaged_pixels = int(cv2.countNonZero(damage_inside_note))
    else:
        damaged_pixels = 0

    percentage = (damaged_pixels / float(total_note_pixels)) * 100.0

    return {
        "damaged_pixels": damaged_pixels,
        "total_note_pixels": total_note_pixels,
        "damage_percentage": round(percentage, 2),
    }


def classify_severity(percentage: float) -> str:
    """
    Classifies damage percentage into experimental project severity levels.
    NOTE: Experimental project thresholds, not official RBI rules.
    """
    if percentage <= 0.0:
        return "None"
    if percentage <= SEVERITY_THRESHOLDS["Minor"]:
        return "Minor"
    if percentage <= SEVERITY_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Severe"


def calculate_severity(
    damage_masks: Dict[str, np.ndarray],
    combined_mask: np.ndarray,
    expected_note_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes per-damage type and combined severity metrics.
    """
    per_damage: Dict[str, Dict[str, Any]] = {}

    for damage_type, mask in damage_masks.items():
        measurement = percentage_from_mask(mask, expected_note_mask)
        measurement["severity"] = classify_severity(measurement["damage_percentage"])
        per_damage[damage_type] = measurement

    combined = percentage_from_mask(combined_mask, expected_note_mask)
    combined["severity"] = classify_severity(combined["damage_percentage"])

    return {
        "per_damage": per_damage,
        "combined": combined,
    }