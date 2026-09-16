"""
thresholds.py
============================================================
Conservative Multi-Label Decision & Evidence Thresholding.
Evaluates Torn, Folded, Burnt, Stain independently.
Normal is 1 iff all four damages are 0.
Assigns status: strong_prediction, moderate_prediction, uncertain.
============================================================
"""

from typing import Dict, Any
from . import config


def apply_evidence_thresholds(
    scores: Dict[str, float],
    thresholds: Dict[str, float] = None,
    ambiguity_band: float = config.AMBIGUITY_BAND,
    strong_margin: float = config.STRONG_PREDICTION_MARGIN
) -> Dict[str, Any]:
    """
    Evaluates independent binary decisions using contrastive evidence scores.
    Assigns classification status:
      - strong_prediction
      - moderate_prediction
      - uncertain
    """
    if thresholds is None:
        thresholds = config.INITIAL_EVIDENCE_THRESHOLDS

    damage_flags = {}
    evidence_values = {}

    for cat in config.DAMAGE_CATEGORIES:
        cat_lower = cat.lower()
        ev_key = f"{cat_lower}_evidence_score"
        ev_score = scores.get(ev_key, 0.0)
        evidence_values[cat] = ev_score
        thresh = thresholds[cat]

        # Independent thresholding: active iff positive evidence meets/exceeds threshold
        is_active = 1 if ev_score >= thresh else 0
        damage_flags[cat] = is_active

    active_cats = [c for c in config.DAMAGE_CATEGORIES if damage_flags[c] == 1]
    total_damages = len(active_cats)

    # Normal derivation logic (Normal = 1 iff no damage is detected)
    if total_damages == 0:
        normal = 1
        predicted_labels = "Normal"
    else:
        normal = 0
        predicted_labels = "; ".join(active_cats)

    # Principled 3-tier status logic:
    # 1. Genuinely uncertain:
    #    - If damage predicted: at least one active damage barely crossed the threshold by < ambiguity_band
    #    - If Normal predicted: the closest damage was within ambiguity_band of crossing into damage
    is_uncertain = False
    if total_damages > 0:
        if any(evidence_values[c] < thresholds[c] + ambiguity_band for c in active_cats):
            is_uncertain = True
    else:
        closest_miss = max(evidence_values[c] - thresholds[c] for c in config.DAMAGE_CATEGORIES)
        if closest_miss > -ambiguity_band:
            is_uncertain = True

    if is_uncertain:
        status = config.STATUS_UNCERTAIN
    else:
        # Check if strong prediction
        if total_damages > 0:
            if all(evidence_values[c] >= thresholds[c] + strong_margin for c in active_cats):
                status = config.STATUS_STRONG
            else:
                status = config.STATUS_MODERATE
        else:
            if all(evidence_values[c] <= thresholds[c] - strong_margin for c in config.DAMAGE_CATEGORIES):
                status = config.STATUS_STRONG
            else:
                status = config.STATUS_MODERATE

    return {
        "Torn": damage_flags["Torn"],
        "Folded": damage_flags["Folded"],
        "Burnt": damage_flags["Burnt"],
        "Stain": damage_flags["Stain"],
        "Normal": normal,
        "predicted_labels": predicted_labels,
        "classification_status": status
    }
