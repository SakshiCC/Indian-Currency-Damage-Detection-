"""
sakshi_interface.py
============================================================
Exposed Interface Hook for Sakshi's Future Segmentation & Severity Module.

Sakshi is responsible for:
  - Damage segmentation masks
  - Damage localization
  - Damage severity (% damaged area)

Sakshi has NOT started her module yet.
This module strictly exposes a clean consumption interface.
It DOES NOT implement segmentation, masks, or severity calculations.
============================================================
"""

from typing import Dict, Any, List, Optional


def get_damage_labels(classification_result: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract final 5-class damage labels dictionary:
      {'Torn': 0/1, 'Folded': 0/1, 'Burnt': 0/1, 'Stain': 0/1, 'Normal': 0/1}
    """
    if not classification_result or not isinstance(classification_result, dict):
        return {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 1}

    if "damage" in classification_result and isinstance(classification_result["damage"], dict):
        classification_result = classification_result["damage"]

    if "final_labels" in classification_result and isinstance(classification_result["final_labels"], dict):
        return dict(classification_result["final_labels"])

    if "final_damage_labels" in classification_result and isinstance(classification_result["final_damage_labels"], dict):
        return dict(classification_result["final_damage_labels"])

    if "damage_labels" in classification_result and isinstance(classification_result["damage_labels"], dict):
        return dict(classification_result["damage_labels"])

    return {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 1}


def get_active_damage_labels(classification_result: Dict[str, Any]) -> List[str]:
    """
    Extract active damage category strings, strictly EXCLUDING 'Normal'.

    Args:
        classification_result: Dictionary returned by DamageClassifier.classify()
                               or CurrencyPipeline.run_analysis()

    Returns:
        List of active damage classes from ["Torn", "Folded", "Burnt", "Stain"].
        Returns [] if no damage is detected (i.e. note is Normal).

    Example:
        >>> res = classifier.classify("note.jpg")
        >>> get_active_damage_labels(res)
        ['Torn', 'Stain']
    """
    if not classification_result or not isinstance(classification_result, dict):
        return []

    # Check if nested in 'damage' key from currency_pipeline
    if "damage" in classification_result and isinstance(classification_result["damage"], dict):
        classification_result = classification_result["damage"]

    # 1. Inspect damage_labels / final_damage_labels dict if present
    damage_labels = (
        classification_result.get("final_damage_labels")
        or classification_result.get("damage_labels")
        or classification_result.get("final_labels")
    )
    if isinstance(damage_labels, dict):
        active = []
        for cat in ["Torn", "Folded", "Burnt", "Stain"]:
            if damage_labels.get(cat, 0) == 1:
                active.append(cat)
        return active

    # 2. Fallback to predicted_labels list
    predicted = classification_result.get("predicted_labels", [])
    if isinstance(predicted, list):
        return [lbl for lbl in predicted if lbl in ["Torn", "Folded", "Burnt", "Stain"]]

    return []


def is_damaged(classification_result: Dict[str, Any]) -> bool:
    """
    Returns True if any physical damage was detected, False if note is Normal.
    """
    return len(get_active_damage_labels(classification_result)) > 0


def get_damage_evidence(classification_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns the dictionary of CLIP contrastive evidence scores for each damage class.
    """
    if "damage" in classification_result and isinstance(classification_result["damage"], dict):
        classification_result = classification_result["damage"]
    return classification_result.get("evidence_scores", {})


def get_sakshi_handoff_payload(
    classification_result: Dict[str, Any],
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the structured handoff payload designed specifically for
    Sakshi's segmentation module to consume.

    Returns:
        Dict containing:
          - active_damage_labels: List[str] (e.g. ['Torn', 'Stain'])
          - is_damaged: bool
          - evidence_scores: Dict[str, float]
          - preprocessing_metadata: Dict[str, Any] (crop boundaries, note detection)
          - image_path: Optional[str]
    """
    if "damage" in classification_result and isinstance(classification_result["damage"], dict):
        inner = classification_result["damage"]
    else:
        inner = classification_result

    active = get_active_damage_labels(inner)
    return {
        "image_path": image_path,
        "is_damaged": len(active) > 0,
        "active_damage_labels": active,
        "evidence_scores": inner.get("evidence_scores", {}),
        "preprocessing_metadata": inner.get("preprocessing_metadata", {}),
        "dynamic_thresholds": inner.get("dynamic_thresholds", {}),
        "classification_status": inner.get("classification_status", "unknown"),
    }
