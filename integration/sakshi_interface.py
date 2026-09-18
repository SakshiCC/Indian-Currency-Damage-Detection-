from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SEVERITY_MODULE_DIR = (
    PROJECT_ROOT
    / "severity-module"
)


# ============================================================
# CONSTANTS
# ============================================================

DAMAGE_TYPES = [
    "Torn",
    "Folded",
    "Burnt",
    "Stain",
]

DEFAULT_DAMAGE_LABELS = {
    "Torn": 0,
    "Folded": 0,
    "Burnt": 0,
    "Stain": 0,
    "Normal": 1,
}


# ============================================================
# DAMAGE LABEL EXTRACTION
# ============================================================

def get_damage_labels(
    classification_result: Dict[str, Any],
) -> Dict[str, int]:

    if not isinstance(
        classification_result,
        dict,
    ):
        return DEFAULT_DAMAGE_LABELS.copy()

    result = classification_result

    nested_damage = result.get(
        "damage"
    )

    if isinstance(
        nested_damage,
        dict,
    ):
        result = nested_damage

    candidates = [
        result.get(
            "final_labels"
        ),
        result.get(
            "final_damage_labels"
        ),
        result.get(
            "damage_labels"
        ),
    ]

    for candidate in candidates:

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        labels = (
            DEFAULT_DAMAGE_LABELS.copy()
        )

        found = False

        for label in labels:

            if label in candidate:

                try:
                    labels[label] = int(
                        candidate[label]
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    labels[label] = int(
                        bool(
                            candidate[label]
                        )
                    )

                found = True

        if found:
            return labels

    # --------------------------------------------------------
    # predicted_labels fallback
    # --------------------------------------------------------

    predicted = result.get(
        "predicted_labels"
    )

    if isinstance(
        predicted,
        str,
    ):

        labels = {
            "Torn": 0,
            "Folded": 0,
            "Burnt": 0,
            "Stain": 0,
            "Normal": 0,
        }

        predicted_values = [
            item.strip()
            for item in (
                predicted
                .replace(",", ";")
                .split(";")
            )
            if item.strip()
        ]

        for value in predicted_values:

            for label in labels:

                if (
                    value.lower()
                    == label.lower()
                ):
                    labels[label] = 1

        if not predicted_values:
            labels["Normal"] = 1

        return labels

    return DEFAULT_DAMAGE_LABELS.copy()


# ============================================================
# ACTIVE DAMAGE LABELS
# ============================================================

def get_active_damage_labels(
    classification_result: Dict[str, Any],
) -> List[str]:

    labels = get_damage_labels(
        classification_result
    )

    return [
        label
        for label in DAMAGE_TYPES
        if labels.get(
            label,
            0,
        ) == 1
    ]


def is_damaged(
    classification_result: Dict[str, Any],
) -> bool:

    return bool(
        get_active_damage_labels(
            classification_result
        )
    )


# ============================================================
# EVIDENCE
# ============================================================

def get_damage_evidence(
    classification_result: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(
        classification_result,
        dict,
    ):
        return {}

    result = classification_result

    nested_damage = result.get(
        "damage"
    )

    if isinstance(
        nested_damage,
        dict,
    ):
        result = nested_damage

    evidence = result.get(
        "evidence_scores",
        {},
    )

    if isinstance(
        evidence,
        dict,
    ):
        return evidence

    return {}


# ============================================================
# HANDOFF PAYLOAD
# ============================================================

def get_sakshi_handoff_payload(
    classification_result: Dict[str, Any],
    image_path: Optional[str] = None,
    denomination_result:
        Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    result = classification_result

    if not isinstance(
        result,
        dict,
    ):
        result = {}

    nested_damage = result.get(
        "damage"
    )

    if isinstance(
        nested_damage,
        dict,
    ):
        result = nested_damage

    denomination_result = (
        denomination_result
        if isinstance(
            denomination_result,
            dict,
        )
        else {}
    )

    return {

        "image_path":
            image_path,

        "is_damaged":
            is_damaged(
                classification_result
            ),

        "active_damage_labels":
            get_active_damage_labels(
                classification_result
            ),

        "evidence_scores":
            get_damage_evidence(
                classification_result
            ),

        "preprocessing_metadata":
            result.get(
                "preprocessing_metadata",
                {},
            ),

        "dynamic_thresholds":
            result.get(
                "dynamic_thresholds",
                {},
            ),

        "classification_status":
            result.get(
                "classification_status"
            ),

        # ----------------------------------------------------
        # YASH INFORMATION
        # ----------------------------------------------------

        "denomination":
            denomination_result.get(
                "denomination"
            )
            or denomination_result.get(
                "value"
            ),

        "denomination_confidence":
            denomination_result.get(
                "confidence"
            ),

        "denomination_accepted":
            denomination_result.get(
                "accepted"
            ),

        "is_background":
            denomination_result.get(
                "is_background",
                False,
            ),
    }


# ============================================================
# LOAD SAKSHI MODULE
# ============================================================

def _load_assessment_function():

    severity_path = str(
        SEVERITY_MODULE_DIR
    )

    if severity_path not in sys.path:
        sys.path.insert(
            0,
            severity_path,
        )

    from interface import assess_damage

    return assess_damage


# ============================================================
# REAL SAKSHI ASSESSMENT
# ============================================================

def run_sakshi_assessment(
    image_input,

    denomination_result:
        Dict[str, Any],

    damage_result:
        Dict[str, Any],
):

    assess_damage = (
        _load_assessment_function()
    )

    active_damage_labels = (
        get_active_damage_labels(
            damage_result
        )
    )

    denomination = (
        denomination_result.get(
            "denomination"
        )
        or denomination_result.get(
            "value"
        )
    )

    denomination_confidence = (
        denomination_result.get(
            "confidence"
        )
    )

    # Do not pass Background as note geometry.
    if denomination_result.get(
        "is_background",
        False,
    ):
        denomination = None

    if (
        denomination is not None
        and str(
            denomination
        ).lower()
        in {
            "unknown",
            "background",
            "none",
        }
    ):
        denomination = None

    return assess_damage(
        image_input=image_input,

        damage_labels=(
            active_damage_labels
        ),

        currency=denomination,

        denomination_confidence=(
            denomination_confidence
        ),

        evidence_scores=(
            damage_result.get(
                "evidence_scores",
                {},
            )
        ),

        dynamic_thresholds=(
            damage_result.get(
                "dynamic_thresholds",
                {},
            )
        ),

        preprocessing_metadata=(
            damage_result.get(
                "preprocessing_metadata",
                {},
            )
        ),

        classification_status=(
            damage_result.get(
                "classification_status"
            )
        ),
    )