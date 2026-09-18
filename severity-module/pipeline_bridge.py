from __future__ import annotations

from typing import Any, Dict, List, Optional

from interface import assess_damage


SUPPORTED_DAMAGE_TYPES = [
    "Torn",
    "Folded",
    "Burnt",
    "Stain",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value,
    default: Optional[float] = None,
) -> Optional[float]:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def extract_denomination(
    pipeline_result: Dict[str, Any],
) -> Optional[str]:

    denomination = (
        pipeline_result.get(
            "denomination",
            {},
        )
    )

    if not isinstance(
        denomination,
        dict,
    ):
        return None

    # Do not use Background as banknote geometry.
    if denomination.get(
        "is_background",
        False,
    ):
        return None

    value = denomination.get(
        "value"
    )

    if value is None:
        value = denomination.get(
            "denomination"
        )

    if value is None:
        return None

    value = str(
        value
    ).strip()

    if value.lower() in {
        "",
        "unknown",
        "background",
        "none",
    }:
        return None

    return value


def extract_denomination_confidence(
    pipeline_result: Dict[str, Any],
) -> Optional[float]:

    denomination = (
        pipeline_result.get(
            "denomination",
            {},
        )
    )

    if not isinstance(
        denomination,
        dict,
    ):
        return None

    return safe_float(
        denomination.get(
            "confidence"
        )
    )


def extract_active_damage_labels(
    pipeline_result: Dict[str, Any],
) -> List[str]:

    # --------------------------------------------------------
    # Preferred source:
    # existing Sakshi handoff
    # --------------------------------------------------------

    handoff = (
        pipeline_result.get(
            "sakshi_handoff",
            {},
        )
    )

    if isinstance(
        handoff,
        dict,
    ):

        active = handoff.get(
            "active_damage_labels"
        )

        if isinstance(
            active,
            (list, tuple),
        ):

            return [
                label
                for label in active
                if label
                in SUPPORTED_DAMAGE_TYPES
            ]

    # --------------------------------------------------------
    # Second source:
    # pipeline's direct Sakshi field
    # --------------------------------------------------------

    active = pipeline_result.get(
        "sakshi_active_damages"
    )

    if isinstance(
        active,
        (list, tuple),
    ):

        return [
            label
            for label in active
            if label
            in SUPPORTED_DAMAGE_TYPES
        ]

    # --------------------------------------------------------
    # Fallback:
    # final Vinit labels
    # --------------------------------------------------------

    damage = pipeline_result.get(
        "damage",
        {},
    )

    if not isinstance(
        damage,
        dict,
    ):
        return []

    labels = (
        damage.get(
            "final_labels"
        )
        or damage.get(
            "damage_labels"
        )
        or {}
    )

    if not isinstance(
        labels,
        dict,
    ):
        return []

    active = []

    for label in SUPPORTED_DAMAGE_TYPES:

        try:

            detected = int(
                labels.get(
                    label,
                    0,
                )
            ) == 1

        except (
            TypeError,
            ValueError,
        ):

            detected = bool(
                labels.get(
                    label,
                    False,
                )
            )

        if detected:
            active.append(
                label
            )

    return active


def extract_vinit_context(
    pipeline_result: Dict[str, Any],
):

    damage = pipeline_result.get(
        "damage",
        {},
    )

    if not isinstance(
        damage,
        dict,
    ):
        damage = {}

    handoff = pipeline_result.get(
        "sakshi_handoff",
        {},
    )

    if not isinstance(
        handoff,
        dict,
    ):
        handoff = {}

    evidence_scores = (
        handoff.get(
            "evidence_scores"
        )
        or damage.get(
            "evidence_scores"
        )
        or {}
    )

    dynamic_thresholds = (
        handoff.get(
            "dynamic_thresholds"
        )
        or damage.get(
            "dynamic_thresholds"
        )
        or {}
    )

    preprocessing_metadata = (
        handoff.get(
            "preprocessing_metadata"
        )
        or pipeline_result.get(
            "preprocessing"
        )
        or {}
    )

    classification_status = (
        handoff.get(
            "classification_status"
        )
        or damage.get(
            "classification_status"
        )
    )

    return {
        "evidence_scores":
            evidence_scores,

        "dynamic_thresholds":
            dynamic_thresholds,

        "preprocessing_metadata":
            preprocessing_metadata,

        "classification_status":
            classification_status,
    }


# ============================================================
# MAIN BRIDGE
# ============================================================

def assess_from_pipeline(
    image_input,
    pipeline_result: Dict[str, Any],
):

    if not isinstance(
        pipeline_result,
        dict,
    ):

        raise TypeError(
            "pipeline_result must be a dictionary."
        )

    denomination = (
        extract_denomination(
            pipeline_result
        )
    )

    denomination_confidence = (
        extract_denomination_confidence(
            pipeline_result
        )
    )

    active_damage_labels = (
        extract_active_damage_labels(
            pipeline_result
        )
    )

    vinit_context = (
        extract_vinit_context(
            pipeline_result
        )
    )

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
            vinit_context[
                "evidence_scores"
            ]
        ),

        dynamic_thresholds=(
            vinit_context[
                "dynamic_thresholds"
            ]
        ),

        preprocessing_metadata=(
            vinit_context[
                "preprocessing_metadata"
            ]
        ),

        classification_status=(
            vinit_context[
                "classification_status"
            ]
        ),
    )