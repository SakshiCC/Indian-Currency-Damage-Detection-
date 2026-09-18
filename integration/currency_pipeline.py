from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from integration.yash_adapter import (
    denomination_adapter,
)

from integration.vinit_adapter import (
    damage_adapter,
)

from integration.sakshi_interface import (
    get_active_damage_labels,
    get_sakshi_handoff_payload,
    run_sakshi_assessment,
)


# ============================================================
# PIPELINE
# ============================================================

class CurrencyAssessmentPipeline:

    def __init__(self):

        self.denomination_adapter = (
            denomination_adapter
        )

        self.damage_adapter = (
            damage_adapter
        )

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        image_input,
        run_severity: bool = True,
    ) -> Dict[str, Any]:

        image_string = str(
            image_input
        )

        # ====================================================
        # 1. YASH - DENOMINATION
        # ====================================================

        denomination_result = (
            self.denomination_adapter
            .predict(
                image_input
            )
        )

        # ====================================================
        # 2. VINIT - DAMAGE CLASSIFICATION
        # ====================================================

        damage_result = (
            self.damage_adapter
            .predict(
                image_input
            )
        )

        # ====================================================
        # 3. SAKSHI HANDOFF
        # ====================================================

        active_damages = (
            get_active_damage_labels(
                damage_result
            )
        )

        sakshi_handoff = (
            get_sakshi_handoff_payload(
                damage_result,

                image_path=(
                    image_string
                ),

                denomination_result=(
                    denomination_result
                ),
            )
        )

        # ====================================================
        # 4. SAKSHI SEGMENTATION / SEVERITY
        # ====================================================

        sakshi_result = None

        sakshi_error = None

        if run_severity:

            try:

                assessment = (
                    run_sakshi_assessment(
                        image_input=(
                            image_input
                        ),

                        denomination_result=(
                            denomination_result
                        ),

                        damage_result=(
                            damage_result
                        ),
                    )
                )

                sakshi_result = (
                    assessment[
                        "result"
                    ]
                )

            except Exception as exc:

                sakshi_error = str(
                    exc
                )

        # ====================================================
        # 5. FORMAT DENOMINATION
        # ====================================================

        denomination = {

            "value":
                denomination_result.get(
                    "denomination"
                ),

            "confidence":
                denomination_result.get(
                    "confidence"
                ),

            "confidence_percent":
                denomination_result.get(
                    "confidence_percent"
                ),

            "class_index":
                denomination_result.get(
                    "class_index"
                ),

            "is_background":
                denomination_result.get(
                    "is_background",
                    False,
                ),

            "accepted":
                denomination_result.get(
                    "accepted"
                ),

            "available":
                denomination_result.get(
                    "available",
                    False,
                ),
        }

        # ====================================================
        # 6. FORMAT DAMAGE
        # ====================================================

        damage = {

            "raw_labels":
                damage_result.get(
                    "raw_damage_labels"
                ),

            "final_labels":
                damage_result.get(
                    "final_damage_labels"
                ),

            "damage_labels":
                damage_result.get(
                    "damage_labels"
                ),

            "predicted_labels":
                damage_result.get(
                    "predicted_labels"
                ),

            "evidence_scores":
                damage_result.get(
                    "evidence_scores",
                    {},
                ),

            "dynamic_thresholds":
                damage_result.get(
                    "dynamic_thresholds",
                    {},
                ),

            "classification_status":
                damage_result.get(
                    "classification_status"
                ),

            "min_evidence_margin":
                damage_result.get(
                    "min_evidence_margin"
                ),

            "calibration_version":
                damage_result.get(
                    "calibration_version"
                ),

            "normal_rule":
                damage_result.get(
                    "normal_rule"
                ),

            "burnt_implies_stain":
                damage_result.get(
                    "burnt_implies_stain"
                ),
        }

        # ====================================================
        # FINAL PIPELINE RESULT
        # ====================================================

        return {

            "success": True,

            "image": {
                "path":
                    image_string,

                "filename":
                    Path(
                        image_string
                    ).name,
            },

            "denomination":
                denomination,

            "damage":
                damage,

            "preprocessing":
                damage_result.get(
                    "preprocessing_metadata",
                    {},
                ),

            "sakshi_active_damages":
                active_damages,

            "sakshi_handoff":
                sakshi_handoff,

            "severity_assessment":
                sakshi_result,

            "severity_error":
                sakshi_error,
        }


# ============================================================
# SINGLETON & ALIASES
# ============================================================

currency_pipeline = (
    CurrencyAssessmentPipeline()
)

CurrencyPipeline = CurrencyAssessmentPipeline


def get_default_pipeline() -> CurrencyAssessmentPipeline:
    return currency_pipeline


def run_currency_analysis(
    image_input,
    run_severity: bool = True,
) -> Dict[str, Any]:
    return currency_pipeline.analyze(
        image_input,
        run_severity=run_severity,
    )