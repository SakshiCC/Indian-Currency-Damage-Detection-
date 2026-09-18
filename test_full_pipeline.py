from __future__ import annotations

import json
import sys
from pathlib import Path

from integration.currency_pipeline import (
    currency_pipeline,
)


def main():

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python test_full_pipeline.py "
            "<image_path>"
        )

        print()

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "FULL CURRENCY NOTE "
        "ASSESSMENT PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Image: {image_path}"
    )

    print()

    result = (
        currency_pipeline.analyze(
            image_path
        )
    )

    # ========================================================
    # YASH
    # ========================================================

    denomination = (
        result[
            "denomination"
        ]
    )

    print(
        "YASH - DENOMINATION"
    )

    print(
        "-" * 70
    )

    print(
        f"Denomination : "
        f"{denomination.get('value')}"
    )

    print(
        f"Confidence   : "
        f"{denomination.get('confidence_percent')}"
    )

    print(
        f"Background   : "
        f"{denomination.get('is_background')}"
    )

    print()

    # ========================================================
    # VINIT
    # ========================================================

    damage = result[
        "damage"
    ]

    print(
        "VINIT - DAMAGE CLASSIFICATION"
    )

    print(
        "-" * 70
    )

    print(
        f"Active Damage : "
        f"{result['sakshi_active_damages']}"
    )

    print(
        f"Status        : "
        f"{damage.get('classification_status')}"
    )

    print(
        f"Evidence      : "
        f"{damage.get('evidence_scores')}"
    )

    print()

    # ========================================================
    # SAKSHI
    # ========================================================

    print(
        "SAKSHI - SEGMENTATION / SEVERITY"
    )

    print(
        "-" * 70
    )

    severity = result.get(
        "severity_assessment"
    )

    if severity is None:

        print(
            "Severity assessment failed."
        )

        print(
            "Error:",
            result.get(
                "severity_error"
            ),
        )

    else:

        print(
            f"Damage Types   : "
            f"{severity.get('damage_types')}"
        )

        print(
            f"Damaged Pixels : "
            f"{severity.get('damaged_pixels')}"
        )

        print(
            f"Note Pixels    : "
            f"{severity.get('total_note_pixels')}"
        )

        print(
            f"Damage %       : "
            f"{severity.get('damage_percentage')}"
        )

        print(
            f"Severity       : "
            f"{severity.get('severity')}"
        )

        print(
            f"Geometry Score : "
            f"{severity.get('geometry_score')}"
        )

        print(
            f"Prior Used     : "
            f"{severity.get('denomination_prior_used')}"
        )

        print(
            f"Recommendation : "
            f"{severity.get('recommendation')}"
        )

    print()
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()