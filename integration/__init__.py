"""
integration
============================================================
Top-Level Integration Layer for Indian Currency Note Analysis.

Integrates:
  - Yash: Denomination Recognition (EfficientNetB0 via yash_adapter)
  - Vinit: Multi-Label Damage Classification (V3 Dynamic CLIP Pipeline)
  - Sakshi: Clean Interface Hook for Future Segmentation & Severity

Zero-modification guarantee:
  Leaves src/, models/, and existing team files completely untouched.
============================================================
"""

import sys
from pathlib import Path

# Ensure vinit-damage-module is in sys.path so clip_damage_classifier can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VINIT_DIR = PROJECT_ROOT / "vinit-damage-module"
if str(VINIT_DIR) not in sys.path:
    sys.path.insert(0, str(VINIT_DIR))

from integration.yash_adapter import (
    YashDenominationAdapter,
    get_yash_adapter,
    predict_denomination_safe,
    is_yash_module_available,
)
from integration.vinit_adapter import (
    VinitDamageAdapter,
    get_vinit_adapter,
    predict_damage_safe,
)
from integration.sakshi_interface import (
    get_damage_labels,
    get_active_damage_labels,
    is_damaged,
    get_damage_evidence,
    get_sakshi_handoff_payload,
)
from integration.currency_pipeline import (
    CurrencyPipeline,
    get_default_pipeline,
    run_currency_analysis,
)

__all__ = [
    "YashDenominationAdapter",
    "get_yash_adapter",
    "predict_denomination_safe",
    "is_yash_module_available",
    "VinitDamageAdapter",
    "get_vinit_adapter",
    "predict_damage_safe",
    "get_damage_labels",
    "get_active_damage_labels",
    "is_damaged",
    "get_damage_evidence",
    "get_sakshi_handoff_payload",
    "CurrencyPipeline",
    "get_default_pipeline",
    "run_currency_analysis",
]
