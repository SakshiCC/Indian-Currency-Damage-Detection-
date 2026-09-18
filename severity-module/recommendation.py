"""
recommendation.py
============================================================
Recommendation Generator for Sakshi's Severity Module.

Generates conservative project-level recommendation guidance
from calculated damage severity.
============================================================
"""

from typing import Dict, List, Any
from config import RECOMMENDATIONS


def generate_recommendation(
    severity: str,
    active_damage_labels: List[str],
    damage_results: Dict[str, Any],
) -> str:
    """
    Generates conservative project-level recommendation text.
    NOTE: These recommendations are project-defined guidance,
    NOT official RBI circulation fitness decisions.
    """
    if not active_damage_labels or "Normal" in active_damage_labels:
        return RECOMMENDATIONS["None"]

    if severity in RECOMMENDATIONS:
        return RECOMMENDATIONS[severity]

    return RECOMMENDATIONS["Moderate"]