"""
test_integration.py
Tests the integration layer:
  - Yash adapter failure safety and clean schema
  - Sakshi interface output contract (excluding Normal)
  - Currency pipeline end-to-end flow
"""

import unittest
from integration.yash_adapter import (
    YashDenominationAdapter,
    predict_denomination_safe,
    is_yash_module_available,
)
from integration.sakshi_interface import (
    get_active_damage_labels,
    is_damaged,
    get_damage_evidence,
    get_sakshi_handoff_payload,
)


class TestIntegration(unittest.TestCase):

    def test_yash_adapter_failure_safety(self):
        """Verifies adapter does NOT crash even if TensorFlow is missing."""
        adapter = YashDenominationAdapter()
        res = adapter.predict("non_existent_image.jpg")
        self.assertIsInstance(res, dict)
        self.assertIn("available", res)
        self.assertIn("denomination", res)
        self.assertIn("confidence", res)
        self.assertIn("is_background", res)
        self.assertIn("accepted", res)

    def test_sakshi_interface_excludes_normal(self):
        """Verifies get_active_damage_labels strictly excludes 'Normal'."""
        # Case 1: Torn + Stain
        input_1 = {
            "damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 1, "Normal": 0},
            "predicted_labels": ["Torn", "Stain"],
        }
        active_1 = get_active_damage_labels(input_1)
        self.assertEqual(active_1, ["Torn", "Stain"])
        self.assertTrue(is_damaged(input_1))

        # Case 2: Clean Note -> Normal = 1
        input_clean = {
            "damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 1},
            "predicted_labels": ["Normal"],
        }
        active_clean = get_active_damage_labels(input_clean)
        self.assertEqual(active_clean, [])
        self.assertFalse(is_damaged(input_clean))

    def test_sakshi_handoff_payload(self):
        """Verifies handoff payload contains all required keys for segmentation."""
        sample_res = {
            "damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0},
            "predicted_labels": ["Torn"],
            "evidence_scores": {"Torn": 0.003, "Folded": -0.002, "Burnt": -0.005, "Stain": -0.001},
            "preprocessing_metadata": {"note_detected": True, "area_ratio": 0.85},
            "dynamic_thresholds": {"Torn": -0.0005, "Folded": 0.0, "Burnt": 0.0038, "Stain": 0.0},
            "classification_status": "strong_prediction",
        }
        payload = get_sakshi_handoff_payload(sample_res, image_path="test_note.jpg")
        self.assertEqual(payload["image_path"], "test_note.jpg")
        self.assertTrue(payload["is_damaged"])
        self.assertEqual(payload["active_damage_labels"], ["Torn"])
        self.assertIn("evidence_scores", payload)
        self.assertIn("preprocessing_metadata", payload)
        self.assertIn("dynamic_thresholds", payload)


if __name__ == "__main__":
    unittest.main()
