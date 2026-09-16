"""
test_integration_pipeline.py
============================================================
Comprehensive test suite for the top-level integration layer:
  1. Yash adapter (read-only, failure-safe, schema conformance)
  2. Vinit adapter (V3.1 calibration, consistency rules, schema conformance)
  3. Combined CurrencyPipeline (orchestration, singleton reuse, schema conformance)
  4. Sakshi interface hooks (clean consumption without segmentation logic)
  5. End-to-end verification on real sample banknote image
============================================================
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

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

SAMPLE_IMAGE_PATH = Path(
    "clip_damage_classifier/data/extracted/Spoilt Indian Banknotes/Spoilt New Indian Bank Notes/New 10 Rupees/10_new_1.jpg"
)


class TestYashAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = YashDenominationAdapter()

    def test_yash_adapter_initialization(self):
        """Adapter safely initializes and reports availability without raising exceptions."""
        self.assertIsInstance(self.adapter.is_available, bool)

    def test_yash_adapter_predict_schema(self):
        """Adapter predict method returns well-defined schema on dummy input."""
        dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        res = self.adapter.predict(dummy_img)

        self.assertIn("available", res)
        self.assertIn("denomination", res)
        self.assertIn("confidence", res)
        self.assertIn("confidence_percent", res)
        self.assertIn("class_index", res)
        self.assertIn("is_background", res)
        self.assertIn("accepted", res)

    def test_yash_adapter_singleton(self):
        """get_yash_adapter returns consistent singleton instance."""
        ad1 = get_yash_adapter()
        ad2 = get_yash_adapter()
        self.assertIs(ad1, ad2)


class TestVinitAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = get_vinit_adapter(device="cpu", calibration_version="v3.1")

    def test_vinit_adapter_initialization(self):
        """Vinit adapter initializes DamageClassifier with V3.1 mode."""
        self.assertTrue(self.adapter.is_available)
        self.assertEqual(self.adapter.calibration_version, "v3.1")

    def test_vinit_adapter_predict_schema(self):
        """Vinit adapter returns complete normalized damage schema."""
        dummy_img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        res = self.adapter.predict(dummy_img)

        self.assertTrue(res["success"])
        self.assertIn("damage_labels", res)
        self.assertIn("raw_damage_labels", res)
        self.assertIn("final_damage_labels", res)
        self.assertIn("predicted_labels", res)
        self.assertIn("evidence_scores", res)
        self.assertIn("dynamic_thresholds", res)
        self.assertIn("classification_status", res)
        self.assertIn("normal_rule", res)
        self.assertEqual(res["normal_rule"], "NOT(Burnt OR Torn)")
        self.assertTrue(res.get("burnt_implies_stain", False))
        self.assertEqual(res["calibration_version"], "v3.1")

    def test_vinit_consistency_rules_invariance(self):
        """Verifies that final_damage_labels strictly obeys Burnt->Stain and Normal rule."""
        dummy_img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        res = self.adapter.predict(dummy_img)
        labels = res["final_damage_labels"]

        # Rule 1: Burnt == 1 ==> Stain == 1
        if labels["Burnt"] == 1:
            self.assertEqual(labels["Stain"], 1)

        # Rule 2: Normal == 1 iff Burnt == 0 AND Torn == 0
        if labels["Burnt"] == 0 and labels["Torn"] == 0:
            self.assertEqual(labels["Normal"], 1)
        else:
            self.assertEqual(labels["Normal"], 0)


class TestSakshiInterface(unittest.TestCase):

    def test_get_damage_labels(self):
        sample_res = {
            "final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 1, "Normal": 0}
        }
        labels = get_damage_labels(sample_res)
        self.assertEqual(labels["Torn"], 1)
        self.assertEqual(labels["Stain"], 1)
        self.assertEqual(labels["Normal"], 0)

    def test_get_active_damage_labels_excludes_normal(self):
        sample_res = {
            "final_damage_labels": {"Torn": 0, "Folded": 1, "Burnt": 0, "Stain": 0, "Normal": 1}
        }
        active = get_active_damage_labels(sample_res)
        self.assertEqual(active, ["Folded"])
        self.assertNotIn("Normal", active)

    def test_is_damaged(self):
        clean_res = {"final_damage_labels": {"Torn": 0, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 1}}
        damaged_res = {"final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 0, "Normal": 0}}
        self.assertFalse(is_damaged(clean_res))
        self.assertTrue(is_damaged(damaged_res))

    def test_sakshi_handoff_payload(self):
        sample_res = {
            "damage": {
                "final_damage_labels": {"Torn": 1, "Folded": 0, "Burnt": 0, "Stain": 1, "Normal": 0},
                "evidence_scores": {"Torn": 0.005},
                "preprocessing_metadata": {"note_detected": True},
            }
        }
        payload = get_sakshi_handoff_payload(sample_res, image_path="test_note.jpg")
        self.assertEqual(payload["image_path"], "test_note.jpg")
        self.assertTrue(payload["is_damaged"])
        self.assertEqual(set(payload["active_damage_labels"]), {"Torn", "Stain"})
        self.assertTrue(payload["preprocessing_metadata"]["note_detected"])


class TestCurrencyPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = get_default_pipeline(device="cpu", calibration_version="v3.1")

    def test_pipeline_schema_on_synthetic_image(self):
        """Pipeline returns structured combined schema matching Section 10."""
        dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        res = self.pipeline.analyze(dummy_img)

        self.assertIn("success", res)
        self.assertIn("image", res)
        self.assertIn("denomination", res)
        self.assertIn("damage", res)
        self.assertIn("preprocessing", res)
        self.assertIn("sakshi_active_damages", res)
        self.assertIn("sakshi_handoff", res)

        # Denomination schema check
        denom = res["denomination"]
        self.assertIn("value", denom)
        self.assertIn("confidence", denom)

        # Damage schema check
        damage = res["damage"]
        self.assertIn("raw_labels", damage)
        self.assertIn("final_labels", damage)
        self.assertIn("predicted_labels", damage)
        self.assertIn("evidence_scores", damage)
        self.assertEqual(damage["calibration_version"], "v3.1")

    def test_pipeline_on_real_banknote_sample(self):
        """Pipeline runs successfully on real benchmark banknote image."""
        if not SAMPLE_IMAGE_PATH.exists():
            self.skipTest(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

        res = self.pipeline.analyze(str(SAMPLE_IMAGE_PATH))

        self.assertTrue(res["success"])
        self.assertEqual(res["image"]["filename"], "10_new_1.jpg")

        damage = res["damage"]
        # Invariant checks
        self.assertEqual(damage["final_labels"]["Torn"], 1)
        self.assertEqual(damage["final_labels"]["Stain"], 1)
        self.assertEqual(damage["final_labels"]["Normal"], 0)
        self.assertIn("Torn", damage["predicted_labels"])
        self.assertIn("Stain", damage["predicted_labels"])

        # Sakshi handoff check
        self.assertTrue(res["sakshi_handoff"]["is_damaged"])
        self.assertIn("Torn", res["sakshi_active_damages"])


if __name__ == "__main__":
    unittest.main()
