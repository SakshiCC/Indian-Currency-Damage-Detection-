"""
test_interface.py
============================================================
Unit tests for the DamageClassifier integration interface.
============================================================
"""

import unittest
import numpy as np
from pathlib import Path
from clip_damage_classifier.interface import DamageClassifier


class TestDamageClassifierInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize classifier once for test suite
        cls.classifier = DamageClassifier(device="cpu", use_preprocessing=True)

    def test_invalid_file_handling(self):
        non_existent = Path("does_not_exist_12345.jpg")
        result = self.classifier.classify(non_existent)
        self.assertFalse(result["success"])
        self.assertIn("Invalid image", result["error"])
        self.assertEqual(result["damage_labels"]["Normal"], 0)

    def test_classify_synthetic_array(self):
        # Create a synthetic note image
        synthetic_img = np.full((120, 200, 3), 150, dtype=np.uint8)
        result = self.classifier.classify(synthetic_img)

        self.assertTrue(result["success"])
        self.assertIn("damage_labels", result)
        self.assertIn("predicted_labels", result)
        self.assertIn("evidence_scores", result)
        self.assertIn("classification_status", result)

        # Check Normal exclusivity invariant
        labels = result["damage_labels"]
        damage_sum = labels["Torn"] + labels["Folded"] + labels["Burnt"] + labels["Stain"]
        if labels["Normal"] == 1:
            self.assertEqual(damage_sum, 0)
        else:
            self.assertGreater(damage_sum, 0)

    def test_batch_classify(self):
        img1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        results = self.classifier.classify_batch([img1, img2])
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["success"])
        self.assertTrue(results[1]["success"])


if __name__ == "__main__":
    unittest.main()
