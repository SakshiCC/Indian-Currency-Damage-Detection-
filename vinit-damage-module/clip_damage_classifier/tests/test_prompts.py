"""
Unit tests for contrastive positive and negative prompt ensembles.
"""
import unittest
from clip_damage_classifier.prompts import POSITIVE_PROMPTS, NEGATIVE_PROMPTS, get_positive_prompts, get_negative_prompts

class TestPrompts(unittest.TestCase):
    def test_all_categories_in_positive_and_negative(self):
        expected = ["Torn", "Folded", "Burnt", "Stain"]
        for cat in expected:
            self.assertIn(cat, POSITIVE_PROMPTS)
            self.assertIn(cat, NEGATIVE_PROMPTS)
            self.assertGreater(len(POSITIVE_PROMPTS[cat]), 3)
            self.assertGreater(len(NEGATIVE_PROMPTS[cat]), 3)

    def test_prompts_are_meaningful_strings(self):
        for cat in POSITIVE_PROMPTS:
            for p in POSITIVE_PROMPTS[cat]:
                self.assertIsInstance(p, str)
                self.assertGreater(len(p.strip()), 15)
            for p in NEGATIVE_PROMPTS[cat]:
                self.assertIsInstance(p, str)
                self.assertGreater(len(p.strip()), 15)

if __name__ == "__main__":
    unittest.main()
