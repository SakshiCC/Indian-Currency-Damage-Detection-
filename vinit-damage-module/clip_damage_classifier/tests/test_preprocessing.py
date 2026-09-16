"""
test_preprocessing.py
============================================================
Unit tests for OpenCV image preprocessing, note detection,
and CLAHE enhancement.
============================================================
"""

import unittest
import numpy as np
from clip_damage_classifier.preprocessing import image_preprocessing as ip
from clip_damage_classifier.preprocessing import note_detection as nd


class TestImagePreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 100x100 RGB image with a distinct centered "note" rectangle
        self.canvas = np.full((100, 100, 3), 40, dtype=np.uint8)
        # Center "note" is brighter (simulating note on dark table)
        self.canvas[20:80, 15:85] = [180, 200, 190]

    def test_load_image_from_array(self):
        loaded = ip.load_image(self.canvas)
        self.assertEqual(loaded.shape, (100, 100, 3))
        self.assertEqual(loaded.dtype, np.uint8)

    def test_is_valid_image_empty(self):
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        is_val, reason = ip.is_valid_image(empty_arr)
        self.assertFalse(is_val)

    def test_is_valid_image_too_small(self):
        tiny_arr = np.zeros((5, 5, 3), dtype=np.uint8)
        is_val, reason = ip.is_valid_image(tiny_arr)
        self.assertFalse(is_val)
        self.assertEqual(reason, "image_too_small")

    def test_clahe_enhancement(self):
        enhanced = ip.apply_clahe(self.canvas)
        self.assertEqual(enhanced.shape, self.canvas.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_resize_image(self):
        resized = ip.resize_image(self.canvas, (64, 48))
        self.assertEqual(resized.shape, (48, 64, 3))

    def test_grayscale_and_thresholding(self):
        gray = ip.to_grayscale(self.canvas)
        self.assertEqual(len(gray.shape), 2)
        otsu = ip.otsu_threshold(gray)
        self.assertEqual(otsu.shape, gray.shape)
        edges = ip.canny_edges(gray)
        self.assertEqual(edges.shape, gray.shape)

    def test_note_detection_and_crop(self):
        cropped, detected = nd.detect_and_crop_note(self.canvas, apply_perspective_correction=False)
        self.assertTrue(detected)
        self.assertLess(cropped.shape[0], self.canvas.shape[0])
        self.assertLess(cropped.shape[1], self.canvas.shape[1])

    def test_note_detection_fallback(self):
        # Uniform solid image has no distinguishable contour
        blank = np.full((100, 100, 3), 128, dtype=np.uint8)
        cropped, detected = nd.detect_and_crop_note(blank)
        self.assertFalse(detected)
        self.assertEqual(cropped.shape, blank.shape)


if __name__ == "__main__":
    unittest.main()
