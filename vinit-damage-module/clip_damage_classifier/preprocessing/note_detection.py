"""
note_detection.py
============================================================
Currency-note boundary detection and background removal.

Approach: classical contour-based detection (Otsu / adaptive
threshold -> morphological cleanup -> largest plausible contour),
NOT a learned detector. This is intentional: no note-detection
model or dataset existed anywhere in the inspected project, so a
classical-CV approach is used first. If/when a labelled
note-detection dataset becomes available, this module is the
natural place to plug in a learned detector without touching
callers (clip_damage/inference.py only calls
`detect_and_crop_note`).

IMPORTANT: this module is intentionally conservative. If no
confident note boundary is found, it returns the ORIGINAL image
rather than an aggressive/incorrect crop, because a bad crop would
silently corrupt downstream CLIP damage scoring.

Owner: Vinit.
============================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required.") from exc

from . import image_preprocessing as ip
from . import perspective_correction as pc

from clip_damage_classifier import config


def find_note_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest contour that plausibly represents a currency
    note against a background.

    Returns None if no contour meets the minimum area ratio - this
    is a normal, expected outcome for close-up / full-frame note
    photos where there is no distinguishable background at all.
    """

    height, width = image.shape[:2]
    image_area = float(height * width)

    gray = ip.to_grayscale(image)
    blurred = ip.gaussian_blur(gray, config.GAUSSIAN_BLUR_KERNEL)

    thresholded = ip.otsu_threshold(blurred)

    # Otsu can select either polarity depending on background
    # brightness; make sure the "foreground" (note) is the smaller,
    # more central mass rather than assuming a fixed polarity.
    closed = ip.morphological_close(thresholded, config.MORPH_KERNEL_SIZE)
    opened = ip.morphological_open(closed, config.MORPH_KERNEL_SIZE)

    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)

    if largest_area / image_area < config.MIN_NOTE_CONTOUR_AREA_RATIO:
        return None

    # A contour covering almost the entire frame usually means the
    # note already fills the photo (no real background to separate),
    # so detection would just be a no-op crop - treat as "not found"
    # to avoid an unnecessary, potentially lossy re-warp.
    if largest_area / image_area > 0.97:
        return None

    return largest


def crop_to_bounding_box(image: np.ndarray, contour: np.ndarray, padding: int = 4) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)

    height, width = image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)

    return image[y0:y1, x0:x1]


def detect_and_crop_note(
    image: np.ndarray,
    apply_perspective_correction: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Full note-detection pipeline: locate the note against its
    background, optionally correct perspective, and crop.

    Returns
    -------
    (processed_image, note_detected)
        note_detected is False when the function fell back to
        returning the original image unchanged.
    """

    contour = find_note_contour(image)

    if contour is None:
        return image, False

    if apply_perspective_correction:
        corrected = pc.correct_perspective(image, contour)
        return corrected, True

    cropped = crop_to_bounding_box(image, contour)
    return cropped, True
