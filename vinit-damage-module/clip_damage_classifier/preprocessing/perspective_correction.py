"""
perspective_correction.py
============================================================
Four-point perspective correction for currency-note images that
were photographed at an angle.

Standard OpenCV four-point-transform technique. Used by
note_detection.py once a plausible note contour/quadrilateral has
been found.

Owner: Vinit.
============================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required.") from exc


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order 4 (x, y) points as: top-left, top-right,
    bottom-right, bottom-left.
    """

    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right

    diffs = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left

    return ordered


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Warp the quadrilateral region defined by `points` (4 corner
    points, any order) into an upright rectangle.
    """

    rect = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        # Degenerate quadrilateral - return the original image
        # rather than producing a zero-sized array.
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    return warped


def correct_perspective(
    image: np.ndarray, contour: Optional[np.ndarray]
) -> np.ndarray:
    """
    Attempt perspective correction using a detected note contour.
    Falls back to returning the original image unchanged if no
    usable 4-point quadrilateral is available - this keeps the
    pipeline robust for note photos that are already front-facing.
    """

    if contour is None:
        return image

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        return four_point_transform(image, approx)

    # Not a clean quadrilateral (e.g. torn note with an irregular
    # silhouette) - use the minimum-area bounding rectangle instead
    # of forcing a 4-point warp onto a shape that isn't one.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    return four_point_transform(image, box.astype("float32"))
