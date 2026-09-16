"""
image_preprocessing.py
============================================================
Low-level, composable image-preprocessing utilities for
currency-note images.

These are intentionally generic building blocks (load, validate,
resize, denoise, threshold, morphology, normalize). Higher-level
logic (note boundary detection, perspective correction, CLIP
input preparation) lives in the sibling modules of this package.

All functions accept/return either a NumPy array (BGR, OpenCV
convention) or raise a clear exception - nothing here silently
swallows a bad image.

Owner: Vinit. This file does not import or modify any existing
project source file; it is fully self-contained.
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python is required for clip_damage_classifier.preprocessing. "
        "Install it with: pip install opencv-python"
    ) from exc

from PIL import Image, UnidentifiedImageError

ImageLike = Union[str, Path, np.ndarray, "Image.Image"]


# ============================================================
# LOADING / VALIDATION
# ============================================================

def load_image(image: ImageLike) -> np.ndarray:
    """
    Load an image into a BGR NumPy array (OpenCV convention),
    regardless of whether the input is a path, a PIL Image, or
    already a NumPy array.

    Raises
    ------
    FileNotFoundError
        If a path is given and the file does not exist.
    ValueError
        If the file exists but cannot be decoded as an image.
    """

    if isinstance(image, np.ndarray):
        return image.copy()

    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, (str, Path)):
        path = Path(image)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Try OpenCV first (fast path).
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if array is not None:
            return array

        # Fall back to PIL for formats/edge cases OpenCV cannot decode
        # (e.g. some palette PNGs, unusual EXIF orientations).
        try:
            with Image.open(path) as pil_image:
                rgb = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not decode image: {path}") from exc

    raise TypeError(
        f"Unsupported image input type: {type(image)!r}. "
        "Expected a path, PIL.Image, or NumPy array."
    )


def is_valid_image(path: ImageLike) -> Tuple[bool, str]:
    """
    Validate that a file exists and can be decoded as an image,
    without raising. Used by the dataset-quality-control pipeline
    to flag corrupted/missing images instead of crashing a batch
    job.

    Returns
    -------
    (is_valid, reason) - reason is "" when is_valid is True.
    """

    try:
        image = load_image(path)
    except FileNotFoundError:
        return False, "missing_file"
    except (ValueError, TypeError) as exc:
        return False, f"corrupted_or_unreadable: {exc}"

    if image is None or image.size == 0:
        return False, "empty_image"

    height, width = image.shape[:2]
    if height < 10 or width < 10:
        return False, "image_too_small"

    return True, ""


# ============================================================
# GEOMETRIC / COLOUR TRANSFORMS
# ============================================================

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize to (width, height) using area interpolation for
    downscaling and cubic interpolation for upscaling."""

    height, width = image.shape[:2]
    target_width, target_height = size

    interpolation = (
        cv2.INTER_AREA
        if (target_width < width or target_height < height)
        else cv2.INTER_CUBIC
    )

    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.

    Useful for currency-note images photographed under uneven
    lighting, which can otherwise wash out faint stains/burns.
    Applied on the luminance channel only, so colour information
    used for stain detection is preserved.
    """

    gray = len(image.shape) == 2
    if gray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, 0)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, kernel_size)


def adaptive_threshold(gray_image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def otsu_threshold(gray_image: np.ndarray) -> np.ndarray:
    _, thresholded = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def canny_edges(
    gray_image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    return cv2.Canny(gray_image, low_threshold, high_threshold)


def morphological_close(
    binary_image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


def morphological_open(
    binary_image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


def dilate(binary_image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.dilate(binary_image, kernel, iterations=iterations)


def erode(binary_image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.erode(binary_image, kernel, iterations=iterations)


def normalize_to_unit_range(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32. NOTE: this is only for
    generic display/analysis use - CLIP inputs must go through
    CLIPProcessor, not this function, to match CLIP's training-time
    normalization exactly (see clip_damage/model.py)."""

    return image.astype(np.float32) / 255.0


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image - the format
    CLIPProcessor expects."""

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
