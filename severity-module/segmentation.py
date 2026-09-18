"""
segmentation.py
============================================================
Damage-Specific Segmentation Module for Sakshi's Severity Module.

Implements strict classical image processing algorithms for:
  - Note Mask Creation / Visible & Expected Geometry
  - Torn Damage Segmentation (strictly clipped to expected_note_mask)
  - Stain Damage Segmentation (strictly inside visible_note_mask, text/portrait false-positive rejection)
  - Burnt Damage Segmentation (charred/dark regions inside visible note)
  - Folded Damage Segmentation (crease lines inside visible note)
  - Multi-Damage Union Fusion (bitwise OR clipped to expected note footprint)
============================================================
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from config import (
    MIN_REGION_AREA,
    STAIN_COLOR_DIFF_THRESHOLD,
    STAIN_MIN_AREA,
    BURNT_INTENSITY_THRESHOLD,
    BURNT_MIN_AREA,
    FOLD_CANNY_LOW,
    FOLD_CANNY_HIGH,
)
from note_geometry import NoteGeometry, estimate_note_geometry


DAMAGE_TYPES = ["Torn", "Folded", "Burnt", "Stain"]


def clean_components(
    mask: np.ndarray,
    minimum_area: int,
    max_aspect_ratio: Optional[float] = None,
    min_solidity: Optional[float] = None,
) -> np.ndarray:
    """Removes small isolated noise components and fine text/edge structures below minimum_area."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return np.zeros_like(mask) if mask is not None else np.array([])

    output = np.zeros_like(mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / float(max(1, min(w, h)))

        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue

        if min_solidity is not None:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(max(1.0, hull_area))
            if solidity < min_solidity:
                continue

        cv2.drawContours(output, [contour], -1, 255, cv2.FILLED)

    return output


def create_inner_boundary_tolerance_mask(expected_mask: np.ndarray) -> np.ndarray:
    """
    Creates a narrow inner boundary tolerance band along expected note edges
    to prevent minor registration noise from being marked as torn.
    """
    if expected_mask is None or cv2.countNonZero(expected_mask) == 0:
        return np.zeros_like(expected_mask)

    h, w = expected_mask.shape[:2]
    thickness = max(2, int(min(h, w) * 0.006))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (thickness * 2 + 1, thickness * 2 + 1)
    )

    eroded = cv2.erode(expected_mask, kernel, iterations=1)
    boundary_band = cv2.subtract(expected_mask, eroded)
    return boundary_band


# ============================================================
# 1. TORN SEGMENTATION
# ============================================================

def segment_torn(
    image: np.ndarray,
    geometry: NoteGeometry,
) -> np.ndarray:
    """
    Segment torn/missing currency note regions.
    Formula: torn_missing_mask = expected_note_mask AND NOT visible_note_mask
    Strict clipping: torn_mask = torn_missing_mask AND expected_note_mask

    CRITICAL: NO pixels outside expected_note_mask can EVER be marked as Torn.
    Background outside expected note footprint remains strictly BLACK (0).
    """
    expected = geometry.expected_mask
    visible = geometry.visible_mask

    if expected is None or visible is None or cv2.countNonZero(expected) == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Missing material inside expected note footprint
    visible_inside_expected = cv2.bitwise_and(visible, expected)
    torn_missing_raw = cv2.subtract(expected, visible_inside_expected)

    # Filter thin registration edge tolerance band
    tolerance_band = create_inner_boundary_tolerance_mask(expected)
    tolerance_inv = cv2.bitwise_not(tolerance_band)
    torn_missing_clean = cv2.bitwise_and(torn_missing_raw, tolerance_inv)

    # Strictly clip to expected note mask
    torn_mask_clipped = cv2.bitwise_and(torn_missing_clean, expected)

    # Filter out tiny noise artifacts, then remove large background-shaped blobs
    expected_area = cv2.countNonZero(expected)
    minimum_area = max(MIN_REGION_AREA, int(expected_area * 0.0015))

    cleaned_torn = clean_components(torn_mask_clipped, minimum_area)

    # --- Fix 2: drop connected components that span the full image width
    # or touch 3+ image edges (classic sign of a background/non-note blob).
    h_img, w_img = cleaned_torn.shape[:2]
    edge_margin_x = max(4, int(w_img * 0.03))
    edge_margin_y = max(4, int(h_img * 0.03))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_torn, connectivity=8)
    result_torn = np.zeros_like(cleaned_torn)

    for lbl in range(1, n_labels):
        x_s = stats[lbl, cv2.CC_STAT_LEFT]
        y_s = stats[lbl, cv2.CC_STAT_TOP]
        w_s = stats[lbl, cv2.CC_STAT_WIDTH]
        h_s = stats[lbl, cv2.CC_STAT_HEIGHT]
        area_s = stats[lbl, cv2.CC_STAT_AREA]

        # Count how many image borders this component touches
        touches_left   = int(x_s <= edge_margin_x)
        touches_top    = int(y_s <= edge_margin_y)
        touches_right  = int(x_s + w_s >= w_img - edge_margin_x)
        touches_bottom = int(y_s + h_s >= h_img - edge_margin_y)
        border_touches = touches_left + touches_top + touches_right + touches_bottom

        # If a component spans nearly the full width AND is a large blob,
        # it is likely a background artifact — drop it.
        width_ratio = w_s / float(w_img)
        area_ratio  = area_s / float(expected_area)

        if border_touches >= 3 and area_ratio > 0.05:
            continue  # background-shaped blob, discard

        if width_ratio > 0.90 and area_ratio > 0.08:
            continue  # nearly full-width horizontal band, discard

        # Keep all other meaningful components
        result_torn[labels == lbl] = 255

    # Enforce strict invariant: torn_missing_mask <= expected_note_mask
    result_torn = cv2.bitwise_and(result_torn, expected)
    return result_torn


# ============================================================
# 2. STAIN SEGMENTATION
# ============================================================

def segment_stain(
    image: np.ndarray,
    geometry: NoteGeometry,
) -> np.ndarray:
    """
    Segment stain damage conservatively inside visible_note_mask ONLY.
    Rejects text, serial numbers, portrait details, patterns, and borders.
    """
    visible_mask = geometry.visible_mask
    if visible_mask is None or cv2.countNonZero(visible_mask) == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    h_img, w_img = image.shape[:2]
    short_side = min(h_img, w_img)

    # Adaptive erosion: ~1.5% of short side to pull away from note edges and
    # torn-edge artifacts (minimum 7 px to avoid text-border contamination).
    erode_k = max(7, int(short_side * 0.015))
    if erode_k % 2 == 0:
        erode_k += 1
    eroded_visible = cv2.erode(
        visible_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (erode_k, erode_k)),
        iterations=1,
    )

    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # High blur spatial background paper estimation (31x31 median filter)
    blurred = cv2.medianBlur(image, 31)
    blurred_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute color/lightness distance inside visible paper
    delta = lab - blurred_lab
    dist = np.sqrt(np.sum(delta * delta, axis=2))

    # Threshold for stain anomaly — adaptive to reject mere printing noise.
    # 35 balances detection on genuine stains vs false positives from torn edges.
    stain_raw = np.where(
        (dist > STAIN_COLOR_DIFF_THRESHOLD) & (eroded_visible > 0), 255, 0
    ).astype(np.uint8)

    # Adaptive morphological kernel: ~0.3% of short side (max 5 px).
    # Keeps stain blobs intact on high-res images while erasing fine text.
    morph_k = min(5, max(3, int(short_side * 0.003)))
    if morph_k % 2 == 0:
        morph_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    stain_opened = cv2.morphologyEx(stain_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    stain_closed = cv2.morphologyEx(stain_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Adaptive min area: scale with expected note size (0.1% of expected, min 300 px)
    expected_area = cv2.countNonZero(geometry.expected_mask)
    adaptive_min_area = max(STAIN_MIN_AREA, int(expected_area * 0.001))

    # Filter out high-aspect ratio / thin edge structures and small regions
    stain_clean = clean_components(
        stain_closed,
        minimum_area=adaptive_min_area,
        max_aspect_ratio=3.0,
        min_solidity=0.50,
    )

    # Constrain strictly to visible note mask
    return cv2.bitwise_and(stain_clean, visible_mask)



# ============================================================
# 3. BURNT SEGMENTATION
# ============================================================

def segment_burnt(
    image: np.ndarray,
    geometry: NoteGeometry,
) -> np.ndarray:
    """
    Segment burnt/charred damage inside visible_note_mask ONLY.
    """
    visible_mask = geometry.visible_mask
    if visible_mask is None or cv2.countNonZero(visible_mask) == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Low intensity (dark/charred) inside visible note
    dark_mask = np.where(
        (v_channel < BURNT_INTENSITY_THRESHOLD)
        & (gray < BURNT_INTENSITY_THRESHOLD)
        & (visible_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    burnt_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    burnt_clean = cv2.morphologyEx(burnt_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned = clean_components(burnt_clean, BURNT_MIN_AREA)
    return cv2.bitwise_and(cleaned, visible_mask)


# ============================================================
# 4. FOLDED SEGMENTATION
# ============================================================

def segment_folded(
    image: np.ndarray,
    geometry: NoteGeometry,
) -> np.ndarray:
    """
    Segment crease / fold lines inside visible_note_mask ONLY.
    """
    visible_mask = geometry.visible_mask
    if visible_mask is None or cv2.countNonZero(visible_mask) == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, FOLD_CANNY_LOW, FOLD_CANNY_HIGH)
    edges = cv2.bitwise_and(edges, visible_mask)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

    creases = cv2.bitwise_or(h_lines, v_lines)
    dilated_creases = cv2.dilate(
        creases, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    cleaned = clean_components(dilated_creases, MIN_REGION_AREA)
    return cv2.bitwise_and(cleaned, visible_mask)


# ============================================================
# MASK FUSION & CONTROLLER
# ============================================================

def combine_masks(
    masks: Dict[str, np.ndarray], shape: tuple, expected_note_mask: np.ndarray
) -> np.ndarray:
    """
    Combines damage masks via bitwise OR (union) and strictly clips
    to expected_note_mask.
    """
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks.values():
        if mask is not None and mask.shape == shape:
            combined = cv2.bitwise_or(combined, mask)

    # Enforce strict clipping rule: combined_mask AND expected_note_mask
    if expected_note_mask is not None and expected_note_mask.shape == shape:
        combined = cv2.bitwise_and(combined, expected_note_mask)

    return combined


def segment_damage(
    image: np.ndarray,
    damage_labels: List[str],
    denomination: Optional[str] = None,
    denomination_confidence: Optional[float] = None,
    preprocessing_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Sakshi's damage segmentation module.
    """
    geometry = estimate_note_geometry(
        image=image,
        denomination=denomination,
        denomination_confidence=denomination_confidence,
        preprocessing_metadata=preprocessing_metadata,
    )

    active_labels = [label for label in damage_labels if label in DAMAGE_TYPES]

    if "Normal" in damage_labels and len(active_labels) == 0:
        active_labels = []

    damage_masks: Dict[str, np.ndarray] = {}

    for label in active_labels:
        if label == "Torn":
            damage_masks[label] = segment_torn(image, geometry)
        elif label == "Stain":
            damage_masks[label] = segment_stain(image, geometry)
        elif label == "Burnt":
            damage_masks[label] = segment_burnt(image, geometry)
        elif label == "Folded":
            damage_masks[label] = segment_folded(image, geometry)

    combined_mask = combine_masks(
        damage_masks, geometry.expected_mask.shape, geometry.expected_mask
    )

    return {
        "geometry": geometry,
        "damage_masks": damage_masks,
        "combined_mask": combined_mask,
    }