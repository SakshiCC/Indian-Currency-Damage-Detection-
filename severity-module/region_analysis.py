"""
region_analysis.py
============================================================
Region Analysis module for Sakshi's Damage Severity Assessment.
Extracts contours, pixel area, bounding box, centroid, and grid
location for each detected damage region.

Applies strict area and percentage filtering to suppress tiny clutter.
============================================================
"""

from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

from config import MIN_REGION_AREA, MIN_REGION_PCT, LOCATION_GRID_NAMES


def get_grid_location(
    centroid: Tuple[int, int],
    note_bbox: Tuple[int, int, int, int],
) -> str:
    """
    Determines approximate 3x3 grid location of a centroid relative to note bbox.
    Grid labels: top-left, top-center, top-right, middle-left, center,
                 middle-right, bottom-left, bottom-center, bottom-right.
    """
    cx, cy = centroid
    nx, ny, nw, nh = note_bbox

    if nw <= 0 or nh <= 0:
        return "center"

    rel_x = (cx - nx) / float(nw)
    rel_y = (cy - ny) / float(nh)

    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))

    col = 0 if rel_x < 0.333 else (1 if rel_x < 0.666 else 2)
    row = 0 if rel_y < 0.333 else (1 if rel_y < 0.666 else 2)

    return LOCATION_GRID_NAMES[row][col]


def analyze_damage_regions(
    damage_mask: np.ndarray,
    note_mask: np.ndarray,
    damage_type: str = "Unknown",
    min_area: int = MIN_REGION_AREA,
    min_pct: float = MIN_REGION_PCT,
) -> List[Dict[str, Any]]:
    """
    Extracts and measures damage region contours from a binary damage mask.
    Filters out regions smaller than min_area or min_pct of total note area.
    """
    if damage_mask is None or cv2.countNonZero(damage_mask) == 0:
        return []

    total_note_pixels = cv2.countNonZero(note_mask) if note_mask is not None else 0
    if total_note_pixels <= 0:
        total_note_pixels = damage_mask.shape[0] * damage_mask.shape[1]

    # Calculate note bounding box for relative grid placement
    if note_mask is not None and cv2.countNonZero(note_mask) > 0:
        note_pts = cv2.findNonZero(note_mask)
        nx, ny, nw, nh = cv2.boundingRect(note_pts)
    else:
        nx, ny, nw, nh = 0, 0, damage_mask.shape[1], damage_mask.shape[0]

    # Clip analysis strictly inside note_mask if provided
    if note_mask is not None:
        mask_to_analyze = cv2.bitwise_and(damage_mask, note_mask)
    else:
        mask_to_analyze = damage_mask.copy()

    contours, _ = cv2.findContours(
        mask_to_analyze, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions: List[Dict[str, Any]] = []

    for contour in contours:
        area_px = int(cv2.contourArea(contour))
        area_pct = round((area_px / float(total_note_pixels)) * 100.0, 2)

        # Filter tiny clutter regions
        if area_px < min_area or area_pct < min_pct:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        loc = get_grid_location((cx, cy), (nx, ny, nw, nh))

        regions.append(
            {
                "damage_type": damage_type,
                "area_pixels": area_px,
                "area_percentage": area_pct,
                "bounding_box": [int(x), int(y), int(w), int(h)],
                "centroid": [int(cx), int(cy)],
                "location": loc,
            }
        )

    # Sort regions by area descending
    regions.sort(key=lambda r: r["area_pixels"], reverse=True)
    return regions
