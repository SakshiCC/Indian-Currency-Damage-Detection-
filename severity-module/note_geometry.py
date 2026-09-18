from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# DENOMINATION ASPECT-RATIO PRIORS
# ============================================================
#
# These are geometric priors used only to assist reconstruction.
# They are NOT pixel dimensions.
#
# The denomination prediction is never allowed to blindly
# override clearly observed note geometry.
# ============================================================

DENOMINATION_ASPECT_RATIOS = {
    "10": 2.02,
    "20": 2.09,
    "50": 2.16,
    "100": 2.19,
    "200": 2.25,
    "500": 2.28,
    "2000": 2.38,
}


@dataclass
class NoteGeometry:
    visible_mask: np.ndarray
    expected_mask: np.ndarray

    contour: Optional[np.ndarray]

    orientation_degrees: float

    expected_width: float
    expected_height: float
    expected_aspect_ratio: float

    visible_area: int
    expected_area: int

    geometry_score: float

    denomination_used: Optional[str]
    denomination_prior_used: bool

    reconstruction_method: str


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_denomination(
    denomination: Optional[str],
) -> Optional[str]:

    if denomination is None:
        return None

    value = str(denomination)

    value = value.replace("₹", "")
    value = value.replace("INR", "")
    value = value.replace("Rs.", "")
    value = value.replace("Rs", "")
    value = value.replace(",", "")
    value = value.strip()

    if value in DENOMINATION_ASPECT_RATIOS:
        return value

    return None


def ensure_bgr(
    image: np.ndarray,
) -> np.ndarray:

    if image is None:
        raise ValueError("Image is None.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    if image.ndim == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    if (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )

    return image.copy()


# ============================================================
# BACKGROUND / FOREGROUND ANALYSIS
# ============================================================

def estimate_background_colour(
    image: np.ndarray,
) -> np.ndarray:
    """
    Estimate background colour using border pixels.

    Currency-note photographs often contain the background
    around the note, so border pixels provide a useful
    background estimate.
    """

    h, w = image.shape[:2]

    border = max(
        2,
        int(min(h, w) * 0.04),
    )

    samples = np.concatenate(
        [
            image[:border, :, :].reshape(-1, 3),
            image[-border:, :, :].reshape(-1, 3),
            image[:, :border, :].reshape(-1, 3),
            image[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    return np.median(
        samples,
        axis=0,
    ).astype(np.float32)


def create_colour_distance_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Detect pixels sufficiently different from the estimated
    image background.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    background_bgr = (
        estimate_background_colour(
            image
        )
    )

    background_sample = np.uint8(
        [[background_bgr]]
    )

    background_lab = cv2.cvtColor(
        background_sample,
        cv2.COLOR_BGR2LAB,
    )[0, 0].astype(np.float32)

    difference = (
        lab - background_lab
    )

    distance = np.sqrt(
        np.sum(
            difference * difference,
            axis=2,
        )
    )

    # Adaptive threshold.
    threshold = float(
        np.percentile(
            distance,
            60,
        )
    )

    threshold = max(
        12.0,
        threshold,
    )

    mask = np.where(
        distance >= threshold,
        255,
        0,
    ).astype(np.uint8)

    return mask


def create_otsu_candidates(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, dark = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    _, light = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    return dark, light


# ============================================================
# CONTOUR SCORING
# ============================================================

def border_contact_score(
    contour: np.ndarray,
    shape,
) -> float:

    h, w = shape[:2]

    x, y, cw, ch = cv2.boundingRect(
        contour
    )

    margin_x = max(
        2,
        int(w * 0.015),
    )

    margin_y = max(
        2,
        int(h * 0.015),
    )

    contacts = 0

    if x <= margin_x:
        contacts += 1

    if y <= margin_y:
        contacts += 1

    if x + cw >= w - margin_x:
        contacts += 1

    if y + ch >= h - margin_y:
        contacts += 1

    return contacts / 4.0


def score_contour(
    contour: np.ndarray,
    image_shape,
) -> float:

    h, w = image_shape[:2]

    image_area = float(
        h * w
    )

    area = float(
        cv2.contourArea(contour)
    )

    if area <= 0:
        return -1.0

    area_ratio = (
        area / image_area
    )

    if area_ratio < 0.025:
        return -1.0

    if area_ratio > 0.92:
        return -1.0

    rect = cv2.minAreaRect(
        contour
    )

    rw, rh = rect[1]

    if rw <= 1 or rh <= 1:
        return -1.0

    rectangle_area = (
        rw * rh
    )

    rectangularity = min(
        1.0,
        area / rectangle_area,
    )

    long_side = max(
        rw,
        rh,
    )

    short_side = min(
        rw,
        rh,
    )

    aspect = (
        long_side / short_side
    )

    if 1.15 <= aspect <= 3.5:
        aspect_score = 1.0
    else:
        aspect_score = 0.20

    area_score = min(
        1.0,
        area_ratio / 0.35,
    )

    border_penalty = (
        border_contact_score(
            contour,
            image_shape,
        )
    )

    score = (
        0.35 * area_score
        + 0.30 * rectangularity
        + 0.25 * aspect_score
        + 0.10 * (
            1.0 - border_penalty
        )
    )

    return float(score)


# ============================================================
# CANDIDATE MASK CLEANUP
# ============================================================

def cleanup_candidate(
    mask: np.ndarray,
) -> np.ndarray:

    h, w = mask.shape[:2]

    base = min(
        h,
        w,
    )

    kernel_size = max(
        3,
        int(base * 0.010),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return cleaned


def find_best_note_contour(
    image: np.ndarray,
) -> Tuple[np.ndarray, float]:

    colour_mask = (
        create_colour_distance_mask(
            image
        )
    )

    dark_mask, light_mask = (
        create_otsu_candidates(
            image
        )
    )

    candidates = [
        colour_mask,
        dark_mask,
        light_mask,
        cv2.bitwise_or(
            colour_mask,
            dark_mask,
        ),
        cv2.bitwise_or(
            colour_mask,
            light_mask,
        ),
    ]

    best_contour = None
    best_score = -1.0

    for candidate in candidates:

        cleaned = cleanup_candidate(
            candidate
        )

        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:

            score = score_contour(
                contour,
                image.shape,
            )

            if score > best_score:
                best_score = score
                best_contour = contour

    if best_contour is None:
        raise RuntimeError(
            "Unable to detect banknote contour."
        )

    return (
        best_contour,
        best_score,
    )


# ============================================================
# VISIBLE PAPER MASK
# ============================================================

def build_visible_mask(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:

    h, w = image.shape[:2]

    contour_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    cv2.drawContours(
        contour_mask,
        [contour],
        -1,
        255,
        cv2.FILLED,
    )

    # Use the contour as the main physical-note region.
    #
    # Only a very small close is allowed. Large morphology
    # could incorrectly fill real tears.

    kernel_size = max(
        3,
        int(
            min(h, w) * 0.004
        ),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    contour_mask = cv2.morphologyEx(
        contour_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return contour_mask


# ============================================================
# ORIENTATION
# ============================================================

def get_orientation(
    contour: np.ndarray,
) -> float:

    rect = cv2.minAreaRect(
        contour
    )

    (_, _), (rw, rh), angle = rect

    if rw < rh:
        angle += 90.0

    return float(angle)


# ============================================================
# ROTATION
# ============================================================

def rotate_mask(
    mask: np.ndarray,
    angle: float,
):

    h, w = mask.shape[:2]

    center = (
        w / 2.0,
        h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos_value = abs(
        matrix[0, 0]
    )

    sin_value = abs(
        matrix[0, 1]
    )

    new_w = int(
        h * sin_value
        + w * cos_value
    )

    new_h = int(
        h * cos_value
        + w * sin_value
    )

    matrix[0, 2] += (
        new_w / 2
        - center[0]
    )

    matrix[1, 2] += (
        new_h / 2
        - center[1]
    )

    rotated = cv2.warpAffine(
        mask,
        matrix,
        (
            new_w,
            new_h,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        rotated,
        matrix,
    )


def inverse_rotate_mask(
    mask: np.ndarray,
    matrix: np.ndarray,
    original_shape,
) -> np.ndarray:

    h, w = original_shape[:2]

    inverse = (
        cv2.invertAffineTransform(
            matrix
        )
    )

    restored = cv2.warpAffine(
        mask,
        inverse,
        (
            w,
            h,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return restored


# ============================================================
# SUPPORT PROFILE
# ============================================================

def calculate_row_spans(
    mask: np.ndarray,
):

    spans = []

    for y in range(
        mask.shape[0]
    ):

        xs = np.where(
            mask[y] > 0
        )[0]

        if len(xs) < 2:
            continue

        spans.append(
            (
                y,
                int(xs[0]),
                int(xs[-1]),
                int(
                    xs[-1]
                    - xs[0]
                    + 1
                ),
            )
        )

    return spans


def calculate_column_spans(
    mask: np.ndarray,
):

    spans = []

    for x in range(
        mask.shape[1]
    ):

        ys = np.where(
            mask[:, x] > 0
        )[0]

        if len(ys) < 2:
            continue

        spans.append(
            (
                x,
                int(ys[0]),
                int(ys[-1]),
                int(
                    ys[-1]
                    - ys[0]
                    + 1
                ),
            )
        )

    return spans


# ============================================================
# EDGE-ANCHORED EXPECTED NOTE RECONSTRUCTION
# ============================================================

def reconstruct_expected_note(
    rotated_visible: np.ndarray,
    aspect_ratio_prior: Optional[float],
):
    """
    Reconstruct expected complete note using the strongest
    surviving rows/columns rather than the foreground centroid.
    """

    ys, xs = np.where(
        rotated_visible > 0
    )

    if len(xs) == 0:
        raise RuntimeError(
            "Visible note mask is empty."
        )

    observed_left = float(
        np.percentile(
            xs,
            0.5,
        )
    )

    observed_right = float(
        np.percentile(
            xs,
            99.5,
        )
    )

    observed_top = float(
        np.percentile(
            ys,
            0.5,
        )
    )

    observed_bottom = float(
        np.percentile(
            ys,
            99.5,
        )
    )

    observed_width = max(
        1.0,
        observed_right
        - observed_left
        + 1.0,
    )

    observed_height = max(
        1.0,
        observed_bottom
        - observed_top
        + 1.0,
    )

    row_spans = calculate_row_spans(
        rotated_visible
    )

    column_spans = (
        calculate_column_spans(
            rotated_visible
        )
    )

    # --------------------------------------------------------
    # Estimate intact long-side width
    # --------------------------------------------------------

    row_widths = np.array(
        [
            row[3]
            for row in row_spans
        ],
        dtype=np.float32,
    )

    if len(row_widths) > 0:

        expected_width = float(
            np.percentile(
                row_widths,
                95,
            )
        )

    else:

        expected_width = (
            observed_width
        )

    expected_width = max(
        expected_width,
        observed_width * 0.96,
    )

    # --------------------------------------------------------
    # Expected height
    # --------------------------------------------------------

    if (
        aspect_ratio_prior is not None
        and aspect_ratio_prior > 1.0
    ):

        ratio_height = (
            expected_width
            / aspect_ratio_prior
        )

        # Critical safety rule:
        #
        # Never allow the denomination prior to create a huge
        # unsupported region.
        #
        # If ratio-derived height differs dramatically from
        # observed geometry, prefer observed evidence.

        if (
            ratio_height
            >= observed_height * 0.80
            and ratio_height
            <= observed_height * 1.35
        ):

            expected_height = max(
                observed_height,
                ratio_height,
            )

            prior_used = True

        else:

            expected_height = (
                observed_height
            )

            prior_used = False

    else:

        expected_height = (
            observed_height
        )

        prior_used = False

    # --------------------------------------------------------
    # Find reliable rows
    # --------------------------------------------------------

    strong_rows = []

    if row_spans:

        max_row_width = max(
            row[3]
            for row in row_spans
        )

        threshold = (
            max_row_width * 0.88
        )

        strong_rows = [
            row
            for row in row_spans
            if row[3] >= threshold
        ]

    # --------------------------------------------------------
    # Horizontal anchoring
    # --------------------------------------------------------

    if strong_rows:

        reliable_left = float(
            np.median(
                [
                    row[1]
                    for row
                    in strong_rows
                ]
            )
        )

        reliable_right = float(
            np.median(
                [
                    row[2]
                    for row
                    in strong_rows
                ]
            )
        )

    else:

        reliable_left = (
            observed_left
        )

        reliable_right = (
            observed_right
        )

    reliable_width = (
        reliable_right
        - reliable_left
        + 1.0
    )

    expected_width = max(
        expected_width,
        reliable_width,
    )

    # Use the most reliable intact width as anchor.
    center_x = (
        reliable_left
        + reliable_right
    ) / 2.0

    expected_left = (
        center_x
        - expected_width / 2.0
    )

    expected_right = (
        center_x
        + expected_width / 2.0
    )

    # --------------------------------------------------------
    # Vertical anchoring
    # --------------------------------------------------------

    #
    # Unlike the previous algorithm, we do not use the
    # foreground centroid.
    #
    # Start with observed top/bottom and only expand when
    # supported by the aspect-ratio prior.
    #

    missing_height = max(
        0.0,
        expected_height
        - observed_height,
    )

    expected_top = (
        observed_top
    )

    expected_bottom = (
        observed_bottom
    )

    if missing_height > 0:

        # Examine width support near top and bottom.
        band = max(
            2,
            int(
                observed_height
                * 0.12
            ),
        )

        top_start = max(
            0,
            int(observed_top),
        )

        top_end = min(
            rotated_visible.shape[0],
            top_start + band,
        )

        bottom_end = min(
            rotated_visible.shape[0],
            int(observed_bottom) + 1,
        )

        bottom_start = max(
            0,
            bottom_end - band,
        )

        top_support = (
            np.count_nonzero(
                rotated_visible[
                    top_start:top_end
                ]
            )
        )

        bottom_support = (
            np.count_nonzero(
                rotated_visible[
                    bottom_start:
                    bottom_end
                ]
            )
        )

        # Less support generally indicates the damaged side.
        if top_support < (
            bottom_support * 0.75
        ):

            expected_top -= (
                missing_height
            )

        elif bottom_support < (
            top_support * 0.75
        ):

            expected_bottom += (
                missing_height
            )

        else:

            # Ambiguous:
            # distribute conservatively.
            expected_top -= (
                missing_height / 2.0
            )

            expected_bottom += (
                missing_height / 2.0
            )

    # --------------------------------------------------------
    # Make sure all observed paper remains inside
    # --------------------------------------------------------

    expected_left = min(
        expected_left,
        observed_left,
    )

    expected_right = max(
        expected_right,
        observed_right,
    )

    expected_top = min(
        expected_top,
        observed_top,
    )

    expected_bottom = max(
        expected_bottom,
        observed_bottom,
    )

    # --------------------------------------------------------
    # Rasterize as convex-hull polygon (tighter than rectangle)
    # --------------------------------------------------------
    #
    # Build the expected mask by:
    #   1. Taking the convex hull of all visible note pixels
    #      (this closely follows surviving piece boundaries)
    #   2. Expanding that hull outward by the missing_height amount
    #      (applied only along the note's short axis)
    #   3. Clipping to the bounding box derived from the
    #      aspect-ratio estimate so we don't over-expand.

    pts = cv2.findNonZero(rotated_visible)
    result = np.zeros_like(rotated_visible)

    if pts is not None and len(pts) >= 4:
        hull = cv2.convexHull(pts)

        # Shift hull points outward along the vertical axis by
        # missing_height to account for torn-away material.
        # Positive y is downward, negative y is upward.
        if missing_height > 0:
            centroid_y = float(
                np.mean([p[0][1] for p in hull])
            )
            expanded_hull = []
            for p in hull:
                x_p, y_p = float(p[0][0]), float(p[0][1])
                # expand only the vertical component
                dy = y_p - centroid_y
                sign = 1.0 if dy >= 0 else -1.0
                y_new = y_p + sign * missing_height * 0.5
                # clip to the expected bounding box
                y_new = max(expected_top, min(expected_bottom, y_new))
                expanded_hull.append([[int(round(x_p)), int(round(y_new))]])
            hull = np.array(expanded_hull, dtype=np.int32)

        # Clip hull points to the expected bounding box
        x1_bb = max(0, int(round(expected_left)))
        x2_bb = min(result.shape[1] - 1, int(round(expected_right)))
        y1_bb = max(0, int(round(expected_top)))
        y2_bb = min(result.shape[0] - 1, int(round(expected_bottom)))

        clipped_hull = []
        for p in hull:
            cx = int(np.clip(p[0][0], x1_bb, x2_bb))
            cy = int(np.clip(p[0][1], y1_bb, y2_bb))
            clipped_hull.append([[cx, cy]])
        clipped_hull = np.array(clipped_hull, dtype=np.int32)

        cv2.fillPoly(result, [clipped_hull], 255)
    else:
        # Fallback: axis-aligned rectangle when too few points
        x1_bb = max(0, int(round(expected_left)))
        x2_bb = min(result.shape[1] - 1, int(round(expected_right)))
        y1_bb = max(0, int(round(expected_top)))
        y2_bb = min(result.shape[0] - 1, int(round(expected_bottom)))
        cv2.rectangle(result, (x1_bb, y1_bb), (x2_bb, y2_bb), 255, cv2.FILLED)

    # Measure final geometry from the filled polygon
    ys_f, xs_f = np.where(result > 0)
    if len(xs_f) > 0:
        final_width = float(xs_f.max() - xs_f.min() + 1)
        final_height = float(ys_f.max() - ys_f.min() + 1)
    else:
        final_width = max(1.0, expected_right - expected_left + 1)
        final_height = max(1.0, expected_bottom - expected_top + 1)

    final_ratio = final_width / max(final_height, 1.0)

    return (
        result,
        final_width,
        final_height,
        final_ratio,
        prior_used,
    )


# ============================================================
# MAIN API
# ============================================================

def estimate_note_geometry(
    image: np.ndarray,
    denomination: Optional[str] = None,
    denomination_confidence: Optional[float] = None,
    preprocessing_metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> NoteGeometry:

    image = ensure_bgr(
        image
    )

    contour, score = (
        find_best_note_contour(
            image
        )
    )

    visible_mask = (
        build_visible_mask(
            image,
            contour,
        )
    )

    orientation = (
        get_orientation(
            contour
        )
    )

    rotated_visible, matrix = (
        rotate_mask(
            visible_mask,
            -orientation,
        )
    )

    denomination_value = (
        normalize_denomination(
            denomination
        )
    )

    aspect_prior = None

    if denomination_value is not None:

        if (
            denomination_confidence
            is None
            or denomination_confidence
            >= 0.60
        ):

            aspect_prior = (
                DENOMINATION_ASPECT_RATIOS[
                    denomination_value
                ]
            )

    (
        rotated_expected,
        expected_width,
        expected_height,
        expected_ratio,
        prior_used,
    ) = reconstruct_expected_note(
        rotated_visible,
        aspect_prior,
    )

    expected_mask = (
        inverse_rotate_mask(
            rotated_expected,
            matrix,
            image.shape,
        )
    )

    # Refine expected_mask: intersect with convex hull of all visible note pixels
    # This prevents the mask extending into pure background regions.
    pts = cv2.findNonZero(visible_mask)
    if pts is not None and len(pts) >= 4:
        hull_pts = cv2.convexHull(pts)
        # Dilate hull slightly to account for torn-edge pixel gaps
        hull_mask = np.zeros_like(visible_mask)
        cv2.fillPoly(hull_mask, [hull_pts], 255)
        # Expand hull by a small margin (~1% of shorter dimension)
        h_vm, w_vm = visible_mask.shape[:2]
        expand_px = max(4, int(min(h_vm, w_vm) * 0.025))
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1)
        )
        hull_mask_dilated = cv2.dilate(hull_mask, expand_kernel, iterations=1)
        expected_mask = cv2.bitwise_and(expected_mask, hull_mask_dilated)

    # Guarantee actual detected paper belongs to expected note.
    expected_mask = cv2.bitwise_or(
        expected_mask,
        visible_mask,
    )

    visible_area = int(
        cv2.countNonZero(
            visible_mask
        )
    )

    expected_area = int(
        cv2.countNonZero(
            expected_mask
        )
    )

    return NoteGeometry(
        visible_mask=visible_mask,
        expected_mask=expected_mask,
        contour=contour,

        orientation_degrees=float(
            orientation
        ),

        expected_width=float(
            expected_width
        ),

        expected_height=float(
            expected_height
        ),

        expected_aspect_ratio=float(
            expected_ratio
        ),

        visible_area=visible_area,
        expected_area=expected_area,

        geometry_score=float(
            score
        ),

        denomination_used=(
            denomination_value
        ),

        denomination_prior_used=(
            prior_used
        ),

        reconstruction_method=(
            "convex_hull_polygon_v3"
        ),
    )