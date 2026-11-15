"""Skeleton detection: the ruled-line structure of a form.

A fixed-layout form is mostly defined by its ruling: the outer border, the
table grid, the boxes around fields. That structure survives scanning,
photocopying and phone photos far better than text does, and it does not
depend on what was typed into the form. This module extracts it with plain
OpenCV morphology:

1. ``binarize``      adaptive threshold -> ink mask (ink = 255)
2. ``line_masks``    close (bridge dashes / scan gaps) then open with long
                     horizontal / vertical kernels -> masks of ruled lines only
3. ``find_junctions``  where a horizontal and a vertical line meet. Each
                     junction is a keypoint with an *arm signature* (which of
                     left/right/up/down has a line), so an L-corner, a T and a
                     cross are distinguishable during matching.
4. ``frame_candidates``  large quadrilaterals in the raw image (form border,
                     paper edge) used to initialise the homography when the
                     page is rotated or in perspective.

The morphological line detector expects lines to be close to axis-aligned.
``fit.fit_page`` guarantees that by rectifying the page with a coarse
homography before calling ``extract_skeleton``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

# Arm bit flags for junction signatures.
ARM_L, ARM_R, ARM_U, ARM_D = 1, 2, 4, 8
_ARM_CHARS = (("L", ARM_L), ("R", ARM_R), ("U", ARM_U), ("D", ARM_D))


def arms_to_str(arms: int) -> str:
    return "".join(c for c, bit in _ARM_CHARS if arms & bit)


def arms_from_str(s: str) -> int:
    lookup = dict(_ARM_CHARS)
    return sum(lookup[c] for c in s.upper() if c in lookup)


def rotate_arms(arms: int, quarter_turns_cw: int) -> int:
    """Rotate an arm signature by k * 90 degrees clockwise (image coordinates)."""
    order = [ARM_U, ARM_R, ARM_D, ARM_L]  # clockwise
    out = 0
    for i, bit in enumerate(order):
        if arms & bit:
            out |= order[(i + quarter_turns_cw) % 4]
    return out


@dataclass
class SkeletonParams:
    """Tunable parameters. Lengths are fractions of min(page width, height),
    so the same defaults work at 100-300 dpi."""

    adaptive_block_frac: float = 0.02   # adaptive threshold window
    adaptive_c: int = 12                 # how much darker than local mean = ink
    bridge_frac: float = 0.004           # gaps in a line that are bridged
    line_min_frac: float = 0.04          # shortest run that counts as a ruled line
    junction_radius_frac: float = 0.003  # dilation used to make H and V lines meet
    arm_len_frac: float = 0.012          # how far to look for an arm
    arm_fill: float = 0.6                # fraction of arm pixels that must be line


@dataclass
class Junction:
    x: float
    y: float
    arms: int  # bitmask of ARM_*

    @property
    def signature(self) -> str:
        return arms_to_str(self.arms)


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    orientation: str  # "H" or "V"


@dataclass
class Skeleton:
    width: int
    height: int
    junctions: List[Junction] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    mask_h: np.ndarray | None = None
    mask_v: np.ndarray | None = None

    def points(self) -> np.ndarray:
        if not self.junctions:
            return np.zeros((0, 2), np.float64)
        return np.array([[j.x, j.y] for j in self.junctions], np.float64)

    def arms(self) -> np.ndarray:
        return np.array([j.arms for j in self.junctions], np.int32)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _odd(n: float, minimum: int = 3) -> int:
    n = max(minimum, int(round(n)))
    return n if n % 2 == 1 else n + 1


def binarize(gray: np.ndarray, params: SkeletonParams | None = None) -> np.ndarray:
    """Adaptive threshold. Returns uint8 mask, ink = 255."""
    p = params or SkeletonParams()
    gray = to_gray(gray)
    block = _odd(min(gray.shape) * p.adaptive_block_frac, 15)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, p.adaptive_c
    )


def line_masks(ink: np.ndarray, params: SkeletonParams | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Masks containing only long horizontal / vertical strokes."""
    p = params or SkeletonParams()
    base = min(ink.shape)
    bridge = max(1, int(round(base * p.bridge_frac)))
    min_len = max(15, int(round(base * p.line_min_frac)))

    def extract(close_k, open_k):
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_k)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_k)
        m = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    mask_h = extract((bridge, 1), (min_len, 1))
    mask_v = extract((1, bridge), (1, min_len))
    return mask_h, mask_v


def _segments_from_mask(mask: np.ndarray, orientation: str) -> List[Segment]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if orientation == "H":
            yc = y + h / 2.0
            segs.append(Segment(float(x), yc, float(x + w - 1), yc, "H"))
        else:
            xc = x + w / 2.0
            segs.append(Segment(xc, float(y), xc, float(y + h - 1), "V"))
    return segs


def find_junctions(mask_h: np.ndarray, mask_v: np.ndarray,
                   params: SkeletonParams | None = None) -> List[Junction]:
    """Junctions = connected components of (dilated H mask) AND (dilated V mask)."""
    p = params or SkeletonParams()
    base = min(mask_h.shape)
    r = max(2, int(round(base * p.junction_radius_frac)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dh = cv2.dilate(mask_h, k)
    dv = cv2.dilate(mask_v, k)
    both = cv2.bitwise_and(dh, dv)
    n, _, _, cents = cv2.connectedComponentsWithStats(both, connectivity=8)

    arm = max(6, int(round(base * p.arm_len_frac)))
    gap = r + 2
    hh, ww = mask_h.shape
    out: List[Junction] = []
    for i in range(1, n):
        cx, cy = cents[i]
        xi, yi = int(round(cx)), int(round(cy))
        arms = 0
        # look along a thin band (+-r) for the line continuing in each direction
        y0, y1 = max(0, yi - r), min(hh, yi + r + 1)
        x0, x1 = max(0, xi - r), min(ww, xi + r + 1)
        left = dh[y0:y1, max(0, xi - gap - arm):max(0, xi - gap)].max(axis=0) if xi - gap > 0 else np.zeros(0)
        right = dh[y0:y1, min(ww, xi + gap):min(ww, xi + gap + arm)].max(axis=0) if xi + gap < ww else np.zeros(0)
        up = dv[max(0, yi - gap - arm):max(0, yi - gap), x0:x1].max(axis=1) if yi - gap > 0 else np.zeros(0)
        down = dv[min(hh, yi + gap):min(hh, yi + gap + arm), x0:x1].max(axis=1) if yi + gap < hh else np.zeros(0)
        for vals, bit in ((left, ARM_L), (right, ARM_R), (up, ARM_U), (down, ARM_D)):
            if vals.size >= arm * 0.5 and (vals > 0).mean() >= p.arm_fill:
                arms |= bit
        # a real junction has at least one horizontal and one vertical arm
        if (arms & (ARM_L | ARM_R)) and (arms & (ARM_U | ARM_D)):
            out.append(Junction(float(cx), float(cy), arms))
    return out


def extract_skeleton(image: np.ndarray, params: SkeletonParams | None = None,
                     keep_masks: bool = False) -> Skeleton:
    """Full skeleton of an (approximately axis-aligned) page image."""
    p = params or SkeletonParams()
    gray = to_gray(image)
    ink = binarize(gray, p)
    mh, mv = line_masks(ink, p)
    sk = Skeleton(width=gray.shape[1], height=gray.shape[0])
    sk.junctions = find_junctions(mh, mv, p)
    sk.segments = _segments_from_mask(mh, "H") + _segments_from_mask(mv, "V")
    if keep_masks:
        sk.mask_h, sk.mask_v = mh, mv
    return sk


# ---------------------------------------------------------------------------
# coarse page frame (rotation / perspective initialisation)
# ---------------------------------------------------------------------------

def order_clockwise(quad: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise (in image coordinates, y down), starting with
    the point closest to the top-left corner of the bounding box."""
    quad = np.asarray(quad, np.float64).reshape(4, 2)
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]  # increasing angle == clockwise when y points down
    start = int(np.argmin(q.sum(axis=1)))
    return np.roll(q, -start, axis=0)


def _quad_area(q: np.ndarray) -> float:
    return float(abs(cv2.contourArea(q.astype(np.float32))))


def frame_candidates(image: np.ndarray, max_candidates: int = 4,
                     min_area_frac: float = 0.15,
                     params: SkeletonParams | None = None) -> List[np.ndarray]:
    """Large quadrilaterals in the image (form border, paper edge, ...).

    Returned clockwise-ordered, largest first. Always returns at least one
    candidate: if no clean quadrilateral exists, the minimum-area rectangle
    around all ink is used.
    """
    gray = to_gray(image)
    h, w = gray.shape
    ink = binarize(gray, params)
    k = _odd(min(h, w) * 0.006, 3)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(h * w)
    diag = float(np.hypot(h, w))
    quads: List[Tuple[float, np.ndarray]] = []
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < min_area_frac * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_clockwise(approx.reshape(4, 2))
        quads.append((_quad_area(q), q))

    quads.sort(key=lambda t: -t[0])
    kept: List[np.ndarray] = []
    for _, q in quads:
        # drop near-duplicates (outer/inner edge of the same thick border)
        if any(np.abs(q - k_).max() < 0.02 * diag for k_ in kept):
            continue
        kept.append(q)
        if len(kept) >= max_candidates:
            break

    if not kept:
        pts = cv2.findNonZero(ink)
        if pts is not None:
            rect = cv2.minAreaRect(pts)
            kept.append(order_clockwise(cv2.boxPoints(rect)))
        else:
            kept.append(order_clockwise(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])))
    return kept
