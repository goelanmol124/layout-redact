"""Correspondence search and RANSAC homography estimation.

There are no descriptors here. Line junctions on a form all look alike; what
identifies them is *where they are* relative to each other. So matching is
geometric and iterative, in the spirit of ICP:

1. start from an initial guess ``H`` (template -> image),
2. project the template keypoints, pair each with the nearest detected
   keypoint that has a compatible arm signature (mutual nearest neighbours,
   within a search radius),
3. estimate a new ``H`` from those pairs with RANSAC (``cv2.findHomography``),
   which throws out wrong pairs (text noise, stamps, missing lines),
4. shrink the radius and repeat.

A homography has 8 degrees of freedom, so it covers translation, scale,
rotation, shear and the perspective of a page photographed at an angle. It
does not model paper curl or folds; see the README's limitations section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FitStats:
    H: Optional[np.ndarray]          # 3x3, template px -> target px
    inliers: int
    correspondences: int
    template_points: int
    mean_error: float                # mean reprojection error of inliers (target px)
    inlier_mask: Optional[np.ndarray] = None   # over template keypoints
    matched_idx: Optional[np.ndarray] = None   # detected index per template keypoint (-1 none)

    @property
    def coverage(self) -> float:
        return self.inliers / max(1, self.template_points)


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), np.float64)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _arm_compatible(a: np.ndarray, b: np.ndarray, max_diff_bits: int) -> np.ndarray:
    """Pairwise compatibility of arm bitmasks (Hamming distance <= max_diff_bits)."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    bits = ((x >> 0) & 1) + ((x >> 1) & 1) + ((x >> 2) & 1) + ((x >> 3) & 1)
    return bits <= max_diff_bits


def match_points(src_proj: np.ndarray, dst: np.ndarray, radius: float,
                 src_arms: Optional[np.ndarray] = None,
                 dst_arms: Optional[np.ndarray] = None,
                 max_arm_diff: int = 1) -> np.ndarray:
    """Mutual-nearest-neighbour matching within ``radius``.

    Returns an int array, one entry per source point: index into ``dst`` or -1.
    """
    n = len(src_proj)
    out = -np.ones(n, np.int64)
    if n == 0 or len(dst) == 0:
        return out
    d2 = ((src_proj[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
    if src_arms is not None and dst_arms is not None:
        d2 = np.where(_arm_compatible(src_arms, dst_arms, max_arm_diff), d2, np.inf)
    j = np.argmin(d2, axis=1)
    i_back = np.argmin(d2, axis=0)
    ok = (d2[np.arange(n), j] <= radius * radius) & (i_back[j] == np.arange(n))
    out[ok] = j[ok]
    return out


def estimate_homography(src: np.ndarray, dst: np.ndarray, threshold: float,
                        max_iters: int = 5000, confidence: float = 0.999,
                        seed: int = 0) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC homography with a least-squares refit on the inliers.

    Returns ``(H, inlier_mask)``; ``H`` is ``None`` if fewer than 4 points or
    RANSAC fails.
    """
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    mask = np.zeros(len(src), bool)
    if len(src) < 4:
        return None, mask
    cv2.setRNGSeed(seed)
    H, m = cv2.findHomography(src, dst, cv2.RANSAC, threshold,
                              maxIters=max_iters, confidence=confidence)
    if H is None or m is None:
        return None, mask
    mask = m.ravel().astype(bool)
    if mask.sum() >= 4:
        H2, _ = cv2.findHomography(src[mask], dst[mask], 0)
        if H2 is not None:
            # re-evaluate inliers with the refined model
            err = np.linalg.norm(project_points(H2, src) - dst, axis=1)
            mask2 = err < threshold
            if mask2.sum() >= mask.sum():
                H, mask = H2, mask2
    return H / H[2, 2], mask


def refine_by_matching(H_init: np.ndarray, tpl_pts: np.ndarray, det_pts: np.ndarray,
                       radii: Sequence[float], ransac_threshold: float,
                       tpl_arms: Optional[np.ndarray] = None,
                       det_arms: Optional[np.ndarray] = None) -> FitStats:
    """Iterative match -> RANSAC -> re-match, with shrinking search radius."""
    H = H_init.copy()
    best = FitStats(None, 0, 0, len(tpl_pts), float("inf"))
    for radius in radii:
        proj = project_points(H, tpl_pts)
        idx = match_points(proj, det_pts, radius, tpl_arms, det_arms)
        sel = idx >= 0
        if sel.sum() < 4:
            break
        H_new, mask = estimate_homography(tpl_pts[sel], det_pts[idx[sel]], ransac_threshold)
        if H_new is None:
            break
        H = H_new
        full_mask = np.zeros(len(tpl_pts), bool)
        full_mask[np.flatnonzero(sel)[mask]] = True
        err = np.linalg.norm(project_points(H, tpl_pts[full_mask]) - det_pts[idx[full_mask]], axis=1)
        stats = FitStats(H, int(full_mask.sum()), int(sel.sum()), len(tpl_pts),
                         float(err.mean()) if len(err) else float("inf"), full_mask, idx)
        if stats.inliers >= best.inliers or best.H is None:
            best = stats
    return best


def fit_confidence(stats: FitStats, max_error: float, min_inliers: int,
                   w_coverage: float = 0.40, w_error: float = 0.35, w_count: float = 0.25) -> float:
    """Confidence in [0, 1] for a fit.

    ``w_coverage * inliers/template_points
      + w_error * (1 - mean_error / max_error)
      + w_count * min(1, inliers / min_inliers)``
    """
    if stats.H is None or stats.inliers == 0:
        return 0.0
    c1 = stats.coverage
    c2 = max(0.0, min(1.0, 1.0 - stats.mean_error / max_error))
    c3 = min(1.0, stats.inliers / float(min_inliers))
    return float(w_coverage * c1 + w_error * c2 + w_count * c3)


def is_sane(H: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> bool:
    """Reject degenerate homographies (folded, tiny, or exploding projections)."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    w, h = src_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    pc = project_points(H, corners)
    # projective denominator must stay positive over the page (no wrap-around)
    den = (H[2, :2] @ corners.T + H[2, 2])
    if np.any(den <= 0):
        return False
    if not cv2.isContourConvex(pc.astype(np.float32).reshape(-1, 1, 2)):
        return False
    area = abs(cv2.contourArea(pc.astype(np.float32)))
    dw, dh = dst_size
    return 0.01 * dw * dh < area < 25.0 * dw * dh
