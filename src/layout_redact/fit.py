"""Fit a template to a page: coarse frame alignment, then RANSAC on junctions.

Stage 1 (coarse)  Find large quadrilaterals in the page (form border, paper
                  edge). For each candidate and each of the 4 corner
                  orderings (0/90/180/270 degree page rotation) build a
                  homography ``H0`` that maps the template frame onto it.

Stage 2 (rectify) Warp the page into the template's coordinate system with
                  ``H0``. Lines are now (almost) axis-aligned, so the
                  morphological skeleton detector works even on rotated or
                  photographed pages.

Stage 3 (fine)    Detect junctions in the rectified page, match them to the
                  template keypoints and estimate the residual homography
                  ``Hr`` with RANSAC (:mod:`layout_redact.homography`).

The hypothesis with the most inliers wins; the final transform is
``H = H0 @ Hr`` (template px -> page px).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

import cv2
import numpy as np

from .homography import FitStats, fit_confidence, is_sane, project_points, refine_by_matching
from .skeleton import SkeletonParams, extract_skeleton, frame_candidates
from .template import Template


@dataclass
class FitParams:
    radii_frac: tuple = (0.02, 0.01, 0.005)   # matching radius per iteration, x template diagonal
    ransac_threshold: float = 6.0             # template px
    max_error: float = 6.0                    # template px, for the confidence score
    min_inliers: int = 12
    max_frame_candidates: int = 4
    try_rotations: bool = True                # also test 90/180/270 degree page rotations
    skeleton: SkeletonParams = field(default_factory=SkeletonParams)

    @classmethod
    def from_template(cls, template: Template, base: Optional["FitParams"] = None) -> "FitParams":
        p = base or cls()
        over = dict(template.params or {})
        sk_over = over.pop("skeleton", {}) or {}
        known = {k: v for k, v in over.items() if k in cls.__dataclass_fields__ and k != "skeleton"}
        if "radii_frac" in known:
            known["radii_frac"] = tuple(known["radii_frac"])
        p = replace(p, **known)
        if sk_over:
            p = replace(p, skeleton=replace(p.skeleton, **sk_over))
        return p


@dataclass
class PageFit:
    template: Template
    H: Optional[np.ndarray]          # template px -> page px
    confidence: float
    stats: FitStats
    H0: Optional[np.ndarray] = None  # coarse frame homography of the chosen hypothesis
    rotation: int = 0                # quarter turns of the chosen hypothesis
    detected_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # page px

    @property
    def ok(self) -> bool:
        return self.H is not None


def _rectify(image: np.ndarray, H0: np.ndarray, size) -> np.ndarray:
    return cv2.warpPerspective(image, H0, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def fit_page(image: np.ndarray, template: Template, params: Optional[FitParams] = None) -> PageFit:
    """Estimate the homography from ``template`` to ``image``."""
    p = FitParams.from_template(template, params)
    tw, th = template.width, template.height
    diag = float(np.hypot(tw, th))
    radii = [r * diag for r in p.radii_frac]
    tpl_pts = template.keypoints
    tpl_arms = template.keypoint_arms
    tpl_frame = template.frame.astype(np.float32)
    ih, iw = image.shape[:2]

    best: Optional[PageFit] = None
    candidates = frame_candidates(image, max_candidates=p.max_frame_candidates, params=p.skeleton)
    rotations = range(4) if p.try_rotations else range(1)
    for quad in candidates:
        for k in rotations:
            dst = np.roll(quad, -k, axis=0).astype(np.float32)
            H0 = cv2.getPerspectiveTransform(tpl_frame, dst)
            if not is_sane(H0, (tw, th), (iw, ih)):
                continue
            rect = _rectify(image, H0, (tw, th))
            sk = extract_skeleton(rect, p.skeleton)
            det = sk.points()
            if len(det) < 4:
                continue
            stats = refine_by_matching(np.eye(3), tpl_pts, det, radii, p.ransac_threshold,
                                       tpl_arms, sk.arms())
            if stats.H is None:
                continue
            H = H0 @ stats.H
            H = H / H[2, 2]
            if not is_sane(H, (tw, th), (iw, ih)):
                continue
            conf = fit_confidence(stats, p.max_error, p.min_inliers)
            cand = PageFit(template, H, conf, stats, H0, k, project_points(H0, det))
            if best is None or (stats.inliers, -stats.mean_error) > (best.stats.inliers, -best.stats.mean_error):
                best = cand

    if best is None:
        return PageFit(template, None, 0.0, FitStats(None, 0, 0, len(tpl_pts), float("inf")))
    return best


def fit_best(image: np.ndarray, templates: List[Template],
             params: Optional[FitParams] = None) -> PageFit:
    """Fit every template and return the most confident fit."""
    if not templates:
        raise ValueError("No templates given")
    fits = [fit_page(image, t, params) for t in templates]
    return max(fits, key=lambda f: f.confidence)
