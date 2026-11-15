"""Project template regions through the fitted homography and paint them out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .fit import FitParams, PageFit, fit_best
from .homography import project_points
from .template import Region, Template

STATUS_REDACTED = "REDACTED"   # confident fit, regions painted
STATUS_FLAGGED = "FLAGGED"     # painted, but a human should check
STATUS_FAILED = "FAILED"       # no trustworthy fit


@dataclass
class Thresholds:
    redacted: float = 0.72   # fit confidence >= this and all regions ok -> REDACTED
    failed: float = 0.50     # fit confidence <  this -> FAILED
    min_coverage: float = 0.35  # fraction of template keypoints matched; below -> FAILED
    region: float = 0.65     # region confidence below this -> FLAGGED
    local_support: int = 4   # inlier keypoints near a region for full local support


@dataclass
class ProjectedRegion:
    name: str
    polygon: np.ndarray          # page px
    confidence: float
    local_inliers: int
    fill: tuple

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon],
            "confidence": round(self.confidence, 3),
            "local_inliers": self.local_inliers,
        }


@dataclass
class PageResult:
    status: str
    fit: PageFit
    regions: List[ProjectedRegion] = field(default_factory=list)
    image: Optional[np.ndarray] = None   # redacted page (None if FAILED and policy == "skip")

    def to_dict(self) -> Dict[str, Any]:
        f = self.fit
        return {
            "status": self.status,
            "template": f.template.name,
            "fit_confidence": round(f.confidence, 3),
            "inliers": f.stats.inliers,
            "template_keypoints": f.stats.template_points,
            "mean_reprojection_error_px": None if not np.isfinite(f.stats.mean_error)
            else round(f.stats.mean_error, 2),
            "rotation_quarter_turns": f.rotation,
            "homography": None if f.H is None else [[float(v) for v in row] for row in f.H],
            "regions": [r.to_dict() for r in self.regions],
        }


def project_region(region: Region, H: np.ndarray) -> np.ndarray:
    """Project a (padded) template region polygon into page coordinates."""
    return project_points(H, region.padded())


def project_regions(fit: PageFit, thresholds: Optional[Thresholds] = None) -> List[ProjectedRegion]:
    """Project every region of the fitted template and score it.

    ``region_confidence = fit_confidence * (0.6 + 0.4 * local_support)``, where
    local support is how many RANSAC inliers lie in or near the region
    (``min(1, n / thresholds.local_support)``). A region far from any matched
    structure depends on extrapolating the homography and scores lower.
    """
    t = thresholds or Thresholds()
    if fit.H is None:
        return []
    tpl = fit.template
    diag = float(np.hypot(tpl.width, tpl.height))
    inl = tpl.keypoints[fit.stats.inlier_mask] if fit.stats.inlier_mask is not None \
        else np.zeros((0, 2))
    out = []
    for r in tpl.regions:
        poly_t = r.padded()
        x0, y0 = poly_t.min(axis=0) - 0.08 * diag
        x1, y1 = poly_t.max(axis=0) + 0.08 * diag
        n = int(((inl[:, 0] >= x0) & (inl[:, 0] <= x1) & (inl[:, 1] >= y0) & (inl[:, 1] <= y1)).sum()) \
            if len(inl) else 0
        local = min(1.0, n / float(t.local_support))
        conf = fit.confidence * (0.6 + 0.4 * local)
        out.append(ProjectedRegion(r.name, project_points(fit.H, poly_t), conf, n, tuple(r.fill)))
    return out


def paint(image: np.ndarray, regions: List[ProjectedRegion]) -> np.ndarray:
    """Solid-fill each polygon. Returns a new image."""
    out = image.copy()
    for r in regions:
        rgb = r.fill
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0])) if out.ndim == 3 else int(np.mean(rgb))
        pts = np.round(r.polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [pts], color, lineType=cv2.LINE_8)
        # outline too, so anti-aliased edges of the projected polygon leave no sliver
        cv2.polylines(out, [pts], True, color, thickness=2, lineType=cv2.LINE_8)
    return out


def decide_status(fit: PageFit, regions: List[ProjectedRegion], t: Thresholds) -> str:
    if fit.H is None or fit.confidence < t.failed or fit.stats.coverage < t.min_coverage:
        return STATUS_FAILED
    if fit.confidence < t.redacted or any(r.confidence < t.region for r in regions):
        return STATUS_FLAGGED
    return STATUS_REDACTED


def redact_page(image: np.ndarray, templates: List[Template], on_fail: str = "skip",
                thresholds: Optional[Thresholds] = None,
                params: Optional[FitParams] = None) -> PageResult:
    """Fit, project and paint one page.

    ``on_fail`` decides what a FAILED page becomes:
    ``"skip"`` (no output image), ``"blank"`` (page filled black) or
    ``"copy"`` (original returned untouched; only for review workflows).
    """
    if on_fail not in ("skip", "blank", "copy"):
        raise ValueError("on_fail must be 'skip', 'blank' or 'copy'")
    t = thresholds or Thresholds()
    fit = fit_best(image, templates, params)
    regions = project_regions(fit, t)
    status = decide_status(fit, regions, t)
    if status == STATUS_FAILED:
        if on_fail == "copy":
            out = image.copy()
        elif on_fail == "blank":
            out = np.zeros_like(image)
        else:
            out = None
        return PageResult(status, fit, regions, out)
    return PageResult(status, fit, regions, paint(image, regions))


# ---------------------------------------------------------------------------
# debug overlay
# ---------------------------------------------------------------------------

_COLORS = {
    STATUS_REDACTED: (60, 170, 40),
    STATUS_FLAGGED: (0, 160, 255),
    STATUS_FAILED: (40, 40, 220),
}


def debug_overlay(image: np.ndarray, result: PageResult) -> np.ndarray:
    """Original page with detected junctions, matched template keypoints,
    projected frame and region outlines drawn on top."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    fit = result.fit
    scale = max(1, int(round(min(vis.shape[:2]) / 600)))
    for x, y in fit.detected_points:
        cv2.circle(vis, (int(x), int(y)), 2 * scale, (200, 120, 0), -1)
    if fit.H is not None:
        tpl = fit.template
        proj = project_points(fit.H, tpl.keypoints)
        mask = fit.stats.inlier_mask if fit.stats.inlier_mask is not None else np.zeros(len(proj), bool)
        for (x, y), m in zip(proj, mask):
            cv2.circle(vis, (int(x), int(y)), 3 * scale, (40, 180, 40) if m else (60, 60, 230),
                       1 if not m else 2)
        frame = project_points(fit.H, np.array([[0, 0], [tpl.width, 0], [tpl.width, tpl.height],
                                                [0, tpl.height]], np.float64))
        cv2.polylines(vis, [frame.astype(np.int32).reshape(-1, 1, 2)], True, (220, 120, 0), scale)
    for r in result.regions:
        pts = np.round(r.polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 120, 255), 2 * scale)
        x, y = pts.reshape(-1, 2).min(axis=0)
        cv2.putText(vis, f"{r.name} {r.confidence:.2f}", (int(x), int(max(12, y - 4))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 90, 220), scale, cv2.LINE_AA)
    label = f"{result.status}  conf={fit.confidence:.2f}  inliers={fit.stats.inliers}/{fit.stats.template_points}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30 * scale), (255, 255, 255), -1)
    cv2.putText(vis, label, (8, 22 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale,
                _COLORS.get(result.status, (0, 0, 0)), 2 * scale, cv2.LINE_AA)
    return vis
