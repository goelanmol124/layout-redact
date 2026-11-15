"""layout-redact: offline, template-based redaction of fixed-layout forms.

Pipeline: skeleton (ruled-line junctions) -> RANSAC homography -> projection
of template redaction regions -> solid fill.
"""

from .fit import FitParams, PageFit, fit_best, fit_page
from .homography import estimate_homography, project_points
from .io import load_image, load_pages, save_image, save_pdf
from .redact import (STATUS_FAILED, STATUS_FLAGGED, STATUS_REDACTED, PageResult, Thresholds,
                     debug_overlay, paint, project_region, project_regions, redact_page)
from .skeleton import SkeletonParams, extract_skeleton, frame_candidates
from .template import Region, Template, build_template

__version__ = "0.1.0"

__all__ = [
    "FitParams", "PageFit", "fit_best", "fit_page",
    "estimate_homography", "project_points",
    "load_image", "load_pages", "save_image", "save_pdf",
    "STATUS_FAILED", "STATUS_FLAGGED", "STATUS_REDACTED", "PageResult", "Thresholds",
    "debug_overlay", "paint", "project_region", "project_regions", "redact_page",
    "SkeletonParams", "extract_skeleton", "frame_candidates",
    "Region", "Template", "build_template",
]
