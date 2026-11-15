"""Homography recovery: pure RANSAC on point sets, and full page fits on synthetic warps."""

import cv2
import numpy as np
import pytest

from layout_redact import fit_page, project_points
from layout_redact import synthetic as sy
from layout_redact.homography import estimate_homography, match_points

PAGE_CORNERS = np.array([[0, 0], [sy.PAGE_W, 0], [sy.PAGE_W, sy.PAGE_H], [0, sy.PAGE_H]], float)


def _random_h(rng):
    src = PAGE_CORNERS.astype(np.float32)
    dst = src + rng.uniform(-150, 150, size=src.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


@pytest.mark.parametrize("seed", range(5))
def test_ransac_recovers_h_with_outliers(seed):
    rng = np.random.default_rng(seed)
    H = _random_h(rng)
    src = rng.uniform([0, 0], [sy.PAGE_W, sy.PAGE_H], size=(120, 2))
    dst = project_points(H, src) + rng.normal(0, 0.5, size=(120, 2))
    n_out = 40  # a third of the pairs are gross outliers
    dst[:n_out] = rng.uniform([0, 0], [sy.PAGE_W, sy.PAGE_H], size=(n_out, 2))

    H_est, mask = estimate_homography(src, dst, threshold=3.0)
    assert H_est is not None
    assert mask[n_out:].mean() > 0.95   # inliers kept
    assert mask[:n_out].mean() < 0.1    # outliers rejected
    err = np.abs(project_points(H_est, PAGE_CORNERS) - project_points(H, PAGE_CORNERS)).max()
    assert err < 2.0


def test_estimate_needs_four_points():
    H, mask = estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)), 3.0)
    assert H is None and not mask.any()


def test_match_points_is_mutual_and_bounded():
    src = np.array([[0, 0], [10, 0], [100, 100]], float)
    dst = np.array([[1, 0], [9, 1], [300, 300]], float)
    idx = match_points(src, dst, radius=5)
    assert idx.tolist() == [0, 1, -1]


def test_match_points_respects_arm_signature():
    src = np.array([[0, 0]], float)
    dst = np.array([[1, 0], [3, 0]], float)
    # src is an L-corner (R|D = 2|8 = 10); dst[0] is a cross (15, 2 bits off), dst[1] matches
    idx = match_points(src, dst, radius=5, src_arms=np.array([10]), dst_arms=np.array([15, 10]),
                       max_arm_diff=1)
    assert idx.tolist() == [1]


SCANS = [
    # (angle, quarter_turns, scale, perspective, background grey)
    (0.0, 0, 1.0, 0.0, 200),
    (4.0, 0, 0.8, 0.02, 120),
    (-7.0, 0, 0.7, 0.04, 60),
    (2.0, 1, 0.9, 0.03, 160),
    (-3.0, 2, 0.75, 0.05, 100),
    (11.0, 3, 0.6, 0.06, 40),
]


@pytest.mark.parametrize("angle,turns,scale,persp,bg", SCANS)
def test_page_fit_recovers_ground_truth(template, angle, turns, scale, persp, bg):
    page = sy.render_form(seed=int(abs(angle) * 10) + turns)
    sp = sy.ScanParams(angle_deg=angle, quarter_turns=turns, scale=scale, perspective=persp,
                       background=bg, seed=turns + 7)
    scan, H_true = sy.simulate_scan(page, sp)
    fit = fit_page(scan, template)
    assert fit.ok
    assert fit.rotation == turns
    assert fit.confidence > 0.8
    err = np.abs(project_points(fit.H, PAGE_CORNERS) - project_points(H_true, PAGE_CORNERS)).max()
    # well under 1% of the scan diagonal, even at the (extrapolated) paper corners
    assert err < 0.005 * np.hypot(*scan.shape[:2])
