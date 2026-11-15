"""Projection of template boxes through a homography, and painting."""

import cv2
import numpy as np

from layout_redact import Region, paint, project_region
from layout_redact.redact import ProjectedRegion


def test_identity_projection_is_box():
    r = Region.from_box("a", (10, 20, 110, 70))
    poly = project_region(r, np.eye(3))
    assert np.allclose(poly, [[10, 20], [110, 20], [110, 70], [10, 70]])


def test_box_order_is_normalised():
    r = Region.from_box("a", (110, 70, 10, 20))
    assert np.allclose(r.polygon.min(axis=0), [10, 20])
    assert np.allclose(r.polygon.max(axis=0), [110, 70])


def test_padding_grows_box():
    r = Region.from_box("a", (10, 20, 110, 70), pad=5)
    poly = project_region(r, np.eye(3))
    assert np.allclose(poly, [[5, 15], [115, 15], [115, 75], [5, 75]])


def test_projection_matches_perspective_transform():
    src = np.float32([[0, 0], [1000, 0], [1000, 1400], [0, 1400]])
    dst = np.float32([[50, 80], [980, 40], [1060, 1450], [10, 1380]])
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    r = Region.from_box("a", (100, 200, 400, 300))
    expected = cv2.perspectiveTransform(r.polygon.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.allclose(project_region(r, H), expected)


def test_similarity_projection_keeps_shape():
    # 90 degree rotation, scale 2, shift
    H = np.array([[0, -2, 500], [2, 0, 10], [0, 0, 1]], float)
    r = Region.from_box("a", (0, 0, 100, 50))
    poly = project_region(r, H)
    side = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)
    assert np.allclose(sorted(side), [100, 100, 200, 200])


def test_paint_fills_polygon_only():
    img = np.full((200, 300, 3), 255, np.uint8)
    poly = np.array([[50, 40], [150, 40], [150, 120], [50, 120]], float)
    out = paint(img, [ProjectedRegion("a", poly, 1.0, 0, (0, 0, 0))])
    assert out[45:115, 55:145].max() == 0
    assert out[150:, :].min() == 255 and out[:, 200:].min() == 255
    assert img.min() == 255  # input untouched


def test_paint_custom_fill_rgb():
    img = np.full((50, 50, 3), 255, np.uint8)
    poly = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], float)
    out = paint(img, [ProjectedRegion("a", poly, 1.0, 0, (255, 0, 0))])
    assert tuple(out[25, 25]) == (0, 0, 255)  # stored as BGR
