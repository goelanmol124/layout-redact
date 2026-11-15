import numpy as np
import pytest

from layout_redact import Template, build_template
from layout_redact import synthetic as sy


def test_template_has_structure(template):
    assert len(template.keypoints) > 50
    assert template.page_size == (sy.PAGE_W, sy.PAGE_H)
    # the outer border is picked up as the frame
    assert np.allclose(template.frame, [[80, 80], [1160, 80], [1160, 1674], [80, 1674]], atol=3)


@pytest.mark.parametrize("ext", [".yaml", ".json"])
def test_roundtrip(template, tmp_path, ext):
    path = tmp_path / f"t{ext}"
    template.save(path)
    t2 = Template.load(path)
    assert t2.name == template.name
    assert np.allclose(t2.keypoints, template.keypoints, atol=0.06)
    assert (t2.keypoint_arms == template.keypoint_arms).all()
    assert [r.name for r in t2.regions] == [r.name for r in template.regions]
    for a, b in zip(t2.regions, template.regions):
        assert np.allclose(a.padded(), b.padded())


def test_polygon_region_and_fill(reference, tmp_path):
    regions = [{"name": "tri", "polygon": [[100, 100], [300, 120], [200, 300]], "fill": [255, 255, 255]}]
    t = build_template(reference, regions, "x")
    t.save(tmp_path / "x.yaml")
    t2 = Template.load(tmp_path / "x.yaml")
    assert t2.regions[0].polygon.shape == (3, 2)
    assert t2.regions[0].fill == (255, 255, 255)


def test_reference_without_lines_is_rejected():
    blank = np.full((800, 600, 3), 255, np.uint8)
    with pytest.raises(ValueError):
        build_template(blank, [], "blank")


def test_bad_format_rejected():
    with pytest.raises(ValueError):
        Template.from_dict({"format": "something-else"})
