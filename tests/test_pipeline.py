"""End-to-end: synthetic scan -> redact -> sensitive pixels gone; CLI smoke tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from layout_redact import STATUS_FAILED, STATUS_REDACTED, redact_page, save_pdf
from layout_redact import synthetic as sy


def _scan(seed):
    page = sy.render_form(seed)
    return sy.simulate_scan(page, sy.random_scan_params(seed))


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_regions_are_blacked_out(template, seed):
    scan, H = _scan(seed)
    res = redact_page(scan, [template])
    assert res.status == STATUS_REDACTED
    gt = sy.region_masks(template.regions, H, scan.shape)
    gray = cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY)
    k = np.ones((5, 5), np.uint8)
    for name, m in gt.items():
        inner = cv2.erode(m, k) > 0          # ignore a 2 px band at the border
        assert gray[inner].max() < 40, name  # fully painted
    # outside all regions (with a margin) the page is unchanged
    union = np.zeros(scan.shape[:2], np.uint8)
    for m in gt.values():
        union |= m
    outside = cv2.dilate(union, np.ones((15, 15), np.uint8)) == 0
    assert np.array_equal(res.image[outside], scan[outside])


def test_unrelated_pages_fail_and_are_not_written(template):
    blank = np.full((1600, 1200, 3), 255, np.uint8)
    other = blank.copy()
    cv2.rectangle(other, (60, 60), (1140, 1540), (0, 0, 0), 3)
    for y in (300, 700, 1300):
        cv2.line(other, (60, y), (1140, y), (0, 0, 0), 2)
    grid = blank.copy()
    for y in range(100, 1500, 90):
        cv2.line(grid, (100, y), (1100, y), (0, 0, 0), 2)
    for x in range(100, 1150, 130):
        cv2.line(grid, (x, 100), (x, 1450), (0, 0, 0), 2)
    for img in (blank, other, grid):
        res = redact_page(img, [template], on_fail="skip")
        assert res.status == STATUS_FAILED
        assert res.image is None
    res = redact_page(blank, [template], on_fail="blank")
    assert res.image.max() == 0


SRC = str(Path(__file__).resolve().parents[1] / "src")


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "layout_redact", *map(str, args)],
                          capture_output=True, text=True, env=env)


def test_cli_build_fit_redact(tmp_path, reference):
    ref_png = tmp_path / "ref.png"
    cv2.imwrite(str(ref_png), reference)
    regions = tmp_path / "regions.yaml"
    regions.write_text("regions:\n  - name: totals\n    box: [960, 1240, 1160, 1420]\n", encoding="utf-8")
    tpl = tmp_path / "form.yaml"
    r = _cli("build-template", ref_png, "--regions", regions, "--name", "form", "-o", tpl)
    assert r.returncode == 0, r.stderr

    scan, _ = _scan(5)
    scan_png = tmp_path / "scan.png"
    cv2.imwrite(str(scan_png), scan)

    r = _cli("fit", scan_png, "-t", tpl)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == STATUS_REDACTED

    out = tmp_path / "out"
    r = _cli("redact", scan_png, "-t", tpl, "-o", out, "--debug")
    assert r.returncode == 0, r.stderr
    assert (out / "scan.redacted.png").exists()
    assert (out / "scan.p1.debug.png").exists()
    report = json.loads((out / "scan.report.json").read_text(encoding="utf-8"))
    assert report["pages"][0]["regions"][0]["name"] == "totals"


def test_cli_failed_page_exit_code(tmp_path, template):
    tpl = tmp_path / "t.yaml"
    template.save(tpl)
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((800, 600, 3), 255, np.uint8))
    out = tmp_path / "out"
    r = _cli("redact", blank, "-t", tpl, "-o", out)
    assert r.returncode == 2
    assert not (out / "blank.redacted.png").exists()


def test_pdf_roundtrip(tmp_path, template):
    if not any(_has(m) for m in ("pypdfium2", "fitz")):
        pytest.skip("no PDF rasteriser installed")
    from layout_redact import load_pages
    scans = [_scan(s)[0] for s in (6, 7)]
    pdf = tmp_path / "two.pdf"
    save_pdf(scans, pdf, dpi=150)
    pages = load_pages(pdf, dpi=150)
    assert len(pages) == 2
    for p in pages:
        assert redact_page(p, [template]).status == STATUS_REDACTED


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False
