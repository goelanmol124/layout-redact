"""End-to-end demo on synthetic data. Nothing here comes from a real document.

    python examples/run_demo.py                 # writes into examples/output/
    python examples/run_demo.py --gallery       # also refreshes docs/images/*.png

Steps:
1. render a blank synthetic "order form" (the reference) and build a template
   from it, with redaction regions for name, address, phone, prices, totals
   and signature;
2. render filled-in copies with fake data and "scan" them: rotate (incl.
   90/180/270 degrees), scale, perspective-warp, blur, add noise and JPEG;
3. redact every scan, write redacted PNGs, debug overlays, JSON reports and a
   two-page PDF that goes through the PDF code path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # run without installing

from layout_redact import (build_template, debug_overlay, extract_skeleton, load_pages,  # noqa: E402
                           redact_page, save_image, save_pdf)
from layout_redact import synthetic as sy  # noqa: E402

SCANS = [
    # name, scan parameters
    ("scan_tilted", sy.ScanParams(angle_deg=-6.0, scale=0.8, perspective=0.03, background=130, seed=11)),
    ("scan_rotated", sy.ScanParams(angle_deg=3.0, quarter_turns=1, scale=0.75, perspective=0.02,
                                   background=70, seed=12)),
    ("scan_photo", sy.ScanParams(angle_deg=9.0, quarter_turns=2, scale=0.7, perspective=0.07, blur=1.3,
                                 noise=9.0, background=45, jpeg_quality=55, seed=13)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--gallery", action="store_true", help="refresh the README images in docs/images")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. reference + template ------------------------------------------------
    reference = sy.render_form(blank=True)
    save_image(reference, out / "reference.png")
    template = build_template(reference, sy.sample_regions(), name="sample-order-form",
                              description="Synthetic order form used by the examples and tests.")
    template.save(HERE / "sample_form.template.yaml")
    print(f"template: {len(template.keypoints)} keypoints, {len(template.regions)} regions")

    # 2+3. scans -> redaction ---------------------------------------------------
    results = {}
    for i, (name, sp) in enumerate(SCANS):
        page = sy.render_form(seed=100 + i)
        scan, _ = sy.simulate_scan(page, sp)
        save_image(scan, out / f"{name}.png")
        res = redact_page(scan, [template])
        results[name] = (scan, res)
        if res.image is not None:
            save_image(res.image, out / f"{name}.redacted.png")
        save_image(debug_overlay(scan, res), out / f"{name}.debug.png")
        (out / f"{name}.report.json").write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
        print(f"{name:13s} {res.status:9s} confidence={res.fit.confidence:.3f} "
              f"inliers={res.fit.stats.inliers}/{res.fit.stats.template_points} "
              f"reproj={res.fit.stats.mean_error:.2f}px rotation={res.fit.rotation * 90}deg")

    # PDF in -> PDF out
    pdf_in = out / "two_scans.pdf"
    save_pdf([results["scan_tilted"][0], results["scan_rotated"][0]], pdf_in)
    try:
        pages = load_pages(pdf_in)
    except ImportError as exc:
        print(f"skipping PDF demo: {exc}")
    else:
        red = [redact_page(p, [template]) for p in pages]
        save_pdf([r.image for r in red if r.image is not None], out / "two_scans.redacted.pdf")
        print("pdf:", ", ".join(r.status for r in red))

    if args.gallery:
        make_gallery(reference, template, results, HERE.parent / "docs" / "images")
    print(f"outputs in {out}")
    return 0


# ---------------------------------------------------------------------------
# README images
# ---------------------------------------------------------------------------

def _thumb(img: np.ndarray, height: int) -> np.ndarray:
    s = height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), height), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def _row(panels, gap=12):
    h = max(p.shape[0] for p in panels)
    parts = []
    for p in panels:
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)])
        parts += [p, np.full((h, gap, 3), 255, np.uint8)]
    return np.hstack(parts[:-1])


def _save_small_png(img: np.ndarray, path: Path, colors: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).quantize(colors=colors, method=Image.MEDIANCUT)
    pil.save(path, optimize=True)
    print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")


def make_gallery(reference, template, results, img_dir: Path) -> None:
    H = 560
    # skeleton of the reference: line masks + junctions
    sk = extract_skeleton(reference, keep_masks=True)
    skel = np.full_like(reference, 255)
    skel[(sk.mask_h > 0) | (sk.mask_v > 0)] = (90, 90, 90)
    for r in template.regions:
        cv2.polylines(skel, [np.round(r.polygon).astype(np.int32).reshape(-1, 1, 2)], True, (0, 120, 255), 4)
    for j in sk.junctions:
        cv2.circle(skel, (int(j.x), int(j.y)), 9, (200, 90, 0), -1)

    scan, res = results["scan_photo"]
    panels = [
        _label(_thumb(reference, H), "1. reference form"),
        _label(_thumb(skel, H), "2. skeleton + regions"),
        _label(_thumb(scan, H), "3. new scan"),
        _label(_thumb(debug_overlay(scan, res), H), "4. RANSAC fit"),
        _label(_thumb(res.image, H), "5. redacted"),
    ]
    _save_small_png(_row(panels), img_dir / "pipeline.png")

    row = []
    for name in ("scan_tilted", "scan_rotated"):
        scan, res = results[name]
        row += [_label(_thumb(scan, 420), name), _label(_thumb(res.image, 420), "redacted")]
    _save_small_png(_row(row), img_dir / "more_scans.png")


if __name__ == "__main__":
    sys.exit(main())
