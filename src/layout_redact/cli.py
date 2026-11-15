"""Command line interface.

    layout-redact skeleton  PAGE                  show detected line structure
    layout-redact build-template REF --regions R  make a template from a reference page
    layout-redact annotate  REF                   draw regions in a GUI, save a template
    layout-redact fit       PAGE -t TEMPLATE      estimate the homography, print a report
    layout-redact redact    PAGE... -t TEMPLATE   redact pages / PDFs / folders

Exit codes for ``fit`` and ``redact``: 0 all pages REDACTED, 1 at least one
FLAGGED, 2 at least one FAILED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

from . import __version__
from .io import PDF_EXTS, is_supported, load_image, load_pages, save_image, save_pdf
from .redact import (STATUS_FAILED, STATUS_FLAGGED, Thresholds, debug_overlay, project_regions,
                     decide_status, redact_page, PageResult)
from .fit import fit_best
from .skeleton import extract_skeleton
from .template import Template, build_template

TEMPLATE_EXTS = {".yaml", ".yml", ".json"}


def _load_templates(paths: List[str]) -> List[Template]:
    out: List[Template] = []
    for p in paths:
        path = Path(p)
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in TEMPLATE_EXTS) \
            if path.is_dir() else [path]
        for f in files:
            out.append(Template.load(f))
    if not out:
        raise SystemExit("No templates found")
    return out


def _expand_inputs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(f for f in path.iterdir() if f.is_file() and is_supported(f)))
        else:
            out.append(path)
    return out


def _exit_code(statuses: List[str]) -> int:
    if STATUS_FAILED in statuses:
        return 2
    if STATUS_FLAGGED in statuses:
        return 1
    return 0


def _thresholds(args) -> Thresholds:
    t = Thresholds()
    if getattr(args, "min_confidence", None) is not None:
        t.redacted = args.min_confidence
    return t


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_skeleton(args) -> int:
    img = load_image(args.input, dpi=args.dpi)
    sk = extract_skeleton(img, keep_masks=True)
    vis = img.copy()
    vis[sk.mask_h > 0] = (0, 170, 0)
    vis[sk.mask_v > 0] = (0, 170, 0)
    r = max(3, int(min(img.shape[:2]) / 250))
    for j in sk.junctions:
        cv2.circle(vis, (int(j.x), int(j.y)), r, (230, 60, 0), -1)
    out = Path(args.output or Path(args.input).with_suffix(".skeleton.png"))
    save_image(vis, out)
    print(f"{len(sk.junctions)} junctions, {len(sk.segments)} line segments -> {out}")
    return 0


def cmd_build_template(args) -> int:
    ref = load_image(args.reference, dpi=args.dpi)
    regions = []
    if args.regions:
        data = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))
        regions = data.get("regions", []) if isinstance(data, dict) else data
    tpl = build_template(ref, regions, name=args.name or Path(args.reference).stem,
                         description=args.description or "")
    tpl.save(args.output)
    print(f"template '{tpl.name}': {len(tpl.keypoints)} keypoints, {len(tpl.regions)} regions -> {args.output}")
    return 0


def cmd_annotate(args) -> int:
    from .annotate import run_annotator
    return run_annotator(args.reference, args.output, template_path=args.template,
                         name=args.name, dpi=args.dpi)


def cmd_fit(args) -> int:
    templates = _load_templates(args.template)
    statuses = []
    t = _thresholds(args)
    for path in _expand_inputs(args.input):
        for i, page in enumerate(load_pages(path, dpi=args.dpi), start=1):
            fit = fit_best(page, templates)
            regions = project_regions(fit, t)
            status = decide_status(fit, regions, t)
            statuses.append(status)
            res = PageResult(status, fit, regions)
            print(json.dumps({"input": str(path), "page": i, **res.to_dict()}, indent=1))
            if args.debug:
                dbg = Path(args.debug)
                dbg.mkdir(parents=True, exist_ok=True)
                save_image(debug_overlay(page, res), dbg / f"{path.stem}.p{i}.debug.png")
    return _exit_code(statuses)


def cmd_redact(args) -> int:
    templates = _load_templates(args.template)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    t = _thresholds(args)
    statuses: List[str] = []
    for path in _expand_inputs(args.input):
        pages = load_pages(path, dpi=args.dpi)
        results = []
        out_pages: List[np.ndarray] = []
        for i, page in enumerate(pages, start=1):
            res = redact_page(page, templates, on_fail=args.on_fail, thresholds=t)
            statuses.append(res.status)
            results.append({"page": i, **res.to_dict(), "written": res.image is not None})
            if res.image is not None:
                out_pages.append(res.image)
            if args.debug:
                save_image(debug_overlay(page, res), outdir / f"{path.stem}.p{i}.debug.png")
            print(f"{path.name} p{i}: {res.status} conf={res.fit.confidence:.2f} "
                  f"template={res.fit.template.name}")

        fmt = args.format
        if fmt == "same":
            fmt = "pdf" if path.suffix.lower() in PDF_EXTS else "png"
        written = []
        if out_pages:
            if fmt == "pdf":
                target = outdir / f"{path.stem}.redacted.pdf"
                save_pdf(out_pages, target, dpi=args.dpi)
                written.append(str(target))
            else:
                for i, img in enumerate(out_pages, start=1):
                    suffix = f".p{i}" if len(out_pages) > 1 else ""
                    target = outdir / f"{path.stem}{suffix}.redacted.png"
                    save_image(img, target)
                    written.append(str(target))
        report = {"input": str(path), "outputs": written, "on_fail": args.on_fail, "pages": results}
        (outdir / f"{path.stem}.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return _exit_code(statuses)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="layout-redact",
        description="Offline redaction of fixed-layout forms via line skeleton + RANSAC homography.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def dpi(sp):
        sp.add_argument("--dpi", type=int, default=150, help="PDF rasterisation resolution (default 150)")

    s = sub.add_parser("skeleton", help="draw the detected line skeleton of a page")
    s.add_argument("input")
    s.add_argument("-o", "--output")
    dpi(s)
    s.set_defaults(func=cmd_skeleton)

    s = sub.add_parser("build-template", help="create a template from a clean reference page")
    s.add_argument("reference")
    s.add_argument("--regions", help="YAML with a 'regions' list (box/polygon in reference px)")
    s.add_argument("--name")
    s.add_argument("--description")
    s.add_argument("-o", "--output", required=True)
    dpi(s)
    s.set_defaults(func=cmd_build_template)

    s = sub.add_parser("annotate", help="draw redaction regions on a reference page (GUI)")
    s.add_argument("reference")
    s.add_argument("-o", "--output", required=True, help="template file to write")
    s.add_argument("-t", "--template", help="existing template to load regions from")
    s.add_argument("--name")
    dpi(s)
    s.set_defaults(func=cmd_annotate)

    for name, func, hlp in (("fit", cmd_fit, "fit template(s) and print the result as JSON"),
                            ("redact", cmd_redact, "redact images, PDFs or folders")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("input", nargs="+", help="image, PDF or folder")
        s.add_argument("-t", "--template", action="append", required=True,
                       help="template file or folder (repeatable; best fit wins per page)")
        s.add_argument("--min-confidence", type=float, default=None,
                       help="fit confidence needed for REDACTED instead of FLAGGED (default 0.72)")
        dpi(s)
        if name == "fit":
            s.add_argument("--debug", metavar="DIR", help="write debug overlays here")
        else:
            s.add_argument("-o", "--output", required=True, help="output folder")
            s.add_argument("--format", choices=["same", "png", "pdf"], default="same")
            s.add_argument("--on-fail", choices=["skip", "blank", "copy"], default="skip",
                           help="FAILED pages: skip (default), blank (fill black) or copy (unredacted!)")
            s.add_argument("--debug", action="store_true", help="also write debug overlays")
        s.set_defaults(func=func)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
