"""Synthetic forms and scans, for examples and tests.

Everything here is invented: a generic order form with fake names, fake
addresses and random numbers, plus a "scanner" that rotates, scales,
perspective-warps, blurs and adds noise, and reports the exact ground-truth
homography it applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .template import Region

PAGE_W, PAGE_H = 1240, 1754   # A4 at 150 dpi

# --- layout (reference px) ---------------------------------------------------
M = 80                          # outer border margin
X0, X1 = M, PAGE_W - M          # 80 .. 1160
Y_TOP, Y_BOTTOM = M, PAGE_H - M  # 80 .. 1674
Y_HEADER = 200
Y_PARTY = (200, 273, 346, 420)  # customer block rows
X_PARTY_SPLIT = 620
X_PARTY_LABEL_L, X_PARTY_LABEL_R = 270, 800
Y_TABLE_TOP, Y_TABLE_HDR = 460, 520
ROW_H, N_ROWS = 60, 12
Y_TABLE_BOTTOM = Y_TABLE_HDR + ROW_H * N_ROWS   # 1240
COLS = (80, 160, 640, 780, 960, 1160)           # #, description, quantity, unit cost, line total
Y_TOTALS = (1240, 1300, 1360, 1420)
X_TOTALS_LABEL = 960
Y_SIG = (1470, 1630)
X_SIG = (760, 1120)

FIRST = ["Alex", "Sam", "Jordan", "Robin", "Casey", "Taylor", "Morgan", "Jamie"]
LAST = ["Sample", "Example", "Placeholder", "Testperson", "Dummy", "Fictional"]
STREETS = ["Example Street", "Sample Road", "Placeholder Lane", "Test Avenue"]
TOWNS = ["Sampletown", "Exampleville", "Testford", "Mockbury"]
ITEMS = ["Steel bracket", "Hex bolt M8", "Cable tie pack", "Hinge, brass", "Washer set",
         "Pipe clamp", "Spring pin", "Rubber gasket", "Mounting plate", "Wing nut"]


def sample_regions() -> List[Region]:
    """The fields of the synthetic form that hold personal or financial data."""
    pad = 2
    return [
        Region.from_box("customer_name", (X_PARTY_LABEL_L, Y_PARTY[0], X_PARTY_SPLIT, Y_PARTY[1]), pad=pad),
        Region.from_box("customer_address", (X_PARTY_LABEL_L, Y_PARTY[1], X_PARTY_SPLIT, Y_PARTY[3]), pad=pad),
        Region.from_box("phone", (X_PARTY_LABEL_R, Y_PARTY[2], X1, Y_PARTY[3]), pad=pad),
        Region.from_box("unit_cost_column", (COLS[3], Y_TABLE_HDR, COLS[4], Y_TABLE_BOTTOM), pad=pad),
        Region.from_box("line_total_column", (COLS[4], Y_TABLE_HDR, COLS[5], Y_TABLE_BOTTOM), pad=pad),
        Region.from_box("totals", (X_TOTALS_LABEL, Y_TOTALS[0], X1, Y_TOTALS[3]), pad=pad),
        Region.from_box("signature", (X_SIG[0], Y_SIG[0], X_SIG[1], Y_SIG[1]), pad=4),
    ]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def render_form(seed: Optional[int] = 0, blank: bool = False) -> np.ndarray:
    """Render the synthetic order form (BGR, PAGE_W x PAGE_H).

    ``blank=True`` draws only the printed layout (good for building templates);
    otherwise fake customer data and line items are filled in, varied by ``seed``.
    """
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(im)
    f_title, f_lbl, f_val, f_small = _font(40), _font(20), _font(22), _font(16)
    ink = (20, 20, 20)

    def hline(y, xa=X0, xb=X1, w=2):
        d.line([(xa, y), (xb, y)], fill=ink, width=w)

    def vline(x, ya, yb, w=2):
        d.line([(x, ya), (x, yb)], fill=ink, width=w)

    # outer border + header
    d.rectangle([X0, Y_TOP, X1, Y_BOTTOM], outline=ink, width=4)
    d.text((X0 + 24, Y_TOP + 30), "ORDER FORM", font=f_title, fill=ink)
    d.text((X1 - 330, Y_TOP + 28), "Synthetic example document", font=f_small, fill=ink)
    d.text((X1 - 330, Y_TOP + 52), "All data on this page is fictional", font=f_small, fill=ink)
    hline(Y_HEADER, w=3)

    # customer block
    for y in Y_PARTY[1:]:
        hline(y)
    vline(X_PARTY_SPLIT, Y_PARTY[0], Y_PARTY[3])
    vline(X_PARTY_LABEL_L, Y_PARTY[0], Y_PARTY[3], w=1)
    vline(X_PARTY_LABEL_R, Y_PARTY[0], Y_PARTY[3], w=1)
    for (lbl, x, row) in (("Customer name", X0, 0), ("Address", X0, 1),
                          ("Reference", X_PARTY_SPLIT, 0), ("Date", X_PARTY_SPLIT, 1),
                          ("Phone", X_PARTY_SPLIT, 2)):
        d.text((x + 14, Y_PARTY[row] + 24), lbl, font=f_lbl, fill=ink)

    # line-items table
    hline(Y_TABLE_TOP, w=3)
    hline(Y_TABLE_HDR, w=3)
    for r in range(1, N_ROWS):
        hline(Y_TABLE_HDR + r * ROW_H, w=1)
    for x in COLS[1:-1]:
        vline(x, Y_TABLE_TOP, Y_TABLE_BOTTOM)
    for x, lbl in zip(COLS[:-1], ("No.", "Item description", "Quantity", "Unit cost", "Line total")):
        d.text((x + 12, Y_TABLE_TOP + 20), lbl, font=f_lbl, fill=ink)
    hline(Y_TABLE_BOTTOM, w=3)

    # totals
    for y in Y_TOTALS[1:]:
        hline(y, COLS[3], X1)
    vline(COLS[3], Y_TOTALS[0], Y_TOTALS[3])
    vline(X_TOTALS_LABEL, Y_TOTALS[0], Y_TOTALS[3])
    for y, lbl in zip(Y_TOTALS, ("Subtotal", "Tax", "Total due")):
        d.text((COLS[3] + 14, y + 20), lbl, font=f_lbl, fill=ink)
    d.text((X0 + 14, Y_TOTALS[0] + 20), "Notes", font=f_lbl, fill=ink)
    hline(Y_TOTALS[3], w=3)

    # signature
    d.rectangle([X_SIG[0], Y_SIG[0], X_SIG[1], Y_SIG[1]], outline=ink, width=2)
    d.text((X_SIG[0], Y_SIG[1] + 8), "Signature", font=f_lbl, fill=ink)
    d.text((X0 + 14, Y_SIG[0]), "Terms: payment within 30 days.", font=f_small, fill=ink)
    d.text((X0 + 14, Y_SIG[0] + 24), "Goods remain property of the seller until paid.", font=f_small, fill=ink)

    if not blank:
        blue = (25, 35, 120)
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        street = f"{rng.integers(1, 200)} {rng.choice(STREETS)}"
        town = f"{rng.choice(TOWNS)} {rng.integers(10000, 99999)}"
        d.text((X_PARTY_LABEL_L + 14, Y_PARTY[0] + 22), name, font=f_val, fill=blue)
        d.text((X_PARTY_LABEL_L + 14, Y_PARTY[1] + 22), street, font=f_val, fill=blue)
        d.text((X_PARTY_LABEL_L + 14, Y_PARTY[2] + 22), town, font=f_val, fill=blue)
        d.text((X_PARTY_LABEL_R + 14, Y_PARTY[0] + 22), f"REF-{rng.integers(1000, 9999)}-X",
               font=f_val, fill=blue)
        d.text((X_PARTY_LABEL_R + 14, Y_PARTY[1] + 22),
               f"20{rng.integers(30, 40)}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}",
               font=f_val, fill=blue)
        d.text((X_PARTY_LABEL_R + 14, Y_PARTY[2] + 22), f"+00 555 01{rng.integers(10, 99)}",
               font=f_val, fill=blue)
        subtotal = 0.0
        for r in range(int(rng.integers(4, N_ROWS + 1))):
            y = Y_TABLE_HDR + r * ROW_H + 18
            qty = int(rng.integers(1, 50))
            cost = float(rng.integers(100, 99999)) / 100.0
            total = qty * cost
            subtotal += total
            d.text((COLS[0] + 14, y), str(r + 1), font=f_val, fill=blue)
            d.text((COLS[1] + 14, y), str(rng.choice(ITEMS)), font=f_val, fill=blue)
            d.text((COLS[2] + 14, y), str(qty), font=f_val, fill=blue)
            d.text((COLS[3] + 14, y), f"{cost:,.2f}", font=f_val, fill=blue)
            d.text((COLS[4] + 14, y), f"{total:,.2f}", font=f_val, fill=blue)
        tax = subtotal * 0.1
        for y, v in zip(Y_TOTALS, (subtotal, tax, subtotal + tax)):
            d.text((X_TOTALS_LABEL + 14, y + 18), f"{v:,.2f}", font=f_val, fill=blue)
        # scribbled signature
        pts = []
        cx, cy = X_SIG[0] + 40, (Y_SIG[0] + Y_SIG[1]) // 2
        for i in range(40):
            cx += int(rng.integers(3, 10))
            cy = int(np.clip(cy + rng.integers(-18, 19), Y_SIG[0] + 20, Y_SIG[1] - 20))
            pts.append((cx, cy))
        d.line(pts, fill=blue, width=3, joint="curve")

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


@dataclass
class ScanParams:
    angle_deg: float = 0.0          # small rotation
    quarter_turns: int = 0          # 0..3, extra 90-degree clockwise rotations
    scale: float = 0.9
    perspective: float = 0.0        # corner jitter as fraction of page size
    blur: float = 0.8
    noise: float = 5.0
    background: int = 150           # grey level of the scanner bed / table
    jpeg_quality: Optional[int] = 80
    seed: int = 0


def random_scan_params(seed: int, max_angle: float = 6.0, allow_quarter_turns: bool = True) -> ScanParams:
    rng = np.random.default_rng(seed)
    return ScanParams(
        angle_deg=float(rng.uniform(-max_angle, max_angle)),
        quarter_turns=int(rng.integers(0, 4)) if allow_quarter_turns else 0,
        scale=float(rng.uniform(0.7, 1.05)),
        perspective=float(rng.uniform(0.0, 0.04)),
        blur=float(rng.uniform(0.4, 1.2)),
        noise=float(rng.uniform(2.0, 8.0)),
        background=int(rng.integers(90, 200)),
        jpeg_quality=int(rng.integers(60, 92)),
        seed=seed,
    )


def simulate_scan(page: np.ndarray, sp: ScanParams) -> Tuple[np.ndarray, np.ndarray]:
    """Warp ``page`` like a scan or photo. Returns ``(scan, H_true)`` where
    ``H_true`` maps page px -> scan px."""
    rng = np.random.default_rng(sp.seed)
    h, w = page.shape[:2]
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)

    # similarity part: centre, scale, rotate (small angle + quarter turns)
    theta = np.deg2rad(sp.angle_deg + 90.0 * sp.quarter_turns)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = (c - [w / 2, h / 2]) @ R.T * sp.scale
    # perspective: jitter corners
    jitter = rng.uniform(-1, 1, size=(4, 2)) * sp.perspective * np.array([w, h]) * sp.scale
    pts = pts + jitter
    margin = 0.06 * max(w, h) * sp.scale
    pts = pts - pts.min(axis=0) + margin
    out_w, out_h = (pts.max(axis=0) + margin).astype(int)

    H = cv2.getPerspectiveTransform(c.astype(np.float32), pts.astype(np.float32)).astype(np.float64)
    bg = (sp.background,) * 3
    scan = cv2.warpPerspective(page, H, (int(out_w), int(out_h)), flags=cv2.INTER_AREA
                               if sp.scale < 1 else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    # uneven lighting
    yy, xx = np.mgrid[0:scan.shape[0], 0:scan.shape[1]].astype(np.float32)
    shade = 1.0 - 0.12 * (xx / scan.shape[1]) * rng.uniform(0.3, 1.0) - 0.06 * (yy / scan.shape[0])
    scan = np.clip(scan.astype(np.float32) * shade[..., None], 0, 255)
    if sp.blur > 0:
        scan = cv2.GaussianBlur(scan, (0, 0), sp.blur)
    if sp.noise > 0:
        scan = scan + rng.normal(0, sp.noise, scan.shape)
    scan = np.clip(scan, 0, 255).astype(np.uint8)
    if sp.jpeg_quality:
        ok, buf = cv2.imencode(".jpg", scan, [cv2.IMWRITE_JPEG_QUALITY, sp.jpeg_quality])
        scan = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return scan, H


def region_masks(regions: List[Region], H: np.ndarray, shape) -> Dict[str, np.ndarray]:
    """Ground-truth masks of each region in scan coordinates (for evaluation)."""
    out = {}
    for r in regions:
        m = np.zeros(shape[:2], np.uint8)
        poly = cv2.perspectiveTransform(r.polygon.reshape(-1, 1, 2), H).reshape(-1, 2)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
        out[r.name] = m
    return out
