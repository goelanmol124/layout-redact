"""Page input/output: rasterise PDFs and images, write PNG or PDF.

All images inside the library are OpenCV-style ``uint8`` BGR arrays.

PDF *input* needs an optional rasteriser: ``pypdfium2`` (preferred) or
``PyMuPDF`` if it happens to be installed. PDF *output* uses Pillow only and
is always raster: every page is re-encoded as an image, so text hidden under
a black box cannot be copied out of the redacted PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np
from PIL import Image

PathLike = Union[str, Path]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}


def is_supported(path: PathLike) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS | PDF_EXTS


def load_pages(path: PathLike, dpi: int = 150) -> List[np.ndarray]:
    """Load every page of an image or PDF as a BGR array.

    Multi-frame TIFFs are returned frame by frame.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return _rasterise_pdf(path, dpi)
    if ext in IMAGE_EXTS:
        pages = []
        with Image.open(path) as im:
            frame = 0
            while True:
                try:
                    im.seek(frame)
                except EOFError:
                    break
                pages.append(_pil_to_bgr(im))
                frame += 1
        return pages
    raise ValueError(f"Unsupported input type: {path}")


def load_image(path: PathLike, dpi: int = 150) -> np.ndarray:
    """Load the first page of an image or PDF."""
    pages = load_pages(path, dpi=dpi)
    if not pages:
        raise ValueError(f"No pages in {path}")
    return pages[0]


def save_image(image: np.ndarray, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _bgr_to_pil(image).save(path)


def save_pdf(pages: Sequence[np.ndarray], path: PathLike, dpi: int = 150) -> None:
    """Write pages as a raster PDF (one image per page)."""
    if not pages:
        raise ValueError("No pages to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = [_bgr_to_pil(p) for p in pages]
    ims[0].save(path, "PDF", resolution=float(dpi), save_all=True, append_images=ims[1:])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    arr = np.asarray(im)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _rasterise_pdf(path: Path, dpi: int) -> List[np.ndarray]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError:
        pdfium = None

    if pdfium is not None:
        pdf = pdfium.PdfDocument(str(path))
        try:
            pages = []
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=dpi / 72.0)
                pages.append(_pil_to_bgr(bitmap.to_pil()))
                page.close()
            return pages
        finally:
            pdf.close()

    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PDF input needs a rasteriser. Install one with: pip install pypdfium2"
        ) from exc

    pages = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(_pil_to_bgr(im))
    return pages
