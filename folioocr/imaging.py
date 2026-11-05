"""Page images in, model-sized images out.

``smart_resize`` reproduces qwen-vl-utils 0.0.14 exactly (the resize
Infinity-Parser2 applies before its processor is called with
``do_resize=False``): both sides become multiples of 32 (16 px patches, 2x2
merge) and the pixel count is clamped to [min_pixels, max_pixels]. Training and
inference must resize the same way, so everything goes through ``fit_image``.

One visual token covers 32x32 = 1024 pixels, so ``max_pixels`` is the visual
token budget times 1024.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

from PIL import Image

FACTOR = 32                    # patch 16 x spatial merge 2
MIN_PIXELS = 2048              # Infinity-Parser2 default
MAX_PIXELS_OFFICIAL = 16777216  # Infinity-Parser2 default (4096 x 4096)
MAX_RATIO = 200
PIXELS_PER_TOKEN = FACTOR * FACTOR

Image.MAX_IMAGE_PIXELS = None   # 300 DPI scans are large; they are trusted local files


def _round(n: float, f: int) -> int:
    return round(n / f) * f


def _ceil(n: float, f: int) -> int:
    return math.ceil(n / f) * f


def _floor(n: float, f: int) -> int:
    return math.floor(n / f) * f


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS_OFFICIAL) -> Tuple[int, int]:
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be >= min_pixels")
    if max(height, width) / max(1, min(height, width)) > MAX_RATIO:
        raise ValueError(f"aspect ratio above {MAX_RATIO}: {width}x{height}")
    h_bar = max(factor, _round(height, factor))
    w_bar = max(factor, _round(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor(height / beta, factor)
        w_bar = _floor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil(height * beta, factor)
        w_bar = _ceil(width * beta, factor)
    return h_bar, w_bar


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def fit_image(img: Image.Image, max_pixels: int, min_pixels: int = MIN_PIXELS,
              factor: int = FACTOR) -> Image.Image:
    """Resize exactly like qwen-vl-utils' ``fetch_image`` (PIL default resample)."""
    img = to_rgb(img)
    w, h = img.size
    rh, rw = smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    return img if (rw, rh) == (w, h) else img.resize((rw, rh))


def visual_tokens(img: Image.Image, max_pixels: int, min_pixels: int = MIN_PIXELS) -> int:
    w, h = img.size
    rh, rw = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
    return (rh * rw) // PIXELS_PER_TOKEN


# --------------------------------------------------------------------- PDF
def pdf_page_count(path: str) -> int:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_pdf(path, dpi: Optional[float] = 300, longest_side: Optional[int] = None,
               pages: Optional[Sequence[int]] = None) -> List[Image.Image]:
    """Render PDF pages (1-based ``pages``; all by default). ``path`` may also
    be the PDF's bytes (e.g. a member read from a tarball).

    ``longest_side`` (e.g. 1288, olmOCR-Bench's common setting) overrides
    ``dpi`` and scales each page so its longer side has that many pixels."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    out = []
    try:
        idx = range(len(pdf)) if pages is None else [p - 1 for p in pages]
        for i in idx:
            page = pdf[i]
            w_pt, h_pt = page.get_size()
            scale = (longest_side / max(w_pt, h_pt)) if longest_side else (dpi or 300) / 72.0
            out.append(to_rgb(page.render(scale=scale).to_pil()))
            page.close()
    finally:
        pdf.close()
    return out


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}


def load_pages(path: str, dpi: Optional[float] = 300, longest_side: Optional[int] = None,
               pages: Optional[Sequence[int]] = None) -> List[Image.Image]:
    """A PDF (all or selected pages) or an image file -> list of RGB images."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return render_pdf(path, dpi=dpi, longest_side=longest_side, pages=pages)
    if ext not in IMAGE_EXTS:
        raise ValueError(f"unsupported file type: {path}")
    img = Image.open(path)
    if ext in (".tif", ".tiff") and getattr(img, "n_frames", 1) > 1:
        frames = []
        for i in range(img.n_frames):
            if pages is None or (i + 1) in pages:
                img.seek(i)
                frames.append(to_rgb(img.copy()))
        return frames
    img = to_rgb(img)
    if longest_side and max(img.size) != longest_side:
        s = longest_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    return [img]
