"""Page-level scan / photo / old-document degradations that also move the boxes.

The pixel operations come from ResumeOCR-Lite's resumeocr/augment.py (line
level) and are applied here to whole pages. Geometric steps (rotation,
perspective, framing on a background, page curl) transform every block and
line box with the same mapping, so labels stay aligned with the pixels.

Conditions:
    clean  born-digital render, at most a light JPEG
    scan   flatbed/office scan: paper tint, slight skew, blur, low DPI, noise,
           sometimes photocopy binarisation, streaks, bleed-through
    photo  phone photo: perspective, background around the page, uneven light,
           shadows, defocus, colour cast
    old    old typewritten/printed archive scan: yellowed paper, faded ink,
           blotches, heavy noise, threshold artefacts
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

Rng = np.random.Generator
CONDITIONS = ("clean", "scan", "photo", "old")


# ----------------------------------------------------------- pixel helpers
def smooth_noise(shape: Tuple[int, int], rng: Rng, cell: float) -> np.ndarray:
    h, w = shape
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    grid = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def paper_field(shape: Tuple[int, int], rng: Rng, strength: float = 1.0) -> np.ndarray:
    """Multiplicative paper texture around 1.0 (mottling + grain + fibres)."""
    h, w = shape
    tex = np.zeros((h, w), np.float32)
    if rng.random() < 0.8:
        tex += smooth_noise((h, w), rng, cell=rng.uniform(40, 240)) * rng.uniform(0.005, 0.035) * strength
    if rng.random() < 0.7:
        g = np.empty((h, w), np.float32)
        cv2.randn(g, 0.0, float(rng.uniform(0.003, 0.02) * strength))
        tex += g
    if rng.random() < 0.2:
        for _ in range(int(rng.integers(2, 12))):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            ang, ln = rng.uniform(0, np.pi), rng.uniform(4, 40)
            cv2.line(tex, (x, y), (int(x + ln * np.cos(ang)), int(y + ln * np.sin(ang))),
                     -float(rng.uniform(0.03, 0.12)), 1, cv2.LINE_AA)
    return 1.0 + tex


def uneven_lighting(img: np.ndarray, rng: Rng, strength: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    sh, sw = max(2, h // 8), max(2, w // 8)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    yy *= h / sh
    xx *= w / sw
    kind = rng.integers(0, 3)
    if kind == 0:
        ang = rng.uniform(0, 2 * np.pi)
        g = xx * np.cos(ang) + yy * np.sin(ang)
        g = (g - g.min()) / max(1e-6, float(g.max() - g.min()))
        field = 1.0 - rng.uniform(0.05, 0.3) * strength * g
    elif kind == 1:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(h, w)
        field = 1.0 - rng.uniform(0.05, 0.25) * strength * np.clip(r * rng.uniform(0.8, 2.0), 0, 1)
    else:
        ang = rng.uniform(0, 2 * np.pi)
        d = (xx - rng.uniform(0, w)) * np.cos(ang) + (yy - rng.uniform(0, h)) * np.sin(ang)
        soft = rng.uniform(3, 0.3 * max(h, w))
        field = 1.0 - rng.uniform(0.1, 0.4) * strength / (1.0 + np.exp(-d / soft))
    field = cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.astype(np.float32) * (field[..., None] if img.ndim == 3 else field)
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if ok else img


def sensor_noise(img: np.ndarray, rng: Rng, sigma: Tuple[float, float] = (2, 10)) -> np.ndarray:
    n = np.empty(img.shape[:2], np.float32)
    cv2.randn(n, 0.0, float(rng.uniform(*sigma)))
    out = img.astype(np.float32) + (n[..., None] if img.ndim == 3 else n)
    return np.clip(out, 0, 255).astype(np.uint8)


def salt_pepper(img: np.ndarray, rng: Rng, amount: float) -> np.ndarray:
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out


def binarize(img: np.ndarray, rng: Rng) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if rng.random() < 0.5:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        blk = int(rng.integers(15, 45)) | 1
        b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, int(rng.integers(5, 15)))
    return b if img.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def resolution_drop(img: np.ndarray, rng: Rng, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    interp = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST][int(rng.choice(3, p=[0.5, 0.4, 0.1]))]
    return cv2.resize(small, (w, h), interpolation=interp)


def scanner_streaks(img: np.ndarray, rng: Rng) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.7:
            x, wd = int(rng.integers(0, w)), int(rng.integers(1, 4))
            out[:, x:x + wd] *= rng.uniform(0.75, 0.97)
        else:
            y, wd = int(rng.integers(0, h)), int(rng.integers(1, 3))
            out[y:y + wd] *= rng.uniform(0.75, 0.97)
    return np.clip(out, 0, 255).astype(np.uint8)


def bleed_through(img: np.ndarray, rng: Rng) -> np.ndarray:
    """Faint mirror image of the page itself (text from the back side)."""
    back = cv2.flip(img, 1)
    back = cv2.GaussianBlur(back, (0, 0), rng.uniform(1.0, 2.5))
    dark = (255 - back.astype(np.float32)) * rng.uniform(0.04, 0.12)
    return np.clip(img.astype(np.float32) - dark, 0, 255).astype(np.uint8)


def fade_ink(img: np.ndarray, rng: Rng, amount: float) -> np.ndarray:
    f = img.astype(np.float32)
    return np.clip(f + (255 - f) * amount, 0, 255).astype(np.uint8)


# ---------------------------------------------------------- geometry
def warp_boxes(boxes: np.ndarray, hm: np.ndarray) -> np.ndarray:
    """Axis-aligned hull of each box after a homography (corners + edge midpoints)."""
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    pts = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x2, y2], 1), np.stack([x1, y2], 1),
                    np.stack([xm, y1], 1), np.stack([xm, y2], 1), np.stack([x1, ym], 1), np.stack([x2, ym], 1)], 1)
    flat = pts.reshape(-1, 2).astype(np.float64)
    p = np.concatenate([flat, np.ones((len(flat), 1))], 1) @ hm.T
    p = (p[:, :2] / p[:, 2:3]).reshape(len(boxes), 8, 2)
    return np.concatenate([p.min(1), p.max(1)], 1).astype(np.float32)


def apply_homography(img: np.ndarray, hm: np.ndarray, out_size: Tuple[int, int], border) -> np.ndarray:
    return cv2.warpPerspective(img, hm, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=border)


def rotation_h(w: int, h: int, deg: float, shear: float = 0.0) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    t1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], np.float64)
    rs = np.array([[c, -s + shear, 0], [s, c, 0], [0, 0, 1]], np.float64)
    t2 = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]], np.float64)
    return t2 @ rs @ t1


def curl_page(img: np.ndarray, rng: Rng) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    h, w = img.shape[:2]
    amp = rng.uniform(0.002, 0.008) * h
    period = rng.uniform(1.2, 3.0) * w
    phase = rng.uniform(0, 2 * np.pi)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_y = yy + amp * np.sin(2 * np.pi * xx / period + phase).astype(np.float32)
    out = cv2.remap(img, xx, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, (amp, period, phase)


def curl_boxes(boxes: np.ndarray, params: Tuple[float, float, float]) -> np.ndarray:
    """A source point (x, y) lands at y - amp*sin(2*pi*x/period + phase)."""
    if len(boxes) == 0:
        return boxes
    amp, period, phase = params
    xs = np.linspace(0, 1, 9)
    out = boxes.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        px = x1 + xs * (x2 - x1)
        d = amp * np.sin(2 * np.pi * px / period + phase)
        out[i, 1] = y1 - d.max()
        out[i, 3] = y2 - d.min()
    return out


def desk_background(h: int, w: int, rng: Rng) -> np.ndarray:
    base = np.array([rng.uniform(40, 200), rng.uniform(40, 190), rng.uniform(40, 180)], np.float32)
    tex = smooth_noise((h, w), rng, cell=rng.uniform(30, 200)) * rng.uniform(5, 30)
    img = base[None, None, :] + tex[..., None]
    if rng.random() < 0.5:            # wood-like streaks
        streak = np.sin(np.linspace(0, rng.uniform(20, 80), w))[None, :] * rng.uniform(4, 15)
        img = img + streak[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- pipeline
def degrade(img_rgb: np.ndarray, boxes: Dict[str, np.ndarray], condition: str, rng: Rng,
            dpi: float) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, Dict]:
    """Returns (RGB image, moved boxes, JPEG quality to save with, applied-steps log)."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    log: Dict = {"condition": condition}
    boxes = {k: v.astype(np.float32).copy() for k, v in boxes.items()}

    def move(hm):
        for k in boxes:
            boxes[k] = warp_boxes(boxes[k], hm)

    if condition == "clean":
        if rng.random() < 0.1:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.6))
            log["blur"] = True
        q = int(rng.integers(85, 96))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), boxes, q, log

    gray = condition == "old" or (condition == "scan" and rng.random() < 0.6)
    if gray:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # paper tone + texture (multiply blend: white becomes paper, ink stays dark)
    if condition == "old":                      # yellowed paper: R > G > B (BGR order here)
        red = rng.uniform(212, 240)
        green = red - rng.uniform(4, 20)
        tone = np.array([green - rng.uniform(15, 50), green, red], np.float32)
        img = fade_ink(img, rng, rng.uniform(0.05, 0.35))
        log["faded"] = True
    elif condition == "scan":
        t = rng.uniform(228, 255)
        tone = np.array([t - rng.uniform(0, 8), t - rng.uniform(0, 5), t], np.float32)
    else:
        tone = np.array([rng.uniform(205, 245)] * 3, np.float32) * np.array([rng.uniform(0.9, 1.0), rng.uniform(0.95, 1.0), 1.0], np.float32)
    paper = paper_field((h, w), rng, strength=2.5 if condition == "old" else 1.0)
    img = np.clip(img.astype(np.float32) / 255.0 * tone[None, None, :] * paper[..., None], 0, 255).astype(np.uint8)
    border = tuple(int(v) for v in tone)

    if condition in ("scan", "old"):
        if rng.random() < 0.12 if condition == "scan" else rng.random() < 0.3:
            img = bleed_through(img, rng)
            log["bleed"] = True
        deg = rng.normal(0, 0.6 if condition == "scan" else 1.0)
        deg = float(np.clip(deg, -2.0, 2.0))
        if abs(deg) > 0.05:
            hm = rotation_h(w, h, deg, shear=rng.normal(0, 0.004))
            img = apply_homography(img, hm, (w, h), border)
            move(hm)
            log["rotate"] = round(deg, 2)
        target_dpi = rng.uniform(90, 200) if condition == "old" else rng.uniform(120, 300)
        if target_dpi < dpi * 0.9 and rng.random() < 0.6:
            img = resolution_drop(img, rng, target_dpi / dpi)
            log["dpi"] = int(target_dpi)
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.9) * dpi / 200)
            log["blur"] = True
        if rng.random() < (0.12 if condition == "scan" else 0.3):
            img = binarize(img, rng)
            log["binarize"] = True
        if rng.random() < 0.1:
            img = scanner_streaks(img, rng)
            log["streaks"] = True
        if rng.random() < 0.25:
            img = uneven_lighting(img, rng, 0.7)
            log["light"] = True
        if condition == "old" and rng.random() < 0.6:
            blot = smooth_noise((h, w), rng, cell=rng.uniform(60, 300))
            mask = np.clip((blot - 0.55) * 3, 0, 1)[..., None] * rng.uniform(15, 45)
            img = np.clip(img.astype(np.float32) - mask, 0, 255).astype(np.uint8)
            log["blotches"] = True
        if rng.random() < (0.5 if condition == "scan" else 0.8):
            img = sensor_noise(img, rng, (2, 8) if condition == "scan" else (4, 14))
        if rng.random() < (0.08 if condition == "scan" else 0.3):
            img = salt_pepper(img, rng, rng.uniform(0.0005, 0.004))
        q = int(rng.integers(40, 96) if condition == "scan" else rng.integers(30, 85))
        img = jpeg(img, q)
        clip_boxes(boxes, w, h)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), boxes, int(rng.integers(88, 95)), log

    # ---- photo: frame the page on a background with a perspective view
    pad = rng.uniform(0.03, 0.12)
    W, H = int(w * (1 + 2 * pad)), int(h * (1 + 2 * pad))
    jit = rng.uniform(0.0, 0.06)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    base = np.float32([[pad * w, pad * h], [pad * w + w, pad * h], [pad * w + w, pad * h + h], [pad * w, pad * h + h]])
    dst = base + rng.uniform(-jit, jit, (4, 2)).astype(np.float32) * np.float32([w, h])
    hm = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    ang = rng.normal(0, 1.5)
    hm = rotation_h(W, H, float(np.clip(ang, -5, 5))) @ hm
    bg = desk_background(H, W, rng)
    warped = cv2.warpPerspective(img, hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), hm, (W, H), flags=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    img = (warped.astype(np.float32) * m + bg.astype(np.float32) * (1 - m)).astype(np.uint8)
    move(hm)
    log["perspective"] = True
    h, w = img.shape[:2]
    if rng.random() < 0.3:
        img, params = curl_page(img, rng)
        for k in boxes:
            boxes[k] = curl_boxes(boxes[k], params)
        log["curl"] = True
    if rng.random() < 0.75:
        img = uneven_lighting(img, rng, rng.uniform(0.8, 1.6))
        log["light"] = True
    if rng.random() < 0.5:
        cast = np.array([rng.uniform(0.85, 1.05), rng.uniform(0.9, 1.05), rng.uniform(0.9, 1.1)], np.float32)
        img = np.clip(img.astype(np.float32) * cast[None, None, :], 0, 255).astype(np.uint8)
    r = rng.random()
    if r < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.4) * dpi / 200)
        log["blur"] = True
    elif r < 0.4:
        k = int(rng.integers(3, 8))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, rot, (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
        log["motion"] = True
    if rng.random() < 0.6:
        img = sensor_noise(img, rng, (2, 9))
    img = jpeg(img, int(rng.integers(50, 93)))
    clip_boxes(boxes, w, h)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), boxes, int(rng.integers(88, 95)), log


def clip_boxes(boxes: Dict[str, np.ndarray], w: int, h: int) -> None:
    for k in boxes:
        b = boxes[k]
        if len(b):
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w - 1)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h - 1)
