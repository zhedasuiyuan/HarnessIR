#!/usr/bin/env python3
"""Stage 2 for IQA: pixel statistics and native-resolution crops.

HarnessIR's tools (OCR, faces, depth, segmentation) gather evidence about the
content the editor must keep. For quality assessment the evidence is in the
pixels, and the obstacle is scale: the model sees a downscaled image, and
downscaling hides exactly what is being judged - noise, blocking, ringing,
fine blur. So the analyst also gets the regions the diagnoser picked, cut out
at native resolution, plus a few classical statistics.

As in HarnessIR the numbers are evidence, never thresholds: each is reported
together with the way it misleads, and the model decides.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Files in these formats are sent exactly as stored. Anything else (BMP, TIFF,
# WebP) is decoded and sent as PNG, because the client labels every non-PNG
# path as JPEG.
SENT_AS_STORED = {".png", ".jpg", ".jpeg"}

# Used when the diagnoser names no region: the centre usually holds the subject.
DEFAULT_REGION = {"bbox": [0.35, 0.35, 0.65, 0.65],
                  "why": "default: centre of the frame"}


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def global_view(path: str | Path, img: Image.Image, max_side: int):
    """The whole image, as the model will see it.

    Sent as stored when it is small enough and a PNG or JPEG, so the model
    judges the exact file. Otherwise downscaled (or converted) and sent as PNG:
    re-encoding as JPEG would add the very artifacts being assessed.
    """
    w, h = img.size
    if max(w, h) <= max_side:
        return str(path) if Path(path).suffix.lower() in SENT_AS_STORED else img
    s = max_side / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def view_size(view: Any, img: Image.Image) -> tuple[int, int]:
    return img.size if isinstance(view, str) else view.size


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _blockiness(y: np.ndarray, block: int = 8) -> float | None:
    """Luminance steps on the 8-pixel grid divided by steps everywhere else."""
    h, w = y.shape
    if h < 2 * block or w < 2 * block:
        return None
    dh = np.abs(np.diff(y, axis=1))          # column j: step from j to j+1
    dv = np.abs(np.diff(y, axis=0))
    grid = np.s_[block - 1::block]
    on_h, off_h = dh[:, grid].mean(), np.delete(dh, grid, axis=1).mean()
    on_v, off_v = dv[grid, :].mean(), np.delete(dv, grid, axis=0).mean()
    return float(0.5 * (on_h / max(off_h, 1e-6) + on_v / max(off_v, 1e-6)))


def measure(img: Image.Image) -> dict[str, Any]:
    """Classical statistics over the whole image, at native resolution."""
    rgb = np.asarray(img, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    h, w = y.shape
    out: dict[str, Any] = {"width": w, "height": h}
    if min(h, w) >= 3:
        # Immerkaer (1996): a Laplacian-difference kernel that cancels smooth
        # structure, then the mean absolute response scaled to a Gaussian sigma.
        c = (y[:-2, :-2] - 2 * y[:-2, 1:-1] + y[:-2, 2:]
             - 2 * y[1:-1, :-2] + 4 * y[1:-1, 1:-1] - 2 * y[1:-1, 2:]
             + y[2:, :-2] - 2 * y[2:, 1:-1] + y[2:, 2:])
        out["noise_sigma"] = math.sqrt(math.pi / 2) * float(np.abs(c).mean()) / 6
        lap = (y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
               - 4 * y[1:-1, 1:-1])
        out["laplacian_var"] = float(lap.var())
    out["blockiness"] = _blockiness(y)
    out["mean_luma"] = float(y.mean())
    out["dark_clip"] = float((y <= 3).mean())
    out["bright_clip"] = float((y >= 252).mean())
    out["rms_contrast"] = float(y.std())
    rg, yb = r - g, 0.5 * (r + g) - b
    out["colorfulness"] = float(math.hypot(rg.std(), yb.std())
                                + 0.3 * math.hypot(rg.mean(), yb.mean()))
    out["cast_rg"], out["cast_yb"] = float(rg.mean()), float(yb.mean())
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in out.items()}


# How each number misleads. Read with the values, never instead of the pictures.
MEANINGS = {
    "noise_sigma": "noise estimate (std on a 0-255 scale). Higher = noisier, "
                   "but fine texture (foliage, fabric, hair) raises it too.",
    "laplacian_var": "edge energy. Higher = more fine detail or sharper edges. "
                     "Depends heavily on content, and noise raises it.",
    "blockiness": "steps on the 8-pixel grid / steps elsewhere. About 1.0 = no "
                  "grid; clearly above 1 suggests JPEG-style blocking. Resizing "
                  "after compression erases the grid, so ~1.0 does not rule "
                  "compression out.",
    "mean_luma": "mean brightness, 0-255.",
    "dark_clip": "fraction of pure-black pixels (crushed shadows); legitimate "
                 "in night scenes and silhouettes.",
    "bright_clip": "fraction of pure-white pixels (blown highlights); legitimate "
                   "for light sources and bright sky.",
    "rms_contrast": "spread of brightness; low = flat or hazy, high = harsh.",
    "colorfulness": "Hasler-Suesstrunk colourfulness; near 0 = grey. Depends "
                    "on content.",
    "cast_rg": "mean red(+)/green(-) offset; far from 0 suggests a colour cast "
               "unless the scene really is that colour.",
    "cast_yb": "mean yellow(+)/blue(-) offset; same caveat.",
}


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:g}"


def render_measurements(m: dict[str, Any]) -> str:
    """The statistics as text, each with its caveat."""
    lines = [f"size: {m.get('width')}x{m.get('height')} px"]
    lines += [f"{key} = {_fmt(m.get(key))}  ({meaning})"
              for key, meaning in MEANINGS.items()]
    return "\n".join(lines)


def render_measurement_pair(m1: dict[str, Any], m2: dict[str, Any]) -> str:
    """Two images' statistics side by side, as ``first / second``."""
    lines = [f"size: {m1.get('width')}x{m1.get('height')} / "
             f"{m2.get('width')}x{m2.get('height')} px"]
    lines += [f"{key} = {_fmt(m1.get(key))} / {_fmt(m2.get(key))}  ({meaning})"
              for key, meaning in MEANINGS.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------

def to_box(bbox: Any, w: int, h: int, min_side: int = 64):
    """``[x0, y0, x1, y1]`` fractions -> a pixel box of at least ``min_side``."""
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if max(x0, y0, x1, y1) > 2.0:            # the model answered in pixels
        x0, x1, y0, y1 = x0 / w, x1 / w, y0 / h, y1 / h

    def span(a: float, b: float, size: int) -> tuple[int, int]:
        a, b = sorted((min(max(a, 0.0), 1.0), min(max(b, 0.0), 1.0)))
        lo, hi = round(a * size), round(b * size)
        need = min(min_side, size)
        if hi - lo < need:
            mid = (lo + hi) // 2
            lo = max(0, min(size - need, mid - need // 2))
            hi = lo + need
        return lo, hi

    (px0, px1), (py0, py1) = span(x0, x1, w), span(y0, y1, h)
    return px0, py0, px1, py1


def crop_box(img: Image.Image, box, *, target: int = 512, max_side: int = 1024,
             max_upscale: int = 4) -> tuple[Image.Image, float]:
    """Cut one box at native resolution and size it for viewing.

    Small crops are enlarged by an integer nearest-neighbour factor, which
    keeps pixel structure (blocking, aliasing, grain) visible instead of
    smoothing it; large crops are reduced to ``max_side``.
    """
    crop = img.crop(tuple(box))
    cw, ch = crop.size
    if max(cw, ch) > max_side:
        s = max_side / max(cw, ch)
        return crop.resize((max(1, round(cw * s)), max(1, round(ch * s))),
                           Image.LANCZOS), s
    k = min(max_upscale, target // max(cw, ch))
    if k > 1:
        return crop.resize((cw * k, ch * k), Image.NEAREST), float(k)
    return crop, 1.0


def crop_regions(img: Image.Image, regions: list[dict[str, Any]], *,
                 max_regions: int = 3, min_side: int = 64) -> list[dict[str, Any]]:
    """The diagnoser's regions, as ``[{"image", "box", "scale", "why"}]``."""
    w, h = img.size
    out = []
    for reg in (regions or [DEFAULT_REGION])[:max_regions]:
        box = to_box((reg or {}).get("bbox"), w, h, min_side)
        if box is None:
            continue
        crop, scale = crop_box(img, box)
        out.append({"image": crop, "box": list(box), "scale": round(scale, 3),
                    "why": str(reg.get("why", ""))})
    return out


def describe_crop(i: int, c: dict[str, Any]) -> str:
    """One line telling the model where a crop came from."""
    x0, y0, x1, y1 = c["box"]
    zoom = (f", enlarged {c['scale']:g}x nearest-neighbour" if c["scale"] > 1
            else (f", reduced to {c['scale']:.2f}x" if c["scale"] < 1 else ""))
    why = f" - {c['why']}" if c.get("why") else ""
    return f"Picture {i}: crop x {x0}-{x1}, y {y0}-{y1} px{zoom}{why}"
