#!/usr/bin/env python3
"""Manifests, taxonomies, splits, and converters for common sources.

Manifest formats (JSON array or JSONL, read by harness.manifest.load_manifest):

    analyze   {"id": "...", "image": "img.png", "labels": ["jpeg"], "group": "ref07"}
    compare   {"id": "...", "image_a": "a.png", "image_b": "b.png",
               "label": "A" | "B" | "tie", "group": "ref07"}

``id`` defaults to the image stem (or ``<a>__vs__<b>``); relative paths resolve
against the manifest's directory, or ``--root``. ``group`` keeps related items
in one split - every distortion of one reference image, for example - so a
rule tuned on one version of a picture is never tested on another version of
it. ``split`` (train/val/test) in an item overrides the hash split.

A taxonomy is a JSON list of ``{"name", "description"}``. The descriptions go
into the analyst's prompt verbatim, so they say what the artifact LOOKS like.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from harness.manifest import load_manifest
from harness.utils import write_json

from .core import TAXONOMY_DIR
from .evidence import load_rgb

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

def load_taxonomy(path: str | Path | None) -> list[dict[str, str]]:
    """Read a taxonomy file; ``None`` means the generic one shipped here."""
    p = Path(path) if path else TAXONOMY_DIR / "generic.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [{"name": k, "description": v} for k, v in data.items()]
    names = [str(t["name"]) for t in data]
    if len(set(names)) != len(names):
        raise ValueError(f"{p}: duplicate type names")
    return [{"name": str(t["name"]), "description": str(t.get("description", ""))}
            for t in data]


def taxonomy_text(taxonomy: list[dict[str, str]]) -> str:
    return "\n".join(f"- {t['name']}: {t['description']}" for t in taxonomy)


def _norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def match_type(raw: Any, taxonomy: list[dict[str, str]]) -> str | None:
    """Map a model's spelling onto a taxonomy name, or ``None``."""
    want = _norm(raw)
    for t in taxonomy:
        if _norm(t["name"]) == want:
            return t["name"]
    return None


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def _resolve(p: str, base: Path) -> str:
    q = Path(p).expanduser()
    return str(q if q.is_absolute() else (base / q).resolve())


def load_items(manifest: str | Path, task: str, *, root: str | Path | None = None,
               taxonomy: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Read and validate a manifest for ``task`` (analyze | compare)."""
    base = Path(root) if root else Path(manifest).resolve().parent
    items = []
    for raw in load_manifest(manifest):
        it = dict(raw)
        if task == "analyze":
            if "image" not in it:
                raise ValueError(f"analyze item without 'image': {raw}")
            it["image"] = _resolve(it["image"], base)
            labels = it.get("labels", [])
            it["labels"] = [labels] if isinstance(labels, str) else list(labels)
            it.setdefault("id", Path(it["image"]).stem)
        elif task == "compare":
            if not {"image_a", "image_b", "label"} <= set(it):
                raise ValueError(f"compare item needs image_a, image_b, label: {raw}")
            it["image_a"] = _resolve(it["image_a"], base)
            it["image_b"] = _resolve(it["image_b"], base)
            label = str(it["label"]).strip()
            it["label"] = "tie" if label.lower() == "tie" else label.upper()
            if it["label"] not in ("A", "B", "tie"):
                raise ValueError(f"label must be A, B or tie: {raw}")
            it.setdefault("id", f"{Path(it['image_a']).stem}__vs__"
                                f"{Path(it['image_b']).stem}")
        else:
            raise ValueError(f"unknown task {task!r}")
        it["id"] = str(it["id"])
        it.setdefault("group", it["id"])
        items.append(it)

    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"duplicate item ids, e.g. {dup}")
    if task == "analyze" and taxonomy is not None:
        names = {t["name"] for t in taxonomy}
        bad = sorted({lab for it in items for lab in it["labels"]} - names)
        if bad:
            raise ValueError(f"labels not in the taxonomy: {bad}")
    return items


def split_items(items: list[dict[str, Any]], fractions=(0.6, 0.2, 0.2),
                seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Deterministic train/val/test split by ``group``.

    A hash, not a shuffle: adding items to a manifest never moves the old ones
    between splits.
    """
    f_train, f_val = fractions[0], fractions[0] + fractions[1]
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for it in items:
        if it.get("split") in SPLITS:
            out[it["split"]].append(it)
            continue
        digest = hashlib.sha1(f"{seed}:{it['group']}".encode("utf-8")).hexdigest()
        u = int(digest[:8], 16) / 0xFFFFFFFF
        out["train" if u < f_train else "val" if u < f_val else "test"].append(it)
    return out


# ---------------------------------------------------------------------------
# KADID-10k
# ---------------------------------------------------------------------------

# Index i is KADID distortion code i+1, in the order of the official database
# page (database.mmsp-kn.de/kadid-10k-database.html). The descriptions restate
# each generating method as what it looks like.
KADID_TYPES = [
    ("gaussian_blur", "even softening of all edges and texture, no preferred direction"),
    ("lens_blur", "defocus-like softening from a circular kernel; small bright points spread into discs"),
    ("motion_blur", "smearing along one straight direction; edges across that direction are streaked"),
    ("color_diffusion", "colours bleed across object edges while brightness detail stays sharp (the a/b colour channels are blurred)"),
    ("color_shift", "a coloured (green/magenta) fringe or offset along strong edges, from a shifted green channel"),
    ("color_quantization", "only a few dozen colours with dithering: false contours, flat colour patches, dot patterns in gradients"),
    ("color_saturation_1", "saturation scaled in HSV space: colours washed out towards grey or unnaturally vivid, with no blur or noise"),
    ("color_saturation_2", "chroma scaled in Lab space: colours washed out towards grey or unnaturally vivid, with no blur or noise"),
    ("jpeg2000", "wavelet compression: blur with ringing around edges and blotchy smooth areas, no 8x8 grid"),
    ("jpeg", "8x8 blocking, mosquito noise around edges, banding in smooth areas"),
    ("white_noise", "grainy random noise on every pixel, in brightness and colour"),
    ("white_noise_color_component", "noise added in YCbCr: blotchy coloured speckles as well as grain"),
    ("impulse_noise", "isolated pure black and white pixels (salt and pepper)"),
    ("multiplicative_noise", "speckle noise whose strength grows with brightness"),
    ("denoise", "over-smoothed, waxy texture with residual blotches, left by a denoiser"),
    ("brighten", "a tone curve lifts brightness while black and white stay fixed: washed-out mid-tones"),
    ("darken", "a tone curve lowers brightness while black and white stay fixed: murky mid-tones"),
    ("mean_shift", "a constant added to every pixel: the whole image uniformly lighter or darker, with clipping at one end"),
    ("jitter", "pixels moved by small random offsets: wobbly, ragged edges and scrambled fine texture"),
    ("non_eccentricity_patch", "small patches moved to nearby positions: local seams and displaced fragments"),
    ("pixelate", "square mosaic blocks from downsizing and nearest-neighbour upsizing"),
    ("quantization", "intensity reduced to a few levels: posterised flat regions with hard contours"),
    ("color_block", "solid blocks of random colour pasted at random positions"),
    ("high_sharpen", "over-sharpening: bright and dark halos along edges, amplified texture and noise"),
    ("contrast_change", "a sigmoid-like curve changes contrast: the image looks flat or harsh"),
]


def kadid_items(root: str | Path, *, min_level: int = 1):
    """Analyze items and taxonomy from a KADID-10k folder.

    Distorted images are named ``Ixx_yy_zz.png`` (reference xx, distortion
    yy, level 1-5); references (``Ixx.png``) are skipped. Low levels can be
    nearly invisible, so ``min_level`` drops them.
    """
    root = Path(root)
    folder = root / "images" if (root / "images").is_dir() else root
    items = []
    for p in sorted(folder.glob("I*_*_*.png")):
        m = re.fullmatch(r"I(\d+)_(\d+)_(\d+)", p.stem)
        if not m:
            continue
        ref, code, level = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= code <= len(KADID_TYPES) or level < min_level:
            continue
        items.append({"id": p.stem, "image": str(p.resolve()),
                      "labels": [KADID_TYPES[code - 1][0]],
                      "group": f"I{ref:02d}", "level": level})
    if not items:
        raise FileNotFoundError(f"no KADID-10k images (Ixx_yy_zz.png) in {folder}")
    taxonomy = [{"name": n, "description": d} for n, d in KADID_TYPES]
    return items, taxonomy


# ---------------------------------------------------------------------------
# MOS -> pairs
# ---------------------------------------------------------------------------

def mos_to_pairs(rows: list[dict[str, str]], *, image_col: str, score_col: str,
                 higher_is_better: bool, margin: float, group_col: str | None = None,
                 max_pairs: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """Preference pairs from per-image scores (MOS, DMOS, Elo...).

    With ``group_col`` (e.g. the reference image) every pair inside a group
    whose score gap is at least ``margin`` is a candidate - the same scene,
    differently processed. Without it, random pairs across the whole set are
    drawn. The A/B order is randomised so the label is not always A. The
    direction of the score must be stated: datasets differ (KADID-10k's DMOS
    is higher-is-better; many DMOS scales are the opposite).
    """
    rng = random.Random(seed)
    groups: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        key = r[group_col] if group_col else ""
        groups.setdefault(key, []).append((r[image_col], float(r[score_col])))

    candidates: list[tuple[str, str, float, float, str]] = []
    for key, members in groups.items():
        if group_col:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.append((members[i][0], members[j][0],
                                       members[i][1], members[j][1], key))
        else:
            for _ in range(max_pairs * 20):
                (a, sa), (b, sb) = rng.sample(members, 2)
                candidates.append((a, b, sa, sb, ""))
    candidates = [c for c in candidates if abs(c[2] - c[3]) >= margin]
    rng.shuffle(candidates)

    pairs, seen = [], set()
    for a, b, sa, sb, key in candidates:
        if len(pairs) >= max_pairs:
            break
        if rng.random() < 0.5:
            a, b, sa, sb = b, a, sb, sa
        pid = f"{Path(a).stem}__vs__{Path(b).stem}"
        if pid in seen or f"{Path(b).stem}__vs__{Path(a).stem}" in seen:
            continue
        seen.add(pid)
        a_better = (sa > sb) == higher_is_better
        pairs.append({"id": pid, "image_a": a, "image_b": b,
                      "label": "A" if a_better else "B",
                      "group": key or pid, "score_a": sa, "score_b": sb})
    return pairs


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Smoke set: known distortions of clean photographs
# ---------------------------------------------------------------------------

SMOKE_TYPES = [
    ("blur", "soft edges and lost fine detail everywhere"),
    ("noise", "random grain on every pixel"),
    ("jpeg", "8x8 blocking, ringing around edges, banding in smooth areas"),
    ("underexposure", "the whole image too dark; shadows crushed"),
    ("overexposure", "the whole image too bright; highlights blown to white"),
    ("pixelation", "square mosaic blocks, as if enlarged from a much smaller image"),
]


def _distort(img: Image.Image, kind: str, level: int, rng: np.random.Generator):
    """``level`` 1-3, mild to strong."""
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur((1.0, 2.0, 3.5)[level - 1]))
    if kind == "noise":
        a = np.asarray(img, dtype=np.float32)
        a = a + rng.normal(0, (8, 16, 30)[level - 1], a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=(30, 15, 6)[level - 1])
        return Image.open(BytesIO(buf.getvalue())).convert("RGB")
    if kind == "underexposure":
        return ImageEnhance.Brightness(img).enhance((0.6, 0.4, 0.25)[level - 1])
    if kind == "overexposure":
        return ImageEnhance.Brightness(img).enhance((1.4, 1.8, 2.4)[level - 1])
    if kind == "pixelation":
        k = (3, 6, 10)[level - 1]
        w, h = img.size
        small = img.resize((max(1, w // k), max(1, h // k)), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)
    raise ValueError(kind)


def _skimage_photos() -> dict[str, Image.Image]:
    """Four clean photographs bundled with scikit-image, if it is installed."""
    import skimage.data as d
    return {n: Image.fromarray(getattr(d, n)()).convert("RGB")
            for n in ("astronaut", "coffee", "chelsea", "rocket")}


def make_smoke(out: str | Path, *, src: str | Path | None = None,
               seed: int = 0) -> dict[str, Path]:
    """Write a small labelled set: every type at 3 levels, plus pairs.

    Analyze labels are the applied type. Compare pairs put level 1 against
    level 3 of the same type and photograph, so the milder one is better - a
    sanity check for the pipeline, not a benchmark.
    """
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if src:
        photos = {p.stem: load_rgb(p) for p in sorted(Path(src).iterdir())
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif")}
    else:
        photos = _skimage_photos()
    if not photos:
        raise FileNotFoundError(f"no images in {src}")
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)

    analyze, compare = [], []
    for name, img in photos.items():
        for kind, _ in SMOKE_TYPES:
            files = {}
            for level in (1, 2, 3):
                rel = f"images/{name}__{kind}_L{level}.png"
                _distort(img, kind, level, rng).save(out / rel)
                files[level] = rel
                analyze.append({"id": f"{name}__{kind}_L{level}", "image": rel,
                                "labels": [kind], "group": name, "level": level})
            a, b, label = files[1], files[3], "A"
            if pick.random() < 0.5:
                a, b, label = b, a, "B"
            compare.append({"id": f"{name}__{kind}_L1_vs_L3", "image_a": a,
                            "image_b": b, "label": label, "group": name})

    paths = {"analyze": out / "analyze.jsonl", "compare": out / "compare.jsonl",
             "taxonomy": out / "taxonomy.json"}
    for key in ("analyze", "compare"):
        rows = analyze if key == "analyze" else compare
        paths[key].write_text("".join(json.dumps(r) + "\n" for r in rows),
                              encoding="utf-8")
    write_json(paths["taxonomy"], [{"name": n, "description": d}
                                   for n, d in SMOKE_TYPES])
    return paths
