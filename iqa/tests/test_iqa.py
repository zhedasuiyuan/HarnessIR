"""Offline tests: no API calls. Run from the repository root:

    python -m pytest iqa/tests -q
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from iqa.__main__ import main as cli
from iqa.analyze import AnalyzeConfig, analyze_image
from iqa.compare import CompareConfig, combine, compare_pair
from iqa.core import BudgetExceeded, CachedVLM, fill
from iqa.data import (kadid_items, load_items, mos_to_pairs, split_items)
from iqa.evidence import crop_box, measure, to_box
from iqa.metrics import analyze_metrics, compare_metrics, main_metric
from iqa.tune import TuneConfig, tune

TAX = [{"name": "jpeg", "description": "blocking"},
       {"name": "blur", "description": "soft"}]


def scene(seed: int = 0, size: int = 203) -> Image.Image:
    """A synthetic photo: gradients, smooth texture and off-grid edges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    low = rng.normal(0, 1, (size // 16 + 2, size // 16 + 2))
    low = np.kron(low, np.ones((16, 16)))[:size, :size]
    base = 80 + 0.4 * xx + 0.2 * yy + 12 * low
    base[37:141, 53:119] += 60
    base[90:171, 131:190] -= 50
    rgb = np.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=-1)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(1.0))


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_fill_is_single_pass_and_strict():
    assert fill("a {{X}} b", X="{{Y}}") == "a {{Y}} b"
    with pytest.raises(ValueError):
        fill("{{X}} {{Y}}", X="1")


class Counting:
    model, temperature = "fake", 0.0

    def __init__(self, replies):
        self.replies, self.n = list(replies), 0

    def chat_json(self, user, images=None, *, system=None):
        self.n += 1
        return self.replies.pop(0)


def test_cache_validation_and_budget(tmp_path):
    def need_ok(obj):
        if not obj.get("ok"):
            raise ValueError("bad")

    client = Counting([{"ok": False}, {"ok": True}])
    vlm = CachedVLM(client, tmp_path)
    assert vlm("s", "u", validate=need_ok) == {"ok": True}
    assert (client.n, vlm.calls) == (2, 2)          # the bad reply was retried
    assert vlm("s", "u", validate=need_ok) == {"ok": True}
    assert (client.n, vlm.hits) == (2, 1)           # ...and only the good one cached

    capped = CachedVLM(Counting([{"ok": True}] * 3), None, max_calls=1)
    capped("s", "1")
    with pytest.raises(BudgetExceeded):
        capped("s", "2")


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def test_measurements_move_with_distortions():
    clean = scene()
    m0 = measure(clean)
    noisy = np.asarray(clean, dtype=np.float32) + \
        np.random.default_rng(1).normal(0, 15, (203, 203, 3))
    m_noise = measure(Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)))
    m_blur = measure(clean.filter(ImageFilter.GaussianBlur(3)))
    buf = BytesIO()
    clean.save(buf, format="JPEG", quality=8)
    m_jpeg = measure(Image.open(BytesIO(buf.getvalue())).convert("RGB"))

    assert m_noise["noise_sigma"] > 3 * m0["noise_sigma"]
    assert m_blur["laplacian_var"] < 0.5 * m0["laplacian_var"]
    assert m_jpeg["blockiness"] > m0["blockiness"] + 0.2


def test_boxes_and_crops():
    assert to_box([0.5, 0.5, 0.52, 0.52], 400, 300, 64) == (172, 121, 236, 185)
    assert to_box([-1, -1, 2, 2], 400, 300) == (0, 0, 400, 300)
    assert to_box([100, 50, 300, 250], 400, 300) == (100, 50, 300, 250)   # pixels
    assert to_box("nonsense", 400, 300) is None
    crop, scale = crop_box(scene(), (10, 10, 74, 74))
    assert scale == 4.0 and crop.size == (256, 256)          # 64 px -> x4 nearest


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def test_split_keeps_groups_together():
    items = [{"id": f"{g}_{k}", "group": g} for g in "abcdefghij" for k in range(5)]
    splits = split_items(items, (0.5, 0.25, 0.25), seed=3)
    where = {it["id"]: s for s, its in splits.items() for it in its}
    for g in "abcdefghij":
        assert len({where[f"{g}_{k}"] for k in range(5)}) == 1
    assert splits == split_items(items, (0.5, 0.25, 0.25), seed=3)


def test_mos_to_pairs_direction_and_margin():
    rows = [{"img": "a.png", "s": "4.0", "g": "1"}, {"img": "b.png", "s": "2.0", "g": "1"},
            {"img": "c.png", "s": "3.9", "g": "1"}]
    for hib in (True, False):
        pairs = mos_to_pairs(rows, image_col="img", score_col="s", group_col="g",
                             higher_is_better=hib, margin=0.5)
        assert len(pairs) == 2                               # a~c is under the margin
        for p in pairs:
            better = p["image_a"] if p["label"] == "A" else p["image_b"]
            worse = p["image_b"] if p["label"] == "A" else p["image_a"]
            assert (better, worse)[0 if hib else 1] != "b.png"


def test_kadid_parsing(tmp_path):
    for name in ("I01_10_03.png", "I01.png", "I02_01_01.png"):
        scene().save(tmp_path / name)
    items, taxonomy = kadid_items(tmp_path, min_level=2)
    assert [(it["id"], it["labels"], it["group"]) for it in items] == \
        [("I01_10_03", ["jpeg"], "I01")]
    assert len(taxonomy) == 25 and taxonomy[0]["name"] == "gaussian_blur"


def test_manifest_validation(tmp_path):
    (tmp_path / "m.jsonl").write_text(json.dumps({"image": "x.png", "labels": ["fog"]}))
    with pytest.raises(ValueError, match="not in the taxonomy"):
        load_items(tmp_path / "m.jsonl", "analyze", taxonomy=TAX)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics():
    recs = [{"gold": ["jpeg"], "final": {"types": ["jpeg"], "primary": "jpeg"}},
            {"gold": ["blur"], "final": {"types": ["jpeg"], "primary": "jpeg"}},
            {"gold": ["blur"], "final": None, "error": "boom"}]
    m = analyze_metrics(recs, ["jpeg", "blur"])
    assert m["primary_acc"] == pytest.approx(1 / 3)
    assert m["per_type"]["jpeg"]["precision"] == 0.5
    assert main_metric("analyze", m) == pytest.approx(1 / 3)

    recs = [{"gold": "A", "final": {"better": "A", "consistent": True}},
            {"gold": "B", "final": {"better": "tie", "consistent": False}},
            {"gold": "A", "final": None, "error": "boom"}]
    m = compare_metrics(recs)
    assert m["acc"] == pytest.approx(1.5 / 3) and m["strict_acc"] == pytest.approx(1 / 3)
    assert m["order_consistency"] == 0.5


def test_combine():
    agree = combine([{"better": "A", "confidence": 0.2}, {"better": "A", "confidence": 0.9}])
    assert (agree["better"], agree["consistent"]) == ("A", True)
    flip = combine([{"better": "A", "confidence": 0.8}, {"better": "B", "confidence": 0.8}])
    assert (flip["better"], flip["consistent"]) == ("tie", False)
    lean = combine([{"better": "A", "confidence": 0.9}, {"better": "B", "confidence": 0.1}])
    assert lean["better"] == "A"


# ---------------------------------------------------------------------------
# Pipelines with a scripted model
# ---------------------------------------------------------------------------

class Scripted:
    """Answers by stage; records what it was sent."""
    model, temperature = "scripted", 0.0

    def __init__(self, judge=None, review=None, decide=None):
        self.sent = []
        self.judge, self.review, self.decide = judge, review, decide

    def chat_json(self, user, images=None, *, system=None):
        self.sent.append((system, user, images))
        if '"zoom_regions"' in system:
            return {"content": "test", "intentional": [], "observations": [],
                    "zoom_regions": [{"bbox": [0.1, 0.1, 0.4, 0.4], "why": "edges"}]}
        if '"claims"' in system:
            return self.review.pop(0)
        if '"artifacts"' in system:
            return self.judge(system, user)
        if '"better"' in system:
            return self.decide(images)
        raise AssertionError("unexpected stage")


def test_analyze_redo_loop(tmp_path):
    path = tmp_path / "x.png"
    scene().save(path)
    judge = lambda s, u: {"artifacts": [{"type": "JPEG", "severity": 3},
                                        {"type": "smudge", "severity": 2}],
                          "primary": "Jpeg", "quality": 2}
    client = Scripted(judge=judge, review=[
        {"claims": [{"type": "jpeg", "verdict": "unsupported", "why": "no grid"}],
         "missed": [], "needs_redo": True, "notes": "look at the sky"},
        {"claims": [], "missed": [], "needs_redo": "false", "notes": ""}])
    cfg = AnalyzeConfig(taxonomy=TAX, guidelines="UNIQUE-GUIDE-TEXT", redo=True)
    rec = analyze_image(CachedVLM(client, None), path, cfg)
    assert rec["error"] is None and len(rec["rounds"]) == 2
    assert rec["selected_round"] == 1
    assert rec["final"]["types"] == ["jpeg"] and rec["final"]["unknown_types"] == ["smudge"]
    second_judge_user = [u for s, u, _ in client.sent if '"artifacts"' in s
                         and '"claims"' not in s][1]
    assert "look at the sky" in second_judge_user and "no grid" in second_judge_user
    reviews = [s + u for s, u, _ in client.sent if '"claims"' in s]
    assert reviews and not any("UNIQUE-GUIDE-TEXT" in r for r in reviews)


def _brightness(im):
    im = Image.open(im) if isinstance(im, (str, Path)) else im
    return float(np.asarray(im.convert("L")).mean())


def test_compare_swap(tmp_path):
    dark, bright = tmp_path / "dark.png", tmp_path / "bright.png"
    scene().save(bright)
    Image.eval(scene(), lambda v: v // 3).save(dark)
    judge = lambda s, u: {"artifacts": [], "primary": "none", "quality": 3}

    def prefers_bright(images):
        first, second = _brightness(images[0]), _brightness(images[1])
        return {"better": "first" if first > second else "second", "confidence": 0.8}

    acfg = AnalyzeConfig(taxonomy=TAX, guidelines="g")
    cfg = CompareConfig(guidelines="c", analyze=acfg)
    rec = compare_pair(CachedVLM(Scripted(judge=judge, decide=prefers_bright), None),
                       dark, bright, cfg)
    assert rec["error"] is None
    assert rec["final"]["better"] == "B" and rec["final"]["consistent"] is True
    assert len(rec["crop_boxes"]) >= 1                   # same size -> crop pairs

    biased = lambda images: {"better": "first", "confidence": 0.8}
    rec = compare_pair(CachedVLM(Scripted(judge=judge, decide=biased), None),
                       dark, bright, cfg)
    assert rec["final"]["better"] == "tie" and rec["final"]["consistent"] is False


def test_tune_finds_the_rule(tmp_path):
    items = []
    for i in range(12):
        p = tmp_path / f"im{i}.png"
        scene(i).save(p)
        items.append({"id": f"im{i}", "image": str(p), "labels": ["jpeg"],
                      "group": f"im{i}"})

    class Learner(Scripted):
        def chat_json(self, user, images=None, *, system=None):
            if '"patterns"' in system:
                return {"patterns": [{"pattern": "calls jpeg blur",
                                      "rule": "PREFER-JPEG"}]}
            if '"candidates"' in system:
                cur = user.split("CURRENT GUIDELINES:\n")[1].split("\n\n")[0]
                return {"candidates": [cur + "\n- PREFER-JPEG"]}
            return super().chat_json(user, images, system=system)

    judge = lambda s, u: {"artifacts": [{"type": "jpeg" if "PREFER-JPEG" in s
                                         else "blur", "severity": 3}]}
    splits = {"train": items[:6], "val": items[6:10], "test": items[10:]}
    cfg = AnalyzeConfig(taxonomy=TAX, guidelines="- start")
    result = tune("analyze", CachedVLM(Learner(judge=judge), None), splits, cfg,
                  cfg.guidelines, tmp_path / "run",
                  TuneConfig(steps=2, beam=1, edits=1, batch=4, val_size=4, workers=1),
                  taxonomy=TAX, log=lambda s: None)
    assert result["initial_val"] == 0.0 and result["best_val"] == 1.0
    assert "PREFER-JPEG" in (tmp_path / "run" / "best_guidelines.txt").read_text()


# ---------------------------------------------------------------------------
# The command line, end to end, with the stub model
# ---------------------------------------------------------------------------

def test_cli_dry_run(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        scene(i).save(src / f"photo{i}.png")
    smoke = tmp_path / "smoke"
    assert cli(["make-smoke", "--src", str(src), "--out", str(smoke)]) == 0
    tax = str(smoke / "taxonomy.json")

    out = tmp_path / "eval_a"
    assert cli(["eval", "--task", "analyze", "--manifest", str(smoke / "analyze.jsonl"),
                "--taxonomy", tax, "--dry-run", "--redo", "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["metrics"]["n"] == 36 and metrics["metrics"]["n_errors"] == 0

    out = tmp_path / "eval_c"
    assert cli(["eval", "--task", "compare", "--manifest", str(smoke / "compare.jsonl"),
                "--taxonomy", tax, "--dry-run", "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["metrics"]["n"] == 12 and metrics["metrics"]["acc"] == 0.5

    out = tmp_path / "tune_a"
    assert cli(["tune", "--task", "analyze", "--manifest", str(smoke / "analyze.jsonl"),
                "--taxonomy", tax, "--dry-run", "--steps", "1", "--fractions",
                "0.5,0.5,0", "--out", str(out)]) == 0
    assert (out / "best_guidelines.txt").is_file() and (out / "report.json").is_file()
