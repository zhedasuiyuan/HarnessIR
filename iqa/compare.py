#!/usr/bin/env python3
"""Pairwise comparison: which of two images has better technical quality.

Harness mode builds on the single-image stages:

    1-3  each image is analysed on its own (diagnosis, statistics, crops,
         judge - cached, so an image shared by many pairs costs once); the
         regions both diagnosers picked are cut from BOTH images when they
         have the same size, giving crop pairs of the same content
    4    the compare judge sees both images, the crop pairs, the statistics
         side by side and both analyses
    4b   with swap on, the pair is judged again in the other order. A
         decision that flips with the order is position bias, so the two
         are combined (see ``combine``) instead of trusting either
    5    with redo on, a reviewer checks the stated differences against the
         pictures; a redo judges again, both orders, with its notes

Plain mode is the baseline: both whole images, one call (two with --swap).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .analyze import AnalyzeConfig, MODES, analyze_image, select_round
from .core import BudgetExceeded, as_bool, fill, load_prompt
from .evidence import (DEFAULT_REGION, crop_box, global_view, load_rgb, measure,
                       render_measurement_pair, to_box, view_size)


@dataclass
class CompareConfig:
    guidelines: str
    analyze: AnalyzeConfig         # for the per-image analyses (harness mode)
    mode: str = "harness"
    swap: bool = True
    redo: bool = False
    max_rounds: int = 3
    max_side: int = 1024
    max_regions: int = 3


def _check_decision(obj: dict[str, Any]) -> None:
    if str(obj.get("better", "")).strip().lower() not in ("first", "second", "tie"):
        raise ValueError(f"better must be first, second or tie: {obj.get('better')!r}")


def _check_review(obj: dict[str, Any]) -> None:
    if as_bool(obj.get("needs_redo")) is None:
        raise ValueError(f"needs_redo is not a boolean: {obj.get('needs_redo')!r}")


def _conf(v: Any) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def _crop_pairs(img_a, img_b, regions: list[dict[str, Any]],
                max_regions: int) -> list[dict[str, Any]]:
    """The same boxes cut from both images. Only meaningful at equal size."""
    w, h = img_a.size
    out = []
    for reg in regions[:max_regions]:
        box = to_box(reg.get("bbox"), w, h)
        if box is None:
            continue
        ca, scale = crop_box(img_a, box)
        cb, _ = crop_box(img_b, box)
        out.append({"a": ca, "b": cb, "box": list(box), "scale": round(scale, 3)})
    return out


def _interleave(a: list, b: list) -> list:
    out = []
    for i in range(max(len(a), len(b))):
        out += ([a[i]] if i < len(a) else []) + ([b[i]] if i < len(b) else [])
    return out


def _analysis_line(rec: dict[str, Any] | None) -> str:
    final = (rec or {}).get("final")
    if not final:
        return "unavailable"
    arts = ", ".join(f"{a['type']} (severity {a['severity']:g})"
                     if a.get("severity") is not None else a["type"]
                     for a in final["artifacts"]) or "no artifacts found"
    q = final.get("quality")
    return (f"{arts}; quality {q:g}/5. {final.get('summary', '')}" if q is not None
            else f"{arts}. {final.get('summary', '')}")


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _pictures(views, imgs, pairs, names=("first", "second")) -> str:
    """Picture list. ``names`` are "first"/"second" for the judge, "A"/"B"
    for the reviewer."""
    lines = []
    for i, (name, view, img) in enumerate(zip(names, views, imgs), 1):
        w, h = img.size
        vw, vh = view_size(view, img)
        shown = "" if (vw, vh) == (w, h) else f", shown at {vw}x{vh}"
        long = f"the {name} image" if name in ("first", "second") else f"image {name}"
        lines.append(f"Picture {i}: {long} ({w}x{h} px{shown}).")
    for k, p in enumerate(pairs):
        x0, y0, x1, y1 = p["box"]
        zoom = f", enlarged {p['scale']:g}x nearest-neighbour" if p["scale"] > 1 else ""
        lines.append(f"Pictures {3 + 2 * k} and {4 + 2 * k}: crop pair {k + 1}, "
                     f"x {x0}-{x1}, y {y0}-{y1} px{zoom} ({names[0]}, then "
                     f"{names[1]}).")
    return "\n".join(lines)


def judge_once(vlm, order: str, ctx: dict[str, Any], cfg: CompareConfig,
               notes: str | None = None) -> dict[str, Any]:
    """One decision with A and B shown in ``order`` ("AB" or "BA")."""
    first, second = (("a", "b") if order == "AB" else ("b", "a"))
    views = [ctx[f"view_{first}"], ctx[f"view_{second}"]]
    imgs = [ctx[f"img_{first}"], ctx[f"img_{second}"]]
    pairs = [{"box": p["box"], "scale": p["scale"], "x": p[first], "y": p[second]}
             for p in ctx["pairs"]]
    parts = [_pictures(views, imgs, pairs)]
    if cfg.mode == "harness":
        parts.append("Measured statistics, first / second:\n" +
                     render_measurement_pair(ctx[f"stats_{first}"],
                                             ctx[f"stats_{second}"]))
        parts.append("Earlier analysis of each image on its own (may be wrong):\n"
                     f"first: {_analysis_line(ctx[f'analysis_{first}'])}\n"
                     f"second: {_analysis_line(ctx[f'analysis_{second}'])}")
    else:
        parts.append("No crops, statistics or earlier analyses are given.")
    if notes:
        parts.append(
            "A reviewer checked an earlier comparison of this pair and raised "
            f"the points below. In them, image A is the {'first' if first == 'a' else 'second'} "
            f"image here and image B the {'second' if first == 'a' else 'first'}. "
            "Look again and decide from scratch; agree with the reviewer only "
            "where the pictures support it.\n" + notes)
    parts.append("Output the JSON only.")

    system = fill(load_prompt("compare_judge.txt"),
                  GUIDELINES=cfg.guidelines.strip() or "(none)")
    images = views + [im for p in pairs for im in (p["x"], p["y"])]
    obj = vlm(system, "\n\n".join(parts), images, validate=_check_decision)
    said = str(obj["better"]).strip().lower()
    better = ("tie" if said == "tie" else
              ("A" if (said == "first") == (first == "a") else "B"))
    return {"order": order, "better": better, "said": said,
            "confidence": _conf(obj.get("confidence")),
            "differences": [str(d) for d in obj.get("differences", [])],
            "reason": str(obj.get("reason", ""))}


def combine(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the per-order decisions into one.

    If every order agrees, that is the answer. If they disagree, the answer
    follows ``p_a`` - each order's answer as a probability that A is better
    (0.5 +/- half its confidence), averaged - with a dead zone around 0.5
    that returns a tie: a preference that reverses with the presentation
    order is weak evidence either way.
    """
    ps = [0.5 + 0.5 * d["confidence"] if d["better"] == "A" else
          0.5 - 0.5 * d["confidence"] if d["better"] == "B" else 0.5
          for d in decisions]
    p_a = sum(ps) / len(ps)
    answers = {d["better"] for d in decisions}
    consistent = None if len(decisions) == 1 else len(answers) == 1
    if len(answers) == 1:
        better = decisions[0]["better"]
    else:
        better = "A" if p_a > 0.55 else "B" if p_a < 0.45 else "tie"
    return {"better": better, "p_a": round(p_a, 4), "consistent": consistent}


def review(vlm, ctx: dict[str, Any], decision: dict[str, Any],
           decisions: list[dict[str, Any]], cfg: CompareConfig) -> dict[str, Any]:
    ab = next(d for d in decisions if d["order"] == "AB")
    parts = [_pictures([ctx["view_a"], ctx["view_b"]], [ctx["img_a"], ctx["img_b"]],
                       ctx["pairs"], names=("A", "B"))]
    if cfg.mode == "harness":
        parts.append("Measured statistics, A / B:\n" +
                     render_measurement_pair(ctx["stats_a"], ctx["stats_b"]))
    verdict = {"A": "A is better", "B": "B is better", "tie": "a tie"}[decision["better"]]
    parts.append(f"The judge's decision: {verdict}.")
    parts.append("Differences the judge said decided it (its 'first' is A and "
                 "'second' is B):\n" +
                 ("\n".join(f"- {d}" for d in ab["differences"]) or "- none given"))
    if decision.get("consistent") is False:
        parts.append("Note: the judge's answer changed when the two images were "
                     "shown in the other order.")
    parts.append("Output the JSON only.")
    images = [ctx["view_a"], ctx["view_b"]] + [im for p in ctx["pairs"]
                                               for im in (p["a"], p["b"])]
    obj = vlm(load_prompt("compare_verify.txt"), "\n\n".join(parts), images,
              validate=_check_review)
    checked = [c for c in obj.get("differences_checked", []) if isinstance(c, dict)]
    bad = sum(c.get("verdict") == "not_visible" for c in checked)
    supported = obj.get("decision_supported") is not False
    return {"needs_redo": as_bool(obj["needs_redo"]), "checked": checked,
            "decision_supported": supported, "notes": str(obj.get("notes", "")),
            "objections": bad + (0 if supported else 1)}


# ---------------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------------

def compare_pair(vlm, path_a: str | Path, path_b: str | Path,
                 cfg: CompareConfig) -> dict[str, Any]:
    """All stages for one pair. Errors are recorded, not raised."""
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rec: dict[str, Any] = {"image_a": str(path_a), "image_b": str(path_b),
                           "mode": cfg.mode, "rounds": [], "final": None,
                           "error": None}
    try:
        ctx: dict[str, Any] = {"pairs": []}
        for key, path in (("a", path_a), ("b", path_b)):
            img = load_rgb(path)
            ctx[f"img_{key}"] = img
            ctx[f"view_{key}"] = global_view(path, img, cfg.max_side)
        if cfg.mode == "harness":
            single = replace(cfg.analyze, mode="harness", redo=False)
            for key, path in (("a", path_a), ("b", path_b)):
                ctx[f"stats_{key}"] = measure(ctx[f"img_{key}"])
                ana = analyze_image(vlm, path, single)
                ctx[f"analysis_{key}"] = ana
                rec[f"analysis_{key}"] = {"final": ana.get("final"),
                                          "error": ana.get("error")}
            if ctx["img_a"].size == ctx["img_b"].size:
                regions = _interleave(
                    (ctx["analysis_a"].get("diagnosis") or {}).get("zoom_regions") or [],
                    (ctx["analysis_b"].get("diagnosis") or {}).get("zoom_regions") or [])
                ctx["pairs"] = _crop_pairs(ctx["img_a"], ctx["img_b"],
                                           regions or [DEFAULT_REGION],
                                           cfg.max_regions)
            rec["crop_boxes"] = [p["box"] for p in ctx["pairs"]]

        rounds = cfg.max_rounds if cfg.redo else 1
        notes = None
        for rnd in range(rounds):
            orders = ["AB", "BA"] if cfg.swap else ["AB"]
            decisions = [judge_once(vlm, o, ctx, cfg, notes) for o in orders]
            entry: dict[str, Any] = {"round": rnd, "decisions": decisions,
                                     "decision": combine(decisions), "review": None}
            rec["rounds"].append(entry)
            if rounds == 1:
                break
            entry["review"] = review(vlm, ctx, entry["decision"], decisions, cfg)
            if not entry["review"]["needs_redo"]:
                break
            notes = entry["review"]["notes"] or "The reviewer found the deciding " \
                                                "differences not visible."

        rec["selected_round"] = select_round(rec["rounds"])
        rec["final"] = rec["rounds"][rec["selected_round"]]["decision"]
    except BudgetExceeded:
        raise
    except Exception as e:                    # noqa: BLE001 - record, do not raise
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec
