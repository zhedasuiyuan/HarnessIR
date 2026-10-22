#!/usr/bin/env python3
"""Single-image analysis: which artifacts are present, and how bad.

HarnessIR's five stages, mapped onto quality assessment:

    HarnessIR                          here
    1 perception and diagnosis    ->   diagnose: describe, separate intended
                                       effects from defects, pick regions
    2 on-demand tools             ->   pixel statistics + native-resolution crops
    3 prompt composition          ->   evidence rendered as text (no model call)
    4 execution (the MFM)         ->   judge: the VLM names artifacts from the
                                       taxonomy, under the tunable guidelines
    5 verification + redo         ->   review against the pictures; a redo
                                       judges again from scratch, with the notes

Two HarnessIR rules carry over. The reviewer never sees the guidelines that
steered the judge, so it checks the answer against the image and the taxonomy
rather than against the instructions. And a redo starts again from the image
with the reviewer's notes, never by editing the previous answer.

Plain mode is the baseline (HarnessIR's run_baseline.py): one judge call with
the whole image only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import BudgetExceeded, as_bool, fill, load_prompt
from .data import match_type, taxonomy_text
from .evidence import (DEFAULT_REGION, crop_regions, describe_crop, global_view,
                       load_rgb, measure, render_measurements, view_size)

MODES = ("harness", "plain")


@dataclass
class AnalyzeConfig:
    taxonomy: list[dict[str, str]]
    guidelines: str
    mode: str = "harness"
    redo: bool = False
    max_rounds: int = 3            # total judge attempts when redo is on
    max_side: int = 1024           # long edge of the whole-image view
    max_regions: int = 3


# ---------------------------------------------------------------------------
# Reply checks. A ValueError makes CachedVLM ask again.
# ---------------------------------------------------------------------------

def _check_diagnosis(obj: dict[str, Any]) -> None:
    if not isinstance(obj.get("zoom_regions", []), list):
        raise ValueError("zoom_regions is not a list")


def _check_analysis(obj: dict[str, Any]) -> None:
    arts = obj.get("artifacts")
    if not isinstance(arts, list) or not all(isinstance(a, dict) for a in arts):
        raise ValueError("artifacts is not a list of objects")


def _check_review(obj: dict[str, Any]) -> None:
    # The accept/redo decision rests on this field, so a reply without it is
    # malformed rather than an implicit "no".
    if as_bool(obj.get("needs_redo")) is None:
        raise ValueError(f"needs_redo is not a boolean: {obj.get('needs_redo')!r}")


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def clean_analysis(obj: dict[str, Any], taxonomy: list[dict[str, str]]) -> dict[str, Any]:
    """Map the reply onto the taxonomy.

    Names outside it are kept under ``unknown_types`` instead of being
    dropped silently: the critic in tune.py reads them, and a model that keeps
    inventing a name is telling the taxonomy something.
    """
    best: dict[str, dict[str, Any]] = {}
    unknown = []
    for a in obj.get("artifacts", []):
        name = match_type(a.get("type", ""), taxonomy)
        if name is None:
            unknown.append(str(a.get("type", "")))
            continue
        entry = {"type": name, "severity": _float(a.get("severity")),
                 "confidence": _float(a.get("confidence")),
                 "where": str(a.get("where", "")),
                 "evidence": str(a.get("evidence", ""))}
        old = best.get(name)
        if old is None or (entry["severity"] or 0) > (old["severity"] or 0):
            best[name] = entry
    primary = match_type(obj.get("primary", ""), taxonomy)
    if primary is None and best:
        primary = max(best.values(), key=lambda e: e["severity"] or 0)["type"]
    return {"artifacts": list(best.values()), "types": sorted(best),
            "primary": primary or "none", "quality": _float(obj.get("quality")),
            "summary": str(obj.get("summary", "")), "unknown_types": unknown}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _pictures(view: Any, img, crops: list[dict[str, Any]]) -> str:
    w, h = img.size
    vw, vh = view_size(view, img)
    shown = "" if (vw, vh) == (w, h) else f", shown at {vw}x{vh}"
    lines = [f"Picture 1: the whole image ({w}x{h} px{shown})."]
    lines += [describe_crop(i, c) for i, c in enumerate(crops, start=2)]
    return "\n".join(lines)


def diagnose(vlm, view, img, stats_text: str, cfg: AnalyzeConfig) -> dict[str, Any]:
    system = fill(load_prompt("analyze_diagnose.txt"),
                  MAX_REGIONS=str(cfg.max_regions),
                  TAXONOMY_NAMES=", ".join(t["name"] for t in cfg.taxonomy))
    user = "\n\n".join([_pictures(view, img, []),
                        "Measured statistics:\n" + stats_text,
                        "Output the JSON only."])
    return vlm(system, user, [view], validate=_check_diagnosis)


def evidence_text(stats_text: str, diag: dict[str, Any]) -> str:
    """Stage 3: the statistics and the triage report, as text for the judge."""
    obs = "\n".join(f"- {o.get('issue', '')} ({o.get('severity', '?')}, "
                    f"{o.get('where', '?')})" for o in diag.get("observations", []))
    intent = "; ".join(str(x) for x in diag.get("intentional", [])) or "none noted"
    return "\n".join([
        "Measured statistics:", stats_text, "",
        "Triage report (an earlier look; may be wrong):",
        f"content: {diag.get('content', '')}",
        f"intentional effects: {intent}",
        "observations:", obs or "- none"])


def judge(vlm, view, img, crops, evidence: str, cfg: AnalyzeConfig,
          notes: str | None = None) -> dict[str, Any]:
    system = fill(load_prompt("analyze_judge.txt"),
                  TAXONOMY=taxonomy_text(cfg.taxonomy),
                  GUIDELINES=cfg.guidelines.strip() or "(none)")
    parts = [_pictures(view, img, crops)]
    parts.append(evidence if evidence else
                 "No crops, statistics or triage report are given for this image.")
    if notes:
        parts.append("A reviewer checked an earlier analysis of this image and "
                     "raised the points below. Look at the pictures again and "
                     "answer from scratch; agree with the reviewer only where "
                     "the pictures support it.\n" + notes)
    parts.append("Output the JSON only.")
    obj = vlm(system, "\n\n".join(parts), [view] + [c["image"] for c in crops],
              validate=_check_analysis)
    return clean_analysis(obj, cfg.taxonomy)


def review(vlm, view, img, crops, evidence: str, analysis: dict[str, Any],
           cfg: AnalyzeConfig) -> dict[str, Any]:
    system = fill(load_prompt("analyze_verify.txt"),
                  TAXONOMY=taxonomy_text(cfg.taxonomy))
    shown = {k: analysis[k] for k in ("artifacts", "primary", "quality", "summary")}
    user = "\n\n".join([_pictures(view, img, crops), evidence,
                        "The analysis to review:\n" + json.dumps(shown, indent=1),
                        "Output the JSON only."])
    obj = vlm(system, user, [view] + [c["image"] for c in crops],
              validate=_check_review)
    claims = [c for c in obj.get("claims", []) if isinstance(c, dict)]
    missed = [m for m in obj.get("missed", []) if isinstance(m, dict)]
    unsupported = [c for c in claims if c.get("verdict") == "unsupported"]
    return {"needs_redo": as_bool(obj["needs_redo"]), "claims": claims,
            "missed": missed, "notes": str(obj.get("notes", "")),
            "objections": len(unsupported) + len(missed)}


def redo_notes(rev: dict[str, Any]) -> str:
    """What the next judge is told: the reviewer's notes plus its findings."""
    lines = [rev["notes"]] if rev["notes"] else []
    lines += [f"- claimed {c.get('type')}: not supported ({c.get('why', '')})"
              for c in rev["claims"] if c.get("verdict") == "unsupported"]
    lines += [f"- possibly missed {m.get('type')}: {m.get('why', '')}"
              for m in rev["missed"]]
    return "\n".join(lines)


def select_round(rounds: list[dict[str, Any]]) -> int:
    """Which attempt to deliver.

    The earliest accepted one - a later round is more work on an answer
    already judged good. If none was accepted, the one with the fewest
    objections, earliest first on ties.
    """
    for i, r in enumerate(rounds):
        if r.get("review") is None or not r["review"]["needs_redo"]:
            return i
    return min(range(len(rounds)),
               key=lambda i: (rounds[i]["review"]["objections"], i))


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------

def analyze_image(vlm, path: str | Path, cfg: AnalyzeConfig) -> dict[str, Any]:
    """All stages for one image. Errors are recorded, not raised - one bad
    file should not end a batch - except a spent budget, which ends the run."""
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rec: dict[str, Any] = {"image": str(path), "mode": cfg.mode, "rounds": [],
                           "final": None, "error": None}
    try:
        img = load_rgb(path)
        view = global_view(path, img, cfg.max_side)
        crops: list[dict[str, Any]] = []
        evidence = ""
        if cfg.mode == "harness":
            stats = measure(img)
            stats_text = render_measurements(stats)
            diag = diagnose(vlm, view, img, stats_text, cfg)
            crops = crop_regions(img, diag.get("zoom_regions") or [DEFAULT_REGION],
                                 max_regions=cfg.max_regions)
            evidence = evidence_text(stats_text, diag)
            rec.update(measurements=stats, diagnosis=diag,
                       crops=[{k: c[k] for k in ("box", "scale", "why")}
                              for c in crops])

        rounds = cfg.max_rounds if (cfg.redo and cfg.mode == "harness") else 1
        notes = None
        for rnd in range(rounds):
            analysis = judge(vlm, view, img, crops, evidence, cfg, notes)
            entry: dict[str, Any] = {"round": rnd, "analysis": analysis,
                                     "review": None}
            rec["rounds"].append(entry)
            if rounds == 1:
                break
            entry["review"] = review(vlm, view, img, crops, evidence, analysis, cfg)
            if not entry["review"]["needs_redo"]:
                break
            notes = redo_notes(entry["review"])

        rec["selected_round"] = select_round(rec["rounds"])
        rec["final"] = rec["rounds"][rec["selected_round"]]["analysis"]
    except BudgetExceeded:
        raise
    except Exception as e:                    # noqa: BLE001 - record, do not raise
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec
