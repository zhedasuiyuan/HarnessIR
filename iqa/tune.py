#!/usr/bin/env python3
"""Prompt tuning from labelled data: rewrite the guidelines block.

HarnessIR is inference-only and label-free. Here labels exist (artifact types,
human preferences), so the one free-text block that steers the judge - the
guidelines - is searched for, with the rest of the harness held fixed.

The search (after ProTeGi, Pryzant et al., EMNLP 2023) is a small beam:

    for each step:
      for each guidelines version in the beam:
        run it on a fresh training batch and collect its mistakes
        critic: name the recurring error patterns, with a general rule each
        editor: write N revised versions that apply those rules
      score every new version on one fixed validation subset
      keep the best `beam` versions seen so far (ties go to the shorter text)

The two signals do different jobs. The validation score, a float, only ranks
versions; the critic's written patterns are what say how to change them.

Only the judge reads the guidelines, so diagnoses (and, for compare, the
per-image analyses) come from the cache after the first pass. The test split
is never used here; the CLI scores the initial and the best version on it
afterwards.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from harness.utils import write_json

from .analyze import analyze_image
from .compare import compare_pair
from .core import BudgetExceeded, fill, load_prompt, pmap
from .data import taxonomy_text
from .metrics import analyze_metrics, compare_metrics, main_metric


@dataclass
class TuneConfig:
    steps: int = 4
    beam: int = 2
    edits: int = 3            # revisions the editor writes per parent
    batch: int = 24           # training items per parent per step
    val_size: int = 48        # validation items every candidate is scored on
    max_examples: int = 12    # mistakes shown to the critic
    max_words: int = 350
    seed: int = 0
    workers: int = 4


# ---------------------------------------------------------------------------
# Running against labels (also used by `python -m iqa eval`)
# ---------------------------------------------------------------------------

def run_items(task: str, vlm, items: list[dict[str, Any]], cfg, workers: int,
              on_done: Callable[[int, int, Any], None] | None = None) -> list[dict]:
    """Run every item; each record gets its ``id`` and ``gold`` label."""
    def one(it: dict[str, Any]) -> dict[str, Any]:
        if task == "analyze":
            rec = analyze_image(vlm, it["image"], cfg)
            rec["gold"] = list(it["labels"])
        else:
            rec = compare_pair(vlm, it["image_a"], it["image_b"], cfg)
            rec["gold"] = it["label"]
        rec["id"] = it["id"]
        return rec
    return pmap(one, items, workers, on_done)


def score(task: str, recs: list[dict], taxonomy: list[dict] | None = None,
          metric: str | None = None) -> tuple[float, dict[str, Any]]:
    if task == "analyze":
        m = analyze_metrics(recs, [t["name"] for t in taxonomy or []])
    else:
        m = compare_metrics(recs)
    return main_metric(task, m, metric), m


def is_wrong(task: str, rec: dict[str, Any]) -> bool:
    final = rec.get("final") if not rec.get("error") else None
    if final is None:
        return True
    if task == "analyze":
        return set(final["types"]) != set(rec["gold"])
    return final["better"] != rec["gold"]


# ---------------------------------------------------------------------------
# What the critic reads
# ---------------------------------------------------------------------------

STAT_KEYS = ("noise_sigma", "laplacian_var", "blockiness", "mean_luma",
             "rms_contrast", "colorfulness")


def _stats_line(m: dict[str, Any] | None) -> str:
    if not m:
        return "n/a"
    return ", ".join(f"{k} {m[k]:g}" for k in STAT_KEYS if m.get(k) is not None)


def _analyze_example(i: int, r: dict[str, Any]) -> str:
    lines = [f"[mistake {i}]", f"label: {', '.join(r['gold']) or 'none'}"]
    final = r.get("final") if not r.get("error") else None
    if final is None:
        lines.append(f"answer: FAILED ({r.get('error')})")
        return "\n".join(lines)
    q = final.get("quality")
    lines.append(f"answer: {', '.join(final['types']) or 'none'}; primary "
                 f"{final['primary']}" + (f"; quality {q:g}" if q is not None else ""))
    for a in final["artifacts"]:
        lines.append(f"  - {a['type']} (severity {a.get('severity')}, "
                     f"confidence {a.get('confidence')}): {a.get('evidence', '')} "
                     f"[{a.get('where', '')}]")
    if final.get("unknown_types"):
        lines.append(f"names outside the list: {', '.join(final['unknown_types'])}")
    lines.append(f"summary: {final.get('summary', '')}")
    if r.get("measurements"):
        lines.append(f"statistics: {_stats_line(r['measurements'])}")
    obs = (r.get("diagnosis") or {}).get("observations") or []
    if obs:
        lines.append("triage saw: " + "; ".join(str(o.get("issue", "")) for o in obs))
    return "\n".join(lines)


def _compare_example(i: int, r: dict[str, Any]) -> str:
    gold = {"A": "A is better", "B": "B is better", "tie": "a tie"}[r["gold"]]
    lines = [f"[mistake {i}]", f"human judgement: {gold}"]
    final = r.get("final") if not r.get("error") else None
    if final is None:
        lines.append(f"answer: FAILED ({r.get('error')})")
        return "\n".join(lines)
    agree = {True: "orders agreed", False: "answer flipped with the order",
             None: "one order only"}[final.get("consistent")]
    lines.append(f"answer: {final['better']} (p_A {final['p_a']:g}; {agree})")
    rnd = r["rounds"][r.get("selected_round", 0)]
    for d in rnd["decisions"]:
        shown = "A first" if d["order"] == "AB" else "B first"
        lines.append(f"  [{shown}] said {d['said']} -> {d['better']}, "
                     f"confidence {d['confidence']:g}: "
                     + "; ".join(d["differences"]) + f" | {d['reason']}")
    for key in ("a", "b"):
        ana = (r.get(f"analysis_{key}") or {}).get("final")
        if ana:
            lines.append(f"analysis {key.upper()}: {', '.join(ana['types']) or 'none'}"
                         f"; quality {ana.get('quality')}")
    return "\n".join(lines)


def error_stats(task: str, recs: list[dict[str, Any]]) -> str:
    wrong = [r for r in recs if is_wrong(task, r)]
    failed = sum(1 for r in recs if r.get("error") or not r.get("final"))
    lines = [f"{len(wrong)} of {len(recs)} items wrong ({failed} failed outright)."]
    if task == "analyze":
        tally: dict[str, Counter] = {}
        confusions: Counter = Counter()
        for r in recs:
            final = r.get("final") if not r.get("error") else None
            pred = set(final["types"]) if final else set()
            gold = set(r["gold"])
            for n in pred | gold:
                c = tally.setdefault(n, Counter())
                c["gold"] += n in gold
                c["missed"] += n in gold and n not in pred
                c["false"] += n in pred and n not in gold
            if len(gold) == 1 and final and final["primary"] not in gold:
                confusions[(next(iter(gold)), final["primary"])] += 1
        lines.append("per type (in labels / missed / claimed wrongly):")
        lines += [f"  {n}: {c['gold']} / {c['missed']} / {c['false']}"
                  for n, c in sorted(tally.items()) if c["missed"] or c["false"]]
        if confusions:
            lines.append("most frequent label -> primary answer confusions:")
            lines += [f"  {g} -> {p}: {k}" for (g, p), k in confusions.most_common(8)]
    else:
        finals = [r.get("final") for r in recs if r.get("final") and not r.get("error")]
        ties = sum(f["better"] == "tie" for f in finals)
        flips = sum(f.get("consistent") is False for f in finals)
        lines.append(f"predicted ties: {ties}; answers that flipped with the "
                     f"order: {flips}")
    return "\n".join(lines)


def pick_mistakes(task: str, recs: list[dict[str, Any]], k: int,
                  rng: random.Random) -> list[dict[str, Any]]:
    """Up to ``k`` mistakes, spread across labels so one frequent confusion
    does not fill the whole sample."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        if is_wrong(task, r):
            key = ",".join(r["gold"]) if task == "analyze" else r["gold"]
            by_label.setdefault(key, []).append(r)
    for group in by_label.values():
        rng.shuffle(group)
    out: list[dict[str, Any]] = []
    while len(out) < k and any(by_label.values()):
        for key in sorted(by_label):
            if by_label[key] and len(out) < k:
                out.append(by_label[key].pop())
    return out


# ---------------------------------------------------------------------------
# Critic and editor
# ---------------------------------------------------------------------------

def _check_patterns(obj: dict[str, Any]) -> None:
    if not isinstance(obj.get("patterns"), list):
        raise ValueError("patterns is not a list")


def _check_candidates(obj: dict[str, Any]) -> None:
    c = obj.get("candidates")
    if not isinstance(c, list) or not all(isinstance(x, str) for x in c):
        raise ValueError("candidates is not a list of strings")


def critic(vlm, task: str, guidelines: str, stats: str, examples: list[str],
           taxonomy: list[dict] | None) -> list[dict[str, Any]]:
    if task == "analyze":
        what = ("name which artifact types from a fixed list are present in an "
                "image (labels give the true types)")
        ref = "The artifact types and their descriptions:\n" + taxonomy_text(taxonomy or [])
    else:
        what = ("decide which of two images has better technical quality "
                "(labels are human preferences)")
        ref = ("The model sees both images, crops of the same regions from both, "
               "pixel statistics and an earlier analysis of each image.")
    system = fill(load_prompt("tune_critic.txt"), TASK=what, REFERENCE=ref)
    user = "\n\n".join(["CURRENT GUIDELINES:\n" + (guidelines.strip() or "(none)"),
                        "ERROR STATISTICS:\n" + stats,
                        "MISTAKES:\n" + "\n\n".join(examples),
                        "Output the JSON only."])
    return [p for p in vlm(system, user, [], validate=_check_patterns)["patterns"]
            if isinstance(p, dict)]


def editor(vlm, guidelines: str, patterns: list[dict[str, Any]], n: int,
           max_words: int) -> list[str]:
    system = fill(load_prompt("tune_editor.txt"), N=str(n), MAX_WORDS=str(max_words))
    user = "\n\n".join(["CURRENT GUIDELINES:\n" + (guidelines.strip() or "(none)"),
                        "ERROR PATTERNS AND PROPOSED RULES:\n"
                        + json.dumps(patterns, indent=1, ensure_ascii=False),
                        "Output the JSON only."])
    out = vlm(system, user, [], validate=_check_candidates)["candidates"]
    return [c.strip() for c in out if c.strip()][:n]


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def tune(task: str, vlm, splits: dict[str, list[dict[str, Any]]], cfg, initial: str,
         out_dir: str | Path, tc: TuneConfig, *, taxonomy: list[dict] | None = None,
         metric: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Search for better guidelines; write every candidate and the best one."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val = splits["train"], splits["val"]
    if not train or not val:
        raise ValueError("tuning needs non-empty train and val splits")
    rng = random.Random(tc.seed)
    val_sub = rng.sample(val, min(tc.val_size, len(val)))

    def evaluate(text: str) -> tuple[float, dict[str, Any]]:
        recs = run_items(task, vlm, val_sub, replace(cfg, guidelines=text), tc.workers)
        return score(task, recs, taxonomy, metric)

    (out / "initial_guidelines.txt").write_text(initial.strip() + "\n", encoding="utf-8")
    cand_log = (out / "candidates.jsonl").open("a", encoding="utf-8")
    step_log = (out / "steps.jsonl").open("a", encoding="utf-8")
    pool: dict[str, float] = {}
    stop = "all steps run"
    try:
        s0, m0 = evaluate(initial)
        pool[initial] = s0
        cand_log.write(json.dumps({"step": 0, "val_score": s0, "val_metrics": m0,
                                   "text": initial}) + "\n")
        log(f"[step 0] initial guidelines: val {s0:.4f} on {len(val_sub)} items")
        beam = [initial]
        for step in range(1, tc.steps + 1):
            new: list[str] = []
            for pi, parent in enumerate(beam):
                batch = rng.sample(train, min(tc.batch, len(train)))
                recs = run_items(task, vlm, batch, replace(cfg, guidelines=parent),
                                 tc.workers)
                bs, _ = score(task, recs, taxonomy, metric)
                wrong = pick_mistakes(task, recs, tc.max_examples, rng)
                entry: dict[str, Any] = {"step": step, "parent": pi, "batch_score": bs,
                                         "mistakes": len(wrong)}
                if wrong:
                    examples = [(_analyze_example if task == "analyze"
                                 else _compare_example)(i, r)
                                for i, r in enumerate(wrong, 1)]
                    patterns = critic(vlm, task, parent, error_stats(task, recs),
                                      examples, taxonomy)
                    entry["patterns"] = patterns
                    if patterns:
                        cands = editor(vlm, parent, patterns, tc.edits, tc.max_words)
                        cands = [c for c in cands if c not in pool and c not in new]
                        new += cands
                        entry["candidates"] = len(cands)
                step_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log(f"[step {step}] parent {pi}: batch {bs:.4f}, {len(wrong)} "
                    f"mistakes, {len(entry.get('patterns', []))} patterns, "
                    f"{entry.get('candidates', 0)} new versions")
            for ci, text in enumerate(new):
                sc, mc = evaluate(text)
                pool[text] = sc
                cand_log.write(json.dumps({"step": step, "val_score": sc,
                                           "val_metrics": mc, "text": text},
                                          ensure_ascii=False) + "\n")
                log(f"[step {step}] version {ci}: val {sc:.4f} "
                    f"({len(text.split())} words)")
            beam = sorted(pool, key=lambda t: (-pool[t], len(t)))[:tc.beam]
            log(f"[step {step}] best so far: val {pool[beam[0]]:.4f}")
            if not new:
                stop = "no new versions (no mistakes, patterns or edits)"
                break
    except BudgetExceeded as e:
        stop = f"stopped early: {e}"
        log(f"[tune] {stop}")
    finally:
        cand_log.close()
        step_log.close()

    if not pool:
        raise RuntimeError(f"tuning produced no scored version ({stop})")
    best = sorted(pool, key=lambda t: (-pool[t], len(t)))[0]
    (out / "best_guidelines.txt").write_text(best.strip() + "\n", encoding="utf-8")
    summary = {"initial_val": pool.get(initial), "best_val": pool[best],
               "best_is_initial": best == initial, "versions_scored": len(pool),
               "val_items": len(val_sub), "stop": stop}
    write_json(out / "tune_summary.json", summary)
    return {**summary, "best": best}
