#!/usr/bin/env python3
"""Scores against labels.

analyze   multi-label artifact typing: micro and macro precision, recall, F1;
          exact-set match; and, over items with exactly one label (KADID-10k,
          the smoke set), whether the primary type is that label.
compare   agreement with human preferences: accuracy over the pairs humans did
          not call a tie, a predicted tie counting as half right; strict
          accuracy; how often the model commits; order consistency.

A failed item (an API error, an unparseable reply) counts as wrong, never as
skipped: a guidelines version that makes the model fail must not score as
if those items did not exist.
"""
from __future__ import annotations

from typing import Any


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def analyze_metrics(recs: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    per = {n: {"tp": 0, "fp": 0, "fn": 0} for n in names}
    exact = single = primary_hits = errors = 0
    for r in recs:
        final = r.get("final") if not r.get("error") else None
        errors += final is None
        gold = set(r["gold"])
        pred = set(final["types"]) if final else set()
        for n in pred | gold:
            slot = per.setdefault(n, {"tp": 0, "fp": 0, "fn": 0})
            slot["tp" if (n in pred and n in gold) else
                 "fp" if n in pred else "fn"] += 1
        exact += pred == gold
        if len(gold) == 1:
            single += 1
            primary_hits += bool(final) and final["primary"] in gold

    for slot in per.values():
        tp, fp, fn = slot["tp"], slot["fp"], slot["fn"]
        slot["precision"] = tp / (tp + fp) if tp + fp else 0.0
        slot["recall"] = tp / (tp + fn) if tp + fn else 0.0
        slot["f1"] = _f1(slot["precision"], slot["recall"])
    tp = sum(s["tp"] for s in per.values())
    fp = sum(s["fp"] for s in per.values())
    fn = sum(s["fn"] for s in per.values())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    present = [s["f1"] for s in per.values() if s["tp"] + s["fn"] > 0]
    n = len(recs)
    return {
        "n": n, "n_errors": errors,
        "micro_precision": micro_p, "micro_recall": micro_r,
        "micro_f1": _f1(micro_p, micro_r),
        "macro_f1": sum(present) / len(present) if present else 0.0,
        "exact_match": exact / n if n else 0.0,
        "single_label_share": single / n if n else 0.0,
        "primary_acc": primary_hits / single if single else None,
        "per_type": per,
    }


def compare_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    credit, strict, decided, n_pairs, errors = 0.0, 0, 0, 0, 0
    three_way, consistent, swapped = 0, 0, 0
    gold_ties = 0
    for r in recs:
        final = r.get("final") if not r.get("error") else None
        errors += final is None
        pred = final["better"] if final else None
        gold = r["gold"]
        three_way += pred == gold
        if final and final.get("consistent") is not None:
            swapped += 1
            consistent += bool(final["consistent"])
        if gold == "tie":
            gold_ties += 1
            continue
        n_pairs += 1
        decided += pred in ("A", "B")
        strict += pred == gold
        credit += 1.0 if pred == gold else 0.5 if pred == "tie" else 0.0
    n = len(recs)
    return {
        "n": n, "n_errors": errors, "n_gold_ties": gold_ties,
        "acc": credit / n_pairs if n_pairs else None,
        "strict_acc": strict / n_pairs if n_pairs else None,
        "decided_rate": decided / n_pairs if n_pairs else None,
        "three_way_acc": three_way / n if (n and gold_ties) else None,
        "order_consistency": consistent / swapped if swapped else None,
    }


def main_metric(task: str, metrics: dict[str, Any], name: str | None = None) -> float:
    """The number tuning maximises.

    By default: for analyze, primary-type accuracy when every item has one
    label, otherwise macro F1 (so rare types count as much as common ones);
    for compare, accuracy with ties as half.
    """
    if name is None:
        if task == "analyze":
            name = ("primary_acc" if metrics["single_label_share"] == 1.0
                    else "macro_f1")
        else:
            name = "acc"
    value = metrics.get(name)
    return float(value) if value is not None else 0.0
