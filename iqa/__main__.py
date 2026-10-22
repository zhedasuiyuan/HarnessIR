#!/usr/bin/env python3
"""Command line. Run from the repository root:  python -m iqa <command> ...

    analyze       images -> artifact report (JSON on stdout)
    compare       two images -> which has better technical quality
    eval          labelled manifest -> predictions.jsonl + metrics.json
    tune          labelled manifest -> best_guidelines.txt, scored on test
    make-smoke    a small labelled set from clean photos, to check the pipeline
    kadid         KADID-10k folder -> analyze manifest + taxonomy
    mos-to-pairs  per-image scores (CSV) -> compare manifest

The model is reached through HarnessIR's client (harness/settings.py):
set HARNESS_API_BASE and HARNESS_API_KEY. --dry-run swaps in a stub that
sends nothing, to check paths and outputs before spending calls.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from harness import settings
from harness.utils import write_json

from .analyze import MODES, AnalyzeConfig, analyze_image
from .compare import CompareConfig, compare_pair
from .core import load_prompt, make_vlm
from .data import (kadid_items, load_items, load_taxonomy, make_smoke,
                   mos_to_pairs, read_csv, split_items)
from .tune import TuneConfig, run_items, score, tune

DEFAULT_CACHE = settings.DEFAULT_OUT_ROOT / "iqa_cache"
DEFAULT_RUNS = settings.DEFAULT_OUT_ROOT / "iqa"


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def _guidelines(path: str | None, task: str) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return load_prompt(f"guidelines_{task}.txt")


def _configs(args, task: str, taxonomy: list[dict[str, str]]):
    acfg = AnalyzeConfig(
        taxonomy=taxonomy,
        guidelines=_guidelines(args.guidelines if task == "analyze"
                               else args.analyze_guidelines, "analyze"),
        mode=args.mode, redo=args.redo, max_rounds=args.max_rounds,
        max_side=args.max_side, max_regions=args.max_regions)
    if task == "analyze":
        return acfg
    swap = args.swap if args.swap is not None else args.mode == "harness"
    return CompareConfig(guidelines=_guidelines(args.guidelines, "compare"),
                         analyze=acfg, mode=args.mode, swap=swap, redo=args.redo,
                         max_rounds=args.max_rounds, max_side=args.max_side,
                         max_regions=args.max_regions)


def _vlm(args, max_calls: int | None = None):
    return make_vlm(model=args.model, temperature=args.temperature,
                    cache_dir=None if args.no_cache else args.cache_dir,
                    dry_run=args.dry_run, max_calls=max_calls)


def _usage(vlm) -> dict[str, Any]:
    from harness.clients import usage_report
    return {"model_calls": vlm.calls, "cache_hits": vlm.hits, "tokens": usage_report()}


def _items(args, taxonomy):
    items = load_items(args.manifest, args.task, root=args.root,
                       taxonomy=taxonomy if args.task == "analyze" else None)
    fractions = tuple(float(x) for x in args.fractions.split(","))
    return items, split_items(items, fractions, args.seed)


def _out_dir(args, kind: str) -> Path:
    if args.out:
        return Path(args.out)
    return DEFAULT_RUNS / f"{kind}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"


def _progress(task: str):
    def show(k: int, n: int, rec: dict[str, Any]) -> None:
        if rec.get("error"):
            tail = f"ERROR {rec['error'][:80]}"
        elif task == "analyze":
            tail = (f"label {','.join(rec['gold']) or 'none'} | "
                    f"answer {','.join(rec['final']['types']) or 'none'}")
        else:
            tail = f"human {rec['gold']} | answer {rec['final']['better']}"
        print(f"[{k}/{n}] {rec['id']}  {tail}", flush=True)
    return show


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def _headline(task: str, metrics: dict[str, Any]) -> str:
    keys = (("primary_acc", "macro_f1", "micro_f1", "exact_match")
            if task == "analyze" else
            ("acc", "strict_acc", "decided_rate", "order_consistency"))
    shown = [f"{k} {metrics[k]:.4f}" for k in keys if metrics.get(k) is not None]
    return ", ".join(shown) + f"  (n {metrics['n']}, errors {metrics['n_errors']})"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_analyze(args) -> int:
    cfg = _configs(args, "analyze", load_taxonomy(args.taxonomy))
    vlm = _vlm(args)
    for path in args.images:
        print(json.dumps(analyze_image(vlm, path, cfg), indent=1, ensure_ascii=False))
    return 0


def cmd_compare(args) -> int:
    cfg = _configs(args, "compare", load_taxonomy(args.taxonomy))
    rec = compare_pair(_vlm(args), args.image_a, args.image_b, cfg)
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    return 0


def cmd_eval(args) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    items, splits = _items(args, taxonomy)
    if args.split != "all":
        items = splits[args.split]
    if args.limit:
        items = items[:args.limit]
    cfg = _configs(args, args.task, taxonomy)
    out = _out_dir(args, "eval")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {"args": vars(args) | {"func": None},
                                     "pipeline": asdict(cfg)})
    vlm = _vlm(args)
    t0 = time.perf_counter()
    recs = run_items(args.task, vlm, items, cfg, args.workers, _progress(args.task))
    value, metrics = score(args.task, recs, taxonomy, args.metric)
    _write_jsonl(out / "predictions.jsonl", recs)
    write_json(out / "metrics.json", {"main": value, "metrics": metrics,
                                      "usage": _usage(vlm),
                                      "elapsed_sec": round(time.perf_counter() - t0, 1)})
    print("=" * 68)
    print(f"{args.task} on {len(items)} items: {_headline(args.task, metrics)}")
    print(f"model calls {vlm.calls}, cache hits {vlm.hits}")
    print(f"output: {out}")
    return 0


def cmd_tune(args) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    _, splits = _items(args, taxonomy)
    print("split sizes: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    cfg = _configs(args, args.task, taxonomy)
    out = _out_dir(args, "tune")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {"args": vars(args) | {"func": None},
                                     "pipeline": asdict(cfg)})
    vlm = _vlm(args)
    vlm.max_calls = args.budget or None
    tc = TuneConfig(steps=args.steps, beam=args.beam, edits=args.edits,
                    batch=args.batch, val_size=args.val_size,
                    max_examples=args.max_examples, max_words=args.max_words,
                    seed=args.seed, workers=args.workers)
    result = tune(args.task, vlm, splits, cfg, cfg.guidelines, out, tc,
                  taxonomy=taxonomy, metric=args.metric)
    search_calls = vlm.calls
    vlm.max_calls = None                     # the budget covers the search only

    report: dict[str, Any] = {"task": args.task, "search": result | {"best": None},
                              "search_model_calls": search_calls, "test": {}}
    if not args.no_test and splits["test"]:
        versions = [("initial", cfg.guidelines)]
        if not result["best_is_initial"]:
            versions.append(("best", result["best"]))
        for name, text in versions:
            recs = run_items(args.task, vlm, splits["test"],
                             replace(cfg, guidelines=text), args.workers)
            value, metrics = score(args.task, recs, taxonomy, args.metric)
            report["test"][name] = {"main": value, "metrics": metrics}
            _write_jsonl(out / f"test_predictions_{name}.jsonl", recs)
    report["usage"] = _usage(vlm)
    write_json(out / "report.json", report)

    print("=" * 68)
    print(f"validation: initial {result['initial_val']:.4f} -> best "
          f"{result['best_val']:.4f} ({result['versions_scored']} versions; "
          f"{result['stop']})")
    for name, r in report["test"].items():
        print(f"test, {name}: {_headline(args.task, r['metrics'])}")
    print(f"model calls {vlm.calls} ({search_calls} in the search), "
          f"cache hits {vlm.hits}")
    print(f"best guidelines: {out / 'best_guidelines.txt'}")
    return 0


def cmd_make_smoke(args) -> int:
    paths = make_smoke(args.out, src=args.src, seed=args.seed)
    for k, p in paths.items():
        print(f"{k}: {p}")
    print(f"\ntry:  python -m iqa eval --task analyze --manifest {paths['analyze']} "
          f"--taxonomy {paths['taxonomy']} --dry-run")
    return 0


def cmd_kadid(args) -> int:
    items, taxonomy = kadid_items(args.root, min_level=args.min_level)
    out = Path(args.out)
    _write_jsonl(out / "analyze.jsonl", items)
    write_json(out / "taxonomy.json", taxonomy)
    groups = len({it["group"] for it in items})
    print(f"{len(items)} images from {groups} references -> {out / 'analyze.jsonl'}")
    print(f"taxonomy ({len(taxonomy)} types) -> {out / 'taxonomy.json'}")
    return 0


def cmd_mos_to_pairs(args) -> int:
    base = Path(args.root) if args.root else Path(args.csv).resolve().parent
    rows = read_csv(args.csv)
    for r in rows:
        p = Path(r[args.image_col]).expanduser()
        r[args.image_col] = str(p if p.is_absolute() else (base / p).resolve())
    pairs = mos_to_pairs(rows, image_col=args.image_col, score_col=args.score_col,
                         higher_is_better=args.higher_is_better == "yes",
                         margin=args.margin, group_col=args.group_col,
                         max_pairs=args.max_pairs, seed=args.seed)
    _write_jsonl(Path(args.out), pairs)
    print(f"{len(pairs)} pairs -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", default=None,
                   help=f"VLM id the endpoint serves (default "
                        f"{settings.VLM_MODEL_NAME})")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--dry-run", action="store_true",
                   help="stub model: no API calls, uninformative answers")


def _pipeline_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("pipeline")
    g.add_argument("--mode", choices=MODES, default="harness",
                   help="harness: all stages; plain: one call, whole image only")
    g.add_argument("--redo", action="store_true",
                   help="stage 5, review and redo (off by default, as in HarnessIR)")
    g.add_argument("--max-rounds", type=int, default=3)
    g.add_argument("--max-side", type=int, default=1024,
                   help="long edge of the whole-image view")
    g.add_argument("--max-regions", type=int, default=3)
    g.add_argument("--taxonomy", default=None,
                   help="artifact taxonomy JSON (default iqa/taxonomy/generic.json)")
    g.add_argument("--guidelines", default=None,
                   help="guidelines for this task's judge (default "
                        "iqa/prompts/guidelines_<task>.txt)")
    g.add_argument("--analyze-guidelines", default=None,
                   help="compare: guidelines for the per-image analyses")
    swap = g.add_mutually_exclusive_group()
    swap.add_argument("--swap", dest="swap", action="store_true", default=None,
                      help="compare: judge both presentation orders "
                           "(default in harness mode)")
    swap.add_argument("--no-swap", dest="swap", action="store_false")


def _data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--task", choices=("analyze", "compare"), required=True)
    g.add_argument("--manifest", required=True)
    g.add_argument("--root", default=None,
                   help="base for relative paths (default: the manifest's folder)")
    g.add_argument("--fractions", default="0.6,0.2,0.2", help="train,val,test")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--metric", default=None,
                   help="metric to report as main / maximise (default: "
                        "primary_acc or macro_f1 for analyze, acc for compare)")
    g.add_argument("--out", default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m iqa", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="artifact report for images")
    p.add_argument("images", nargs="+")
    _model_args(p)
    _pipeline_args(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("compare", help="which of two images is better")
    p.add_argument("image_a")
    p.add_argument("image_b")
    _model_args(p)
    _pipeline_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("eval", help="score a labelled manifest")
    _data_args(p)
    _model_args(p)
    _pipeline_args(p)
    p.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("tune", help="tune the guidelines on a labelled manifest")
    _data_args(p)
    _model_args(p)
    _pipeline_args(p)
    g = p.add_argument_group("search")
    g.add_argument("--steps", type=int, default=4)
    g.add_argument("--beam", type=int, default=2)
    g.add_argument("--edits", type=int, default=3)
    g.add_argument("--batch", type=int, default=24)
    g.add_argument("--val-size", type=int, default=48)
    g.add_argument("--max-examples", type=int, default=12)
    g.add_argument("--max-words", type=int, default=350)
    g.add_argument("--budget", type=int, default=3000,
                   help="max API calls for the search (0 = no limit); the "
                        "final test evaluation is not counted")
    g.add_argument("--no-test", action="store_true")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("make-smoke", help="small labelled set from clean photos")
    p.add_argument("--out", default=str(settings.DEFAULT_OUT_ROOT / "iqa_smoke"))
    p.add_argument("--src", default=None,
                   help="folder of clean photos (default: scikit-image samples)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_make_smoke)

    p = sub.add_parser("kadid", help="KADID-10k -> manifest + taxonomy")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-level", type=int, default=1,
                   help="drop distortion levels below this (1-5)")
    p.set_defaults(func=cmd_kadid)

    p = sub.add_parser("mos-to-pairs", help="per-image scores -> preference pairs")
    p.add_argument("--csv", required=True)
    p.add_argument("--image-col", required=True)
    p.add_argument("--score-col", required=True)
    p.add_argument("--higher-is-better", choices=("yes", "no"), required=True)
    p.add_argument("--group-col", default=None,
                   help="pair only within groups (e.g. the reference image)")
    p.add_argument("--margin", type=float, default=0.5,
                   help="minimum score gap for a pair")
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--root", default=None,
                   help="base for relative image paths (default: the CSV's folder)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_mos_to_pairs)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
