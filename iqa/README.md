# HarnessIQA: the HarnessIR harness, repurposed for image quality assessment

Two tasks, one harness, plus a tuner that HarnessIR does not have:

| Task | Input | Output | Labels used for tuning |
|---|---|---|---|
| `analyze` | one image | artifact types from a taxonomy (with severity, location, evidence), a primary type, a 1–5 quality score | artifact labels per image |
| `compare` | two images | which has better technical quality (`A` / `B` / `tie`) and `p_A` | human preferences per pair |
| `tune` | a labelled manifest | rewritten **guidelines** for the judge, scored on a held-out test split | either of the above |

Everything lives in `iqa/`. It reuses HarnessIR's model client (`harness/clients.py`, `harness/settings.py`) and helpers, and none of its restoration code.

## How HarnessIR maps onto IQA

| HarnessIR stage | HarnessIQA |
|---|---|
| 1 perception and diagnosis | **diagnose** (VLM): describe the content, separate intended effects (bokeh, grain, night) from defects, pick up to 3 regions to inspect |
| 2 on-demand tools (OCR, faces, depth, segmentation) | **evidence**: pixel statistics (noise, edge energy, 8×8 blockiness, clipping, contrast, colour) and the chosen regions cropped at native resolution |
| 3 prompt composition | evidence rendered as text, each number with the way it misleads (no model call) |
| 4 execution by the image model | **judge** (VLM): names artifacts from the taxonomy, under the tunable guidelines; for `compare`, decides between the two images |
| 5 verification and redo from the original input | **review** (VLM, with `--redo`): checks the claims against the pictures; a redo judges again from scratch with the reviewer's notes |

Rules that carry over unchanged: statistics are evidence, never thresholds; the reviewer never sees the guidelines (as HarnessIR's verifier never sees the prompt), so it checks the answer against the image and the taxonomy instead of the instructions; a redo restarts from the image instead of editing the previous answer; selection takes the earliest accepted round.

Added for IQA:
- **Crops at native resolution.** The model sees a downscaled image, which hides noise, blocking and fine blur. Small crops are enlarged with nearest-neighbour scaling so pixel structure stays visible. Files are never re-encoded as JPEG.
- **Order swap for `compare`.** Each pair is judged in both presentation orders. If the answers agree, that is the decision; if they flip, the per-order confidences are averaged and a near-even result becomes a tie.
- **Prompt tuning** (`iqa/tune.py`), below.

`--mode plain` is the baseline (HarnessIR's `run_baseline.py`): one call with the whole image (two images for `compare`), the same judge prompt and guidelines, nothing else.

## Setup

Only `pillow`, `numpy` and `requests` are needed (scikit-image is optional, for the default smoke images). The restoration tools in `requirements.txt` (torch, PaddleOCR, SAM 3...) are not.

The model is reached through HarnessIR's settings:

```bash
export HARNESS_API_BASE="https://generativelanguage.googleapis.com"   # or your gateway
export HARNESS_API_KEY="<your key>"
```

Google's own endpoint takes the key in `x-goog-api-key`; any other host gets `Authorization: Bearer`, as before (`harness/settings.py: auth_headers`). The default model is `settings.VLM_MODEL_NAME` (`gemini-3.7-flash`). For Gemini Flash 3.8 pass `--model <the id your endpoint serves>`.

Run everything from the repository root. Outputs go to `runs/` (git-ignored); model replies are cached in `runs/iqa_cache/`, keyed on the exact request, so a rerun costs nothing.

## Data

Manifests are JSONL (or a JSON array). Relative paths resolve against the manifest's folder, or `--root`.

```jsonc
// analyze
{"id": "I07_10_04", "image": "images/I07_10_04.png", "labels": ["jpeg"], "group": "I07"}
// compare
{"id": "x__vs__y", "image_a": "x.png", "image_b": "y.png", "label": "A", "group": "ref07"}
```

`labels` may hold several types (multi-label). `label` is `A`, `B` or `tie`. `group` keeps related items in one split (every distortion of one reference image, for example), so a rule tuned on one version of a picture is never tested on another version of it; `split` (`train`/`val`/`test`) overrides the hash split.

A taxonomy is a JSON list of `{"name", "description"}`. The descriptions go into the judge's prompt verbatim, so write what each artifact **looks like**. `iqa/taxonomy/generic.json` (14 real-world types) is the default.

Converters:

```bash
# KADID-10k: 25 synthetic distortion types, 5 levels; labels from the file names
python -m iqa kadid --root /data/kadid10k --out data/kadid --min-level 3

# Per-image scores (MOS, DMOS, Elo) -> preference pairs. State the direction:
# KADID-10k's DMOS is higher-is-better, many DMOS scales are the opposite.
python -m iqa mos-to-pairs --csv /data/kadid10k/dmos.csv --image-col dist_img \
    --score-col dmos --higher-is-better yes --group-col ref_img --margin 0.5 \
    --root /data/kadid10k/images --out data/kadid/pairs.jsonl
```

Check the CSV's column names first; the ones above are an example. Some KADID-10k types are nearly indistinguishable by eye (the two saturation types; white noise and white noise in a colour component), which caps typing accuracy; consider merging them in your own taxonomy.

## Commands

```bash
# 1. A labelled smoke set (6 known distortions x 3 levels of clean photos), then a dry run:
#    the stub model sends nothing and checks paths, prompts and outputs.
python -m iqa make-smoke                      # -> runs/iqa_smoke/
python -m iqa eval --task analyze --manifest runs/iqa_smoke/analyze.jsonl \
    --taxonomy runs/iqa_smoke/taxonomy.json --dry-run

# 2. One image, one pair
python -m iqa analyze photo.png --taxonomy data/kadid/taxonomy.json
python -m iqa compare a.png b.png

# 3. Score a manifest: baseline vs harness, single pass vs redo
python -m iqa eval --task analyze --manifest data/kadid/analyze.jsonl \
    --taxonomy data/kadid/taxonomy.json --split test --mode plain
python -m iqa eval --task analyze --manifest data/kadid/analyze.jsonl \
    --taxonomy data/kadid/taxonomy.json --split test --redo

# 4. Tune the guidelines, then use them
python -m iqa tune --task analyze --manifest data/kadid/analyze.jsonl \
    --taxonomy data/kadid/taxonomy.json --out runs/iqa/tune_kadid
python -m iqa eval --task analyze --manifest data/kadid/analyze.jsonl \
    --taxonomy data/kadid/taxonomy.json --split test --redo \
    --guidelines runs/iqa/tune_kadid/best_guidelines.txt
```

`eval` writes `config.json`, `predictions.jsonl` (every stage's output per item) and `metrics.json`. Metrics: for `analyze`, micro/macro precision, recall and F1, exact-set match, and primary-type accuracy over single-label items (the default main metric when every item has one label; macro F1 otherwise); for `compare`, accuracy over non-tie human pairs with a predicted tie counting as half (the main metric), strict accuracy, how often the model commits, and order consistency. A failed item counts as wrong.

## Prompt tuning

`tune` changes one block of text, the judge's guidelines (`iqa/prompts/guidelines_analyze.txt` or `guidelines_compare.txt`); the rest of the harness stays fixed. It is a small beam search after ProTeGi (Pryzant et al., EMNLP 2023):

1. Run each guidelines version in the beam on a fresh batch of training items and collect its mistakes.
2. **Critic** (text-only call): reads the mistakes - label, answer, the judge's own evidence, statistics - and names recurring error patterns, each with one general rule.
3. **Editor** (text-only call): writes `--edits` revised versions applying those rules.
4. Score every new version on one fixed validation subset; keep the best `--beam` versions seen so far.

The validation score (a float) only ranks versions; the critic's written patterns are what say how to change them. The test split is never used in the search; `report.json` scores the initial and the best version on it. The run directory also holds every version with its validation score (`candidates.jsonl`), each step's patterns (`steps.jsonl`) and `best_guidelines.txt`.

Tuning runs whatever pipeline the flags describe; leave `--redo` off while tuning (cheaper) and evaluate the tuned guidelines with `--redo` afterwards.

## Cost

Model calls per item (diagnoses are cached, so a repeated image costs only its judge calls):

| | plain | harness | harness `--redo`, worst case (3 rounds) |
|---|---|---|---|
| `analyze` | 1 | 2 | 7 |
| `compare` | 1 (2 with `--swap`) | 4 per new image + 2 | + 1 review per round, + 2 judges per extra round |

With the defaults (4 steps, beam 2, 3 edits, batch 24, 48 validation items), tuning `analyze` takes about 1,500–1,700 calls; `--budget` (default 3,000) stops the search early, and the test evaluation afterwards (about 2 × test size × calls per item) is not counted. `metrics.json` and `report.json` record calls, cache hits and token counts.

## Tests

```bash
python -m pytest iqa/tests -q        # offline, no API calls
```

## Files

```
iqa/
  __main__.py   command line
  analyze.py    single-image stages
  compare.py    pairwise stages and the order-swap merge
  evidence.py   statistics and crops
  tune.py       prompt tuning (also runs labelled items for eval)
  metrics.py    scores against labels
  data.py       manifests, splits, taxonomies, KADID-10k, MOS -> pairs, smoke set
  core.py       cached model calls, prompt files, thread pool, offline stub
  prompts/      stage prompts; guidelines_*.txt are the tunable blocks
  taxonomy/     generic.json
```
