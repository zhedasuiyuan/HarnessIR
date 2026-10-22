#!/usr/bin/env python3
"""Shared plumbing: prompt files, the cached VLM call, a thread pool.

Every stage calls the model through ``CachedVLM``, which wraps HarnessIR's
``VLMClient`` (harness/clients.py) with three things the IQA runs need:

* a disk cache keyed on the exact request. Prompt tuning re-runs the same
  calls many times - the diagnosis of an image does not depend on the
  guidelines being tuned, and a candidate that survives one step is evaluated
  again in the next - so repeats must cost nothing;
* reply validation. Only replies that pass are cached, so a malformed reply
  is retried rather than replayed forever;
* a call budget, so a tuning run cannot spend more than it was given.

``StubVLM`` stands in for the client offline (``--dry-run`` and the tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from harness.utils import strip_comments

IQA_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = IQA_DIR / "prompts"
TAXONOMY_DIR = IQA_DIR / "taxonomy"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    """Read ``iqa/prompts/<name>`` without its ``#`` authoring header."""
    return strip_comments((PROMPTS_DIR / name).read_text(encoding="utf-8"))


def fill(template: str, **slots: str) -> str:
    """Fill ``{{NAME}}`` slots in one pass.

    One pass, so text inserted into one slot (tuned guidelines, say) is never
    scanned for further slots. A slot left unfilled raises: a blank in the
    prompt would silently change what the model is asked.
    """
    missing: set[str] = set()

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key in slots:
            return slots[key]
        missing.add(key)
        return m.group(0)

    out = re.sub(r"\{\{([A-Z_]+)\}\}", sub, template)
    if missing:
        raise ValueError(f"unfilled prompt slots: {sorted(missing)}")
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    """The run asked for more API calls than it was allowed."""


def _fingerprint(im: Any) -> str:
    """Content hash of one picture, a path or a PIL image."""
    if isinstance(im, (str, Path)):
        return hashlib.sha256(Path(im).read_bytes()).hexdigest()
    head = f"{im.mode}{im.size}".encode("ascii")
    return hashlib.sha256(head + im.tobytes()).hexdigest()


class CachedVLM:
    """``VLMClient.chat_json`` behind a disk cache, a validator and a budget."""

    def __init__(self, client: Any, cache_dir: str | Path | None, *,
                 max_calls: int | None = None, attempts: int = 2) -> None:
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_calls = max_calls
        self.attempts = attempts
        self.calls = 0          # requests actually sent
        self.hits = 0           # requests answered from the cache
        self._lock = threading.Lock()

    def _key(self, system: str, user: str, images: list) -> str:
        ident = [getattr(self.client, "model", "?"),
                 getattr(self.client, "temperature", None),
                 system, user, [_fingerprint(im) for im in images]]
        blob = json.dumps(ident, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def __call__(self, system: str, user: str, images: list | None = None, *,
                 validate: Callable[[dict], None] | None = None) -> dict[str, Any]:
        """One request. ``validate`` raises ``ValueError`` on a bad reply."""
        images = list(images or [])
        path = None
        if self.cache_dir is not None:
            key = self._key(system, user, images)
            path = self.cache_dir / key[:2] / f"{key}.json"
            if path.is_file():
                with self._lock:
                    self.hits += 1
                return json.loads(path.read_text(encoding="utf-8"))

        problem: Exception | None = None
        for _ in range(self.attempts):
            with self._lock:
                if self.max_calls is not None and self.calls >= self.max_calls:
                    raise BudgetExceeded(
                        f"the call budget ({self.max_calls}) is used up")
                self.calls += 1
            obj = self.client.chat_json(user, images, system=system)
            try:
                if validate is not None:
                    validate(obj)
            except ValueError as e:
                problem = e
                continue
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(
                    f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(obj, ensure_ascii=False),
                               encoding="utf-8")
                tmp.replace(path)
            return obj
        raise ValueError(
            f"reply failed validation {self.attempts} times: {problem}")


class StubVLM:
    """Offline stand-in for ``VLMClient``: well-formed, uninformative replies.

    ``--dry-run`` uses it to push a manifest through every stage - paths,
    crops, prompt slots, metrics, output files - without a single API call.
    The stage is recognised by the reply keys its system prompt asks for.
    """

    model = "stub"
    temperature = 0.0

    def chat_json(self, user: str, images: list | None = None, *,
                  system: str | None = None) -> dict[str, Any]:
        s = system or ""
        if '"zoom_regions"' in s:
            return {"content": "", "intentional": [], "observations": [],
                    "zoom_regions": [{"bbox": [0.35, 0.35, 0.65, 0.65],
                                      "why": "stub"}]}
        if '"claims"' in s:
            return {"claims": [], "missed": [], "needs_redo": False, "notes": ""}
        if '"artifacts"' in s:
            return {"artifacts": [], "primary": "none", "quality": 3.0,
                    "summary": "stub"}
        if '"differences_checked"' in s:
            return {"differences_checked": [], "decision_supported": True,
                    "needs_redo": False, "notes": ""}
        if '"better"' in s:
            return {"better": "tie", "confidence": 0.0, "differences": [],
                    "reason": "stub"}
        if '"patterns"' in s:
            return {"patterns": []}
        if '"candidates"' in s:
            return {"candidates": []}
        raise ValueError("StubVLM: no stage recognised in the system prompt")


def make_vlm(*, model: str | None, temperature: float,
             cache_dir: str | Path | None, dry_run: bool = False,
             max_calls: int | None = None) -> CachedVLM:
    """The model every stage of one run shares.

    A dry run gets no cache: stub replies must never be replayed as real ones.
    """
    if dry_run:
        return CachedVLM(StubVLM(), None)
    from harness.clients import VLMClient
    return CachedVLM(VLMClient(model=model, temperature=temperature), cache_dir,
                     max_calls=max_calls)


def as_bool(v: Any) -> bool | None:
    """A JSON boolean, or its string spelling; ``None`` for anything else."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

def pmap(fn: Callable[[Any], Any], items: Iterable[Any], workers: int,
         on_done: Callable[[int, int, Any], None] | None = None) -> list[Any]:
    """``[fn(x) for x in items]`` on a thread pool, results in input order.

    ``on_done(k, n, result)`` is called as each item finishes, for progress
    lines. The calls are I/O-bound HTTP requests, so threads are enough.
    """
    items = list(items)
    n = len(items)
    out: list[Any] = [None] * n
    if workers <= 1 or n <= 1:
        for i, x in enumerate(items):
            out[i] = fn(x)
            if on_done:
                on_done(i + 1, n, out[i])
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for k, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            out[i] = fut.result()
            if on_done:
                on_done(k, n, out[i])
    return out
