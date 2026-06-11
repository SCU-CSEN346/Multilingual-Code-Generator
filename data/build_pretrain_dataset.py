"""
Build a mixed continued-pretraining dataset for a code LM.

Sources (streamed from the Hub — no full download):
  - UKPLab/SLTrans         (LLVM IR <-> source pairs; primary IRCoder signal)
  - allenai/peS2o          (open scientific text)
  - bigcode/the-stack      (permissively licensed source code)

Mixing target (token-weighted): 70 / 15 / 15.

Output: JSONL shards under OUT_DIR. Each line:
    {"text": "...", "source": "sltrans" | "pes2o" | "the_stack", "meta": {...}}

The token budget is approximate — we use a fast whitespace token estimate by default
to avoid pulling a heavy tokenizer into the streaming loop. Swap in a real tokenizer
(see TOKENIZER section) if you want exact counts against your model's vocab.

Usage:
    pip install "datasets>=2.18" huggingface_hub tqdm
    huggingface-cli login        # SLTrans + the-stack are gated; you must accept their terms
    python build_pretrain_dataset.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import socket

from datasets import interleave_datasets, load_dataset
from tqdm import tqdm

# Install hf-transfer and set this env var for significantly faster shard downloads.
# pip install hf-transfer
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# Raise the per-shard download timeout (default 10s is too short for HF CDN under load).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# Apply a 90s socket-level timeout to every connection in this process.
# This covers HF Hub file-listing API calls (which have no timeout by default)
# and prevents indefinite hangs at 'resolving data files'.
socket.setdefaulttimeout(90)

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class SourceSpec:
    name: str                       # short id used in output records
    hf_path: str                    # HF dataset path
    hf_config: str | None           # config name, if any
    split: str                      # split to stream
    target_fraction: float          # share of the total token budget
    text_fn: Callable[[dict], str]  # extracts the training text from a row
    meta_fn: Callable[[dict], dict] # extracts a small metadata dict
    # the-stack is organized by language subset; SLTrans by source language.
    # If `subsets` is set, we round-robin over them, each loaded as a separate stream.
    subsets: list[str] | None = None


# Total tokens in the final dataset (approximate).
TOTAL_TOKEN_BUDGET = 1_500_000_000

# Per-record length filters (in estimated tokens).
MIN_TOKENS_PER_RECORD = 32
MAX_TOKENS_PER_RECORD = 8192

# Output.
OUT_DIR = Path("./pretrain_mix")
SHARD_RECORDS = 50_000               # records per .jsonl shard
SEED = 17

# Reservoir / sampling. We don't reservoir-sample (would require knowing N);
# instead we accept records with probability `keep_prob` per source, tuned so
# the stream yields roughly the target token count before exhaustion. Set to
# 1.0 to take everything until budget is hit.
KEEP_PROB = {
    "sltrans":   1.0,
    "pes2o":     1.0,
    "the_stack": 0.5,   # the-stack is huge; subsample to diversify languages
}

# For the-stack, list languages you want represented. Keep this short for a
# focused replication; expand for broader code coverage.
THE_STACK_LANGS = [
    "python", "c", "cpp", "rust", "go", "java", "javascript", "typescript",
]

# SLTrans subsets (source languages whose IR we want). None => use default split.
SLTRANS_SUBSETS = [
    f"{lang}/{split}"
    for lang in ["C", "C++", "D", "Fortran", "Go", "Haskell", "Nim", "Objective-C", "Python", "Rust", "Swift"]
    for split in ["Perf_Optimized", "Size_Optimized"]
]

# ============================================================================
# TEXT / META EXTRACTORS
# ============================================================================
# These are intentionally defensive — different dataset versions name fields
# differently. Adjust if a `KeyError` shows up in your run.

def _first_present(row: dict, keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return default


def sltrans_text(row: dict) -> str:
    """SLTrans pairs source code with its LLVM IR. Concatenate with a marker so
    the model learns to associate them — IRCoder-style."""
    src = _first_present(row, ["source", "code", "src", "input"])
    ir  = _first_present(row, ["ir", "llvm_ir", "llvm", "target", "output"])
    if not src or not ir:
        return ""
    return f"<source>\n{src}\n</source>\n<llvm_ir>\n{ir}\n</llvm_ir>"


def sltrans_meta(row: dict) -> dict:
    return {
        "lang": _first_present(row, ["language", "lang", "source_lang"]),
    }


def pes2o_text(row: dict) -> str:
    return _first_present(row, ["text", "content"])


def pes2o_meta(row: dict) -> dict:
    return {
        "id":     _first_present(row, ["id", "doc_id"]),
        "source": _first_present(row, ["source", "venue"]),
    }


def the_stack_text(row: dict) -> str:
    return _first_present(row, ["content", "text", "code"])


def the_stack_meta(row: dict) -> dict:
    return {
        "lang":    _first_present(row, ["lang", "language"]),
        "repo":    _first_present(row, ["max_stars_repo_name", "repo_name"]),
        "license": _first_present(row, ["license"]),
    }


# ============================================================================
# SOURCES
# ============================================================================

SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="sltrans",
        hf_path="UKPLab/SLTrans",
        hf_config=None,
        split="train",
        target_fraction=0.70,
        text_fn=sltrans_text,
        meta_fn=sltrans_meta,
        subsets=SLTRANS_SUBSETS,
    ),
    SourceSpec(
        name="pes2o",
        hf_path="allenai/peS2o",
        hf_config="v2",
        split="train",
        target_fraction=0.15,
        text_fn=pes2o_text,
        meta_fn=pes2o_meta,
    ),
    SourceSpec(
        name="the_stack",
        hf_path="bigcode/the-stack",
        hf_config=None,
        # the-stack uses `data_dir` per language rather than HF configs.
        split="train",
        target_fraction=0.15,
        text_fn=the_stack_text,
        meta_fn=the_stack_meta,
        subsets=THE_STACK_LANGS,
    ),
]


# ============================================================================
# TOKEN ESTIMATOR
# ============================================================================
# Whitespace-based estimate. For BPE tokenizers, real tokens ~= 1.3x words for
# natural language and ~1.5–2x for code. We bake a 1.5x correction in here so
# the budget is honest enough for planning. If you want exact counts:
#
#   from transformers import AutoTokenizer
#   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B")
#   def estimate_tokens(text: str) -> int:
#       return len(tok.encode(text, add_special_tokens=False))

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.5)


# ============================================================================
# STREAMING
# ============================================================================

def open_stream(spec: SourceSpec, subset: str | None):
    """Return an IterableDataset for a (source, subset) pair, or None if unavailable."""
    kwargs = {"split": spec.split, "streaming": True}
    if spec.hf_config is not None:
        kwargs["name"] = spec.hf_config

    # the-stack uses data_dir to select a language.
    if spec.hf_path == "bigcode/the-stack" and subset is not None:
        kwargs["data_dir"] = f"data/{subset}"

    # SLTrans subsets are encoded as "Lang/Split" (e.g. "Python/Perf_Optimized").
    if spec.hf_path == "UKPLab/SLTrans" and subset is not None:
        lang, slt_split = subset.rsplit("/", 1)
        kwargs["name"] = lang
        kwargs["split"] = slt_split

    _TRANSIENT = ("ssl", "timeout", "handshake", "connection", "timed out")
    for attempt in range(5):
        try:
            return load_dataset(spec.hf_path, **kwargs)
        except ValueError as e:
            if "Bad split" in str(e):
                return None
            raise
        except Exception as e:
            if attempt < 4 and any(k in str(e).lower() for k in _TRANSIENT):
                time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
                continue
            raise
    return None


def round_robin(spec: SourceSpec) -> Iterator[dict]:
    """Yield rows from the source, interleaving across subsets if any.

    Data-file resolution (the HF Hub HTTP round-trips that show as
    'resolving data files') is parallelised across subsets so all
    metadata fetches happen concurrently instead of one-by-one.
    """
    if not spec.subsets:
        ds = open_stream(spec, None)
        if ds is not None:
            yield from ds
        return

    # Resolve all subset streams in parallel — resolution is I/O-bound so
    # threads eliminate most of the serial 'resolving data files' wait.
    with ThreadPoolExecutor(max_workers=min(4, len(spec.subsets))) as pool:
        results = list(pool.map(open_stream, [spec] * len(spec.subsets), spec.subsets))

    datasets = [ds for ds in results if ds is not None]
    if not datasets:
        return

    yield from interleave_datasets(datasets, stopping_strategy="all_exhausted")


# ============================================================================
# WRITER
# ============================================================================

class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, records_per_shard: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.records_per_shard = records_per_shard
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._shard_idx = 0
        self._records_in_shard = 0
        self._fh = None
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self.out_dir / f"{self.prefix}-{self._shard_idx:05d}.jsonl"
        self._fh = path.open("w", encoding="utf-8")
        self._records_in_shard = 0
        self._shard_idx += 1

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._records_in_shard += 1
        if self._records_in_shard >= self.records_per_shard:
            self._open_new_shard()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ============================================================================
# MAIN
# ============================================================================

def sample_source(spec: SourceSpec, target_tokens: int, writer: ShardWriter,
                  rng: random.Random) -> int:
    """Stream `spec` until ~target_tokens have been written. Returns tokens written."""
    keep_prob = KEEP_PROB.get(spec.name, 1.0)
    tokens_written = 0
    records_written = 0
    rows_seen = 0
    rows_skipped_filter = 0
    rows_skipped_subsample = 0
    rows_skipped_empty = 0
    started = time.time()

    # tqdm bar measured in tokens — the unit that actually matters for the budget.
    # `unit_scale=True` formats large counts as 1.23M / 1.23B automatically.
    bar = tqdm(
        total=target_tokens,
        unit="tok",
        unit_scale=True,
        desc=f"{spec.name:>10}",
        dynamic_ncols=True,
        smoothing=0.05,
    )

    def _refresh_postfix() -> None:
        elapsed = max(time.time() - started, 1e-6)
        bar.set_postfix({
            "records": f"{records_written:,}",
            "rows":    f"{rows_seen:,}",
            "tok/s":   f"{tokens_written/elapsed:,.0f}",
            "skip":    f"{rows_skipped_filter+rows_skipped_subsample+rows_skipped_empty:,}",
        }, refresh=False)

    try:
        for row in round_robin(spec):
            rows_seen += 1

            if keep_prob < 1.0 and rng.random() > keep_prob:
                rows_skipped_subsample += 1
                continue

            try:
                text = spec.text_fn(row)
            except Exception as e:
                if rows_seen <= 3:
                    bar.write(f"[{spec.name}] text_fn error on row {rows_seen}: {e!r}")
                rows_skipped_empty += 1
                continue

            if not text:
                rows_skipped_empty += 1
                continue

            n_tok = estimate_tokens(text)
            if n_tok < MIN_TOKENS_PER_RECORD or n_tok > MAX_TOKENS_PER_RECORD:
                rows_skipped_filter += 1
                continue

            record = {
                "text":   text,
                "source": spec.name,
                "meta":   spec.meta_fn(row),
                "est_tokens": n_tok,
            }
            writer.write(record)
            tokens_written += n_tok
            records_written += 1

            # Don't overshoot the bar (tqdm clamps, but `min` keeps the math clean).
            bar.update(min(n_tok, target_tokens - bar.n))

            # Refresh the postfix every ~1k records — cheaper than every step.
            if records_written % 1_000 == 0:
                _refresh_postfix()

            if tokens_written >= target_tokens:
                break

        _refresh_postfix()
    finally:
        bar.close()

    elapsed = time.time() - started
    print(f"[{spec.name}] DONE: {records_written:,} records, "
          f"{tokens_written:,} tokens, {rows_seen:,} rows seen, "
          f"skipped(filter={rows_skipped_filter:,} subsample={rows_skipped_subsample:,} "
          f"empty={rows_skipped_empty:,}), {elapsed:.0f}s")
    return tokens_written


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity-check fractions sum to ~1.
    total_frac = sum(s.target_fraction for s in SOURCES)
    if abs(total_frac - 1.0) > 1e-3:
        print(f"WARN: source fractions sum to {total_frac}, not 1.0", file=sys.stderr)

    # Banner — reassures the user something is happening before HF streams open.
    print("=" * 64)
    print(f"Building mixed pretrain corpus  →  {OUT_DIR.resolve()}")
    print(f"Total token budget: {TOTAL_TOKEN_BUDGET:,}")
    print(f"Per-record range: {MIN_TOKENS_PER_RECORD}–{MAX_TOKENS_PER_RECORD} tokens")
    for s in SOURCES:
        target = int(TOTAL_TOKEN_BUDGET * s.target_fraction)
        kp = KEEP_PROB.get(s.name, 1.0)
        subs = f", subsets={s.subsets}" if s.subsets else ""
        print(f"  - {s.name:>10}: {s.target_fraction:>5.0%} → {target:>15,} tok  "
              f"[keep_prob={kp}{subs}]")
    print("=" * 64)

    summary = {}
    for spec in SOURCES:
        target = int(TOTAL_TOKEN_BUDGET * spec.target_fraction)
        writer = ShardWriter(OUT_DIR, prefix=spec.name,
                             records_per_shard=SHARD_RECORDS)
        try:
            written = sample_source(spec, target, writer, rng)
        finally:
            writer.close()
        summary[spec.name] = {"target": target, "written": written}

    # Manifest.
    manifest = {
        "total_token_budget": TOTAL_TOKEN_BUDGET,
        "seed": SEED,
        "sources": [
            {"name": s.name, "hf_path": s.hf_path,
             "fraction": s.target_fraction, "subsets": s.subsets}
            for s in SOURCES
        ],
        "tokens_written": summary,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== SUMMARY ===")
    grand_total = sum(v["written"] for v in summary.values())
    for name, v in summary.items():
        pct = 100 * v["written"] / grand_total if grand_total else 0
        print(f"  {name:>10}: {v['written']:>15,} tokens ({pct:5.1f}%)")
    print(f"  {'TOTAL':>10}: {grand_total:>15,} tokens")
    print(f"\nOutput: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()