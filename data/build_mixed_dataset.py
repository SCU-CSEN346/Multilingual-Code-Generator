#!/usr/bin/env python3
"""
build_mixed_dataset.py — Five-source mixed pre-training corpus builder.

Sources
-------
  SLTrans          UKPLab/SLTrans (balanced language x IR-type)   --sltrans-tokens
  peS2o            allenai/peS2o  (open scientific papers)        --pes2o-tokens
  TheStack (code)  bigcode/the-stack, non-LLVM languages          --stack-tokens
  TheStack (LLVM)  bigcode/the-stack, lang=llvm  (unpaired IR)    --stack-llvm-tokens
  OpenWebMath      open-web-math/open-web-math (math web text)    --owm-tokens

Set any cap to 0 to skip that source.
TheStack code and LLVM IR are streamed from the same HF dataset but kept
as separate shards (the_stack-*.jsonl vs stack_llvm-*.jsonl) so downstream
pipelines can weight them independently.

Output
------
  JSONL shards under --output-dir, one filename prefix per source:
    sltrans-00000.jsonl, pes2o-00000.jsonl, the_stack-00000.jsonl, ...
  Each record:
    {"text": "...", "source": "...", "meta": {...}, "est_tokens": N}
  Plus manifest.json summarising the run.

Usage
-----
  pip install "datasets>=2.18" pyarrow pandas tqdm
  huggingface-cli login          # peS2o and the-stack are gated

  python build_mixed_dataset.py
  python build_mixed_dataset.py --sltrans-tokens 500e6 --owm-tokens 200e6
  python build_mixed_dataset.py --stack-tokens 0              # skip code
  python build_mixed_dataset.py --stack-langs python,rust,go
  python build_mixed_dataset.py --stack-llvm-tokens 100e6     # 100M unpaired IR
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

socket.setdefaulttimeout(90)

# ── constants ──────────────────────────────────────────────────────────────────
SLTRANS_PROBE_ROWS = 200
SLTRANS_HF_REPO    = "UKPLab/SLTrans"

THE_STACK_LANGS = [
    "python", "c", "c++", "rust", "go",
    "java", "javascript", "typescript",
]

_TRANSIENT_ERRORS = ("ssl", "timeout", "handshake", "connection", "timed out")


# ── token estimation ───────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.5)


# ── JSONL shard writer ─────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, records_per_shard: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._dir, self._pfx, self._rps = out_dir, prefix, records_per_shard
        self._idx = self._n = 0
        self._fh = None
        self._roll()

    def _roll(self):
        if self._fh:
            self._fh.close()
        self._fh = (self._dir / f"{self._pfx}-{self._idx:05d}.jsonl").open("w", encoding="utf-8")
        self._n = 0
        self._idx += 1

    def write(self, record: dict):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n >= self._rps:
            self._roll()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ── SLTrans (UKPLab/SLTrans on the HuggingFace hub) ─────────────────────────────
# The hub repo lays parquet files out as
#     datasets/<repo>/<language>/<Perf|Size>_Optimized-*.parquet
# one directory per language config, mirroring the local layout this used to
# read from disk.  We read the remote parquet files directly through
# HfFileSystem (range requests), so footer/row-group reads only fetch the bytes
# they need instead of downloading every shard up front.

def _sltrans_fs():
    """Return an HfFileSystem for streaming SLTrans parquet shards from the hub."""
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def _sltrans_pf(path: str, fs) -> pq.ParquetFile:
    """Open a remote SLTrans parquet shard as a (seekable) ParquetFile."""
    return pq.ParquetFile(fs.open(path, "rb"))


def _sltrans_find_groups(repo: str, fs) -> dict[tuple[str, str], list[str]]:
    """Return {(language, ir_type): [sorted hub shard paths]}."""
    groups: dict[tuple[str, str], list[str]] = {}
    for path in fs.glob(f"datasets/{repo}/*/*.parquet"):
        lang = path.split("/")[-2]
        name = path.rsplit("/", 1)[-1]
        m = re.match(r"^(Perf_Optimized|Size_Optimized)", name)
        if m:
            groups.setdefault((lang, m.group(1)), []).append(path)
    for k in groups:
        groups[k].sort()
    return groups


def _pq_nrows(files: list[str], fs) -> int:
    return sum(_sltrans_pf(f, fs).metadata.num_rows for f in files)


def _est_tok_df(src: pd.Series, ir: pd.Series) -> pd.Series:
    src_w = src.fillna("").str.split().str.len().fillna(0)
    ir_w  = ir.fillna("").str.split().str.len().fillna(0)
    return ((src_w + ir_w + 5) * 1.5).astype(int)


def _probe_avg_tokens(files: list[str], n: int, rng: random.Random, fs) -> float:
    frames = []
    seed = rng.randint(0, 2**31)
    for f in files:
        df = _sltrans_pf(f, fs).read_row_group(0).to_pandas()
        if not df.empty:
            frames.append(df.sample(min(n, len(df)), random_state=seed))
        if sum(len(x) for x in frames) >= n:
            break
    if not frames:
        return 0.0
    p = pd.concat(frames, ignore_index=True).head(n)
    p = p.dropna(subset=["Source_Code", "IR_Original"])
    p = p[(p["Source_Code"] != "") & (p["IR_Original"] != "")]
    return float(_est_tok_df(p["Source_Code"], p["IR_Original"]).mean()) if len(p) else 0.0


def _sltrans_allocate(
    groups: dict[tuple[str, str], list[str]],
    total: int,
    rng: random.Random,
    fs,
) -> dict[tuple[str, str], int]:
    """Equal-share budget with deficit redistribution for small groups."""
    keys = sorted(groups)
    avail: dict[tuple[str, str], int] = {}
    for k in tqdm(keys, desc="      probe", unit="grp", leave=False):
        rows = _pq_nrows(groups[k], fs)
        avg  = _probe_avg_tokens(groups[k], SLTRANS_PROBE_ROWS, rng, fs)
        avail[k] = int(rows * avg)
        tqdm.write(
            f"      {k[0]:>15}/{k[1]:<16}  ~{avail[k]:>14,} tok"
            f"  ({rows:,} rows, avg {avg:.0f})"
        )
    budgets = {k: total // len(keys) for k in keys}
    for _ in range(len(keys)):
        capped  = {k: min(budgets[k], avail[k]) for k in keys}
        deficit = sum(budgets[k] - capped[k] for k in keys)
        if not deficit:
            break
        room = [k for k in keys if capped[k] < avail[k]]
        if not room:
            break
        bonus = deficit // len(room)
        for k in room:
            capped[k] = min(capped[k] + bonus, avail[k])
        budgets = capped
    return budgets


def write_sltrans(
    repo: str,
    budget: int,
    writer: ShardWriter,
    rng: random.Random,
    min_tokens: int,
    fs,
) -> int:
    groups = _sltrans_find_groups(repo, fs)
    if not groups:
        print(f"    WARNING: no SLTrans parquet files found in {repo}", file=sys.stderr)
        return 0

    budgets = _sltrans_allocate(groups, budget, rng, fs)
    total_written = 0
    bar = tqdm(total=budget, unit="tok", unit_scale=True,
               desc="      write", dynamic_ncols=True)

    for (lang, ir_type) in sorted(groups):
        g_budget  = budgets[(lang, ir_type)]
        g_written = 0
        files = list(groups[(lang, ir_type)])
        rng.shuffle(files)

        for f in files:
            if g_written >= g_budget:
                break
            pf = _sltrans_pf(f, fs)
            for gi in range(pf.num_row_groups):
                if g_written >= g_budget:
                    break
                df = pf.read_row_group(gi).to_pandas()
                df = df.dropna(subset=["Source_Code", "IR_Original"])
                df = df[(df["Source_Code"] != "") & (df["IR_Original"] != "")]
                if df.empty:
                    continue
                df = df.sample(frac=1, random_state=rng.randint(0, 2**31)).reset_index(drop=True)
                df["_t"] = _est_tok_df(df["Source_Code"], df["IR_Original"])

                remaining = g_budget - g_written
                cutoff = max(int((df["_t"].cumsum() <= remaining).sum()), 1)
                for row in df.iloc[:cutoff].to_dict("records"):
                    toks = int(row["_t"])
                    if toks < min_tokens:
                        continue
                    text = (
                        f"<source>\n{row['Source_Code']}\n</source>\n"
                        f"<llvm_ir>\n{row['IR_Original']}\n</llvm_ir>"
                    )
                    writer.write({
                        "text": text,
                        "source": "sltrans",
                        "meta": {"language": lang, "ir_type": ir_type},
                        "est_tokens": toks,
                    })
                    g_written    += toks
                    total_written += toks
                    bar.update(min(toks, budget - bar.n))
    bar.close()
    return total_written


# ── HuggingFace streaming ──────────────────────────────────────────────────────

def _hf_open(
    hf_path: str,
    split: str = "train",
    hf_config: str | None = None,
    data_dir: str | None = None,
):
    """Open one HF streaming dataset with exponential-backoff retry."""
    from datasets import load_dataset

    kw: dict = {"split": split, "streaming": True, "trust_remote_code": True}
    if hf_config:
        kw["name"] = hf_config
    if data_dir:
        kw["data_dir"] = data_dir

    for attempt in range(5):
        try:
            return load_dataset(hf_path, **kw)
        except ValueError as e:
            if "Bad split" in str(e):
                return None
            raise
        except Exception as e:
            if attempt < 4 and any(k in str(e).lower() for k in _TRANSIENT_ERRORS):
                time.sleep(2 ** attempt)
                continue
            raise
    return None


def _hf_iter(
    hf_path: str,
    split: str = "train",
    hf_config: str | None = None,
):
    """Yield rows from a HuggingFace streaming dataset."""
    ds = _hf_open(hf_path, split=split, hf_config=hf_config)
    if ds is not None:
        yield from ds


def write_hf_source(
    source_name: str,
    budget: int,
    writer: ShardWriter,
    rng: random.Random,
    min_tokens: int,
    hf_path: str,
    text_fn,
    meta_fn,
    hf_config: str | None = None,
    split: str = "train",
    lang_filter: set[str] | None = None,
) -> int:
    written = skipped = 0
    bar = tqdm(total=budget, unit="tok", unit_scale=True,
               desc=f"      {source_name:<12}", dynamic_ncols=True, smoothing=0.05)
    try:
        for row in _hf_iter(hf_path, split=split, hf_config=hf_config):
            if lang_filter is not None:
                lang = (row.get("lang") or row.get("language") or "").lower()
                if lang not in lang_filter:
                    skipped += 1
                    continue
            text = text_fn(row)
            if not text:
                skipped += 1
                continue
            toks = estimate_tokens(text)
            if toks < min_tokens:
                skipped += 1
                continue
            writer.write({
                "text": text,
                "source": source_name,
                "meta": meta_fn(row),
                "est_tokens": toks,
            })
            written += toks
            bar.update(min(toks, budget - bar.n))
            if written >= budget:
                break
    finally:
        bar.close()
    print(f"      done: {written:,} tokens written, {skipped:,} rows skipped")
    return written


# ── text / meta extractors ─────────────────────────────────────────────────────

def _get(row: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return default


def pes2o_text(row):  return _get(row, "text", "content")
def pes2o_meta(row):  return {"id": _get(row, "id", "doc_id"), "source": _get(row, "source", "venue")}

def stack_text(row):  return _get(row, "content", "text", "code")
def stack_meta(row):  return {
    "lang":    _get(row, "lang", "language"),
    "repo":    _get(row, "max_stars_repo_name", "repo_name"),
    "license": _get(row, "license"),
}

def owm_text(row):    return _get(row, "text")
def owm_meta(row):    return {"url": _get(row, "url")}


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sltrans-repo",   default=SLTRANS_HF_REPO,
                    help=f"HuggingFace dataset repo for SLTrans (default: {SLTRANS_HF_REPO})")
    ap.add_argument("--sltrans-tokens", type=float, default=700_000_000,
                    help="Token cap for SLTrans   (default: 700M, 0=skip)")
    ap.add_argument("--pes2o-tokens",   type=float, default=150_000_000,
                    help="Token cap for peS2o     (default: 150M, 0=skip)")
    ap.add_argument("--stack-tokens",      type=float, default=100_000_000,
                    help="Token cap for TheStack source code  (default: 100M, 0=skip)")
    ap.add_argument("--stack-llvm-tokens", type=float, default=0,
                    help="Token cap for TheStack unpaired LLVM IR  (default: 0=skip)")
    ap.add_argument("--owm-tokens",        type=float, default=50_000_000,
                    help="Token cap for OpenWebMath (default: 50M, 0=skip)")
    ap.add_argument("--stack-langs",       default=",".join(THE_STACK_LANGS),
                    help="Comma-separated TheStack source-code language subsets (never includes llvm)")
    ap.add_argument("--output-dir",     default="./mixed_pretrain",
                    help="Output directory for JSONL shards (default: ./mixed_pretrain)")
    ap.add_argument("--shard-size",     type=int, default=50_000,
                    help="Records per JSONL shard (default: 50000)")
    ap.add_argument("--min-tokens",     type=int, default=32,
                    help="Drop records shorter than this (est. tokens, default: 32)")
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    rng         = random.Random(args.seed)
    out_dir     = Path(args.output_dir)
    # Ensure "llvm" never leaks into the source-code filter.
    stack_langs = [s.strip() for s in args.stack_langs.split(",")
                   if s.strip() and s.strip().lower() != "llvm"]

    budgets = {
        "sltrans":     int(args.sltrans_tokens),
        "pes2o":       int(args.pes2o_tokens),
        "the_stack":   int(args.stack_tokens),
        "stack_llvm":  int(args.stack_llvm_tokens),
        "openwebmath": int(args.owm_tokens),
    }
    active_sources = [name for name, tok in budgets.items() if tok > 0]
    total_budget   = sum(budgets.values())

    print("=" * 64)
    print("Mixed pre-training dataset builder")
    print(f"  Output : {out_dir.resolve()}")
    print(f"  Seed   : {args.seed}")
    print()
    for name, toks in budgets.items():
        if toks > 0:
            print(f"  {name:<14}  {toks:>15,} tokens")
        else:
            print(f"  {name:<14}  (skipped)")
    print(f"  {'TOTAL':<14}  {total_budget:>15,} tokens")
    print("=" * 64)

    summary: dict[str, int] = {}
    n_active = len(active_sources)
    step = 1

    # ── SLTrans ────────────────────────────────────────────────────────────────
    if budgets["sltrans"] > 0:
        print(f"\n[{step}/{n_active}] SLTrans  ({args.sltrans_repo}, balanced language x IR-type)")
        step += 1
        w = ShardWriter(out_dir, "sltrans", args.shard_size)
        try:
            summary["sltrans"] = write_sltrans(
                args.sltrans_repo, budgets["sltrans"], w, rng, args.min_tokens,
                _sltrans_fs(),
            )
        finally:
            w.close()

    # ── peS2o ──────────────────────────────────────────────────────────────────
    if budgets["pes2o"] > 0:
        print(f"\n[{step}/{n_active}] peS2o  (allenai/peS2o, config=v2)")
        step += 1
        w = ShardWriter(out_dir, "pes2o", args.shard_size)
        try:
            summary["pes2o"] = write_hf_source(
                "pes2o", budgets["pes2o"], w, rng, args.min_tokens,
                hf_path="allenai/peS2o",
                text_fn=pes2o_text, meta_fn=pes2o_meta,
                hf_config="v2",
            )
        finally:
            w.close()

    # ── TheStack ───────────────────────────────────────────────────────────────
    if budgets["the_stack"] > 0:
        print(f"\n[{step}/{n_active}] TheStack  (bigcode/the-stack, {len(stack_langs)} language subsets)")
        print(f"      langs: {', '.join(stack_langs)}")
        step += 1
        w = ShardWriter(out_dir, "the_stack", args.shard_size)
        try:
            summary["the_stack"] = write_hf_source(
                "the_stack", budgets["the_stack"], w, rng, args.min_tokens,
                hf_path="bigcode/the-stack",
                text_fn=stack_text, meta_fn=stack_meta,
                lang_filter={l.lower() for l in stack_langs},
            )
        finally:
            w.close()

    # ── TheStack LLVM IR (unpaired) ────────────────────────────────────────────
    if budgets["stack_llvm"] > 0:
        print(f"\n[{step}/{n_active}] TheStack LLVM IR  (bigcode/the-stack, lang=llvm, unpaired)")
        step += 1
        w = ShardWriter(out_dir, "stack_llvm", args.shard_size)
        try:
            summary["stack_llvm"] = write_hf_source(
                "stack_llvm", budgets["stack_llvm"], w, rng, args.min_tokens,
                hf_path="bigcode/the-stack",
                text_fn=stack_text, meta_fn=stack_meta,
                lang_filter={"llvm"},
            )
        finally:
            w.close()

    # ── OpenWebMath ────────────────────────────────────────────────────────────
    if budgets["openwebmath"] > 0:
        print(f"\n[{step}/{n_active}] OpenWebMath  (open-web-math/open-web-math)")
        w = ShardWriter(out_dir, "openwebmath", args.shard_size)
        try:
            summary["openwebmath"] = write_hf_source(
                "openwebmath", budgets["openwebmath"], w, rng, args.min_tokens,
                hf_path="open-web-math/open-web-math",
                text_fn=owm_text, meta_fn=owm_meta,
            )
        finally:
            w.close()

    # ── manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "seed":                    args.seed,
        "min_tokens_per_record":   args.min_tokens,
        "sources": {
            "sltrans":     {"hf_path":  args.sltrans_repo,                 "target_tokens": budgets["sltrans"]},
            "pes2o":       {"hf_path":  "allenai/peS2o",                  "target_tokens": budgets["pes2o"]},
            "the_stack":   {"hf_path":  "bigcode/the-stack",              "target_tokens": budgets["the_stack"],     "langs": stack_langs},
            "stack_llvm":  {"hf_path":  "bigcode/the-stack",              "target_tokens": budgets["stack_llvm"],    "langs": ["llvm"]},
            "openwebmath": {"hf_path":  "open-web-math/open-web-math",    "target_tokens": budgets["openwebmath"]},
        },
        "tokens_written": summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── summary ────────────────────────────────────────────────────────────────
    grand = sum(summary.values())
    print("\n" + "=" * 62)
    print(f"{'Source':<14}  {'Target':>15}  {'Written':>15}  {'Share':>6}")
    print("-" * 58)
    for name in ["sltrans", "pes2o", "the_stack", "stack_llvm", "openwebmath"]:
        if budgets[name] == 0:
            continue
        written = summary.get(name, 0)
        pct     = 100 * written / grand if grand else 0
        print(f"{name:<14}  {budgets[name]:>15,}  {written:>15,}  {pct:>5.1f}%")
    print("-" * 58)
    print(f"{'TOTAL':<14}  {total_budget:>15,}  {grand:>15,}  100.0%")
    print(f"\nOutput: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
