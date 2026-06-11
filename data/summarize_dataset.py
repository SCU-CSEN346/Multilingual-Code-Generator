#!/usr/bin/env python3
"""
Summarize the mixed_pretrain JSONL dataset.

Prints:
  - Total record and token counts
  - Per-source breakdown (records, tokens, token distribution)
  - Schema / column structure for each source
  - Meta field distributions
  - Sample records

Usage:
    python summarize_dataset.py
    python summarize_dataset.py --dir mixed_pretrain --samples 2
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def fmt(n: int) -> str:
    return f"{n:,}"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(data: list[int], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir",     default="mixed_pretrain",
                    help="Directory containing JSONL shards (default: mixed_pretrain)")
    ap.add_argument("--samples", type=int, default=1,
                    help="Number of sample records to print per source (default: 1)")
    args = ap.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        print(f"ERROR: directory not found: {data_dir}")
        return

    shards = sorted(data_dir.glob("*.jsonl"))
    if not shards:
        print(f"No JSONL files found in {data_dir}")
        return

    # ── collect stats ──────────────────────────────────────────────────────────
    source_records:  dict[str, int]         = Counter()
    source_tokens:   dict[str, int]         = Counter()
    source_tok_dist: dict[str, list[int]]   = defaultdict(list)
    source_meta:     dict[str, list[dict]]  = defaultdict(list)
    source_samples:  dict[str, list[dict]]  = defaultdict(list)
    # (src, lang) → tokens / records / per-record token list
    srclang_tokens:  dict[tuple[str,str], int]       = Counter()
    srclang_records: dict[tuple[str,str], int]       = Counter()
    srclang_dist:    dict[tuple[str,str], list[int]] = defaultdict(list)
    total_records = 0

    for shard in shards:
        for rec in load_jsonl(shard):
            src   = rec.get("source", "unknown")
            toks  = rec.get("est_tokens", 0)
            meta  = rec.get("meta", {})
            total_records += 1
            source_records[src] += 1
            source_tokens[src]  += toks
            source_tok_dist[src].append(toks)
            source_meta[src].append(meta)
            if len(source_samples[src]) < args.samples:
                source_samples[src].append(rec)
            # per-(source, language) aggregation
            lang = (meta.get("language") or meta.get("lang") or "").strip()
            if lang:
                key = (src, lang)
                srclang_tokens[key]  += toks
                srclang_records[key] += 1
                srclang_dist[key].append(toks)

    total_tokens = sum(source_tokens.values())
    sources = sorted(source_records)

    # ── header ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Directory    : {data_dir.resolve()}")
    print(f"Shards       : {len(shards)}")
    print(f"Total records: {fmt(total_records)}")
    print(f"Total tokens : {fmt(total_tokens)}  (~{total_tokens/1e9:.2f}B)")

    # ── per-source overview ────────────────────────────────────────────────────
    print()
    print("PER-SOURCE OVERVIEW")
    print("-" * 70)
    print(f"{'Source':<14} {'Records':>10} {'Tokens':>15} {'Share':>7}  Token dist (p25/p50/p75/p99)")
    print("-" * 70)
    for src in sources:
        recs  = source_records[src]
        toks  = source_tokens[src]
        dist  = source_tok_dist[src]
        share = 100 * toks / total_tokens if total_tokens else 0
        p25   = int(percentile(dist, 25))
        p50   = int(percentile(dist, 50))
        p75   = int(percentile(dist, 75))
        p99   = int(percentile(dist, 99))
        print(f"{src:<14} {fmt(recs):>10} {fmt(toks):>15} {share:>6.1f}%  "
              f"{fmt(p25)} / {fmt(p50)} / {fmt(p75)} / {fmt(p99)}")
    print("-" * 70)
    print(f"{'TOTAL':<14} {fmt(total_records):>10} {fmt(total_tokens):>15}")

    # ── column / schema breakdown ──────────────────────────────────────────────
    print()
    print("COLUMN STRUCTURE")
    print("-" * 70)
    print("All records share these top-level fields:")
    print("  text         string   Formatted training text")
    print("  source       string   Dataset origin (see sources above)")
    print("  meta         object   Source-specific metadata (see below)")
    print("  est_tokens   int      Whitespace token estimate (words x 1.5)")
    print()
    print("text format by source:")
    print("  sltrans      '<source>\\n{code}\\n</source>\\n<llvm_ir>\\n{ir}\\n</llvm_ir>'")
    print("  pes2o        plain prose (scientific paper abstract + body)")
    print("  the_stack    plain source code")
    print("  openwebmath  plain web text (math-heavy)")

    # ── meta field breakdown ───────────────────────────────────────────────────
    print()
    print("META FIELD BREAKDOWN")
    print("-" * 70)
    meta_schemas = {
        "sltrans":     ["language (C/C++/Python/...)", "ir_type (Perf_Optimized|Size_Optimized)"],
        "pes2o":       ["id (document id)", "source (venue/collection)"],
        "the_stack":   ["lang (programming language)", "repo (repo name)", "license"],
        "openwebmath": ["url (source URL)"],
    }
    for src in sources:
        fields = meta_schemas.get(src, ["(unknown)"])
        print(f"  {src}:")
        for f in fields:
            print(f"    - {f}")

    # ── per-source meta distributions ──────────────────────────────────────────
    print()
    print("META VALUE DISTRIBUTIONS")
    print("-" * 70)

    if "sltrans" in source_meta:
        lang_counts  = Counter(m.get("language", "") for m in source_meta["sltrans"])
        ir_counts    = Counter(m.get("ir_type", "")  for m in source_meta["sltrans"])
        print("sltrans / language:")
        for lang, cnt in lang_counts.most_common():
            print(f"  {lang:<18} {fmt(cnt):>8} records")
        print("sltrans / ir_type:")
        for ir, cnt in ir_counts.most_common():
            print(f"  {ir:<26} {fmt(cnt):>8} records")
        print()

    if "the_stack" in source_meta:
        lang_counts = Counter(m.get("lang", "") for m in source_meta["the_stack"])
        print("the_stack / lang:")
        for lang, cnt in lang_counts.most_common():
            print(f"  {lang:<20} {fmt(cnt):>8} records")
        print()

    if "pes2o" in source_meta:
        src_counts = Counter(m.get("source", "") for m in source_meta["pes2o"])
        print(f"pes2o / source  ({len(src_counts)} distinct values, top 5):")
        for s, cnt in src_counts.most_common(5):
            print(f"  {s:<30} {fmt(cnt):>8} records")
        print()

    if "openwebmath" in source_meta:
        n = len(source_meta["openwebmath"])
        print(f"openwebmath / url: {fmt(n)} unique entries (one per record)")
        print()

    # ── token breakdown by source × language ──────────────────────────────────
    src_with_langs = sorted({src for src, _ in srclang_tokens})
    if src_with_langs:
        print()
        print("TOKEN BREAKDOWN BY LANGUAGE PER SOURCE")
        col = f"{'Language':<22} {'Records':>10} {'Tokens':>15} {'Share':>7}  Token dist (p25/p50/p75/p99)"
        for src in src_with_langs:
            src_total = source_tokens[src]
            langs_for_src = sorted(
                {lang for s, lang in srclang_tokens if s == src},
                key=lambda l: srclang_tokens[(src, l)],
                reverse=True,
            )
            print()
            print(f"  [{src}]  ({fmt(src_total)} total tokens)")
            print("  " + "-" * 68)
            print("  " + col)
            print("  " + "-" * 68)
            lang_subtotal = 0
            rec_subtotal  = 0
            for lang in langs_for_src:
                key   = (src, lang)
                recs  = srclang_records[key]
                toks  = srclang_tokens[key]
                dist  = srclang_dist[key]
                share = 100 * toks / src_total if src_total else 0
                p25   = int(percentile(dist, 25))
                p50   = int(percentile(dist, 50))
                p75   = int(percentile(dist, 75))
                p99   = int(percentile(dist, 99))
                print(f"  {lang:<22} {fmt(recs):>10} {fmt(toks):>15} {share:>6.1f}%  "
                      f"{fmt(p25)} / {fmt(p50)} / {fmt(p75)} / {fmt(p99)}")
                lang_subtotal += toks
                rec_subtotal  += recs
            print("  " + "-" * 68)
            print(f"  {'SUBTOTAL':<22} {fmt(rec_subtotal):>10} {fmt(lang_subtotal):>15}")
            unlabeled = src_total - lang_subtotal
            if unlabeled:
                share = 100 * unlabeled / src_total
                print(f"  (no language tag: {fmt(unlabeled)} tokens, {share:.1f}% of {src})")

    # ── sample records ─────────────────────────────────────────────────────────
    if args.samples > 0:
        print("SAMPLE RECORDS")
        print("=" * 70)
        for src in sources:
            for i, rec in enumerate(source_samples[src]):
                print(f"--- {src}  sample {i+1}/{args.samples} ---")
                print(f"  source     : {rec['source']}")
                print(f"  est_tokens : {fmt(rec['est_tokens'])}")
                print(f"  meta       : {rec['meta']}")
                preview = rec['text'].replace('\n', ' ')[:200]
                print(f"  text       : {preview!r}")
                print()


if __name__ == "__main__":
    main()
