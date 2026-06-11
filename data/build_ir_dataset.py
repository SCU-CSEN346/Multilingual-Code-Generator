#!/usr/bin/env python3
"""
Transform the mixed_pretrain JSONL shards into a unified parquet dataset
where every record has a source_code column and an llvm_ir column.

SLTrans records already contain both — the <source> / <llvm_ir> tags are
parsed out into separate columns.  Records from other sources (peS2o,
TheStack, OpenWebMath) have their text placed in source_code and llvm_ir
set to null, keeping the schema consistent.

Output schema
-------------
  source_code    string    source code or prose text
  llvm_ir        string    LLVM IR  (null for non-SLTrans records)
  language       string    programming language or content category
  ir_type        string    Perf_Optimized / Size_Optimized  (null if not SLTrans)
  source_dataset string    sltrans | pes2o | the_stack | openwebmath
  est_tokens     int64     whitespace token estimate

Usage
-----
    python build_ir_dataset.py
    python build_ir_dataset.py --input mixed_pretrain --output ir_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── SLTrans text parsing ───────────────────────────────────────────────────────
# Matches the format written by build_mixed_dataset.py:
#   <source>\n{code}\n</source>\n<llvm_ir>\n{ir}\n</llvm_ir>
_SOURCE_PAT = re.compile(r"<source>\n(.*?)\n</source>",  re.DOTALL)
_IR_PAT     = re.compile(r"<llvm_ir>\n(.*?)\n</llvm_ir>", re.DOTALL)


def parse_sltrans(text: str) -> tuple[str, str]:
    src = _SOURCE_PAT.search(text)
    ir  = _IR_PAT.search(text)
    return (src.group(1) if src else ""), (ir.group(1) if ir else "")


# ── language normalisation ─────────────────────────────────────────────────────
# Maps non-SLTrans sources to a human-readable language/category label.
def _language(source: str, meta: dict) -> str:
    if source == "the_stack":
        return meta.get("lang") or meta.get("language") or "code"
    if source == "pes2o":
        return "scientific"
    if source == "openwebmath":
        return "math"
    return ""


# ── record transformer ─────────────────────────────────────────────────────────
def transform(rec: dict) -> dict:
    source = rec["source"]
    text   = rec.get("text", "")
    meta   = rec.get("meta", {})

    if source == "sltrans":
        source_code, llvm_ir = parse_sltrans(text)
        language = meta.get("language", "")
        ir_type  = meta.get("ir_type", None)
    elif source == "stack_llvm":
        # Unpaired IR from TheStack: the text IS the IR, no corresponding source
        source_code = None
        llvm_ir     = text
        language    = "LLVM"
        ir_type     = None
    else:
        source_code = text
        llvm_ir     = None
        language    = _language(source, meta)
        ir_type     = None

    return {
        "source_code":    source_code,
        "llvm_ir":        llvm_ir,
        "language":       language,
        "ir_type":        ir_type,
        "source_dataset": source,
        "est_tokens":     rec.get("est_tokens", 0),
    }


# ── parquet schema ─────────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("source_code",    pa.large_utf8()),
    pa.field("llvm_ir",        pa.large_utf8()),
    pa.field("language",       pa.large_utf8()),
    pa.field("ir_type",        pa.large_utf8()),
    pa.field("source_dataset", pa.large_utf8()),
    pa.field("est_tokens",     pa.int64()),
])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",      default="mixed_pretrain",
                    help="Directory containing JSONL shards (default: mixed_pretrain)")
    ap.add_argument("--output",     default="ir_dataset.parquet",
                    help="Output parquet path (default: ir_dataset.parquet)")
    ap.add_argument("--batch-size", type=int, default=10_000,
                    help="Records per parquet row group (default: 10000)")
    args = ap.parse_args()

    data_dir = Path(args.input)
    if not data_dir.is_dir():
        print(f"ERROR: input directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    shards = sorted(data_dir.glob("*.jsonl"))
    if not shards:
        print(f"No JSONL files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    print(f"Input  : {data_dir.resolve()}  ({len(shards)} shards)")
    print(f"Output : {out_path.resolve()}")
    print(f"Schema : {[f.name for f in SCHEMA]}")
    print()

    writer  = pq.ParquetWriter(out_path, SCHEMA)
    batch: list[dict] = []
    total = sltrans_ok = sltrans_partial = 0

    def flush(rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        # Ensure all columns present even if batch is from a single source
        for col in [f.name for f in SCHEMA]:
            if col not in df.columns:
                df[col] = None
        df["est_tokens"] = df["est_tokens"].fillna(0).astype("int64")
        table = pa.Table.from_pandas(df[[ f.name for f in SCHEMA ]], schema=SCHEMA)
        writer.write_table(table)

    for shard in tqdm(shards, desc="shards", unit="file"):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                row = transform(rec)

                if rec["source"] == "sltrans":
                    if row["source_code"] and row["llvm_ir"]:
                        sltrans_ok += 1
                    else:
                        sltrans_partial += 1

                batch.append(row)
                total += 1

                if len(batch) >= args.batch_size:
                    flush(batch)
                    batch.clear()

    if batch:
        flush(batch)

    writer.close()

    # ── summary ────────────────────────────────────────────────────────────────
    pf = pq.ParquetFile(out_path)
    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Total records written : {total:,}")
    print(f"SLTrans (both fields) : {sltrans_ok:,}")
    if sltrans_partial:
        print(f"SLTrans (parse miss)  : {sltrans_partial:,}  <- check text format")
    print()
    print("Output schema:")
    print(pf.schema_arrow)
    print()

    # Quick per-source count from the written file
    df = pf.read(columns=["source_dataset"]).to_pandas()
    print("Records per source_dataset:")
    for src, cnt in df["source_dataset"].value_counts().items():
        print(f"  {src:<14} {cnt:>10,}")
    print()
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
