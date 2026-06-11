#!/usr/bin/env python3
"""
tokenize_for_training.py — Tokenize and chunk ir_dataset.parquet into a
HuggingFace dataset ready for continued_pretrain.py.

Pipeline position: AFTER build_ir_dataset.py, BEFORE continued_pretrain.py

Input
-----
  Parquet file produced by build_ir_dataset.py with columns:
    source_code, llvm_ir, language, ir_type, source_dataset, est_tokens

Processing
----------
  1. Assemble one text string per record:
       sltrans    → <source>\\n{code}\\n</source>\\n<llvm_ir>\\n{ir}\\n</llvm_ir>
       stack_llvm → {llvm_ir}  (unpaired IR, no source)
       others     → {source_code}  (peS2o, TheStack, OpenWebMath)
  2. Tokenize each string without truncation, append an EOS token as a
     document separator.
  3. Concatenate all token sequences into one stream, split into
     fixed-length blocks of --block-size tokens.
  4. Emit records with input_ids / attention_mask / labels columns.
     labels == input_ids (the Trainer shifts internally for causal LM loss).

Output
------
  Arrow dataset saved with save_to_disk(), optionally pushed to the Hub.
  Load locally:  datasets.load_from_disk("./tokenized_dataset")
  Load from Hub: datasets.load_dataset("your-org/your-dataset")

  NOTE: continued_pretrain.py uses load_dataset(...), so either push to Hub
  or replace that call with load_from_disk() for a purely local workflow.

Usage
-----
  python tokenize_for_training.py \\
      --input ir_dataset.parquet \\
      --model bigcode/starcoderbase-1b \\
      --output ./tokenized_dataset

  python tokenize_for_training.py \\
      --input ir_dataset.parquet \\
      --model codellama/CodeLlama-7b-hf \\
      --output ./tokenized_dataset \\
      --validation-split 0.05 \\
      --push-to-hub your-org/dataset-name \\
      --token YOUR_HF_TOKEN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from itertools import chain

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Special tokens the IRCoder paper adds to every model's vocabulary.
_IR_SPECIAL_TOKENS = ["<source_to_llvm>", "<llvm_to_source>"]
_PAD_TOKEN = "<|pad|>"

# Records per batch during the group_texts step.  Larger batches waste fewer
# tokens at chunk boundaries.
_GROUP_BATCH_SIZE = 5_000


# ── tokenizer setup ────────────────────────────────────────────────────────────

def build_tokenizer(model_name: str, block_size: int | None, token: str | None) -> AutoTokenizer:
    """Load tokenizer and register the IRCoder special tokens."""
    tok = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        truncation_side="right",
        trust_remote_code=True,
        token=token,
    )

    tokens_to_add: dict = {}

    if tok.pad_token is None:
        tokens_to_add["pad_token"] = _PAD_TOKEN

    extra = [t for t in _IR_SPECIAL_TOKENS if t not in tok.get_vocab()]
    existing_extra = list(tok.extra_special_tokens or [])
    new_extra = [t for t in extra if t not in existing_extra]
    if new_extra:
        tokens_to_add["additional_special_tokens"] = existing_extra + new_extra

    if tokens_to_add:
        tok.add_special_tokens(tokens_to_add)

    if block_size is not None:
        tok.model_max_length = block_size
    elif tok.model_max_length > 1_000_000:
        # HuggingFace sets model_max_length to a huge sentinel when the tokenizer
        # config doesn't specify a context length. Fall back to the value used by
        # the IRCoder paper for all StarCoder/DeepSeek/CodeLlama models.
        tok.model_max_length = 4096
        print(f"  WARNING: tokenizer did not report a context length; defaulting "
              f"to {tok.model_max_length}. Pass --block-size to override.")

    return tok


# ── text assembly ──────────────────────────────────────────────────────────────

def _assemble_text_batch(batch: dict) -> dict:
    """Derive a single 'text' string for each record."""
    texts: list[str | None] = []
    for src, ir, source in zip(
        batch["source_code"], batch["llvm_ir"], batch["source_dataset"]
    ):
        if source == "sltrans":
            if src and ir:
                texts.append(
                    f"<source>\n{src}\n</source>\n<llvm_ir>\n{ir}\n</llvm_ir>"
                )
            else:
                texts.append(None)
        elif source == "stack_llvm":
            texts.append(str(ir) if ir else None)
        else:
            texts.append(str(src) if src else None)
    return {"text": texts}


def _is_valid(example: dict) -> bool:
    t = example["text"]
    return t is not None and t != ""


# ── tokenisation ───────────────────────────────────────────────────────────────

def _tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize text without truncation; append EOS as a document separator."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(
            f"Tokenizer for {tokenizer.name_or_path!r} has no eos_token_id. "
            "Set one explicitly before running this script."
        )

    encoded = tokenizer(
        batch["text"],
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return {
        "input_ids":      [ids + [eos] for ids in encoded["input_ids"]],
        "attention_mask": [am  + [1]   for am  in encoded["attention_mask"]],
    }


# ── chunking ───────────────────────────────────────────────────────────────────

def _group_texts(batch: dict, block_size: int) -> dict:
    """Concatenate token sequences and split into fixed-length blocks."""
    all_ids = list(chain.from_iterable(batch["input_ids"]))
    total = (len(all_ids) // block_size) * block_size
    if total == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    num_blocks = total // block_size
    ids_list = [all_ids[i : i + block_size] for i in range(0, total, block_size)]
    am_row = [1] * block_size

    return {
        "input_ids": ids_list,
        "attention_mask": [am_row] * num_blocks,
        "labels": ids_list,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", default="ir_dataset.parquet",
        help="Parquet file from build_ir_dataset.py (default: ir_dataset.parquet)",
    )
    ap.add_argument(
        "--model", required=True,
        help="HuggingFace model name — selects the tokenizer "
             "(e.g. bigcode/starcoderbase-1b)",
    )
    ap.add_argument(
        "--output", default="./tokenized_dataset",
        help="Directory for the Arrow dataset (default: ./tokenized_dataset)",
    )
    ap.add_argument(
        "--block-size", type=int, default=None,
        help="Token block length. Defaults to the tokenizer's model_max_length "
             "(4096 for StarCoder/DeepSeek/CodeLlama, 2048 for CodeGen).",
    )
    ap.add_argument(
        "--num-workers", type=int, default=1,
        help="Parallel workers for dataset.map (default: 1). "
             "Increase on Linux; keep at 1 on Windows to avoid spawn issues.",
    )
    ap.add_argument(
        "--validation-split", type=float, default=0.0,
        help="Fraction of blocks to hold out as validation (default: 0 = no split). "
             "The training script can also split at runtime via "
             "--validation_split_percentage.",
    )
    ap.add_argument(
        "--push-to-hub", default=None, metavar="HUB_DATASET_ID",
        help="Push the finished dataset to the Hub (e.g. your-org/dataset-name).",
    )
    ap.add_argument(
        "--token", default=None,
        help="HuggingFace token (required for gated models and Hub push).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print(f"Model / tokenizer  : {args.model}")
    tokenizer = build_tokenizer(args.model, args.block_size, args.token)
    block_size = tokenizer.model_max_length
    print(f"Block size         : {block_size} tokens")
    print(f"Vocab size (final) : {len(tokenizer)}")
    print(f"EOS token          : {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print()

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"[1/4] Loading {in_path.resolve()} ...")
    ds: Dataset = Dataset.from_parquet(str(in_path))
    print(f"      {len(ds):,} records loaded")

    # ── assemble text ─────────────────────────────────────────────────────────
    print("\n[2/4] Assembling text column ...")
    ds = ds.map(
        _assemble_text_batch,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=ds.column_names,
        desc="assemble",
    )
    before = len(ds)
    ds = ds.filter(_is_valid, num_proc=args.num_workers, desc="filter-empty")
    dropped = before - len(ds)
    print(f"      {len(ds):,} records with text  ({dropped:,} dropped — empty/null fields)")

    # ── tokenize ──────────────────────────────────────────────────────────────
    print("\n[3/4] Tokenizing (no truncation, EOS appended) ...")
    ds = ds.map(
        _tokenize_batch,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1_000,
        writer_batch_size=500,
        num_proc=args.num_workers,
        remove_columns=["text"],
        desc="tokenize",
    )

    # ── chunk ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Chunking into {block_size}-token blocks ...")
    ds = ds.map(
        _group_texts,
        fn_kwargs={"block_size": block_size},
        batched=True,
        batch_size=_GROUP_BATCH_SIZE,
        num_proc=args.num_workers,
        desc="chunk",
    )
    total_tokens = len(ds) * block_size
    print(f"      {len(ds):,} blocks  ({total_tokens:,} tokens)")
    if len(ds) == 0:
        print("ERROR: chunking produced 0 blocks. Check that --block-size is set "
              "correctly and that the input dataset is non-empty.", file=sys.stderr)
        sys.exit(1)
    if "labels" not in ds.column_names:
        ds = ds.add_column("labels", ds["input_ids"])

    # ── optional validation split ─────────────────────────────────────────────
    out_ds: Dataset | DatasetDict
    if args.validation_split > 0.0:
        split = ds.train_test_split(
            test_size=args.validation_split, seed=42, shuffle=True
        )
        out_ds = DatasetDict({"train": split["train"], "validation": split["test"]})
        print(
            f"\n      train      : {len(split['train']):,} blocks"
            f"\n      validation : {len(split['test']):,} blocks"
        )
    else:
        out_ds = ds

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving parquet to {out_dir.resolve()} ...")

    splits: dict[str, Dataset] = (
        dict(out_ds.items()) if isinstance(out_ds, DatasetDict) else {"train": out_ds}
    )
    for split_name, split_ds in splits.items():
        dest = out_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(dest))
        print(f"  Wrote {dest.name}  ({len(split_ds):,} blocks)")

    if args.push_to_hub:
        print(f"\nPushing to Hub: {args.push_to_hub} ...")
        out_ds.push_to_hub(args.push_to_hub, token=args.token)
        print("Pushed.")

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    for split_name, split_ds in splits.items():
        print(
            f"  {split_name:<12} {len(split_ds):>10,} blocks"
            f"  ({len(split_ds) * block_size:,} tokens)"
        )

    schema_ds = next(iter(splits.values()))
    print(f"\nDataset schema : {list(schema_ds.column_names)}")
    print(f"Output         : {out_dir.resolve()}")
    if args.push_to_hub:
        print(f"Hub            : {args.push_to_hub}")
    print()
    print("Pass to continued_pretrain.py with:")
    print(f"  --dataset_name {str(out_dir.resolve())!r}")


if __name__ == "__main__":
    main()
