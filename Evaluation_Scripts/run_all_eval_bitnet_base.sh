#!/bin/bash
BASE_DIR="/WAVE/scratch2/CSEN-346-Sp26/Multilingual-Code-Generator/recode/datasets/perturbed/humaneval/full"

find "$BASE_DIR" -name "*_s0.jsonl" | while read FILE; do

    task_name=$(basename "$FILE" .jsonl)

    category=$(basename "$(dirname "$FILE")")

    sbatch \
        --export=TEST_FILE="$FILE",TASK_NAME="$task_name",CATEGORY="$category" \
        run_recode_bitnet_base_s0.slurm

done
