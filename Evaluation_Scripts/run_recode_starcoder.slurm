#!/bin/bash
#SBATCH --job-name=recode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HUB_DISABLE_TORCHVISION=1

module purge

module load PyTorch/2.10.0-Python-3.12-CPU

cd /WAVE/scratch2/CSEN-346-Sp26/Multilingual-Code-Generator/recode
mkdir -p logs
source ~/venvs/humaneval/bin/activate
METHOD=${METHOD:-format}
SEED=${SEED:-0}

MODEL="starcoderbase-1b"


OUTPUT_DIR="${OUTROOT}/${MODEL}/full/humaneval/${CATEGORY}/${TASK_NAME}"

mkdir -p "$OUTPUT_DIR"
python3 evaluate-public-models/evaluate_model.py \
    --model_name_or_path bigcode/$MODEL \
    --tokenizer_name bigcode/$MODEL \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --programming_lang python \
    --num_samples_per_example 1 \
    --use_stopping_criteria \
    --override_previous_results

python3 evaluate-public-models/evaluate_model.py \
    --model_name_or_path bigcode/$MODEL \
    --tokenizer_name bigcode/$MODEL \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --programming_lang python \
    --run_eval_only

echo "DONE: $METHOD / s$SEED"
