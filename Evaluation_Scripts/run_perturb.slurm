#!/bin/bash

#SBATCH --job-name=perturb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
module purge
module load PyTorch/2.10.0-Python-3.12
cd /WAVE/scratch2/CSEN-346-Sp26/Multilingual-Code-Generator/recode
mkdir -p logs

METHODS="natgen"
SEEDS=0

if [[ "$METHOD" == "func_name" ]]; then
    TASK="code_generation"
    AUG_METHOD=0
elif [[ "$METHOD" == "format" ]]; then
    TASK="partial_code"
    AUG_METHOD=0
elif [[ "$METHOD" == "natgen" ]]; then
    TASK="partial_code"
    AUG_METHOD=0
fi
python3 perturb.py \
    --data humaneval \
    --task "$TASK" \
    --method "$METHOD" \
    --aug_method "$AUG_METHOD" \
    --seed "$SEED" \
    --overwrite
