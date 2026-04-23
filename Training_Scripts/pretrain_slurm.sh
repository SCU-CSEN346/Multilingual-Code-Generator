#!/bin/bash

# --- 1. ARGUMENT HANDLING ---
# We expect 5 args now: [mode] [sif] [scripts] [train_json] [ds_json]
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 [--interactive | --batch] <sif_path> <scripts_dir> <training_json> <deepspeed_json>"
    exit 1
fi

MODE=$1
IMAGE_SIF=$(readlink -f "$2")
HOST_SCRIPTS_DIR=$(readlink -f "$3")
HOST_TRAIN_JSON=$(readlink -f "$4")
HOST_DS_JSON=$(readlink -f "$5")

# --- 2. WORKSPACE SETUP ---
export MY_SCRATCH="/WAVE/scratch/CSEN-346-Sp26/$USER"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
NEW_WORKSPACE="$MY_SCRATCH/$RUN_ID"
FAKE_HOME="$MY_SCRATCH/fake_home"

mkdir -p "$NEW_WORKSPACE/output" "$FAKE_HOME"
cp -r "$HOST_SCRIPTS_DIR/." "$NEW_WORKSPACE/"
cp "$HOST_TRAIN_JSON" "$NEW_WORKSPACE/train_config.json"
cp "$HOST_DS_JSON" "$NEW_WORKSPACE/ds_config.json"

# --- 3. PREPARE THE COMMAND ---
# Flatten JSON to CLI flags (using the container's python)
JSON_FLAGS=$(python3 -c "import json; d=json.load(open('$HOST_TRAIN_JSON')); print(' '.join([f'--{k} {v}' for k,v in d.items()]))")

# Construct the core singularity command
EXEC_CMD="singularity exec --nv -C \
    --env HF_TOKEN=\$HF_TOKEN \
    --env WANDB_API_KEY=\$WANDB_API_KEY \
    --env PYTHONUNBUFFERED=1 \
    --home $FAKE_HOME:/home/$USER \
    --bind $NEW_WORKSPACE:/workspace \
    --pwd /workspace \
    $IMAGE_SIF \
    accelerate launch --num_processes=1 --main_process_port=29699 \
    continued_pretrain.py \
    $JSON_FLAGS \
    --output_dir /workspace/output \
    --deepspeed ds_config.json \
    --token \$HF_TOKEN \
    --wandb_token \$WANDB_API_KEY"

# --- 4. SLURM PARAMETERS ---
# 2-day time limit, high memory for the V100
SLURM_ARGS="--partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=120G --time=2-00:00:00 --job-name=ircoder_$USER"

# --- 5. EXECUTION ---
if [ "$MODE" == "--interactive" ]; then
    echo ">>> Starting INTERACTIVE session (srun)..."
    srun $SLURM_ARGS bash -c "$EXEC_CMD"
elif [ "$MODE" == "--batch" ]; then
    LOG_FILE="$NEW_WORKSPACE/slurm_output_%j.log"
    echo ">>> Submitting BATCH job (sbatch)..."
    echo ">>> Logs will be saved to: $LOG_FILE"
    sbatch $SLURM_ARGS --output="$LOG_FILE" --wrap="$EXEC_CMD"
else
    echo "ERROR: First argument must be --interactive or --batch"
    exit 1
fi