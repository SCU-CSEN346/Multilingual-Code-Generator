#!/bin/bash
#SBATCH --job-name=build_ircoder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build_%j.log

# --- 1. RESOLVE ABSOLUTE PATHS ---
# readlink -f turns any relative path into a full absolute path
# This allows you to run the script from anywhere.
if [ -z "$1" ]; then
    echo "Usage: sbatch build_image.slurm <path_to_def_file>"
    exit 1
fi

ABS_DEF_PATH=$(readlink -f "$1")
DEF_DIR=$(dirname "$ABS_DEF_PATH")
DEF_FILENAME=$(basename "$ABS_DEF_PATH")
SIF_FILENAME="${DEF_FILENAME%.def}.sif"

# --- 2. PERMISSION CHECK ---
if [ ! -r "$ABS_DEF_PATH" ]; then
    echo "ERROR: Compute node $(hostname) cannot read the file at:"
    echo "       $ABS_DEF_PATH"
    echo "       Please ensure the folder has 'read' and 'execute' permissions for others (chmod 755)."
    exit 1
fi

# --- 3. LOCAL WORKSPACE SETUP ---
# We use /tmp to bypass network filesystem (NFS/Lustre) 'fakeroot' issues
NODE_TMP="/tmp/${USER}_build_${SLURM_JOB_ID}"
mkdir -p "$NODE_TMP"
export SINGULARITY_TMPDIR="$NODE_TMP"
export SINGULARITY_CACHEDIR="$NODE_TMP"

# --- 4. THE BUILD ---
echo ">>> User: $USER"
echo ">>> Building from: $ABS_DEF_PATH"
echo ">>> Outputting to: $DEF_DIR/$SIF_FILENAME"

# Copy to local node to ensure Singularity has zero permission issues during build
cp "$ABS_DEF_PATH" "$NODE_TMP/"

cd "$NODE_TMP"
singularity build --fakeroot "$SIF_FILENAME" "$DEF_FILENAME"

# --- 5. MOVE PRODUCT TO ORIGINAL LOCATION ---
if [ -f "$SIF_FILENAME" ]; then
    echo ">>> Build successful. Moving image to $DEF_DIR/"
    mv "$SIF_FILENAME" "$DEF_DIR/"
else
    echo ">>> FATAL: Build failed. Check the logs above for compiler errors."
    exit 1
fi

# Cleanup
rm -rf "$NODE_TMP"