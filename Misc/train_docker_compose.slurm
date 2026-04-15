#!/bin/bash
#SBATCH --job-name=ircoder_build
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=build_%j.log

# Stop the script if any command fails
set -e

# 1. Setup paths
BUILD_DIR="/dev/shm/$USER/podman_temp"
FINAL_DEST="/WAVE/scratch/CSEN-346-Sp26/$USER/ircoder-training.tar"

# Create the directories
mkdir -p $BUILD_DIR/root $BUILD_DIR/run $BUILD_DIR/xdg
mkdir -p $(dirname $FINAL_DEST)

# 2. Set Environment Variables
export MAX_JOBS=$SLURM_CPUS_PER_TASK
# fix shell issues when runing docker file with podman
export BUILDAH_FORMAT=docker
# fix podman runtime not using correct location
export XDG_RUNTIME_DIR="$BUILD_DIR/xdg"

# IMPORTANT: Unset this to prevent crun from trying to use D-Bus/systemd
unset DBUS_SESSION_BUS_ADDRESS

# 3. The Podman Build Command (Now with cgroupfs and events-backend)
echo "Starting build..."
podman --root $BUILD_DIR/root \
       --runroot $BUILD_DIR/run \
       --cgroup-manager=cgroupfs \
       --events-backend=file \
       --storage-driver overlay \
       --storage-opt "overlay.mount_program=/usr/bin/fuse-overlayfs" \
       --storage-opt "overlay.force_mask=700" \
       build -t ircoder-training:latest -f Misc/Training.Dockerfile .

# 4. Export the image
echo "Saving image to $FINAL_DEST..."
podman --root $BUILD_DIR/root save ircoder-training:latest -o $FINAL_DEST

# 5. Cleanup
rm -rf $BUILD_DIR
echo "Job complete."