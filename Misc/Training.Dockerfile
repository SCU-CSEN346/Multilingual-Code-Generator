# This file edited from ircoder with the help of Gemini 3 pro
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# 1. Setup Global Environment Variables
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/Miniforge/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# 2. Setup System Utilities (Full IRCoder list)
RUN apt-get update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        apt-utils autoconf automake bc build-essential ca-certificates check cmake \
        curl dmidecode emacs g++ gcc git iproute2 jq kmod libaio-dev \
        libcurl4-openssl-dev libgl1-mesa-glx libglib2.0-0 libgomp1 libibverbs-dev \
        libnuma-dev libnuma1 libomp-dev libsm6 libssl-dev libsubunit-dev \
        libsubunit0 libtool libxext6 libxrender-dev make moreutils net-tools \
        ninja-build openssh-client openssh-server openssl pkg-config python3-dev \
        software-properties-common sudo unzip util-linux vim wget zlib1g-dev \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Setup Mamba and the "Skeleton" Environment
RUN wget -O /tmp/Miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge.sh -b -p /Miniforge \
    && /Miniforge/bin/mamba create -y -n pre-train python=3.11 \
    # Install everything EXCEPT pytorch here
    && /Miniforge/bin/mamba install -y -n pre-train -c conda-forge \
        "setuptools<70" packaging wheel ninja "numpy<2" pandas scikit-learn wandb \
    && /Miniforge/bin/mamba clean -a -f -y

# 4. GUARANTEED GPU PYTORCH (The Fix)
# We use the official PyTorch wheels which are much more reliable than Conda versions
RUN /Miniforge/bin/mamba run -n pre-train pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Diagnostic Step: Verify before the long builds
RUN /Miniforge/bin/mamba run -n pre-train python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'Linked CUDA: {torch.version.cuda}')"
# 6. Install Flash Attention
RUN export MAX_JOBS=10 \
    && export FLASH_ATTENTION_FORCE_BUILD=TRUE \
    && /Miniforge/bin/mamba run -n pre-train pip install flash-attn==2.4.2 --no-build-isolation

# 7. Install optimized NVIDIA Apex
RUN export MAX_JOBS=10 \
    && git clone https://github.com/NVIDIA/apex /tmp/apex \
    && cd /tmp/apex && git checkout 23.08 \
    && /Miniforge/bin/mamba run -n pre-train pip install -v --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        --config-settings "--build-option=--fast_layer_norm" \
        --config-settings "--build-option=--fmha" ./ \
    && rm -rf /tmp/apex

# 8. Install Deepspeed
RUN export MAX_JOBS=10 \
    && export DS_BUILD_AIO=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 \
    && /Miniforge/bin/mamba run -n pre-train pip install deepspeed==0.12.6 --no-build-isolation

# 9. Final IRCoder Dependencies
RUN /Miniforge/bin/mamba run -n pre-train pip install \
    transformers==4.36.2 datasets accelerate peft bitsandbytes==0.42.0 autoawq==0.1.8