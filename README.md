# Multilingual-Code-Generator
This project builds upon the work of: [*IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators*](https://arxiv.org/pdf/2403.03894).
We are doing Option 2. We are improving on the solution by getting better results.


## Setup
This section outlines basics to set things up. *It is also a work in progress*

### Environment Variables
To prevent API keys being accidentally uploaded to git, they should be stored as environment variables.
The following environment variables must be set to use the continued pretrain script.
* HF_TOKEN
* WANDB_API_KEY

### Accepting Terms
TO access starcoder from Hugging Face, you must
1. Log into HF
2. Visit https://huggingface.co/bigcode/starcoderbase-1b and click *Agree ane access repository*
3. Visit https://huggingface.co/bigcode/starcoderbase-3b and click *Agree ane access repository*
4. Visit https://huggingface.co/bigcode/starcoderbase-7b and click *Agree ane access repository*


## Train Image
To allow a consistent image for training purposes, a base singularity (apptainer) image is provided. The original authors provided a docker image to use, but it had two issues: 1. singularity is preferred to docker in HPC environments. 2. The docker image the original authors provide did not include python package versions and has broken with time.

### V100 Image
We provide an image specifically tuned to work with an Nvidia Tesla V100. It is tunned to run as similar to the original docker image, but with multiple benefits:
* Based on an Nvidia base image instead of an arbitrary Ubuntu image.
    * This required a very particular version as V100 support was dropped around late 2024.
* Updated PyTorch version
    * This was needed to allow SDPA Attention.
* Pinned all major package versions
    * Allows for significantly better reproducibility. 

### V100 Bit Image
This image is built similar to the V100 image, but includes the bitnet python package for training.

### Compiling an Image
Images can be compiled with the `Misc/build_image.slurm` script. To compile the ircoder_v100.def file run
```bash
sbatch Misc/build_image.slurm Misc/ircoder_v100.def"
```
This will produce a `.sif` file that you can use for training.


## Continued Pre-Training
This section details running continued pre-training

### Python Modificaitons
Continued Pre-Training involves using a modified version of the original author's `Training_Scripts/continued_pretrain.py`. However, the script has been modified for multiple reasons:
* Deprecated function replaced
    * To update to newer versions of some packages the updated equivalents of some deprecated functions used by the authors have been used.
* Specify SDPA Attention
    * This is done to improve runtime
* Used NormalFloat-4 Quantization for forward pass
    * Improves runtime and decreases memory usage
* Added Hugging Face Uploading
* Added LoRA config for BitNet-b1.58

### Tesla V100 Limitations
The Tesla V100 slowed much of our development and limited what we could test. Here are a few ways we were limited:
* No BF16 support
* No Flash Attention
    * This was attempted. There are repos that claim to backport Flash Attention to the V100, but none of them seemed to play well with our singularity container.
* May have Broken DeepSpeed
    * We could not get DeepSpeed to work in a reasonable amount of time, so it was dropped
* May have prevented QLoRA
    * This also didn't play nicely with the environemnt
* **Limited Max Sequence Length**
    * This likely greatly hurt our results. We were unable to increase sequence length due to VRAM and runtime limitations.

### Running Continued Pre-Train
A script was created to submit the training run using SLURM and the desired training image. The code was also modified to support Json files for arguments. These arguments life in the `models` folder and archive the exace models we trained.

To run, use the following:
```bash
Training_Scripts/pretrain_slurm.sh [--interactive | --batch] <sif_path> <scripts_dir> <training_json>
```
For an example run try:
```bash
Training_Scripts/pretrain_slurm.sh --interactive Misc/ircoder_v100.sif Training_Scripts/ models/example.json
```

### Getting Results
The results will be live uploaded to your Weights and Biases account. This is the easiest way to monitor a run's progress and check train/val metrics.

The model weights will also be uploaded to Hugging Face if you provide the `hf_group` argument (and have the proper token to upload there).


## Submission History
**Submission 1: Introduction & Initial commit**
- Stephen McCabe: paper write up/introduction
- Anuj Patnaik: abstract write up/ initial code set up
- Bradley Gore: ran the code with the dockerfile. Did the environment setup. Wrote the slurm file

**Submission 2: Methods & First Training Attempts**
- Stephen McCabe: Report sections written (experimental setup, datasets, evaluation setup), initial dataset, previous report sections revised/completed (related work)
- Anuj Patnaik: Worked on the Parameter Efficient Fine Tuning and the Experimental Setup in the paper
- Bradley Gore: Created an improved training image. Created scripts for running training scripts using HPC slurm. Got continued-pretrain to run. Performed initial testing using FP16 to try to speed up training speed.

**Submission 3: Training & Evaluation Runs**
- Stephen McCabe: Set up dataset pipeline. Compiled and preprocessed mixed source dataset for the continued pre-training. Some more sections of the report written, as much as is possible with results data still on the way.
- Anuj Patnaik: Created perturbed datasets for each of the tasks. Also, ran the evaluate model scripts to come up with the benchmarks. There is a slight issue with getting the actual values. The results will be posted by the next submission.
- Bradley Gore:Added NF4 quantization. Switched from random Ubuntu image to Nvidia based pytorch image. Updated packages to work with newer image, not break code, and add support for SDPA. Sucessfully pre-trained starcoder-1b on ir dataset with 250M Tokens.

**Submission 4: More Evaluation & BitNet**
- Stephen McCabe: Data pipeline improvements for ease of use. Data preprocessing for microsoft model training. Additional edits and writing for ethical considerations.
- Anuj Patnaik: Ran the evaluation model scripts for pass 1 for each of the tasks with the starcoder model. Still waiting on the results for the ircoder model we developed. Added my results to Table 1 on the report.
- Bradley Gore: Added build environments and settings to train microsoft/bitnet-b1.58-2B-4T-bf16. Updated source code and added more config options. Debugged and began pretraining for bitnet. Wrote first draft of Limitations and Ethical Considerations sections of the paper.

**Submission 5: Presentation**
- Stephen McCabe: Presentation visuals and content, writing and editing for slide deck. Failed to fix MultiPL-E docker image, rip.
- Anuj Patnaik: Ran evaluation model scripts for the BITNET base and the BITNET IR. Still waiting on one of the results for BITNET base. Also, I worked on the slides. 
- Bradley Gore: Worked on slide content. Added visuals to slides. Outlined Presentation. Looked into the different ReCode codebases (provided by paper, vs original) to try to find why our validation scores differ.
