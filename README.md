# Multilingual-Code-Generator
This project builds upon the work of: [*IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators*](https://arxiv.org/pdf/2403.03894).
We are doing Option 2. We are improving on the solution by getting better results.

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
- Stephen McCabe: ...
- Anuj Patnaik: ...
- Bradley Gore:Added NF4 quantization. Switched from random Ubuntu image to Nvidia based pytorch image. Updated packages to work with newer image, not break code, and add support for SDPA. Sucessfully pre-trained starcoder-1b on ir dataset with 250M Tokens.


## SETUP
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
