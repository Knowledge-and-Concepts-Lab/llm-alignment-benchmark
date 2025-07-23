#!/bin/bash

export PATH="/opt/conda/bin:$PATH"

STAGING_DIR="/staging/studdiford/causal"

HF_CACHE_DIR="./temp_cache"


HF_TEMP_CACHE_DIR="/staging/studdiford/causal/temp_cache_9b.tar.gz"

set -e

LOG_FILE="causal_plot.log"

exec > >(tee -i $LOG_FILE) 2>&1

eval "$(conda shell.bash hook)"
conda activate myenv

##

export HF_HOME="$HF_CACHE_DIR"
export MODEL_DIR="gemma-2-9b"

mkdir triton_cache
export TRITON_CACHE_DIR=triton_cache

tar -xf SAE_concepts.tar

export TRANSFORMERS_CACHE="./temp_cache"
export HF_HOME="./temp_cache"

# ✅ Insert the actual HF API key from the specified environment variable
export HF_TOKEN='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'
export HUGGINGFACE_KEY=$HF_TOKEN

if [ -z "$HUGGINGFACE_KEY" ]; then
    echo "Error: Hugging Face API key not set."
    exit 1
fi

echo "Logging into Hugging Face..."
echo "$HUGGINGFACE_KEY" | huggingface-cli login --token $HUGGINGFACE_KEY

# ✅ Conditionally copy and extract the temp cache if from_cache=True
if [ "False" == "True" ]; then
    echo "Copying and extracting temp cache..."
    cp $HF_TEMP_CACHE_DIR ./
    tar -xzvf temp_cache_9b.tar.gz
fi

echo "Running script with experiment: prompt_1_a, model: gemma_2_9b_it, stimuli: icl_roundthings_size"

cd SAE_concepts
accelerate launch -m script.exp_battery "prompt_1_a" "gemma_2_9b_it" "icl_roundthings_size"

echo "removing model dir"
rm -rf "$MODEL_NAME"
rm -rf "${MODEL_NAME}.tar.gz"

# ✅ Conditionally remove temp cache only if it was used
if [ "False" == "True" ]; then
    rm -rf temp_cache_9b.tar.gz
fi

rm -rf triton_cache
rm -rf temp_cache

echo "Job completed successfully!"