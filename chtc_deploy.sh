#!/bin/bash

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <CHTC_USERNAME> <experiment_name> <model_config_name> <stimuli_key> <HF_VAR_NAME> [from_cache]"
    exit 1
fi

CHTC_USER="$1"
EXPERIMENT_NAME="$2"
MODEL_CONFIG_NAME="$3"
STIMULI_KEY="$4"
HF_VAR_NAME="$5"  # Name of the environment variable containing the HF API key
FROM_CACHE="${6:-False}"  # Default to False if not provided

# Get the Hugging Face API key from the environment variable
HF_API_KEY="${!HF_VAR_NAME}"

if [ -z "$HF_API_KEY" ]; then
    echo "Error: Hugging Face API key not found in environment variable $HF_VAR_NAME"
    exit 1
fi

CHTC_SERVER="ap2002.chtc.wisc.edu"
REMOTE_DIR="~/SAE_concepts"

LOCAL_FOLDER="$(basename "$PWD")"
LOCAL_TARBALL="${LOCAL_FOLDER}.tar"

echo "Establishing SSH connection..."
ssh -fN -M -S my_chtc_socket $CHTC_USER@$CHTC_SERVER

echo "Ensuring remote directory exists on CHTC..."
ssh -S my_chtc_socket $CHTC_USER@$CHTC_SERVER "mkdir -p $REMOTE_DIR"

echo "Creating tarball of $LOCAL_FOLDER..."
tar -cvf $LOCAL_TARBALL -C .. "$LOCAL_FOLDER"

echo "Dynamically writing experiment.sh with args..."
cat > experiment.sh << EOF
#!/bin/bash

export PATH="/opt/conda/bin:\$PATH"

STAGING_DIR="/staging/studdiford/causal"

HF_CACHE_DIR="./temp_cache"
MODEL_NAME="$MODEL_CONFIG_NAME"  # Dynamically insert model config name

HF_SAVE_DIR="\$STAGING_DIR/huggingface_save/\${MODEL_NAME}.tar.gz"

HF_TEMP_CACHE_DIR="/staging/studdiford/causal/temp_cache_9b.tar.gz"

set -e

LOG_FILE="causal_plot.log"

exec > >(tee -i \$LOG_FILE) 2>&1

eval "\$(conda shell.bash hook)"
conda activate myenv

##

export HF_HOME="\$HF_CACHE_DIR"
export HF_SAVE="\$HF_SAVE_DIR"
export MODEL_DIR="gemma-2-9b"

mkdir triton_cache
export TRITON_CACHE_DIR=triton_cache

tar -xf SAE_concepts.tar

export TRANSFORMERS_CACHE="./temp_cache"
export HF_HOME="./temp_cache"

# ✅ Insert the actual HF API key from the specified environment variable
export HF_TOKEN='$HF_API_KEY'
export HUGGINGFACE_KEY=\$HF_TOKEN

if [ -z "\$HUGGINGFACE_KEY" ]; then
    echo "Error: Hugging Face API key not set."
    exit 1
fi

echo "Logging into Hugging Face..."
echo "\$HUGGINGFACE_KEY" | huggingface-cli login --token \$HUGGINGFACE_KEY

# ✅ Conditionally copy and extract the temp cache if from_cache=True
if [ "$FROM_CACHE" == "True" ]; then
    echo "Copying and extracting temp cache..."
    cp \$HF_TEMP_CACHE_DIR ./
    tar -xzvf temp_cache_9b.tar.gz
fi

echo "Running script with experiment: $EXPERIMENT_NAME, model: $MODEL_CONFIG_NAME, stimuli: $STIMULI_KEY"

cd SAE_concepts
accelerate launch -m script.exp_battery "$EXPERIMENT_NAME" "$MODEL_CONFIG_NAME" "$STIMULI_KEY"

echo "removing model dir"
rm -rf "\$MODEL_NAME"
rm -rf "\${MODEL_NAME}.tar.gz"

if [ "$FROM_CACHE" == "True" ]; then
    rm -rf temp_cache_9b.tar.gz
fi

rm -rf triton_cache
rm -rf temp_cache

echo "Job completed successfully!"
EOF

chmod +x experiment.sh  # Ensure script is executable

echo "Transferring files to CHTC server..."
scp -o ControlPath=my_chtc_socket $LOCAL_TARBALL job.submit experiment.sh $CHTC_USER@$CHTC_SERVER:$REMOTE_DIR/

echo "Connecting to CHTC and submitting job..."
ssh -S my_chtc_socket $CHTC_USER@$CHTC_SERVER << EOF
    export HF_TOKEN='$HF_API_KEY'
    cd $REMOTE_DIR
    tar -xvf $LOCAL_TARBALL  # Extract tarball
    condor_submit SAE_concepts/exp_run.sub  # Submit job to HTCondor
    rm -rf SAE_concepts
EOF

ssh -S my_chtc_socket -O exit $CHTC_USER@$CHTC_SERVER

echo "Job submitted successfully!"

rm "$LOCAL_TARBALL"