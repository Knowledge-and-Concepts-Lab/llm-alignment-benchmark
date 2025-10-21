#!/bin/bash


set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <CHTC_USERNAME> <experiment_name> <model_config_name> <stimuli_key> <HF_VAR_NAME> [from_cache] [key=value ...]"
    exit 1
fi

CHTC_USER="$1"
EXPERIMENT_NAME="$2"
MODEL_CONFIG_NAME="$3"
STIMULI_KEY="$4"
HF_VAR_NAME="$5"

FROM_CACHE="False"
shift 5
if [ "$#" -gt 0 ] && { [ "${1:-}" = "True" ] || [ "${1:-}" = "False" ]; }; then
    FROM_CACHE="$1"
    shift 1
fi
EXTRA_ARGS=( "$@" )
EXTRA_ARGS_STR="${EXTRA_ARGS[*]}"

# Resolve HF token from env var name
HF_API_KEY="${!HF_VAR_NAME:-}"
if [ -z "$HF_API_KEY" ]; then
    echo "Error: Hugging Face API key not found in env var $HF_VAR_NAME"
    exit 1
fi

CHTC_SERVER="ap2002.chtc.wisc.edu"
REMOTE_DIR="~/llm-alignment-benchmark"

LOCAL_FOLDER="$(basename "$PWD")"
LOCAL_TARBALL="${LOCAL_FOLDER}.tar"

echo "Creating tarball of $LOCAL_FOLDER..."
tar -cvf "$LOCAL_TARBALL" -C .. "$LOCAL_FOLDER"

echo "Dynamically writing experiment.sh with args..."
cat > experiment.sh << EOF
#!/bin/bash
set -euo pipefail

export PATH="/opt/conda/bin:\$PATH"
eval "\$(conda shell.bash hook)"
conda activate myenv

tar -xf "$LOCAL_TARBALL"
cd "$LOCAL_FOLDER"

export HF_TOKEN='$HF_API_KEY'
export HUGGINGFACE_KEY="\$HF_TOKEN"

echo "Logging into Hugging Face..."
echo "\$HUGGINGFACE_KEY" | huggingface-cli login --token "\$HUGGINGFACE_KEY"

echo "Running: $EXPERIMENT_NAME $MODEL_CONFIG_NAME $STIMULI_KEY $EXTRA_ARGS_STR"
python script/exp_battery.py "$EXPERIMENT_NAME" "$MODEL_CONFIG_NAME" "$STIMULI_KEY" $EXTRA_ARGS_STR

echo "[DEBUG] Finished experiment, results (if any) in: \$(pwd)/results"

# cleanup scratch
rm -rf triton_cache temp_cache temp_cache.tar.gz

echo "Job completed."
EOF

chmod +x experiment.sh

echo "Establishing SSH connection..."
ssh -fN -M -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER"

echo "Ensuring remote directory exists on CHTC..."
ssh -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER" "mkdir -p $REMOTE_DIR"

echo "Transferring payload to CHTC..."
scp -o ControlPath=my_chtc_socket "$LOCAL_TARBALL" experiment.sh "$CHTC_USER@$CHTC_SERVER:$REMOTE_DIR/"

echo "Submitting job..."
ssh -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER" "
  cd $REMOTE_DIR
  tar -xvf \"$LOCAL_TARBALL\"
  condor_submit $LOCAL_FOLDER/exp_run.sub
  rm -rf $LOCAL_FOLDER
"

ssh -S my_chtc_socket -O exit "$CHTC_USER@$CHTC_SERVER"

echo "Job submitted. Results will remain on submit host under $REMOTE_DIR/$LOCAL_FOLDER/results/"
echo "Done."
