#!/bin/bash

# Usage:
#   ./chtc_deploy.sh <CHTC_USERNAME> <experiment_name> <model_config_name> <stimuli_key> <HF_VAR_NAME> [from_cache] [key=value ...]
# Notes:
#   - HF_VAR_NAME is the name of the env var that holds your HF token (e.g., HF_TOKEN)
#   - from_cache is optional ("True" or "False", default False)
#   - any extra key=value pairs are forwarded to python script/exp_battery.py

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

echo "Establishing SSH connection..."
ssh -fN -M -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER"

echo "Ensuring remote directory exists on CHTC..."
ssh -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER" "mkdir -p $REMOTE_DIR"

echo "Creating tarball of $LOCAL_FOLDER..."
tar -cvf "$LOCAL_TARBALL" -C .. "$LOCAL_FOLDER"

echo "Dynamically writing experiment.sh with args..."
cat > experiment.sh << EOF
#!/bin/bash
set -euo pipefail

export PATH="/opt/conda/bin:\$PATH"
STAGING_DIR="/staging/studdiford/causal"

HF_CACHE_DIR="./temp_cache"
MODEL_NAME="$MODEL_CONFIG_NAME"

HF_SAVE_DIR="\$STAGING_DIR/huggingface_save/\${MODEL_NAME}.tar.gz"
HF_TEMP_CACHE_ARCHIVE="/staging/studdiford/causal/temp_cache.tar.gz"

LOG_FILE="causal_plot.log"
exec > >(tee -i "\$LOG_FILE") 2>&1

eval "\$(conda shell.bash hook)"
conda activate myenv

export HF_HOME="\$HF_CACHE_DIR"
export HF_SAVE="\$HF_SAVE_DIR"
export MODEL_DIR="gemma-2-9b"

mkdir -p triton_cache
export TRITON_CACHE_DIR=triton_cache

tar -xf "$LOCAL_TARBALL"

export TRANSFORMERS_CACHE="./temp_cache"
export HF_HOME="./temp_cache"

export HF_TOKEN='$HF_API_KEY'
export HUGGINGFACE_KEY="\$HF_TOKEN"

if [ -z "\$HUGGINGFACE_KEY" ]; then
  echo "Error: Hugging Face API key not set."
  exit 1
fi

echo "\$HUGGINGFACE_KEY" | huggingface-cli login --token "\$HUGGINGFACE_KEY"

if [ "$FROM_CACHE" == "True" ]; then
  if [ -f "\$HF_TEMP_CACHE_ARCHIVE" ]; then
    echo "Preloading temp cache..."
    cp "\$HF_TEMP_CACHE_ARCHIVE" ./temp_cache.tar.gz
    tar -xzf temp_cache.tar.gz
  else
    echo "[WARN] Missing cache archive at \$HF_TEMP_CACHE_ARCHIVE"
  fi
fi

echo "Running: $EXPERIMENT_NAME $MODEL_CONFIG_NAME $STIMULI_KEY $EXTRA_ARGS_STR"
cd "$LOCAL_FOLDER"
python script/exp_battery.py "$EXPERIMENT_NAME" "$MODEL_CONFIG_NAME" "$STIMULI_KEY" $EXTRA_ARGS_STR

# --- bundle results for return ---
cd ..
if [ -d "./results" ]; then
  echo "Bundling results..."
  tar -czf results.tar.gz results
  echo "Created results.tar.gz"
else
  echo "[WARN] No ./results dir to bundle."
fi

# cleanup
rm -rf triton_cache temp_cache temp_cache.tar.gz

echo "Job completed."
EOF

chmod +x experiment.sh

echo "Transferring payload to CHTC..."
scp -o ControlPath=my_chtc_socket "$LOCAL_TARBALL" experiment.sh "$CHTC_USER@$CHTC_SERVER:$REMOTE_DIR/"

echo "Submitting job..."
# Capture ClusterId so we can wait and then fetch outputs.
SUBMIT_OUT=$(ssh -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER" bash -lc "
  export HF_TOKEN='$HF_API_KEY'
  cd $REMOTE_DIR
  tar -xvf \"$LOCAL_TARBALL\"
  # Ensure exp_run.sub contains transfer_output_files=results.tar.gz, when_to_transfer_output=ON_EXIT
  condor_submit $LOCAL_FOLDER/exp_run.sub
")
echo "\$SUBMIT_OUT"

# Parse ClusterId (works with standard condor_submit output)
CLUSTER_ID=$(echo "$SUBMIT_OUT" | sed -n 's/.*cluster \([0-9]\+\).*/\1/p' | head -n1)
if [ -z "\$CLUSTER_ID" ]; then
  echo "[WARN] Could not parse ClusterId; skipping wait/merge."
  ssh -S my_chtc_socket -O exit "$CHTC_USER@$CHTC_SERVER"
  rm -f "$LOCAL_TARBALL"
  exit 0
fi
echo "Submitted ClusterId: \$CLUSTER_ID"

echo "Waiting for job completion on CHTC..."
ssh -S my_chtc_socket "$CHTC_USER@$CHTC_SERVER" bash -lc "
  cd $REMOTE_DIR
  condor_wait -echo chtc.\$CLUSTER_ID.*.log
"

echo "Fetching results.tar.gz to local machine (if present)..."
scp -o ControlPath=my_chtc_socket "$CHTC_USER@$CHTC_SERVER:$REMOTE_DIR/results.tar.gz" ./ || echo "[INFO] No results.tar.gz found on submit host."

echo "Closing SSH control connection..."
ssh -S my_chtc_socket -O exit "$CHTC_USER@$CHTC_SERVER"

# --- Merge step on local source machine ---
if [ -f results.tar.gz ]; then
  echo "Unpacking results.tar.gz to a temp directory..."
  TMP_UNPACK="./_results_tmp_unpack"
  rm -rf "\$TMP_UNPACK"
  mkdir -p "\$TMP_UNPACK"
  tar -xzf results.tar.gz -C "\$TMP_UNPACK"

  # Show what would be new (xref)
  echo "Diff (new files that would be added):"
  rsync -av --ignore-existing --dry-run "\$TMP_UNPACK/results/" "./results/" | sed -n 's/^>f..t.... //p'

  echo "Copying only new files into ./results/ (preserving structure)..."
  rsync -av --ignore-existing "\$TMP_UNPACK/results/" "./results/"

  echo "Clean up temp unpack."
  rm -rf "\$TMP_UNPACK"
  echo "Merge complete."
else
  echo "[INFO] No results.tar.gz locally; nothing to merge."
fi

echo "Cleaning local tarball..."
rm -f "$LOCAL_TARBALL"
echo "Done."
