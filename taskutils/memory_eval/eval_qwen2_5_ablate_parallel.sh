#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_qwen2_5_ablate_parallel.sh

set -xeuo pipefail
unset http_proxy https_proxy

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate verl-071
echo "$(which python)"

cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export ABLATE_CONFIG=parallel
export STEP=80
LOG_DIR="results_qwen2.5-7B-ablate/parallel-step_${STEP}"
mkdir -p "$LOG_DIR"
python run_rjob_qwen25_ablate.py 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M).txt"
