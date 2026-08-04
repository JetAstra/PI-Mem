#!/bin/bash
# HQA ablation: keep only the first parallel extraction/merge pass.
# Uses all eight visible GPUs as four TP=2 vLLM replicas.

set -xeuo pipefail
unset http_proxy https_proxy

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo "$(which python)"

cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export ABLATE_CONFIG=parallel_max_passes_1
export STEP=240

LOG_DIR="results_qwen2.5-7B-ablate/parallel-max-passes-1-step_${STEP}"
mkdir -p "$LOG_DIR"
python run_rjob_qwen25_ablate.py 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M).txt"
