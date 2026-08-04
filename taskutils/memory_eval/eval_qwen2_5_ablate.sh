#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_qwen2_5_ablate.sh

# set -xeuo pipefail
set -xu
unset http_proxy https_proxy

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
# conda activate /mnt/shared-storage-user/liudawei/envs/qwenlongl1_5
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo $(which python)
cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# export ABLATE_CONFIG=parallel
# export STEP=240
# LOG_DIR="results_qwen2.5-7B-ablate/parallel-step_${STEP}"
# mkdir -p "$LOG_DIR"
# python run_rjob_qwen25_ablate.py 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M).txt"

export ABLATE_CONFIG=parallel_no_merge
export STEP=240
LOG_DIR="results_qwen2.5-7B-ablate/parallel-no-merge-step_${STEP}"
mkdir -p "$LOG_DIR"
python run_rjob_qwen25_ablate.py 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M).txt"

export ABLATE_CONFIG=parallel_no_check
LOG_DIR="results_qwen2.5-7B-ablate/parallel-no-check-step_${STEP}"
mkdir -p "$LOG_DIR"
python run_rjob_qwen25_ablate.py 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M).txt"

# export ABLATE_CONFIG=memagent
# LOG_DIR=results_qwen2.5-7B-ablate/RL-MemAgent-7B
# mkdir -p "$LOG_DIR"
# python run_rjob_qwen25_ablate.py 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M).txt"


bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
