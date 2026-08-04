#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5_baseline.sh

# set -xeuo pipefail
set -xu

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071    # vllm 0.18.0
# conda activate /mnt/shared-storage-user/liudawei/envs/qwenlongl1_5  # vllm 0.7.0
echo $(which python)

cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


# LOG_DIR=results_qwen3.5-35B-parallel-boxed/qwen3.5-35B-baseline-boxed
# LOG_DIR=logs/mix_baseline-Qwen35_35B_15k_4k_recurrent_boxed-Qwen35_35B_openai
LOG_DIR=results_qwen3.5-35B-testspeed/qwen3.5-35B-parallel-baseline-boxed
mkdir -p $LOG_DIR
python run_rjob_qwen35.py 2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M).txt


LOG_DIR=results_qwen3.5-35B-testspeed/qwen3.5-35B-baseline-yarn8.0
mkdir -p $LOG_DIR
python run_rjob_qwen35_yarn.py 2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M).txt


bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh