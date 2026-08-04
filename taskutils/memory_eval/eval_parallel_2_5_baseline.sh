#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_2_5_baseline.sh
# set -xeuo pipefail
set -xu

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071    # vllm 0.18.0
# conda activate /mnt/shared-storage-user/liudawei/envs/qwenlongl1_5  # vllm 0.7.0
echo $(which python)

cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


# LOG_DIR=results_qwen3.5-35B-parallel-boxed/qwen3.5-35B-baseline-boxed
LOG_DIR=logs/mix_baseline-qwen2.5_recurrent_boxed_debug
# LOG_DIR=results_qwen3.5-35B-bsz1
mkdir -p $LOG_DIR

python run_rjob_qwen25_baseline.py 2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M).txt

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5.sh
# bash /mnt/shared-storage-user/liudawei/home/Self-Distillation/scripts/train_sft_tooluse_sweep1.sh
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5_baseline.sh