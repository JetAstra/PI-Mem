#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5_debug.sh

# set -xeuo pipefail
set -x

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo $(which python)

cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# export CUDA_VISIBLE_DEVICES=0,1,2,3
STEPS=(46)

for STEP in "${STEPS[@]}"; do
    # LOG_DIR=results_qwen3.5-35B-parallel-boxed/step_${STEP}
    LOG_DIR=results_interns2-preview-parallel-boxed-debug/step_${STEP}
    mkdir -p $LOG_DIR

    STEP="$STEP" python run_rjob_qwen35_debug.py 2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M).txt

done

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
# bash /mnt/shared-storage-user/liudawei/home/Self-Distillation/scripts/train_sft_tooluse_sweep1.sh
