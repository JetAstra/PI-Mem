#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel1.sh

# set -xeuo pipefail
set -xu
unset http_proxy https_proxy

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
# conda activate /mnt/shared-storage-user/liudawei/envs/qwenlongl1_5
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo $(which python)
cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

STEPS=(240)

for STEP in "${STEPS[@]}"; do
    LOG_DIR=results_qwen2.5-7B-parallel-CORRECT/step_${STEP}
    mkdir -p $LOG_DIR

    STEP="$STEP" python run_rjob_qwen25.py 2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M%S).txt

done

# bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar.sh