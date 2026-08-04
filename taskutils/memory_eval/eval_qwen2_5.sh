#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_qwen2_5.sh


# set -xeuo pipefail
set -xu

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo $(which python)

nvidia-smi
export RAY_CGRAPH_get_timeout=1800

unset http_proxy https_proxy
cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# export CUDA_VISIBLE_DEVICES=0,1,2,3
STEPS=(240)

for STEP in "${STEPS[@]}"; do
    # LOG_DIR=results_qwen3.5-35B-parallel-boxed/step_${STEP}
    LOG_DIR=results-multivalue
    mkdir -p $LOG_DIR

    STEP="$STEP" python run_rjob_qwen25.py 2>&1 | tee $LOG_DIR/training_$(date +%Y%m%d_%H%M).txt

done

bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5.sh

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
