#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5.sh

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
STEPS=(80)
RUNS=(run{1..40})

for STEP in "${STEPS[@]}"; do
    COMMON_RUN_DIR=results-qwen3.5-openai-repeat/qwen3.5-openai/repeat_runs

    for RUN_NAME in "${RUNS[@]}"; do
        RUN_DIR=${COMMON_RUN_DIR}/${RUN_NAME}
        mkdir -p "$RUN_DIR"

        STEP="$STEP" RUN_NAME="$RUN_NAME" python run_rjob_qwen35.py 2>&1 \
            | tee "$RUN_DIR/training_$(date +%Y%m%d_%H%M).txt"
    done
done

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
