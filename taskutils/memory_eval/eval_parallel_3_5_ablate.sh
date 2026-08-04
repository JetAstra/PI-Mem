#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/eval_parallel_3_5_ablate.sh

set -euo pipefail

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo "$(which python)"

nvidia-smi
export RAY_CGRAPH_get_timeout=1800

unset http_proxy https_proxy
cd /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RESULTS_ROOT=results_qwen3.5-35B-parallel-training-free-ablate
ABLATIONS=(
    control
)

for ABLATE_CONFIG in "${ABLATIONS[@]}"; do
    RUN_DIR="${RESULTS_ROOT}/${ABLATE_CONFIG}"
    mkdir -p "$RUN_DIR"

    ABLATE_CONFIG="$ABLATE_CONFIG" \
    ABLATE_RESULTS_ROOT="$RESULTS_ROOT" \
        python run_rjob_qwen35_ablate.py 2>&1 \
        | tee "$RUN_DIR/eval_$(date +%Y%m%d_%H%M%S).txt"
done

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
