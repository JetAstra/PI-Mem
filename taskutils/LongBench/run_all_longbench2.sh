#!/usr/bin/env bash
# Recurrent agent full LongBench-v2 run: base + RL step80.
# Submit this as one 8-GPU job.

set -xu

cd /mnt/shared-storage-user/liudawei/home/verl-new

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo "=== Our Env: $(which python) ==="

export SERVE_PORT="${SERVE_PORT:-8000}"
export DASH_PORT="${DASH_PORT:-8265}"

SAVE_DIR="taskutils/LongBench/results/all_configs_recurrent"
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_recurrent_$(date +%Y%m%d_%H%M%S).log"

{
  python taskutils/LongBench/run_rjob_qwen35.py \
    --config qwen35-recurrent-rl-step80,qwen35-recurrent-base \
    --save_dir "${SAVE_DIR}" \
    --port "${SERVE_PORT}" \
    --dash_port "${DASH_PORT}" \
    --split_by_sub_domain \
    --force \
    --skip_result
} 2>&1 | tee "${LOG_FILE}"

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
