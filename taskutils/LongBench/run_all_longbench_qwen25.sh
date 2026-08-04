#!/usr/bin/env bash
# Qwen2.5-7B LongBench-v2 run: vanilla, yarn, recurrent RL, parallel RL, parallel RL v4.
# Submit this as one GPU job.
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/LongBench/run_all_longbench_qwen25.sh
set -xu

cd /mnt/shared-storage-user/liudawei/home/verl-new

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071
echo "=== Our Env: $(which python) ==="

export SERVE_PORT="${SERVE_PORT:-8000}"
export DASH_PORT="${DASH_PORT:-8265}"

SAVE_DIR="taskutils/LongBench/results/qwen25_all_configs"
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_qwen25_$(date +%Y%m%d_%H%M%S).log"

# --config qwen25-parallel-rl-v4,qwen25-parallel-rl,qwen25-recurrent-rl,qwen25-vanilla,qwen25-vanilla-yarn4.0
{
  python taskutils/LongBench/run_rjob_qwen35.py \
    --config qwen25-vanilla-yarn4.0,qwen25-vanilla \
    --save_dir "${SAVE_DIR}" \
    --port "${SERVE_PORT}" \
    --dash_port "${DASH_PORT}" \
    --split_by_sub_domain \
    --force
} 2>&1 | tee "${LOG_FILE}"
