#!/usr/bin/env bash
# LongBench-v2 RAG baseline run: Qwen3.5-35B first, then Qwen2.5-7B.
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/LongBench/run_all_longbench_rag.sh
set -xu

cd /mnt/shared-storage-user/liudawei/home/verl-new

source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate verl-071
echo "=== Our Env: $(which python) ==="

export SERVE_PORT="${SERVE_PORT:-8000}"
export DASH_PORT="${DASH_PORT:-8265}"


CONFIGS="${CONFIGS:-qwen35-rag}"
SAVE_DIR="${SAVE_DIR:-taskutils/LongBench/results/rag_qwen35_qwen25}"
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_rag_$(date +%Y%m%d_%H%M%S).log"

{
  python taskutils/LongBench/run_rjob_qwen35.py \
    --config "${CONFIGS}" \
    --save_dir "${SAVE_DIR}" \
    --port "${SERVE_PORT}" \
    --dash_port "${DASH_PORT}" \
    --split_by_sub_domain \
    --force
} 2>&1 | tee "${LOG_FILE}"

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh

