#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export DASH_PORT="${DASH_PORT:-8265}"

# Available, in any comma-separated order:
# vanilla,yarn,rag,memagent-base,memagent-trained,pi-mem-base,pi-mem-trained
CONFIGS="${CONFIGS:-pi-mem-trained}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/results/qwen35}"
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# Override these only when using the local Yarn or MemAgent-trained checkpoints.
export QWEN35_YARN_MODEL="${QWEN35_YARN_MODEL:-${REPO_ROOT}/models/Qwen3.5-35B-A3B-yarn}"
export QWEN35_MEMAGENT_MODEL="${QWEN35_MEMAGENT_MODEL:-${REPO_ROOT}/models/Qwen3.5-35B-A3B-MemAgent}"

python taskutils/LongBench/run_qwen35.py \
  --configs "${CONFIGS}" \
  --save-dir "${SAVE_DIR}" \
  --port "${SERVE_PORT}" \
  --dash-port "${DASH_PORT}" \
  --split-by-sub-domain \
  "$@" 2>&1 | tee "${LOG_FILE}"
