#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export DASH_PORT="${DASH_PORT:-8265}"
export DATA_ROOT="${DATA_ROOT:-hf://datasets/JetLM/PI-Mem-Data/hotpotqa_eval}"
export MEMORY_DATA_ROOT="${MEMORY_DATA_ROOT:-${REPO_ROOT}/taskutils/memory_data}"

# Available, in any comma-separated order:
# vanilla,yarn,rag,memagent-base,memagent-trained,pi-mem-base,pi-mem-trained
CONFIGS="${CONFIGS:-pi-mem-trained}"
# Full evaluation by default. The order below is also the execution order.
TASKS="${TASKS:-hqa,ood}"

# HQA lengths (all by default). Example: HQA_LENGTHS=800,1600,3200
HQA_LENGTHS="${HQA_LENGTHS:-50,100,200,400,800,1600,3200,6400,12800,25600}"

# RULER OOD subsets and lengths (all by default).
# Example: OOD_TASKS=qa_1,vt OOD_LENGTHS=8192,16384
OOD_TASKS="${OOD_TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,fwe,qa_1}"
OOD_LENGTHS="${OOD_LENGTHS:-8192,16384,32768,65536,131072,262144,524288,1048576}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_qwen25}"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

export QWEN25_YARN_MODEL="${QWEN25_YARN_MODEL:-${REPO_ROOT}/models/Qwen2.5-7B-Instruct-yarn}"

python taskutils/memory_eval/run_qwen25.py \
  --configs "${CONFIGS}" \
  --tasks "${TASKS}" \
  --hqa-lengths "${HQA_LENGTHS}" \
  --ood-tasks "${OOD_TASKS}" \
  --ood-lengths "${OOD_LENGTHS}" \
  --results-dir "${RESULTS_DIR}" \
  --data-root "${DATA_ROOT}" \
  --memory-data-root "${MEMORY_DATA_ROOT}" \
  --port "${SERVE_PORT}" \
  --dash-port "${DASH_PORT}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
