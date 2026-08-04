#!/usr/bin/env bash
set -xu
# bash /mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_eval/run_rag_infmem_full_qwen35_first.sh
REPO_ROOT="/mnt/shared-storage-user/liudawei/home/verl-new"
CONDA_SH="/mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh"

cd "${REPO_ROOT}/taskutils/memory_eval"

source "${CONDA_SH}"
conda activate verl-071

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

# Full RULER OOD + full HQA. Do not inherit a previous ad-hoc TASK_SPEC.
export TASK_MODE="${TASK_MODE:-full}"
unset TASK_SPEC

export NUM_SAMPLES="${NUM_SAMPLES:-64}"
export N_PROC="${N_PROC:-128}"

# Run Qwen3.5 first. Override CONFIG_FILTER before invoking this script if needed.
export CONFIG_FILTER="${CONFIG_FILTER:-qwen35_infmem,qwen35_rag,qwen25_infmem,qwen25_rag}"

# run_rag_infmem.py will write one timestamped log per config under this tree.
export SPLIT_CONFIG_LOGS="${SPLIT_CONFIG_LOGS:-1}"
export RAG_INFMEM_LOG_ROOT="${RAG_INFMEM_LOG_ROOT:-results_rag_infmem/logs}"
export RUN_LABEL="${RUN_LABEL:-full_qwen35_first_ns${NUM_SAMPLES}_np${N_PROC}}"

if [[ "${CLEAN_START:-1}" == "1" ]]; then
  serve shutdown -a http://localhost:${DASH_PORT:-8265} <<<'y' >/dev/null 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
fi

echo "CONFIG_FILTER=${CONFIG_FILTER}"
echo "TASK_MODE=${TASK_MODE}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "N_PROC=${N_PROC}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "RAG_INFMEM_LOG_ROOT=${RAG_INFMEM_LOG_ROOT}"

python run_rag_infmem.py

bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh
