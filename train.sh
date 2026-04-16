# bash /mnt/shared-storage-user/dllm-share/liudawei/verl/train.sh
export PROJ_DIR="/mnt/shared-storage-user/dllm-share/liudawei/verl"
cd "${PROJ_DIR}"
source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/dllm-share/songhaixu/miniforge3/envs/qwenlongl1_5
export PYTHONPATH="/mnt/shared-storage-user/dllm-share/liudawei/verl"

which python
python -c "import verl; print(verl)"

export PORT=6381
export WANDB_MODE=offline
export WANDB_DIR="${PROJ_DIR}/wandb_local_logs"
export WANDB_PROJECT="QwenLong-L1.5"

export LLM_JUDGE=Y
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VERIFIER_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-7B-Instruct/"
export VERIFIER_HOST="100.97.88.235"
export VERIFIER_PORT="23547"


# 关键
# ===================== 函数和核心逻辑（无修改） =====================
check_ray_status() {
    until ray status >/dev/null 2>&1; do
        echo "Waiting for Ray cluster to be ready..."
        sleep 5
    done
}

ray status
echo "Starting single-node Ray cluster with 1 GPUs..."
ray start --head --port=${PORT} --num-gpus=8 --num-cpus=128 --include-dashboard=false
check_ray_status
echo "Ray single-node cluster started successfully (1 GPUs)"

echo "Starting RL task (WANDB logs saved locally)..."
bash ${PROJ_DIR}/examples/grpo_trainer/run_qwen3moe-30b_fsdp.sh 2>&1 | tee ${PROJ_DIR}/logs/rl_log_$(date +%Y%m%d_%H%M%S).txt

ray stop
echo "RL task finished, Ray cluster stopped"
echo "WANDB local logs saved to: ${WANDB_DIR:-./wandb}"

wait