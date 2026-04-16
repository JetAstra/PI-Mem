#!/bin/bash
# bash /mnt/shared-storage-user/dllm-share/liudawei/verl/run_memory_qwen3moe-30b_parallel_grpo.sh
set -x
set -e
source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/dllm-share/songhaixu/miniforge3/envs/qwenlongl1_5

cd /mnt/shared-storage-user/dllm-share/liudawei/verl/
echo "$(which python)"

export LLM_JUDGE=Y
export VERIFIER_PATH=/mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-32B-Instruct/
export VERIFIER_HOST=100.97.88.235
export VERIFIER_PORT=23547

export WANDB_MODE=offline
export WANDB_DIR="/mnt/shared-storage-user/dllm-share/liudawei/verl/"
# export CUDA_LAUNCH_BLOCKING=1
# pick a writable, mounted place
export WANDB_DATA_DIR=/mnt/shared-storage-user/dllm-share/liudawei/wandb_data
export WANDB_CACHE_DIR=/mnt/shared-storage-user/dllm-share/liudawei/wandb_cache  # keep yours
export WANDB_CONFIG_DIR=/mnt/shared-storage-user/dllm-share/liudawei/wandb_config # keep yours
export XDG_DATA_HOME=/mnt/shared-storage-user/dllm-share/liudawei/.local/share

mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$XDG_DATA_HOME"

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=4
NGPUS_PER_NODE=8
PROJ_ROOT=/mnt/shared-storage-user/liudawei/songhaixu/checkpoints

MODEL_PATH=/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-30B-A3B
TRAIN_PATH="/mnt/shared-storage-user/dllm-share/liudawei/verl/data/hotpotqa/hotpotqa_train_32k.parquet"
VAL_PATH="/mnt/shared-storage-user/dllm-share/liudawei/verl/data/hotpotqa/hotpotqa_dev.parquet"
EXPERIMENT_NAME="Qwen3-30BA3B-8GPU-4nodes-parallel-grpo"
mkdir -p "${PROJ_ROOT}/${EXPERIMENT_NAME}"

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=32000
CHUNK_SIZE=8000
MAX_NEW_TOKEN=4000
MEMORY_LEN=4000
MERGE_LEN=8000

# export HYDRA_FULL_ERROR=1
# recurrent.memory.config.max_prompt_length 只有一小段 query 长度
# main_ppo 的 overlong cfg 传不进去，可能要改代码
python -u -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=$CHUNK_SIZE \
    recurrent.memory.config.max_prompt_length=512 \
    recurrent.memory.config.max_memorization_length=$MEMORY_LEN \
    recurrent.memory.config.max_final_response_length=$MAX_NEW_TOKEN \
    recurrent.memory.config.max_passes=3 \
    recurrent.memory.config.max_merge_length=$MERGE_LEN \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.truncation='middle' \
    data.prompt_key=prompt \
    +data.context_key='context' \
    data.filter_overlong_prompts_workers=96 \
    data.filter_overlong_prompts=True \
    data.train_batch_size=128 \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    +algorithm.filter_groups.enable=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=102400 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=102400 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=102400 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
)    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.model.use_liger=True \
    reward_model.reward_manager='dapo_parallel' \
    trainer.logger=['console','wandb'] \
    trainer.project_name='QwenLong-L1-MemAgent' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=-1 \
    trainer.save_freq=5 \
    trainer.rollout_data_dir=${PROJ_ROOT}/${EXPERIMENT_NAME}/rollout \
    trainer.default_local_dir=${PROJ_ROOT}/${EXPERIMENT_NAME}/ckpt \
    trainer.resume_mode=auto \
    trainer.total_epochs=30 "${@:1}" \
    2>&1 | tee ${PROJ_ROOT}/${EXPERIMENT_NAME}/memagent_log_$(date +%Y%m%d_%H%M%S).txt