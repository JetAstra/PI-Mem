#!/bin/bash
# bash /mnt/shared-storage-user/liudawei/home/verl/run_parallel_7B.sh
set -x
set -e
source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/dllm-share/songhaixu/miniforge3/envs/qwenlongl1_5

cd /mnt/shared-storage-user/liudawei/home/verl/
echo "$(which python)"

export LLM_JUDGE=N
export VERIFIER_PATH=/mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-32B-Instruct/
export VERIFIER_HOST=100.97.166.235
export VERIFIER_PORT=23547

export VERL_LOGGING_LEVEL=DEBUG

export WANDB_MODE=offline

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=1
NGPUS_PER_NODE=2
PROJ_ROOT=/mnt/shared-storage-user/liudawei/work_dirs_ckpt/verl

MODEL_PATH=/mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-7B-Instruct
TRAIN_PATH="data/hotpotqa/hotpotqa_train_32k.parquet"
VAL_PATH="data/hotpotqa/hotpotqa_dev.parquet"
EXPERIMENT_NAME="Qwen2.5-7B-8GPU-1nodes-parallel-debug"
mkdir -p "${PROJ_ROOT}/${EXPERIMENT_NAME}"

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=32768
CHUNK_SIZE=5000
MAX_NEW_TOKEN=1024
MEMORY_LEN=1024

# export HYDRA_FULL_ERROR=1
# recurrent.memory.config.max_prompt_length 只有一小段 query 长度
# main_ppo 的 overlong cfg 传不进去，可能要改代码
python3 -u -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=$CHUNK_SIZE \
    recurrent.memory.config.max_prompt_length=1024 \
    recurrent.memory.config.max_memorization_length=$MEMORY_LEN \
    recurrent.memory.config.max_final_response_length=$MAX_NEW_TOKEN \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger=['console'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=32 \
    data.train_batch_size=4 \
    data.truncation='middle' \
    data.prompt_key=prompt \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='dapo_parallel' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model','optimizer','extra'] \
    trainer.critic_warmup=0 \
    trainer.project_name='memagent' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=-1 \
    trainer.save_freq=50 \
    trainer.default_hdfs_dir=null \
    trainer.rollout_data_dir=${PROJ_ROOT}/${EXPERIMENT_NAME}/rollout \
    trainer.default_local_dir=${PROJ_ROOT}/${EXPERIMENT_NAME}/ckpt \
    trainer.total_epochs=30 \
    2>&1 | tee ${PROJ_ROOT}/${EXPERIMENT_NAME}/training_$(date +%Y%m%d_%H%M%S).txt