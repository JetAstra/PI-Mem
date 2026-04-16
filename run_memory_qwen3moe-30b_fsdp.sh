#!/bin/bash
# bash /mnt/shared-storage-user/dllm-share/liudawei/verl/run_memory_qwen3moe-30b_fsdp.sh
set -x
source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/dllm-share/songhaixu/miniforge3/envs/qwenlongl1_5
# export WANDB_MODE=offline

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=1
NGPUS_PER_NODE=8
PROJ_ROOT=/mnt/shared-storage-user/dllm-share/liudawei/verl

MODEL_PATH=/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-30B-A3B-Instruct-2507
TRAIN_PATH="data/hotpotqa/hotpotqa_train_32k.parquet"
VAL_PATH="data/hotpotqa/hotpotqa_dev.parquet"
EXP=memory_agent/7B
PROJ_DIR=${PROJ_ROOT}/${EXP}

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=32000
MAX_NEW_TOKEN=4000
MEMORY_LEN=4000

export HYDRA_FULL_ERROR=1
# recurrent.memory.config.max_prompt_length 只有一小段 query 长度
python -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=4000 \
    recurrent.memory.config.max_prompt_length=1024 \
    recurrent.memory.config.max_memorization_length=$MEMORY_LEN \
    recurrent.memory.config.max_final_response_length=$MAX_NEW_TOKEN \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.truncation='middle' \
    data.prompt_key=prompt \
    +data.context_key='context' \
    data.filter_overlong_prompts_workers=32 \
    data.filter_overlong_prompts=True \
    data.train_batch_size=2 \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=60000 \
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
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    actor_rollout_ref.model.use_liger=True \
    trainer.logger=['console'] \
    trainer.project_name='QwenLong-L1' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.default_local_dir=./checkpoints/${EXPERIMENT_NAME} \
    trainer.resume_mode=auto \
    trainer.total_epochs=30 "${@:1}"