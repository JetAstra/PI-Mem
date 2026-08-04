#!/usr/bin/env bash
# Qwen3.5-35B-A3B MoE GRPO RL with Megatron (single node, 8 GPUs, geo3k dataset)
#
# notes on vllm:
#     by 20260225, the latest vllm nightly does not support qwen3.5 rollout, to use this script, you need to 
#         1. wait until vllm supports qwen3.5 officially, and build a verl docker with that version of vllm
#         2. self build a verl docker image with vllm from source code with qwen3.5 support (main branch 20260225 is OK)
#     I succeeded in running this script with the main branch of vllm on 20260225, yet there are still some minor issues
#     the vllm qwen3.5 during initialization, need to be fixed. Also, the cuda_graph is somehow not working, need to be 
#     fixed, either by verl team with supoorts to vllm0.16, or by vllm team.
# Requirements:
#   - 8 GPUs (80GB each, e.g. 1x8 H100/H200)
#   - Additional packages on top of the base image:
#       pip install --upgrade transformers
#       pip install flash-linear-attention
#       pip install -U git+https://github.com/ISEEKYAN/mbridge.git
#   - Megatron-LM==0.16.0
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does
#   NOT support packed sequences (THD format) in Megatron-LM. Therefore:
#     - model.use_remove_padding=False           (deprecated option, will be removed in the future forces bshd compute format)
#     - actor.megatron.use_remove_padding=False  (forces bshd compute format)
#     - actor.use_dynamic_bsz=False              (required for bshd mode)
#
#   Once Megatron-LM adds THD support for Qwen3.5 GDN, use_remove_padding
#   can be set to True for better performance.
#
# Tested parallelism config (8 GPUs / 1 node):
#   TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8
#

# bash /mnt/shared-storage-user/liudawei/home/verl-new/examples/parallel_trainer/run_qwen3_5_35b_megatron.sh
cd /mnt/shared-storage-user/liudawei/home/verl-new/
source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liudawei/envs/verl-071

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export WANDB_MODE=offline

export HOME=/mnt/shared-storage-user/liudawei/home/
export HF_HOME=/mnt/shared-storage-user/liudawei/home/.cache/huggingface/
export HF_DATASETS_CACHE=/mnt/shared-storage-user/liudawei/.cache/huggingface/datasets

export VLLM_RPC_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=120

#### for debug?
# dont use export PYTORCH_ALLOC_CONF=expandable_segments:True!!
export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# set -xeuo pipefail

########################### Quick Config ###########################

# ---- user-adjustable ----
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-4}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-4}
SP=${SP:-True}

REF_OFFLOAD=${REF_OFFLOAD:-True}
OLD_MICRO_BSZ=${OLD_MICRO_BSZ:-2}

rollout_name="vllm"
project_name='ParallelAgent'
exp_name='interns2_megatron_debug'
adv_estimator=grpo

# HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/shared-storage-user/liudawei/home/verl/models/Intern-S2-preview-Qwen3.5"}
# train_path=${train_path:-"data/hotpotqa/hotpotqa_train_32k.parquet"}
train_path=/mnt/shared-storage-user/liudawei/home/verl/data/hotpotqa_train_sample/hotpotqa_train_doc1000.parquet
test_path=${test_path:-"data/hotpotqa/hotpotqa_dev.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${PWD}/checkpoints/${project_name}/${exp_name}"}
mkdir -p $CKPTS_DIR
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
########################### Parameter Arrays ###########################

resp_len=${resp_len:-4096}
DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=4
    data.max_prompt_length=140000
    data.max_response_length=${resp_len}
    data.truncation='middle'
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=32
    +data.context_key='context'
    +data.apply_chat_template_kwargs.enable_thinking=False
)

RECURRENT=(
    recurrent.enable=memory
    recurrent.memory.config.chunk_size=15000
    recurrent.memory.config.max_chunks=4
    recurrent.memory.config.max_passes=3
    recurrent.memory.config.max_merge_length=4096
    recurrent.memory.config.max_memorization_length=${resp_len}
    recurrent.memory.config.max_final_response_length=${resp_len}
    recurrent.memory.config.pass_reward_coef=0.2
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr_warmup_steps=20
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=True
    actor_rollout_ref.actor.clip_ratio_high=0.20
    actor_rollout_ref.actor.ppo_mini_batch_size=2
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.sequence_parallel=${SP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.max_num_batched_tokens=102400
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${OLD_MICRO_BSZ}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${OLD_MICRO_BSZ}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.sequence_parallel=${SP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${REF_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=False
    +algorithm.filter_groups.enable=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.rollout_data_dir=${CKPTS_DIR}/dump
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.total_epochs=15
)

EXTRA=(
    model_engine=megatron
)

########################### Launch ###########################

python3 -u -m verl.trainer.main_ppo_debug \
    "${DATA[@]}" \
    "${RECURRENT[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee ${CKPTS_DIR}/training-$(date +%Y%m%d)_$(date +%H%M%S).log


# bash /mnt/shared-storage-user/liudawei/home/LLaMA-Factory/examples/shell/lmf-qwen3-dense-ar-rlaunch8.sh