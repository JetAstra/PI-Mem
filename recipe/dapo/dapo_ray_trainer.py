# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
import loguru

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_1D_grpo_advantage,
    compute_advantage,
    compute_response_mask,
    pad_dataproto_to_divisor,
    unpad_dataproto,
)
from verl.utils.debug import marked_timer


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        reward_batch = None
        reward_tensor_batch = None
        sample_index_batch = None
        selected_prompt_uids = []
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                pad_size = 0

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                local_train_batch = None
                local_reward_batch = None
                local_reward_tensor = None
                local_sample_index = None

                # [MemAgent] pop generation keys: recurrent dataset decides keys, non-recurrent keeps original logic.
                if self.config.recurrent.enable:
                    batch_keys_to_pop, non_tensor_batch_keys_to_pop = self.train_dataset.get_bactch_keys()
                else:
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        if not self.config.recurrent.enable:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                        ####################
                        # [MemAgent] Below is all about agents - the "LLM + forloop"
                        ####################
                        else:
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                raise NotImplementedError("REMAX is not implemented for recurrent.")

                            prompt_uids = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                            new_batch.non_tensor_batch["uid"] = prompt_uids
                            original_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                            from recurrent.utils import final_batch

                            local_train_batch, final_mask, local_sample_index = self.generation_manager.run_llm_loop(gen_batch, timing_raw)
                            assert final_mask.sum().item() == len(original_batch.batch), \
                                "The number of final responses should be equal to the number of prompts." \
                                f"{len(original_batch.non_tensor_batch['uid'])} != {len(original_batch.batch)}"
                            # This is a simplified diagram to show how sample_index works.
                            # DataProto and 2D tensors represented as a list of samples.

                            # ex. batch = [s1, s2, s3, s4]
                            #     gen_batch = [s1_turn1, s2_turn1, s3_turn1, s4_turn1, s1_turn2, s3_turn2, s3_turn3, s1_final, s2_final, s3_final, s4_final]
                            #     final_mask = [      F,        F,        F,        F,        F,        F,        F,        T,        T,        T,        T]
                            #     sample_index = [    0,        1,        2,        3,        0,        2,        2,        0,        1,        2,        3]
                            
                            # then, batch[sample_index] will be
                            #                 [      s1,       s2,       s3,       s4,       s1,       s3,       s3,       s1,       s2,       s3,       s4]
                            # We map info from original_sample to gen_batch_output now, e.x. in reward computation

                            repeated_prompt_uids = original_batch.non_tensor_batch["uid"]
                            # [MemAgent] turn-level uid mapping: prompt uid -> all turns via sample_index.
                            local_train_batch.non_tensor_batch["uid"] = repeated_prompt_uids[local_sample_index]
                            # [MemAgent] reward is computed on final-batch + original prompt info.
                            local_reward_batch = final_batch(local_train_batch, final_mask, local_sample_index).union(original_batch)

                    if not self.config.recurrent.enable and self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if not self.config.recurrent.enable:
                        new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                        # repeat to align with repeated responses in rollout
                        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        local_train_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.config.recurrent.enable:
                            # [MemAgent] explicit constraints in recurrent mode.
                            if self.use_rm:
                                raise NotImplementedError("RM is not implemented for recurrent.")
                            if self.config.algorithm.use_kl_in_reward:
                                raise NotImplementedError("KL-in-reward is not implemented for recurrent.")

                        reward_input_batch = local_reward_batch if self.config.recurrent.enable else local_train_batch

                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(local_train_batch)
                            local_train_batch = local_train_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list] = {}
                        try:
                            reward_result = self.reward_fn(reward_input_batch, return_dict=True)
                            local_reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            local_reward_tensor = self.reward_fn(reward_input_batch)

                        reward_input_batch.batch["token_level_scores"] = local_reward_tensor

                        if reward_extra_infos_dict:
                            reward_input_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.recurrent.enable:
                            # 在没有 KL in reward 时, token_level_rewards == token_level_scores
                            reward_input_batch.batch["token_level_rewards"] = reward_input_batch.batch["token_level_scores"]
                        else:
                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                local_train_batch, kl_metrics = apply_kl_penalty(
                                    local_train_batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                local_train_batch.batch["token_level_rewards"] = local_train_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = local_train_batch
                        if self.config.recurrent.enable:
                            reward_batch = local_reward_batch
                            reward_tensor_batch = local_reward_tensor
                            sample_index_batch = local_sample_index
                    else:
                        # [MemAgent] recurrent filter works on final-batch metric, but deletes/keeps all turns per prompt.
                        metric_name = self.config.algorithm.filter_groups.metric
                        filter_batch = reward_input_batch

                        if metric_name == "seq_final_reward":
                            filter_batch.non_tensor_batch["seq_final_reward"] = filter_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            filter_batch.non_tensor_batch["seq_reward"] = filter_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        else:
                            raise ValueError(f"Unsupported filter_groups.metric: {metric_name}")

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(filter_batch.non_tensor_batch["uid"], filter_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)
                        selected_prompt_uids.extend(kept_prompt_uids)

                        kept_prompt_uid_set = set(kept_prompt_uids)
                        if self.config.recurrent.enable:  # [MemAgent] Select valid batches
                            kept_reward_idxs = [idx for idx, prompt_uid in enumerate(local_reward_batch.non_tensor_batch["uid"]) if prompt_uid in kept_prompt_uid_set]
                            kept_turn_idxs = [idx for idx, prompt_uid in enumerate(local_train_batch.non_tensor_batch["uid"]) if prompt_uid in kept_prompt_uid_set]

                            filtered_train_batch = local_train_batch[kept_turn_idxs]
                            filtered_reward_batch = local_reward_batch[kept_reward_idxs]
                            filtered_reward_tensor = local_reward_tensor[kept_reward_idxs]

                            remap = torch.full((len(local_reward_batch),), -1, dtype=torch.long, device=local_sample_index.device)
                            if kept_reward_idxs:
                                kept_reward_idxs_tensor = torch.tensor(kept_reward_idxs, dtype=torch.long, device=local_sample_index.device)
                                remap[kept_reward_idxs_tensor] = torch.arange(len(kept_reward_idxs), device=local_sample_index.device)
                            kept_turn_idxs_tensor = torch.tensor(kept_turn_idxs, dtype=torch.long, device=local_sample_index.device)
                            filtered_sample_index = remap[local_sample_index[kept_turn_idxs_tensor]]

                            batch = filtered_train_batch if batch is None else DataProto.concat([batch, filtered_train_batch])
                            if reward_batch is None:
                                reward_batch = filtered_reward_batch
                                reward_tensor_batch = filtered_reward_tensor
                                sample_index_batch = filtered_sample_index
                            else:
                                offset = len(reward_batch)
                                reward_batch = DataProto.concat([reward_batch, filtered_reward_batch])
                                reward_tensor_batch = torch.cat([reward_tensor_batch, filtered_reward_tensor], dim=0)
                                sample_index_batch = torch.cat([sample_index_batch, filtered_sample_index + offset], dim=0)
                        else:
                            kept_traj_idxs = [idx for idx, prompt_uid in enumerate(local_train_batch.non_tensor_batch["uid"]) if prompt_uid in kept_prompt_uid_set]
                            local_train_batch = local_train_batch[kept_traj_idxs]
                            batch = local_train_batch if batch is None else DataProto.concat([batch, local_train_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            loguru.logger.info(f"[DAPO filter groups] {num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                loguru.logger.info(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )

                        # [MemAgent] align by unique prompts, keep all turns of selected prompts.
                        if self.config.recurrent.enable:
                            selected_prompt_uids = selected_prompt_uids[:prompt_bsz]
                            selected_prompt_uid_set = set(selected_prompt_uids)

                            kept_reward_idxs = [idx for idx, prompt_uid in enumerate(reward_batch.non_tensor_batch["uid"]) if prompt_uid in selected_prompt_uid_set]
                            kept_turn_idxs = [idx for idx, prompt_uid in enumerate(batch.non_tensor_batch["uid"]) if prompt_uid in selected_prompt_uid_set]

                            kept_turn_idxs_tensor = torch.tensor(kept_turn_idxs, dtype=torch.long, device=sample_index_batch.device)
                            trimmed_sample_index = sample_index_batch[kept_turn_idxs_tensor]

                            remap = torch.full((len(reward_batch),), -1, dtype=torch.long, device=sample_index_batch.device)
                            if kept_reward_idxs:
                                kept_reward_idxs_tensor = torch.tensor(kept_reward_idxs, dtype=torch.long, device=sample_index_batch.device)
                                remap[kept_reward_idxs_tensor] = torch.arange(len(kept_reward_idxs), device=sample_index_batch.device)

                            sample_index_batch = remap[trimmed_sample_index]
                            reward_batch = reward_batch[kept_reward_idxs]
                            reward_tensor_batch = reward_tensor_batch[kept_reward_idxs]
                            batch = batch[kept_turn_idxs]
                        else:
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    # [MemAgent] recurrent path must keep order to preserve sample_index/final mapping.
                    if self.config.trainer.balance_batch and (not self.config.recurrent.enable):
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # [MemAgent] pad for log_prob. 注意这里padding是纯复制，所以计算entropy时可能有偏差
                    if self.config.recurrent.enable:
                        batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        if not self.config.recurrent.enable:
                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                        else:
                            # [MemAgent] recurrent: 1D GRPO scalar advantage then broadcast to token-level by sample_index.
                            batch = unpad_dataproto(batch, pad_size=pad_size)
                            if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                                raise NotImplementedError("Only GRPO is implemented for recurrent DAPO.")

                            advantage_scalar = compute_1D_grpo_advantage(
                                token_level_rewards=reward_tensor_batch,
                                index=reward_batch.non_tensor_batch["uid"],
                                use_adv=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            )
                            advantage_scalar = advantage_scalar[sample_index_batch]

                            response_length = batch.batch["responses"].size(-1)
                            eos_mask = batch.batch["response_mask"]
                            advantages = advantage_scalar.unsqueeze(-1).tile([1, response_length]) * eos_mask
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = advantages
                            batch.batch["token_level_scores"] = reward_tensor_batch[sample_index_batch]
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        if self.config.recurrent.enable:
                            # [MemAgent] pad recurrent batch for actor update world-size divisibility.
                            wsz = self.actor_rollout_wg.world_size
                            if len(batch) % wsz != 0:
                                from recurrent.utils import graceful_padding, indexing_proto

                                padding_index, no_padding_mask = graceful_padding(len(batch), wsz)
                                batch = indexing_proto(batch, padding_index)
                                batch.batch["attention_mask"][~no_padding_mask, :] = 0
                                batch.batch["response_mask"][~no_padding_mask, :] = 0
                                batch.batch["no_padding_mask"] = no_padding_mask
                                batch.meta_info["padded"] = True
                            else:
                                batch.batch["no_padding_mask"] = torch.ones(len(batch), dtype=torch.bool)

                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            from recurrent.utils import clip_long_string
                            loguru.logger.info(f"[Before Dump] Batch keys: {batch.batch.keys()}")
                            prompt_token_ids = batch.batch["prompts"].cpu().tolist()
                            response_token_ids = batch.batch["responses"].cpu().tolist()
                            total_lens = batch.batch["attention_mask"].sum(-1)   # prompt + response
                            resp_lens = batch.batch["response_mask"].sum(-1)     # response
                            prompt_lens = total_lens - resp_lens                 # prompt
                            prompt_token_ids = [row[-l:] for row, l in zip(prompt_token_ids, prompt_lens)]
                            response_token_ids = [row[:l] for row, l in zip(response_token_ids, resp_lens)]

                            inputs = self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=False)
                            # *optianal* clip long prompts
                            inputs = [clip_long_string(i, max_length=4000) for i in inputs]
                            outputs = self.tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # collect metrics
                if batch.meta_info.get("padded", False):
                    # [MemAgent] remove actor-update padding before metrics.
                    from recurrent.utils import indexing_proto

                    batch = indexing_proto(batch, batch.batch["no_padding_mask"])

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                reward_batch = None
                reward_tensor_batch = None
                sample_index_batch = None
                selected_prompt_uids = []
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
