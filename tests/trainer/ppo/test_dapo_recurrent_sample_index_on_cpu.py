# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import torch
import unittest

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_1D_grpo_advantage


def _build_local_recurrent_batches(prompt_to_traj_turns: dict[str, list[int]], reward_base: int = 0):
    """Build one recurrent gen batch with variable turns per trajectory."""
    reward_uids: list[str] = []
    reward_rows: list[torch.Tensor] = []
    turn_uids: list[str] = []
    local_sample_index: list[int] = []
    response_masks: list[torch.Tensor] = []

    seq_len = 5
    reward_id = reward_base

    for uid, traj_turns in prompt_to_traj_turns.items():
        for traj_i, turn_count in enumerate(traj_turns):
            reward_uids.append(uid)
            # Unique reward profile for each final sample / trajectory.
            reward_rows.append(torch.tensor([float(reward_id), float(reward_id) + 0.5, 0.0], dtype=torch.float32))
            reward_idx = len(reward_uids) - 1

            for turn_j in range(turn_count):
                turn_uids.append(uid)
                local_sample_index.append(reward_idx)
                valid_len = 1 + ((traj_i + turn_j + reward_id) % seq_len)
                response_masks.append(
                    torch.tensor([1] * valid_len + [0] * (seq_len - valid_len), dtype=torch.float32)
                )

            reward_id += 1

    local_train_batch = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((len(turn_uids), seq_len), dtype=torch.long),
            "response_mask": torch.stack(response_masks, dim=0),
        },
        non_tensors={"uid": np.array(turn_uids, dtype=object)},
    )
    local_reward_batch = DataProto.from_dict(
        tensors={"placeholder": torch.zeros((len(reward_uids), 1), dtype=torch.float32)},
        non_tensors={"uid": np.array(reward_uids, dtype=object)},
    )

    return (
        local_train_batch,
        local_reward_batch,
        torch.stack(reward_rows, dim=0),
        torch.tensor(local_sample_index, dtype=torch.long),
    )


def _accumulate_like_dapo_recurrent(
    local_batches: list[tuple[DataProto, DataProto, torch.Tensor, torch.Tensor]],
    kept_prompts_per_gen_batch: list[list[str]],
    prompt_bsz: int,
):
    """Replicate recurrent filter/concat/trim logic in recipe/dapo/dapo_ray_trainer.py."""
    batch = None
    reward_batch = None
    reward_tensor_batch = None
    sample_index_batch = None

    selected_prompt_uids: list[str] = []
    num_prompt_in_batch = 0

    history = []

    for (local_train_batch, local_reward_batch, local_reward_tensor, local_sample_index), kept_prompt_uids in zip(
        local_batches, kept_prompts_per_gen_batch
    ):
        num_prompt_in_batch += len(kept_prompt_uids)
        selected_prompt_uids.extend(kept_prompt_uids)
        kept_prompt_uid_set = set(kept_prompt_uids)

        kept_reward_idxs = [
            idx for idx, prompt_uid in enumerate(local_reward_batch.non_tensor_batch["uid"]) if prompt_uid in kept_prompt_uid_set
        ]
        kept_turn_idxs = [
            idx for idx, prompt_uid in enumerate(local_train_batch.non_tensor_batch["uid"]) if prompt_uid in kept_prompt_uid_set
        ]

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

        history.append(
            {
                "reward_len": len(reward_batch),
                "sample_index_batch": sample_index_batch.clone(),
                "batch_len": len(batch),
            }
        )

    assert num_prompt_in_batch >= prompt_bsz, "test fixture must generate enough prompts"

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

    return batch, reward_batch, reward_tensor_batch, sample_index_batch, history, selected_prompt_uids


def _manual_group_centered_advantage(reward_tensor_batch: torch.Tensor, reward_uids: np.ndarray) -> torch.Tensor:
    """Manual 1D GRPO advantage with use_adv=False (subtract per-prompt mean only)."""
    scores = reward_tensor_batch.sum(dim=-1)
    uid_to_indices = {}
    for i, uid in enumerate(reward_uids):
        uid_to_indices.setdefault(uid, []).append(i)

    out = scores.clone()
    for uid, indices in uid_to_indices.items():
        if len(indices) == 1:
            mean = torch.tensor(0.0, dtype=scores.dtype)
        else:
            mean = scores[indices].mean()
        out[indices] = scores[indices] - mean
    return out


def test_recurrent_sample_index_batch_variable_turns_cross_gen_batches_and_trim():
    # Two gen batches; second batch contributes extra prompts; then trim to prompt_bsz=3.
    # Prompt D should be dropped after trim, with full sample_index remap staying correct.
    local1 = _build_local_recurrent_batches(
        {
            "A": [2, 4],
            "B": [3, 1],
        },
        reward_base=10,
    )
    local2 = _build_local_recurrent_batches(
        {
            "C": [5, 2],
            "D": [4, 1],
        },
        reward_base=30,
    )

    batch, reward_batch, reward_tensor_batch, sample_index_batch, history, selected_prompt_uids = _accumulate_like_dapo_recurrent(
        local_batches=[local1, local2],
        kept_prompts_per_gen_batch=[["A", "B"], ["C", "D"]],
        prompt_bsz=3,
    )

    # Cross-gen offset check before final trim/remap:
    # after gen#1 reward_len=4, so all gen#2 indices should be shifted by +4 in history[1].
    sample_idx_after_gen2 = history[1]["sample_index_batch"]
    gen1_turns = history[0]["batch_len"]
    assert torch.all(sample_idx_after_gen2[gen1_turns:] >= history[0]["reward_len"])

    # Prompt trim check.
    assert selected_prompt_uids == ["A", "B", "C"]
    assert set(reward_batch.non_tensor_batch["uid"]) == {"A", "B", "C"}
    assert set(batch.non_tensor_batch["uid"]) == {"A", "B", "C"}

    # sample_index_batch must always map each turn to final-batch reward with the same prompt uid.
    mapped_turn_uids = np.array([reward_batch.non_tensor_batch["uid"][idx] for idx in sample_index_batch.tolist()], dtype=object)
    assert np.array_equal(mapped_turn_uids, batch.non_tensor_batch["uid"])
    assert sample_index_batch.min().item() >= 0
    assert sample_index_batch.max().item() < len(reward_batch)

    # Final-batch-only advantage then broadcast back to all turns.
    advantage_scalar = compute_1D_grpo_advantage(
        token_level_rewards=reward_tensor_batch,
        index=reward_batch.non_tensor_batch["uid"],
        use_adv=False,
    )
    manual_adv = _manual_group_centered_advantage(reward_tensor_batch, reward_batch.non_tensor_batch["uid"])
    torch.testing.assert_close(advantage_scalar, manual_adv)

    turn_scalar = advantage_scalar[sample_index_batch]
    response_mask = batch.batch["response_mask"]
    response_length = batch.batch["responses"].size(-1)
    advantages = turn_scalar.unsqueeze(-1).tile([1, response_length]) * response_mask

    # Every valid token equals the turn scalar; masked tokens are zero.
    for i in range(len(batch)):
        valid_mask = response_mask[i].bool()
        if valid_mask.any():
            torch.testing.assert_close(advantages[i, valid_mask], turn_scalar[i].expand(valid_mask.sum()))
        if (~valid_mask).any():
            torch.testing.assert_close(advantages[i, ~valid_mask], torch.zeros((~valid_mask).sum(), dtype=advantages.dtype))


def test_recurrent_sample_index_batch_variable_turns_single_gen_batch_no_trim():
    # No cross-gen concat path; still verify variable-turn mapping and 1D advantage broadcast.
    local = _build_local_recurrent_batches(
        {
            "P0": [1, 4, 2],
            "P1": [3, 1, 5],
        },
        reward_base=100,
    )

    batch, reward_batch, reward_tensor_batch, sample_index_batch, _, selected_prompt_uids = _accumulate_like_dapo_recurrent(
        local_batches=[local],
        kept_prompts_per_gen_batch=[["P0", "P1"]],
        prompt_bsz=2,
    )

    assert selected_prompt_uids == ["P0", "P1"]

    mapped_turn_uids = np.array([reward_batch.non_tensor_batch["uid"][idx] for idx in sample_index_batch.tolist()], dtype=object)
    assert np.array_equal(mapped_turn_uids, batch.non_tensor_batch["uid"])

    advantage_scalar = compute_1D_grpo_advantage(
        token_level_rewards=reward_tensor_batch,
        index=reward_batch.non_tensor_batch["uid"],
        use_adv=False,
    )
    turn_scalar = advantage_scalar[sample_index_batch]

    response_mask = batch.batch["response_mask"]
    advantages = turn_scalar.unsqueeze(-1).tile([1, batch.batch["responses"].size(-1)]) * response_mask

    assert advantages.shape[0] == len(batch)
    assert advantages.shape[1] == batch.batch["responses"].size(-1)
    # At least one masked position should exist in this fixture.
    assert torch.any(response_mask == 0)


class TestDAPORecurrentSampleIndex(unittest.TestCase):
    def test_cross_gen_batches_and_trim(self):
        test_recurrent_sample_index_batch_variable_turns_cross_gen_batches_and_trim()

    def test_single_gen_batch_no_trim(self):
        test_recurrent_sample_index_batch_variable_turns_single_gen_batch_no_trim()


if __name__ == "__main__":
    pass
    # unittest.main()
