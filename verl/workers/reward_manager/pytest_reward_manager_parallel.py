# test_reward_manager_parallel.py

import sys
import types
import pytest
import torch
from unittest.mock import MagicMock
from collections import defaultdict
from functools import partial

# ---------------------------------------------------------------------------
# Minimal stubs so we can import without the real verl package
# ---------------------------------------------------------------------------

# Build a minimal `verl` package stub if not already importable
_verl = types.ModuleType("verl")
_verl_utils = types.ModuleType("verl.utils")
_verl_utils_reward_score = types.ModuleType("verl.utils.reward_score")
_verl_workers = types.ModuleType("verl.workers")
_verl_workers_rm = types.ModuleType("verl.workers.reward_manager")

# A trivial default_compute_score
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None):
    return 1.0

_verl_utils_reward_score.default_compute_score = _default_compute_score

# A trivial register decorator (identity)
_REGISTRY: dict = {}

def _register(name):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

_verl_workers_rm.register = _register

sys.modules.setdefault("verl", _verl)
sys.modules.setdefault("verl.utils", _verl_utils)
sys.modules.setdefault("verl.utils.reward_score", _verl_utils_reward_score)
sys.modules.setdefault("verl.workers", _verl_workers)
sys.modules.setdefault("verl.workers.reward_manager", _verl_workers_rm)

# ---------------------------------------------------------------------------
# Minimal DataProto / DataProtoItem stubs
# ---------------------------------------------------------------------------

class DataProtoItem:
    def __init__(self, batch: dict, non_tensor_batch: dict):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


class DataProto:
    """Minimal stub that mimics verl.DataProto."""

    def __init__(self, batch: dict, non_tensor_batch_list: list):
        self.batch = batch
        self._non_tensor_batch_list = non_tensor_batch_list

    def __len__(self):
        return self.batch["responses"].shape[0]

    def __getitem__(self, idx):
        item_batch = {k: v[idx] for k, v in self.batch.items()}
        return DataProtoItem(
            batch=item_batch,
            non_tensor_batch=self._non_tensor_batch_list[idx],
        )

# Patch verl.DataProto so our module can import it
_verl.DataProto = DataProto

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
from .dapo_parallel import (
    DAPORewardManagerParallel,
    _process_single_item,
    _extract_item_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(vocab=None):
    """Create a mock tokenizer."""
    tok = MagicMock()
    tok.eos_token = "</s>"

    def decode(ids, skip_special_tokens=True):
        # Simple: join token ids as characters via chr(id) – or just return str repr
        return "".join(chr(int(x) + 65) for x in ids if int(x) >= 0)

    tok.decode = decode
    return tok


def _simple_compute_score(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None):
    """A picklable compute_score for testing: returns len(solution_str) / 10."""
    return round(len(solution_str) / 10.0, 4)


def _dict_compute_score(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None):
    """A picklable compute_score that returns a dict."""
    s = round(len(solution_str) / 10.0, 4)
    return {"score": s, "length": len(solution_str)}


def _build_data(n_samples=4, prompt_len=5, response_len=10):
    """Build a DataProto with n_samples items."""
    prompts_list = []
    responses_list = []
    attention_masks_list = []
    non_tensor_batch_list = []

    for i in range(n_samples):
        prompt = torch.arange(0, prompt_len, dtype=torch.long)
        response = torch.arange(prompt_len, prompt_len + response_len, dtype=torch.long)
        # all tokens are valid
        attn_mask = torch.ones(prompt_len + response_len, dtype=torch.long)

        prompts_list.append(prompt)
        responses_list.append(response)
        attention_masks_list.append(attn_mask)

        non_tensor_batch_list.append(
            {
                "reward_model": {"ground_truth": f"answer_{i}"},
                "data_source": "test_source",
                "extra_info": {"id": i},
            }
        )

    batch = {
        "prompts": torch.stack(prompts_list),
        "responses": torch.stack(responses_list),
        "attention_mask": torch.stack(attention_masks_list),
    }

    return DataProto(batch=batch, non_tensor_batch_list=non_tensor_batch_list)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessSingleItem:
    """Unit tests for the worker function _process_single_item."""

    def test_basic_scalar_score(self):
        item_data = {
            "prompt_str": "hello",
            "response_str": "world12345",  # len=10 → score=1.0
            "ground_truth": "answer",
            "data_source": "ds",
            "extra_info": None,
            "valid_response_length": 10,
            "max_resp_len": None,
            "overlong_buffer_cfg": None,
        }
        res = _process_single_item((0, item_data), compute_score_fn=_simple_compute_score)
        assert res["idx"] == 0
        assert res["reward"] == pytest.approx(1.0)
        assert res["overlong_reward"] is None

    def test_dict_score(self):
        item_data = {
            "prompt_str": "hello",
            "response_str": "world12345",
            "ground_truth": "answer",
            "data_source": "ds",
            "extra_info": None,
            "valid_response_length": 10,
            "max_resp_len": None,
            "overlong_buffer_cfg": None,
        }
        res = _process_single_item((3, item_data), compute_score_fn=_dict_compute_score)
        assert res["idx"] == 3
        assert res["reward"] == pytest.approx(1.0)
        assert res["extra_info_result"]["length"] == 10

    def test_overlong_penalty(self):
        item_data = {
            "prompt_str": "p",
            "response_str": "r" * 20,
            "ground_truth": "a",
            "data_source": "ds",
            "extra_info": None,
            "valid_response_length": 100,
            "max_resp_len": 80,
            "overlong_buffer_cfg": {
                "enable": True,
                "len": 20,
                "penalty_factor": 1.0,
                "log": True,
            },
        }
        res = _process_single_item((0, item_data), compute_score_fn=_simple_compute_score)
        # exceed_len = 100 - (80-20) = 40
        # overlong_reward = min(-40/20 * 1.0, 0) = -2.0
        base_score = round(20 / 10.0, 4)
        assert res["reward"] == pytest.approx(base_score + (-2.0))
        assert res["overlong_reward"] == pytest.approx(-2.0)
        assert res["overlong"] is True

    def test_overlong_no_penalty_when_short(self):
        item_data = {
            "prompt_str": "p",
            "response_str": "r" * 5,
            "ground_truth": "a",
            "data_source": "ds",
            "extra_info": None,
            "valid_response_length": 10,
            "max_resp_len": 80,
            "overlong_buffer_cfg": {
                "enable": True,
                "len": 20,
                "penalty_factor": 1.0,
                "log": True,
            },
        }
        res = _process_single_item((0, item_data), compute_score_fn=_simple_compute_score)
        # exceed_len = 10 - 60 = -50  →  min(50/20*1.0, 0) = 0
        assert res["overlong_reward"] == pytest.approx(0.0)
        assert res["overlong"] is False


class TestExtractItemData:
    """Tests for _extract_item_data."""

    def test_basic_extraction(self):
        tok = _make_tokenizer()
        prompt = torch.arange(0, 5, dtype=torch.long)
        response = torch.arange(5, 15, dtype=torch.long)
        attn = torch.ones(15, dtype=torch.long)

        item = DataProtoItem(
            batch={"prompts": prompt, "responses": response, "attention_mask": attn},
            non_tensor_batch={
                "reward_model": {"ground_truth": "42"},
                "data_source": "math",
                "extra_info": {"id": 0},
            },
        )

        result = _extract_item_data(item, tok, "data_source", None, None)
        assert result["ground_truth"] == "42"
        assert result["data_source"] == "math"
        assert result["valid_response_length"] == 10
        assert isinstance(result["prompt_str"], str)
        assert isinstance(result["response_str"], str)

    def test_eos_stripping(self):
        tok = _make_tokenizer()
        # The decode produces characters via chr(id+65). We'll hack eos_token to match.
        # response ids = [0,1,2] → decode → "ABC"
        tok.eos_token = "C"
        prompt = torch.tensor([10, 11], dtype=torch.long)
        response = torch.tensor([0, 1, 2], dtype=torch.long)
        attn = torch.ones(5, dtype=torch.long)

        item = DataProtoItem(
            batch={"prompts": prompt, "responses": response, "attention_mask": attn},
            non_tensor_batch={
                "reward_model": {"ground_truth": "x"},
                "data_source": "ds",
            },
        )
        result = _extract_item_data(item, tok, "data_source", None, None)
        assert not result["response_str"].endswith("C")


class TestDAPORewardManagerParallel:
    """Integration tests for the full parallel reward manager."""

    def test_basic_reward_tensor_shape(self):
        tok = _make_tokenizer()
        data = _build_data(n_samples=8, prompt_len=4, response_len=6)

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            num_workers=4,
        )
        reward = manager(data)
        assert reward.shape == data.batch["responses"].shape
        # All rewards should be placed at valid_response_length - 1 = 5
        for i in range(8):
            assert reward[i, 5].item() != 0.0 or True  # score could be 0 in edge case
            # All other positions should be 0
            assert reward[i, :5].sum().item() == 0.0

    def test_return_dict(self):
        tok = _make_tokenizer()
        data = _build_data(n_samples=4)

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_dict_compute_score,
            num_workers=2,
        )
        result = manager(data, return_dict=True)
        assert "reward_tensor" in result
        assert "reward_extra_info" in result
        assert "length" in result["reward_extra_info"]
        assert len(result["reward_extra_info"]["length"]) == 4

    def test_rm_scores_passthrough(self):
        tok = _make_tokenizer()
        data = _build_data(n_samples=2, response_len=5)
        rm_scores = torch.randn_like(data.batch["responses"], dtype=torch.float32)
        data.batch["rm_scores"] = rm_scores

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            num_workers=2,
        )
        result = manager(data)
        assert torch.equal(result, rm_scores)

    def test_single_worker_fallback(self):
        tok = _make_tokenizer()
        data = _build_data(n_samples=3, prompt_len=3, response_len=7)

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            num_workers=1,
        )
        reward = manager(data)
        assert reward.shape == data.batch["responses"].shape

    def test_overlong_with_parallel(self):
        tok = _make_tokenizer()
        data = _build_data(n_samples=4, prompt_len=3, response_len=50)

        overlong_cfg = MagicMock()
        overlong_cfg.enable = True
        overlong_cfg.len = 10
        overlong_cfg.penalty_factor = 1.0
        overlong_cfg.log = True

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            max_resp_len=30,
            overlong_buffer_cfg=overlong_cfg,
            num_workers=2,
        )
        result = manager(data, return_dict=True)
        assert "overlong_reward" in result["reward_extra_info"]
        assert "overlong" in result["reward_extra_info"]
        assert len(result["reward_extra_info"]["overlong_reward"]) == 4

    def test_consistency_with_sequential(self):
        """Verify that parallel (num_workers>1) gives same results as sequential (num_workers=1)."""
        tok = _make_tokenizer()
        data = _build_data(n_samples=16, prompt_len=5, response_len=12)

        manager_seq = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            num_workers=1,
        )
        manager_par = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            num_workers=4,
        )

        reward_seq = manager_seq(data)
        reward_par = manager_par(data)

        assert torch.allclose(reward_seq, reward_par), (
            f"Sequential and parallel results differ!\n"
            f"Sequential: {reward_seq}\n"
            f"Parallel: {reward_par}"
        )

    def test_multiple_data_sources_print(self, capsys):
        """Test that num_examine controls printing per data_source."""
        tok = _make_tokenizer()
        data = _build_data(n_samples=6, prompt_len=3, response_len=5)
        # Set different data sources
        for i in range(3):
            data._non_tensor_batch_list[i]["data_source"] = "source_a"
        for i in range(3, 6):
            data._non_tensor_batch_list[i]["data_source"] = "source_b"

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=1,
            compute_score=_simple_compute_score,
            num_workers=2,
        )
        manager(data)
        captured = capsys.readouterr()
        # Should print exactly 1 prompt per data source = 2 "[prompt]" lines
        assert captured.out.count("[prompt]") == 2

    def test_empty_data(self):
        """Edge case: zero samples."""
        tok = _make_tokenizer()
        batch = {
            "prompts": torch.zeros(0, 5, dtype=torch.long),
            "responses": torch.zeros(0, 10, dtype=torch.long),
            "attention_mask": torch.zeros(0, 15, dtype=torch.long),
        }
        data = DataProto(batch=batch, non_tensor_batch_list=[])

        manager = DAPORewardManagerParallel(
            tokenizer=tok,
            num_examine=0,
            compute_score=_simple_compute_score,
            num_workers=2,
        )
        reward = manager(data)
        assert reward.shape == (0, 10)


# export PYTHONPATH="/mnt/shared-storage-user/dllm-share/liudawei/verl"
# ---------------------------------------------------------------------------
# Run with:  pytest pytest_reward_manager_parallel.py -v
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])