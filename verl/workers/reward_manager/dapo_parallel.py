# reward_manager_parallel.py

from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Union

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def _process_single_item(
    args: Tuple[int, Dict[str, Any]],
    compute_score_fn,
) -> Dict[str, Any]:
    """
    Worker function executed in a subprocess.
    Receives pre-extracted plain Python data (no torch tensors, no tokenizer objects).
    Returns a plain dict with results.
    """
    idx, item_data = args

    prompt_str = item_data["prompt_str"]
    response_str = item_data["response_str"]
    ground_truth = item_data["ground_truth"]
    data_source = item_data["data_source"]
    extra_info = item_data["extra_info"]
    valid_response_length = item_data["valid_response_length"]
    max_resp_len = item_data["max_resp_len"]
    overlong_buffer_cfg = item_data["overlong_buffer_cfg"]

    # Compute score
    result = compute_score_fn(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        prompt_str=prompt_str,
    )

    score: float
    extra_info_result: Dict[str, Any] = {}
    if isinstance(result, dict):
        score = result["score"]
        extra_info_result = dict(result)
    else:
        score = result

    reward = score

    overlong_reward = None
    overlong = None
    if overlong_buffer_cfg is not None and overlong_buffer_cfg["enable"]:
        overlong_buffer_len = overlong_buffer_cfg["len"]
        expected_len = max_resp_len - overlong_buffer_len
        exceed_len = valid_response_length - expected_len
        overlong_penalty_factor = overlong_buffer_cfg["penalty_factor"]
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        reward += overlong_reward
        overlong = overlong_reward < 0

    return {
        "idx": idx,
        "reward": reward,
        "score": score,
        "valid_response_length": valid_response_length,
        "data_source": data_source,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "result": result,
        "extra_info_result": extra_info_result,
        "overlong_reward": overlong_reward,
        "overlong": overlong,
    }


def _extract_item_data(
    data_item,
    tokenizer,
    reward_fn_key: str,
    max_resp_len: Optional[int],
    overlong_buffer_cfg,
) -> Dict[str, Any]:
    """
    Extract all necessary data from a DataProtoItem into plain Python types.
    This runs in the main process so we can use the tokenizer here.
    """
    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum().item()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum().item()
    valid_response_ids = response_ids[:valid_response_length]

    # Decode in main process (tokenizer is not always picklable)
    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    eos_token = tokenizer.eos_token
    if eos_token and response_str.endswith(eos_token):
        response_str = response_str[: -len(eos_token)]

    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    data_source = data_item.non_tensor_batch[reward_fn_key]
    extra_info = data_item.non_tensor_batch.get("extra_info", None)

    # Convert overlong_buffer_cfg to a plain dict (it might be an OmegaConf object)
    overlong_cfg_dict = None
    if overlong_buffer_cfg is not None:
        overlong_cfg_dict = {
            "enable": overlong_buffer_cfg.enable if hasattr(overlong_buffer_cfg, "enable") else overlong_buffer_cfg.get("enable", False),
            "len": overlong_buffer_cfg.len if hasattr(overlong_buffer_cfg, "len") else overlong_buffer_cfg.get("len", 0),
            "penalty_factor": overlong_buffer_cfg.penalty_factor if hasattr(overlong_buffer_cfg, "penalty_factor") else overlong_buffer_cfg.get("penalty_factor", 1.0),
            "log": overlong_buffer_cfg.log if hasattr(overlong_buffer_cfg, "log") else overlong_buffer_cfg.get("log", False),
        }

    return {
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "data_source": data_source,
        "extra_info": extra_info,
        "valid_response_length": valid_response_length,
        "max_resp_len": max_resp_len,
        "overlong_buffer_cfg": overlong_cfg_dict,
    }


@register("dapo_parallel")
class DAPORewardManagerParallel:
    """The reward manager with multiprocessing parallel reward computation."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        num_workers: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_workers = num_workers

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Parallel version of reward computation using multiprocessing.Pool."""

        # Fast path: if rm_scores already exist
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # ---- Step 1: Extract data in main process (tokenizer decode happens here) ----
        items_data: List[Tuple[int, Dict[str, Any]]] = []
        for i in range(len(data)):
            data_item = data[i]
            item_data = _extract_item_data(
                data_item,
                self.tokenizer,
                self.reward_fn_key,
                self.max_resp_len,
                self.overlong_buffer_cfg,
            )
            items_data.append((i, item_data))

        # ---- Step 2: Parallel compute_score in worker processes ----
        worker_fn = partial(_process_single_item, compute_score_fn=self.compute_score)

        effective_workers = min(self.num_workers, len(items_data))

        if effective_workers <= 1:
            # Fallback to sequential if only 1 worker or 1 item
            results = [worker_fn(item) for item in items_data]
        else:
            with Pool(processes=effective_workers) as pool:
                results = pool.map(worker_fn, items_data)

        # ---- Step 3: Assemble results back in main process ----
        # Sort by idx to maintain order (pool.map preserves order, but be safe)
        results.sort(key=lambda x: x["idx"])

        already_print_data_sources: Dict[str, int] = {}

        for res in results:
            idx = res["idx"]
            reward = res["reward"]
            valid_response_length = res["valid_response_length"]
            data_source = res["data_source"]
            result = res["result"]

            # Place reward at the last valid token position
            reward_tensor[idx, valid_response_length - 1] = reward

            # Collect extra info
            if res["extra_info_result"]:
                for key, value in res["extra_info_result"].items():
                    reward_extra_info[key].append(value)

            # Overlong logging
            overlong_cfg_dict = items_data[idx][1]["overlong_buffer_cfg"]
            if overlong_cfg_dict is not None and overlong_cfg_dict["enable"] and overlong_cfg_dict["log"]:
                reward_extra_info["overlong_reward"].append(res["overlong_reward"])
                reward_extra_info["overlong"].append(res["overlong"])

            # Print examination
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", res["prompt_str"])
                print("[response]", res["response_str"])
                print("[ground_truth]", res["ground_truth"])
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", res["score"])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor