#!/usr/bin/env python3
"""Benchmark reward computation throughput for long/docqa/docmath with DAPO parallel manager."""

import argparse
import contextlib
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.dapo_parallel import DAPORewardManagerParallel


LONG_DS = "long_toc_choices_benchmark"
DOCQA_DS = "multihoprag_benchmark"
DOCMATH_DS = "docmath_benchmark"


@dataclass
class SampleSet:
    prompt_map: Dict[int, str]
    response_map: Dict[int, str]
    ground_truths: List[str]
    data_source: str


class MapTokenizer:
    """Tokenizer stub that maps first token id to predefined prompt/response text."""

    def __init__(self, prompt_map: Dict[int, str], response_map: Dict[int, str]):
        self.prompt_map = prompt_map
        self.response_map = response_map
        self.eos_token = None

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if len(ids) == 0:
            return ""
        key = int(ids[0])
        if key in self.prompt_map:
            return self.prompt_map[key]
        if key in self.response_map:
            return self.response_map[key]
        return ""


def _prompt_template(i: int) -> str:
    return (
        f"<text>\nDoc snippet {i}\n</text>\n"
        f"Question {i}: pick the best answer.\n"
        "Format your response as follows:\n"
        "<think>...</think> final answer."
    )


def build_long_samples(n: int) -> SampleSet:
    prompt_map = {}
    response_map = {}
    ground_truths = []
    for i in range(n):
        pkey = 100_000 + i
        rkey = 200_000 + i
        prompt_map[pkey] = _prompt_template(i)
        answer = "A" if i % 2 == 0 else "B"
        response_map[rkey] = f"<think>reasoning {i}</think>The correct answer is ({answer})"
        ground_truths.append(f"The correct answer is ({answer})")
    return SampleSet(prompt_map=prompt_map, response_map=response_map, ground_truths=ground_truths, data_source=LONG_DS)


def build_docqa_samples(n: int, trigger_judge: bool) -> SampleSet:
    prompt_map = {}
    response_map = {}
    ground_truths = []
    for i in range(n):
        pkey = 100_000 + i
        rkey = 200_000 + i
        prompt_map[pkey] = _prompt_template(i)
        if trigger_judge:
            pred = f"<think>reasoning {i}</think>the answer is wrong_{i}."
            gt = f"the answer is right_{i}."
        else:
            pred = f"<think>reasoning {i}</think>the answer is right_{i}."
            gt = f"the answer is right_{i}."
        response_map[rkey] = pred
        ground_truths.append(gt)
    return SampleSet(prompt_map=prompt_map, response_map=response_map, ground_truths=ground_truths, data_source=DOCQA_DS)


def build_docmath_samples(n: int, trigger_judge: bool) -> SampleSet:
    prompt_map = {}
    response_map = {}
    ground_truths = []
    for i in range(n):
        pkey = 100_000 + i
        rkey = 200_000 + i
        prompt_map[pkey] = _prompt_template(i)
        gt_num = i % 7 + 1
        pred_num = gt_num + 1 if trigger_judge else gt_num
        response_map[rkey] = f"<think>reasoning {i}</think>the answer is {pred_num}."
        ground_truths.append(f"the answer is {gt_num}.")
    return SampleSet(prompt_map=prompt_map, response_map=response_map, ground_truths=ground_truths, data_source=DOCMATH_DS)


def build_data_proto(samples: SampleSet, n: int, prompt_len: int = 2, response_len: int = 2) -> DataProto:
    prompt_ids = []
    response_ids = []
    attn_masks = []
    reward_models = []
    data_sources = []
    extra_infos = []

    for i in range(n):
        pkey = 100_000 + i
        rkey = 200_000 + i
        prompt_ids.append(torch.tensor([pkey] + [0] * (prompt_len - 1), dtype=torch.long))
        response_ids.append(torch.tensor([rkey] + [0] * (response_len - 1), dtype=torch.long))
        attn_masks.append(torch.ones(prompt_len + response_len, dtype=torch.long))
        reward_models.append({"ground_truth": samples.ground_truths[i]})
        data_sources.append(samples.data_source)
        extra_infos.append({"id": i})

    tensors = {
        "prompts": torch.stack(prompt_ids),
        "responses": torch.stack(response_ids),
        "attention_mask": torch.stack(attn_masks),
    }
    non_tensors = {
        "reward_model": np.array(reward_models, dtype=object),
        "data_source": np.array(data_sources, dtype=object),
        "extra_info": np.array(extra_infos, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def compute_score_silent(data_source, solution_str, ground_truth, extra_info=None, prompt_str=None, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            prompt_str=prompt_str,
            **kwargs,
        )


def parse_workers(text: str) -> List[int]:
    workers = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        workers.append(int(part))
    if not workers:
        raise ValueError("workers list is empty")
    return workers


def run_single_bench(data: DataProto, tokenizer, workers: int, repeats: int, warmup: int, silent_reward_logs: bool):
    compute_fn = compute_score_silent if silent_reward_logs else default_compute_score
    manager = DAPORewardManagerParallel(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_fn,
        num_workers=workers,
    )

    for _ in range(warmup):
        _ = manager(data)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        reward = manager(data)
        dt = time.perf_counter() - t0
        times.append(dt)
        if reward.shape[0] != len(data):
            raise RuntimeError(f"Unexpected reward shape: {reward.shape}")
    return times


def print_result_table(task: str, samples: int, workers: Sequence[int], time_map: Dict[int, List[float]]):
    base = min(workers)
    base_avg = sum(time_map[base]) / len(time_map[base])
    print(f"\n=== Task: {task} | samples={samples} ===")
    print("workers | avg_sec | p50_sec | samples_per_sec | speedup_vs_base")
    for w in workers:
        vals = sorted(time_map[w])
        avg = sum(vals) / len(vals)
        p50 = vals[len(vals) // 2]
        throughput = samples / avg if avg > 0 else float("inf")
        speedup = base_avg / avg if avg > 0 else float("inf")
        print(f"{w:>7d} | {avg:>7.3f} | {p50:>7.3f} | {throughput:>15.2f} | {speedup:>15.2f}")


def build_task_samples(task: str, samples: int, trigger_judge: bool) -> SampleSet:
    if task == "long":
        return build_long_samples(samples)
    if task == "docqa":
        return build_docqa_samples(samples, trigger_judge=trigger_judge)
    if task == "docmath":
        return build_docmath_samples(samples, trigger_judge=trigger_judge)
    raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark reward computation for long/docqa/docmath under different CPU workers.")
    parser.add_argument("--task", choices=["long", "docqa", "docmath", "all"], default="all")
    parser.add_argument("--samples", type=int, default=512, help="Number of samples in one batch.")
    parser.add_argument("--workers", type=str, default="1,2,4,8", help="Comma-separated worker counts.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats per worker.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per worker.")
    parser.add_argument("--trigger-judge", action="store_true", help="Force docqa/docmath mismatch so LLM judge path is exercised.")
    parser.add_argument("--llm-judge", choices=["Y", "N"], default=None, help="Override LLM_JUDGE env for this run.")
    parser.add_argument("--keep-reward-logs", action="store_true", help="Keep verbose prints inside reward functions.")
    args = parser.parse_args()

    if args.llm_judge is not None:
        os.environ["LLM_JUDGE"] = args.llm_judge

    workers = sorted(set(parse_workers(args.workers)))
    tasks = ["long", "docqa", "docmath"] if args.task == "all" else [args.task]

    print("Benchmark config:")
    print(f"  LLM_JUDGE={os.getenv('LLM_JUDGE', '<unset>')}")
    print(f"  VERIFIER_HOST={os.getenv('VERIFIER_HOST', '<unset>')}")
    print(f"  VERIFIER_PORT={os.getenv('VERIFIER_PORT', '<unset>')}")
    print(f"  VERIFIER_PATH={os.getenv('VERIFIER_PATH', '<unset>')}")
    print(f"  tasks={tasks}, samples={args.samples}, workers={workers}, repeats={args.repeats}, warmup={args.warmup}")
    silent_reward_logs = not args.keep_reward_logs
    print(f"  trigger_judge={args.trigger_judge}, silent_reward_logs={silent_reward_logs}")

    for task in tasks:
        sample_set = build_task_samples(task, args.samples, trigger_judge=args.trigger_judge)
        tokenizer = MapTokenizer(prompt_map=sample_set.prompt_map, response_map=sample_set.response_map)
        data = build_data_proto(sample_set, args.samples)

        time_map = {}
        for w in workers:
            times = run_single_bench(
                data=data,
                tokenizer=tokenizer,
                workers=w,
                repeats=args.repeats,
                warmup=args.warmup,
                silent_reward_logs=silent_reward_logs,
            )
            time_map[w] = times
        print_result_table(task=task, samples=args.samples, workers=workers, time_map=time_map)


if __name__ == "__main__":
    main()
