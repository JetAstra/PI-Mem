# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import argparse
import json
import os

import tiktoken
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from utils import clip_long_string, extract_solution, load_existing_ids, update_answer
from utils.envs import DATAROOT, MEMORY_DATA_ROOT


### From RULER
def string_match_all(pred, ref):
    return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)


def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {"sub_em": 0, "total_num": 0}
    for pred, gold in zip(predictions, goldens):
        metrics["sub_em"] += string_match_all(pred, gold)
    metrics["total_num"] = len(goldens)
    for k, _ in metrics.items():
        if k == "total_num":
            continue
        metrics[k] = round((metrics[k] / metrics["total_num"]), 2)
    return metrics


def calc_qa_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {"f1": 0, "prec": 0, "recall": 0, "em": 0, "sub_em": 0, "total_num": 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == "total_num":
            continue
        metrics[k] = round((metrics[k] / metrics["total_num"]), 2)
    return metrics


DUMP_CLIP_LENGTH = 1000


def clip_for_dump(value):
    if isinstance(value, str):
        return clip_long_string(value, max_length=DUMP_CLIP_LENGTH)
    if isinstance(value, list):
        return [clip_for_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: clip_for_dump(item) for key, item in value.items()}
    return value


def get_pred(data, args, out_file):
    model = args.model
    if (
        "gpt" in model
        or "o1" in model
        or "o3" in model
        or "o4" in model
        or "gemini" in model
        or "claude" in model
    ):
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
    if args.api == "openai":
        from utils import extract_answer
        from utils.openai_api import async_query_llm
    elif args.api == "completion":
        from utils import extract_answer
        from utils.completion_api import async_query_llm
    elif args.api == "rag":
        from utils import extract_answer
        from utils.openai_retrieval import async_query_llm
    elif args.api == "recurrent":
        from utils import extract_answer
        from utils.recurrent import async_query_llm
    elif args.api == "recurrent-boxed":
        from utils import extract_boxed_answer as extract_answer
        from utils.recurrent_boxed import async_query_llm
    elif args.api == "boxed":
        from utils import extract_boxed_answer as extract_answer
        from utils.boxed import async_query_llm
    elif args.api == "parallel":
        from utils import extract_answer
        from utils.parallel import async_query_llm
    elif args.api == "parallel-boxed":
        from utils import extract_boxed_answer as extract_answer
        from utils.parallel_boxed import async_query_llm
    elif args.api == "parallel-boxed-no-merge":
        from utils import extract_boxed_answer as extract_answer
        from utils.parallel_boxed_no_merge import async_query_llm
    elif args.api == "parallel-boxed-no-check":
        from utils import extract_boxed_answer as extract_answer
        from utils.parallel_boxed_no_check import async_query_llm
    elif args.api == "parallel-boxed-v4":
        from utils import extract_boxed_answer as extract_answer
        from utils.parallel_boxed_ver4 import async_query_llm
    else:
        print(f"Invalid API: {args.api}")
        raise ValueError
    coros = []
    for item in data:
        kwargs = {}
        if args.api in (
            "parallel-boxed",
            "parallel-boxed-no-merge",
            "parallel-boxed-no-check",
            "parallel-boxed-v4",
        ):
            kwargs["use_answer_prefix"] = args.use_answer_prefix
        coro = async_query_llm(
            item,
            model,
            tokenizer,
            temperature=0.7,
            top_p=0.95,
            **kwargs,
        )
        coros.append(coro)
    import uvloop
    from utils.aio import async_main, close_async_client

    outputs = uvloop.run(async_main(coros, args.n_proc))
    uvloop.run(async_main([close_async_client()]))
    from collections import defaultdict

    scores = defaultdict(list)
    fout = open(out_file, "w" if args.force else "a", encoding="utf-8")
    for i, (output, item) in enumerate(zip(outputs, data)):
        response = ""
        trace = None
        if isinstance(output, dict):
            response = (output.get("response") or output.get("final") or "").strip()
            trace = output.get("trace", output.get("conversation"))
            for k in (
                "parallel_passes_used",
                "parallel_converged",
                "parallel_max_passes",
                "conversation",
                "step",
                "max_step",
                "perf",
            ):
                if k in output:
                    item[k] = output[k]
        elif output:
            response = output.strip()

        if response == "":
            continue
        pred, _ = extract_solution(response)
        item["response"] = response
        item["answer"] = item.pop("outputs")
        item["pred"] = extract_answer(pred) if pred else extract_answer(response)
        if "qa" in args.split:
            if item["pred"]:
                metrics = calc_qa_metrics([item["pred"]], [item["answer"][0]])
            else:
                metrics = {
                    "f1": 0,
                    "prec": 0,
                    "recall": 0,
                    "em": 0,
                    "sub_em": 0,
                    "total_num": 0,
                }
            item["judge_sub_em"] = metrics["sub_em"]
            item["judge_em"] = metrics["em"]
            item["judge_f1"] = metrics["f1"]
            scores["em"].append(item["judge_em"])
            scores["f1"].append(item["judge_f1"])
            scores["sub_em"].append(item["judge_sub_em"])
        else:
            item["judge_sub_em"] = (
                calc_metrics([item["pred"]], [item["answer"]])["sub_em"]
                if item["pred"]
                else 0
            )
            scores["sub_em"].append(item["judge_sub_em"])
        if trace is not None:
            # item["trace"] = clip_for_dump(trace)
            item["trace"] = trace
            item["input_clipped"] = clip_long_string(
                item.get("input", ""), max_length=DUMP_CLIP_LENGTH
            )
            item["context_clipped"] = clip_long_string(
                item.get("context", ""), max_length=DUMP_CLIP_LENGTH
            )
        for key in ("conversation", "perf"):
            if key in item:
                item[key] = clip_for_dump(item[key])
        item.pop("context")
        fout.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")
        if i == 0:
            print("=" * 40 + "New Item Start" + "=" * 40)
            print(item["response"])
            print("-" * 80)
            print(item["pred"])
            print("-" * 80)
            print(item["answer"])
            print("-" * 80)
            print(item["judge_sub_em"])
            print("=" * 40 + "New Item End" + "=" * 40)
    print(f"ruler_general [{args.length}]")
    for k, v in scores.items():
        print(f"{k}: {round(sum(v) * 100 /len(v), 2)}")
    print(f"Total: {len(data)}")


# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs


# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


DOCS = None


def set_context(item):
    global DOCS
    if DOCS is None:
        if args.split == "qa_1":
            _, DOCS = read_squad(os.path.join(MEMORY_DATA_ROOT, "squad.json"))
        elif args.split == "qa_2":
            _, DOCS = read_hotpotqa(
                os.path.join(MEMORY_DATA_ROOT, "hotpotqa_dev.json")
            )
        else:
            raise ValueError
    all_docs = [DOCS[idx] for idx in item["context"]]
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    item["context"] = context
    return item


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")

    dataset = concatenate_datasets(
        [
            load_dataset(
                "json",
                data_files=f"{DATAROOT}/eval_{args.split}_{args.length}.json",
                split="train",
            ),
        ]
    )
    if args.num_samples is not None:
        print(f"Random sampling: {args.num_samples}")
        dataset = dataset.shuffle(seed=42).select(range(min(args.num_samples, len(dataset))))
    if isinstance(dataset[0]["context"], list):
        dataset = [[set_context(item) for item in dataset]]
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy

    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])

    # cache
    has_data = load_existing_ids(out_file)
    data = []
    for item in data_all:
        if item["_id"] not in has_data or args.force:
            data.append(item)
        elif args.force:
            data.append(item)

    get_pred(data, args, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="niah_single_1",
        choices=[
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
            "vt",
            "cwe",
            "fwe",
            "qa_1",
            "qa_2",
        ],
        help="split of the dataset",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=8192,
        choices=[
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
            1048576 * 2,
            1048576 * 4,
            10000000,
            5000000,
        ],
    )
    parser.add_argument("--save_dir", "-s", type=str, default="results/ruler_general")
    parser.add_argument(
        "--save_file", "-f", type=str, default="Qwen2.5-7B-Instruct-recurrent"
    )
    parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default="/mnt/hdfs/hongli/model/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--n_proc", "-n", type=int, default=64)
    parser.add_argument("--api", "-a", type=str, default="recurrent")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--use_answer_prefix",
        action="store_true",
        help="use dataset answer_prefix in the final parallel-boxed prompt",
    )
    parser.add_argument("--force", action="store_true", help="force to overrite")
    args = parser.parse_args()
    main()
