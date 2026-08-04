import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer


LOCAL_NO_PROXY = ["127.0.0.1", "localhost", "::1"]
PROXY_ENV_VARS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
MEMORY_EVAL_DIR = REPO_ROOT / "taskutils" / "memory_eval"

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"


def ensure_local_no_proxy():
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [x.strip() for x in current.split(",") if x.strip()]
        for item in LOCAL_NO_PROXY:
            if item not in parts:
                parts.append(item)
        os.environ[key] = ",".join(parts)


ensure_local_no_proxy()

model_map = json.loads(
    (BASE_DIR / "config" / "model2path.json").read_text(encoding="utf-8")
)
maxlen_map = json.loads(
    (BASE_DIR / "config" / "model2maxlen.json").read_text(encoding="utf-8")
)

template_rag = (BASE_DIR / "prompts" / "0shot_rag.txt").read_text(encoding="utf-8")
template_no_context = (BASE_DIR / "prompts" / "0shot_no_context.txt").read_text(
    encoding="utf-8"
)
template_0shot = (BASE_DIR / "prompts" / "0shot.txt").read_text(encoding="utf-8")
template_0shot_cot = (BASE_DIR / "prompts" / "0shot_cot.txt").read_text(
    encoding="utf-8"
)
template_0shot_cot_ans = (BASE_DIR / "prompts" / "0shot_cot_ans.txt").read_text(
    encoding="utf-8"
)


def clip_long_string(string, max_length=2000):
    if string is None:
        return None
    if len(string) <= max_length:
        return string
    marker = "\n\n...(truncated)\n\n"
    target_len = max_length - len(marker)
    return string[: target_len // 2] + marker + string[-target_len // 2 :]


def parse_json_objects(content, source=None, allow_partial=False):
    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    valid_end = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as exc:
            if allow_partial:
                label = f" in {source}" if source else ""
                print(f"Warning: ignoring invalid trailing JSON{label}: {exc}")
                return objects, valid_end, True
            raise
        objects.append(obj)
        valid_end = idx
    return objects, valid_end, False


def load_existing_ids(out_file, repair_partial=False):
    if not os.path.exists(out_file):
        return {}
    path = Path(out_file)
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return {}
    ids = {}
    objects, valid_end, repaired = parse_json_objects(
        content, source=out_file, allow_partial=repair_partial
    )
    if repaired:
        repaired_content = content[:valid_end].rstrip()
        if repaired_content:
            repaired_content += "\n"
        path.write_text(repaired_content, encoding="utf-8")
    for obj in objects:
        if isinstance(obj, dict) and "_id" in obj:
            ids[obj["_id"]] = 0
    return ids


def dump_jsonl(out_file, records, indent=None, append=True):
    if indent is not None and indent <= 0:
        indent = None
    mode = "a" if append else "w"
    with open(out_file, mode, encoding="utf-8") as fout:
        for record in records:
            if record is None:
                continue
            fout.write(json.dumps(record, ensure_ascii=False, indent=indent) + "\n")
            fout.flush()


def sanitize_filename(value):
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    return name.strip("._") or "unknown"


def build_split_out_file(args, sub_domain):
    return os.path.join(args.save_dir, sanitize_filename(sub_domain) + ".jsonl")


def dump_jsonl_by_sub_domain(records, args):
    grouped = defaultdict(list)
    for record in records:
        if record is not None:
            grouped[record["sub_domain"]].append(record)
    for sub_domain in sorted(grouped):
        dump_jsonl(
            build_split_out_file(args, sub_domain),
            grouped[sub_domain],
            args.json_indent,
            append=False,
        )


def get_tokenizer(model, tokenizer_path):
    if any(name in model for name in ("gpt", "o1", "o3", "o4")):
        return tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    if tokenizer_path is None:
        tokenizer_path = model_map[model]
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def encode_text(tokenizer, text):
    try:
        return tokenizer.encode(text, disallowed_special=())
    except TypeError:
        return tokenizer.encode(text)


def decode_tokens(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def truncate_prompt(prompt, model, tokenizer):
    max_len = int(os.getenv("MAX_INPUT_LEN", str(maxlen_map.get(model, 120000))))
    input_ids = encode_text(tokenizer, prompt)
    if len(input_ids) <= max_len:
        return prompt
    input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
    return decode_tokens(tokenizer, input_ids)


def extract_longbench_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return normalize_choice(response)


def normalize_choice(answer):
    if answer is None:
        return None
    answer = str(answer).strip()
    if not answer:
        return None
    stripped = answer.strip().strip(".。,:;，：；").strip()
    stripped = stripped.strip("()[]{}").strip()
    if re.fullmatch(r"[A-D]", stripped):
        return stripped
    match = re.search(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", answer)
    if match:
        return match.group(1)
    return None


def build_longbench_prompt(item, context, template):
    return (
        template.replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def build_agent_input(item):
    return (
        f"What is the correct answer to this question: {item['question'].strip()}\n"
        "Choices:\n"
        f"(A) {item['choice_A'].strip()}\n"
        f"(B) {item['choice_B'].strip()}\n"
        f"(C) {item['choice_C'].strip()}\n"
        f"(D) {item['choice_D'].strip()}\n\n"
        "Answer with the letter A, B, C, or D."
    )


def select_context_and_template(item, args):
    context = item["context"]
    if args.rag > 0:
        template = template_rag
        retrieved = item.get("retrieved_context", [])[: args.rag]
        retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
        context = "\n\n".join(
            [
                f"Retrieved chunk {idx + 1}: {x['content']}"
                for idx, x in enumerate(retrieved)
            ]
        )
    elif args.no_context:
        template = template_no_context
    elif args.cot:
        template = template_0shot_cot
    else:
        template = template_0shot
    return context, template


async def post_chat_completion(
    session,
    url,
    api_key,
    model,
    prompt,
    max_tokens,
    temperature=1.0,
    top_p=0.95,
    enable_thinking=False,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    for attempt in range(5):
        try:
            async with session.post(
                url=url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"status={resp.status}, model={model}, body={text[:500]}")
                    await asyncio.sleep(1)
                    continue
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f'Error Occurs: "{exc}"        Retry ...')
            await asyncio.sleep(1)
    print("Max tries. Failed.")
    return ""


async def async_query_openai(item, args, tokenizer):
    from aiohttp import ClientSession, ClientTimeout

    request_model = args.api_model or model_map.get(args.model, args.model)
    max_output_len = int(os.getenv("MAX_OUTPUT_LEN", "128"))
    context, template = select_context_and_template(item, args)
    prompt = build_longbench_prompt(item, context, template)
    prompt = truncate_prompt(prompt, args.model, tokenizer)
    api_base = args.api_base.rstrip("/")
    url = f"{api_base}/chat/completions"

    async with ClientSession(timeout=ClientTimeout(total=86400)) as session:
        output = await post_chat_completion(
            session,
            url,
            args.api_key,
            request_model,
            prompt,
            max_output_len,
            enable_thinking=args.enable_thinking,
        )
        if output == "":
            return None
        if args.cot:
            cot_response = output.strip()
            prompt = (
                template_0shot_cot_ans.replace("$DOC$", context.strip())
                .replace("$Q$", item["question"].strip())
                .replace("$C_A$", item["choice_A"].strip())
                .replace("$C_B$", item["choice_B"].strip())
                .replace("$C_C$", item["choice_C"].strip())
                .replace("$C_D$", item["choice_D"].strip())
                .replace("$COT$", cot_response)
            )
            prompt = truncate_prompt(prompt, args.model, tokenizer)
            output = await post_chat_completion(
                session,
                url,
                args.api_key,
                request_model,
                prompt,
                max_output_len,
                enable_thinking=args.enable_thinking,
            )
            if output == "":
                return None
        else:
            cot_response = None

    response = output.strip()
    record = build_output_record(item, response, context)
    if cot_response is not None:
        record["response_cot"] = cot_response
    record["pred"] = extract_longbench_answer(response)
    record["judge"] = record["pred"] == item["answer"]
    return record


def configure_memory_eval_imports(args):
    parsed = urlparse(args.api_base)
    if parsed.hostname:
        os.environ["SERVE_HOST"] = parsed.hostname
    if parsed.port:
        os.environ["SERVE_PORT"] = str(parsed.port)
    if str(MEMORY_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(MEMORY_EVAL_DIR))


def import_agent_api(args):
    configure_memory_eval_imports(args)
    from utils import extract_boxed_answer
    from utils.aio import async_main, close_async_client

    if args.api == "recurrent-boxed":
        from utils.recurrent_boxed import async_query_llm
    elif args.api == "parallel-boxed":
        from utils.parallel_boxed import async_query_llm
    elif args.api == "parallel-boxed-v2":
        from utils.parallel_boxed_ver2 import async_query_llm
    elif args.api == "parallel-boxed-v3":
        from utils.parallel_boxed_ver3 import async_query_llm
    elif args.api == "parallel-boxed-v4":
        from utils.parallel_boxed_ver4 import async_query_llm
    elif args.api == "parallel-boxed-v5":
        from utils.parallel_boxed_ver5 import async_query_llm
    elif args.api == "parallel-boxed-v6":
        from utils.parallel_boxed_ver6 import async_query_llm
    elif args.api == "parallel-boxed-v7":
        from utils.parallel_boxed_ver7 import async_query_llm
    elif args.api == "parallel-boxed-v8":
        from utils.parallel_boxed_ver8 import async_query_llm
    elif args.api == "parallel-boxed-v9":
        from utils.parallel_boxed_ver9 import async_query_llm
    elif args.api == "parallel-boxed-v10":
        from utils.parallel_boxed_ver10 import async_query_llm
    elif args.api == "parallel-boxed-v11":
        from utils.parallel_boxed_ver11 import async_query_llm
    elif args.api == "parallel-boxed-v12":
        from utils.parallel_boxed_ver12 import async_query_llm
    elif args.api == "parallel-boxed-v13":
        from utils.parallel_boxed_ver13 import async_query_llm
    elif args.api == "parallel-boxed-v14":
        from utils.parallel_boxed_ver14 import async_query_llm
    elif args.api == "parallel-boxed-v15":
        from utils.parallel_boxed_ver15 import async_query_llm
    elif args.api == "parallel-boxed-v16":
        from utils.parallel_boxed_ver16 import async_query_llm
    elif args.api == "rag":
        from utils.openai_retrieval import async_query_llm
    else:
        raise ValueError(f"Unsupported agent API: {args.api}")
    return async_query_llm, extract_boxed_answer, async_main, close_async_client


async def async_query_agent(item, args, tokenizer, async_query_llm):
    request_model = args.api_model or args.model
    agent_item = deepcopy(item)
    agent_item["input"] = build_agent_input(item)
    agent_item["context"] = item["context"]
    kwargs = {}
    if args.api in (
        "parallel-boxed",
        "parallel-boxed-v2",
        "parallel-boxed-v3",
        "parallel-boxed-v4",
        "parallel-boxed-v5",
        "parallel-boxed-v6",
        "parallel-boxed-v7",
        "parallel-boxed-v8",
        "parallel-boxed-v9",
        "parallel-boxed-v10",
        "parallel-boxed-v11",
        "parallel-boxed-v12",
        "parallel-boxed-v13",
        "parallel-boxed-v14",
        "parallel-boxed-v15",
        "parallel-boxed-v16",
    ):
        kwargs["use_answer_prefix"] = False
    return await async_query_llm(
        agent_item,
        request_model,
        tokenizer,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        top_p=0.95,
        **kwargs,
    )


def build_output_record(item, response, context):
    record = {
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "response": response,
        "context_clipped": clip_long_string(context),
    }
    if "retrieved_context" in item:
        record["retrieved_context"] = item["retrieved_context"]
    return record


def run_async(coro):
    try:
        import uvloop

        return uvloop.run(coro)
    except ImportError:
        return asyncio.run(coro)


def get_openai_pred(data, args, out_file):
    tokenizer = get_tokenizer(args.model, args.tokenizer)
    configure_memory_eval_imports(args)
    from utils.aio import async_main

    coros = [async_query_openai(item, args, tokenizer) for item in data]
    records = run_async(async_main(coros, args.n_proc))
    if out_file is not None:
        dump_jsonl(out_file, records, args.json_indent)
    return records


def get_agent_pred(data, args, out_file):
    tokenizer = get_tokenizer(args.model, args.tokenizer)
    async_query_llm, extract_boxed_answer, async_main, close_async_client = (
        import_agent_api(args)
    )
    coros = [async_query_agent(item, args, tokenizer, async_query_llm) for item in data]
    outputs = run_async(async_main(coros, args.n_proc))
    run_async(close_async_client())

    records = []
    for output, item in zip(outputs, data):
        if isinstance(output, dict):
            response = output.get("response", "").strip()
        elif output:
            response = str(output).strip()
        else:
            response = ""
        if response == "":
            continue

        record = build_output_record(item, response, item["context"])
        if args.api == "rag":
            record["pred"] = normalize_choice(response)
        else:
            try:
                boxed = extract_boxed_answer(response)
            except AssertionError:
                boxed = None
            record["pred"] = normalize_choice(boxed)
        record["judge"] = record["pred"] == item["answer"]
        record["input_clipped"] = clip_long_string(build_agent_input(item))
        if isinstance(output, dict):
            trace = output.get("trace")
            if trace is not None:
                record["trace"] = trace
            for key in (
                "parallel_passes_used",
                "parallel_converged",
                "parallel_max_passes",
            ):
                if key in output:
                    record[key] = output[key]
        records.append(record)
    if out_file is not None:
        dump_jsonl(out_file, records, args.json_indent)
    return records


def build_out_file(args):
    if args.save_file:
        return os.path.join(args.save_dir, args.save_file + ".jsonl")
    suffix = ""
    if args.rag > 0:
        suffix = f"_rag_{args.rag}"
    elif args.no_context:
        suffix = "_no_context"
    elif args.cot:
        suffix = "_cot"
    return os.path.join(args.save_dir, args.model.split("/")[-1] + suffix + ".jsonl")


def load_longbench_data(args):
    if args.data_path:
        dataset = json.load(open(args.data_path, "r", encoding="utf-8"))
    else:
        dataset = load_dataset("THUDM/LongBench-v2", split="train")

    data_all = []
    for item in dataset:
        record = {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        if "retrieved_context" in item:
            record["retrieved_context"] = item["retrieved_context"]
        data_all.append(record)
    return data_all


def apply_filters(data_all, args):
    if args.domains:
        domains = {x.strip() for x in args.domains.split(",") if x.strip()}
        data_all = [item for item in data_all if item["domain"] in domains]
    if args.sub_domains:
        sub_domains = {x.strip() for x in args.sub_domains.split(",") if x.strip()}
        data_all = [item for item in data_all if item["sub_domain"] in sub_domains]
    if args.lengths:
        lengths = {x.strip() for x in args.lengths.split(",") if x.strip()}
        data_all = [item for item in data_all if item["length"] in lengths]
    if args.max_samples_per_sub_domain is not None:
        counts = {}
        sampled = []
        for item in data_all:
            key = item["sub_domain"]
            count = counts.get(key, 0)
            if count < args.max_samples_per_sub_domain:
                sampled.append(item)
                counts[key] = count + 1
        data_all = sampled
    if args.max_samples is not None:
        data_all = data_all[: args.max_samples]
    return data_all


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = build_out_file(args)
    data_all = apply_filters(load_longbench_data(args), args)

    if args.split_by_sub_domain:
        data = data_all
    else:
        if args.force and os.path.exists(out_file):
            os.remove(out_file)
        has_data = load_existing_ids(out_file, repair_partial=True)
        data = [item for item in data_all if item["_id"] not in has_data]
    if not data:
        if args.split_by_sub_domain:
            print("No data to process.")
        else:
            print(f"No new data to process for {out_file}")
        return

    if args.api == "openai":
        records = get_openai_pred(data, args, None if args.split_by_sub_domain else out_file)
    elif args.api in (
        "recurrent-boxed",
        "parallel-boxed",
        "parallel-boxed-v2",
        "parallel-boxed-v3",
        "parallel-boxed-v4",
        "parallel-boxed-v5",
        "parallel-boxed-v6",
        "parallel-boxed-v7",
        "parallel-boxed-v8",
        "parallel-boxed-v9",
        "parallel-boxed-v10",
        "parallel-boxed-v11",
        "parallel-boxed-v12",
        "parallel-boxed-v13",
        "parallel-boxed-v14",
        "parallel-boxed-v15",
        "parallel-boxed-v16",
        "rag",
    ):
        if args.cot or args.no_context or args.rag > 0:
            raise ValueError("Agent APIs support the standard context setting only.")
        records = get_agent_pred(data, args, None if args.split_by_sub_domain else out_file)
    else:
        raise ValueError(f"Invalid API: {args.api}")

    if args.split_by_sub_domain:
        dump_jsonl_by_sub_domain(records, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--save_file", "-f", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path/name. Defaults to config/model2path.json[model].",
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default=None,
        help="Model id sent to the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api",
        "-a",
        type=str,
        default="openai",
        choices=[
            "openai",
            "recurrent-boxed",
            "parallel-boxed",
            "parallel-boxed-v2",
            "parallel-boxed-v3",
            "parallel-boxed-v4",
            "parallel-boxed-v5",
            "parallel-boxed-v6",
            "parallel-boxed-v7",
            "parallel-boxed-v8",
            "parallel-boxed-v9",
            "parallel-boxed-v10",
            "parallel-boxed-v11",
            "parallel-boxed-v12",
            "parallel-boxed-v13",
            "parallel-boxed-v14",
            "parallel-boxed-v15",
            "parallel-boxed-v16",
            "rag",
        ],
    )
    parser.add_argument("--api_base", type=str, default=URL)
    parser.add_argument("--api_key", type=str, default=API_KEY)
    parser.add_argument("--enable_thinking", dest="enable_thinking", action="store_true")
    parser.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        default=False,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(BASE_DIR / "data" / "data.json"),
        help="Local LongBench v2 data.json. Set empty to load THUDM/LongBench-v2.",
    )
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--sub_domains", type=str, default=None)
    parser.add_argument("--lengths", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_sub_domain", type=int, default=None)
    parser.add_argument(
        "--split_by_sub_domain",
        action="store_true",
        help="Write one JSONL per LongBench sub_domain. Resume is not applied in this mode.",
    )
    parser.add_argument("--cot", "-cot", action="store_true")
    parser.add_argument("--no_context", "-nc", action="store_true")
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument(
        "--json_indent",
        type=int,
        default=4,
        help="Indent each output JSON object. Set <=0 for compact one-line JSONL.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main()
