import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# DEFAULT_HQA_TESTS = [50, 100, 200, 400, 800, 1600, 3200]
DEFAULT_HQA_TESTS = [3200]
DEFAULT_GENERAL_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
]
# DEFAULT_GENERAL_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
DEFAULT_GENERAL_LENGTHS = [524288]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run taskutils/memory_eval against Qwen3.5-35B-A3B via a vLLM OpenAI-compatible server."
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-35B-A3B"),
        help="Local model path or HF repo id for Qwen3.5-35B-A3B.",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.getenv("QWEN35_TOKENIZER", None),
        help="Tokenizer path/repo id. Defaults to --model-path.",
    )
    parser.add_argument(
        "--served-model-name",
        default=os.getenv("QWEN35_SERVED_MODEL_NAME", None),
        help="Model name exposed by the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "completion", "parallel", "all"],
        default=os.getenv("MEMORY_EVAL_API", "all"),
        help="Which memory_eval backend to run.",
    )
    parser.add_argument(
        "--suite",
        choices=["hqa", "ood", "all"],
        default=os.getenv("MEMORY_EVAL_SUITE", "all"),
        help="Which evaluation suite to run.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=int(os.getenv("QWEN35_TP", "4")),
        help="Tensor parallel size for vLLM serve.",
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=int(os.getenv("MEMORY_EVAL_NPROC", "32")),
        help="Concurrent requests used by ruler scripts.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("SERVE_HOST", "127.0.0.1"),
        help="Host for the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SERVE_PORT", "8000")),
        help="Port for the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("QWEN35_MAX_MODEL_LEN", "140000")),
        help="vLLM max model len. 140k is enough for the bundled 64k/128k ruler tests.",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=int(os.getenv("MAX_INPUT_LEN", "140000")),
        help="Client-side truncation limit for openai/completion mode.",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=int(os.getenv("MAX_OUTPUT_LEN", "10000")),
        help="Client-side max output tokens.",
    )
    parser.add_argument(
        "--parallel-max-context-len",
        type=int,
        default=int(os.getenv("RECURRENT_MAX_CONTEXT_LEN", "140000")),
        help="Context limit used by the parallel evaluator.",
    )
    parser.add_argument(
        "--parallel-chunk-size",
        type=int,
        default=int(os.getenv("RECURRENT_CHUNK_SIZE", "16384")),
        help="Chunk size used by the parallel evaluator.",
    )
    parser.add_argument(
        "--parallel-max-new",
        type=int,
        default=int(os.getenv("RECURRENT_MAX_NEW", "10240")),
        help="Per-call max new tokens for the parallel evaluator.",
    )
    parser.add_argument(
        "--parallel-max-passes",
        type=int,
        default=int(os.getenv("PARALLEL_MAX_PASSES", "3")),
        help="Maximum number of refinement passes for the parallel evaluator.",
    )
    parser.add_argument(
        "--parallel-merge-max-tokens",
        type=int,
        default=int(os.getenv("PARALLEL_MERGE_MAX_TOKENS", "10240")),
        help="Max tokens for the merge step in the parallel evaluator.",
    )
    parser.add_argument(
        "--result-root",
        default=os.getenv(
            "MEMORY_EVAL_RESULT_ROOT",
            "results_qwen35",
        ),
        help="Directory root used to store evaluation outputs.",
    )
    parser.add_argument(
        "--trace-root",
        default=os.getenv(
            "MEMORY_EVAL_TRACE_ROOT",
            "results_qwen35",
        ),
        help="Directory root used to store parallel traces.",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=int(os.getenv("QWEN35_SERVER_TIMEOUT", "1800")),
        help="Seconds to wait for vLLM to become healthy.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("QWEN35_GPU_MEMORY_UTILIZATION", "0.85")),
        help="Passed to vLLM serve.",
    )
    parser.add_argument(
        "--model-impl",
        choices=["auto", "vllm", "transformers"],
        default=os.getenv("QWEN35_MODEL_IMPL", "auto"),
        help="vLLM model implementation backend.",
    )
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("QWEN35_ATTENTION_BACKEND", None),
        help="Optional vLLM attention backend override, e.g. FLASH_ATTN.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        default=os.getenv("QWEN35_DISABLE_CUSTOM_ALL_REDUCE", "0") == "1",
        help="Pass --disable-custom-all-reduce to vLLM serve.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.getenv("QWEN35_ENFORCE_EAGER", "0") == "1",
        help="Pass --enforce-eager to vLLM serve.",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Use an already running OpenAI-compatible server instead of starting vLLM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing jsonl outputs.",
    )
    return parser.parse_args()


def resolve_model_name(model_path: str, served_model_name: str | None) -> str:
    if served_model_name:
        return served_model_name
    candidate = Path(model_path)
    return candidate.name if candidate.exists() or "/" in model_path else model_path


def get_models(host: str, port: int) -> dict | None:
    url = f"http://{host}:{port}/v1/models"
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def wait_for_server(host: str, port: int, model_name: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = get_models(host, port)
        if data and any(m.get("id") == model_name for m in data.get("data", [])):
            print(f"[server] ready: {model_name} on {host}:{port}")
            return
        print("[server] waiting for vLLM ...")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for model {model_name} on {host}:{port}")


def build_server_cmd(args, model_name: str) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--served-model-name",
        model_name,
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--model-impl",
        args.model_impl,
        "--no-enable-log-requests",
        "--language-model-only",
    ]
    if args.attention_backend:
        cmd.extend(["--attention-backend", args.attention_backend])
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def start_server(args, model_name: str) -> subprocess.Popen:
    cmd = build_server_cmd(args, model_name)
    print("[server] command:")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
    wait_for_server(args.host, args.port, model_name, args.server_timeout)
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


def env_with_runtime(args) -> dict[str, str]:
    env = os.environ.copy()
    env["SERVE_HOST"] = args.host
    env["SERVE_PORT"] = str(args.port)
    env["VLLM_USE_V1"] = os.getenv("VLLM_USE_V1", env.get("VLLM_USE_V1", "1"))
    env["MAX_INPUT_LEN"] = str(args.max_input_len)
    env["MAX_OUTPUT_LEN"] = str(args.max_output_len)
    env["RECURRENT_MAX_CONTEXT_LEN"] = str(args.parallel_max_context_len)
    env["RECURRENT_CHUNK_SIZE"] = str(args.parallel_chunk_size)
    env["RECURRENT_MAX_NEW"] = str(args.parallel_max_new)
    env["PARALLEL_MAX_PASSES"] = str(args.parallel_max_passes)
    env["PARALLEL_MERGE_MAX_TOKENS"] = str(args.parallel_merge_max_tokens)
    return env


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("[eval] command:")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def run_hqa(
    args, api: str, model_name: str, tokenizer: str, env: dict[str, str]
) -> None:
    save_root = os.path.join(args.result_root, api)
    save_file = f"{model_name}-{api}"
    for length in DEFAULT_HQA_TESTS:
        cmd = [
            sys.executable,
            "ruler_hqa.py",
            "--model",
            model_name,
            "--length",
            str(length),
            "--save_dir",
            os.path.join(save_root, f"ruler_hqa_{length}"),
            "--save_file",
            save_file,
            "--tokenizer",
            tokenizer,
            "--api",
            api,
            "--n_proc",
            str(args.n_proc),
        ]
        if api == "parallel":
            cmd.extend(
                [
                    "--trace_dir",
                    os.path.join(args.trace_root, api, f"ruler_hqa_{length}"),
                ]
            )
        if args.force:
            cmd.append("--force")
        run_cmd(cmd, env)


def run_ood(
    args, api: str, model_name: str, tokenizer: str, env: dict[str, str]
) -> None:
    save_root = os.path.join(args.result_root, api)
    save_file = f"{model_name}-{api}"
    for task in DEFAULT_GENERAL_TASKS:
        for length in DEFAULT_GENERAL_LENGTHS:
            cmd = [
                sys.executable,
                "ruler_general.py",
                "--model",
                model_name,
                "--split",
                task,
                "--length",
                str(length),
                "--save_dir",
                os.path.join(save_root, f"ruler_{task}_{length}"),
                "--save_file",
                save_file,
                "--tokenizer",
                tokenizer,
                "--api",
                api,
                "--n_proc",
                str(args.n_proc),
            ]
            if api == "parallel":
                cmd.extend(
                    [
                        "--trace_dir",
                        os.path.join(args.trace_root, api, f"ruler_{task}_{length}"),
                    ]
                )
            if args.force:
                cmd.append("--force")
            run_cmd(cmd, env)


def main():
    args = parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tokenizer = args.tokenizer or args.model_path
    model_name = resolve_model_name(args.model_path, args.served_model_name)
    env = env_with_runtime(args)

    server_proc = None
    try:
        if args.no_start_server:
            wait_for_server(args.host, args.port, model_name, args.server_timeout)
        else:
            server_proc = start_server(args, model_name)

        apis = ["openai", "completion", "parallel"] if args.api == "all" else [args.api]
        for api in apis:
            if args.suite in ("hqa", "all"):
                run_hqa(args, api, model_name, tokenizer, env)
            if args.suite in ("ood", "all"):
                run_ood(args, api, model_name, tokenizer, env)
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()
