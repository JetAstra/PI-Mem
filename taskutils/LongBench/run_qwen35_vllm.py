import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3.5-35B-A3B"
DEFAULT_MODEL_NAME = "Qwen3.5-35B-A3B"
DEFAULT_API_KEY = "token-abc123"
LOCAL_NO_PROXY = ["127.0.0.1", "localhost", "::1"]
PROXY_ENV_VARS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def ensure_local_no_proxy(env):
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        parts = [x.strip() for x in current.split(",") if x.strip()]
        for item in LOCAL_NO_PROXY:
            if item not in parts:
                parts.append(item)
        env[key] = ",".join(parts)
    return env


def run_command(cmd, cwd=None, env=None):
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def shutdown_existing_serve(dash_port):
    subprocess.run(
        ["serve", "shutdown", "-a", f"http://localhost:{dash_port}"],
        input="yes\n",
        text=True,
        env=ensure_local_no_proxy(os.environ.copy()),
        check=False,
    )


def start_server(args, api_model):
    serve_script = REPO_ROOT / "serve" / "llm0180.py"
    env = ensure_local_no_proxy(os.environ.copy())
    env["SERVE_PORT"] = str(args.port)
    env["DASH_PORT"] = str(args.dash_port)

    if args.shutdown_existing:
        shutdown_existing_serve(args.dash_port)

    cmd = [
        sys.executable,
        str(serve_script),
        "--model",
        args.model_path,
        "--tp",
        str(args.tp),
        "--port",
        str(args.port),
        "--dash-port",
        str(args.dash_port),
    ]
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.hf_overrides is not None:
        cmd.extend(["--hf-overrides", args.hf_overrides])

    print("serving command:", flush=True)
    print(" ".join(cmd), flush=True)
    print(f"expect model id: {api_model}", flush=True)
    return subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)


def fetch_model_ids(port):
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["id"] for item in payload.get("data", [])]


def wait_for_server(port, api_model, timeout, process=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            model_ids = fetch_model_ids(port)
            if api_model in model_ids:
                print(f"server ready: {api_model}", flush=True)
                return
            print(f"waiting for model id {api_model}; available={model_ids}", flush=True)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"waiting for server: {exc}", flush=True)
        time.sleep(5)
    raise TimeoutError(f"server was not ready within {timeout} seconds")


def stop_server(process):
    if process is None:
        return
    if process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=30)


def run_pred(args, api_model):
    api_base = f"http://127.0.0.1:{args.port}/v1"
    cmd = [
        sys.executable,
        "pred.py",
        "--model",
        args.model_name,
        "--tokenizer",
        args.model_path,
        "--api_model",
        api_model,
        "--api_base",
        api_base,
        "--api_key",
        args.api_key,
        "--data_path",
        args.data_path,
        "--save_dir",
        args.save_dir,
        "--n_proc",
        str(args.n_proc),
        "--json_indent",
        str(args.json_indent),
    ]
    if args.cot:
        cmd.append("--cot")
    if args.no_context:
        cmd.append("--no_context")
    if args.rag > 0:
        cmd.extend(["--rag", str(args.rag)])
    if args.enable_thinking:
        cmd.append("--enable_thinking")
    else:
        cmd.append("--disable_thinking")
    if args.domains:
        cmd.extend(["--domains", args.domains])
    if args.sub_domains:
        cmd.extend(["--sub_domains", args.sub_domains])
    if args.lengths:
        cmd.extend(["--lengths", args.lengths])
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.max_samples_per_sub_domain is not None:
        cmd.extend(
            [
                "--max_samples_per_sub_domain",
                str(args.max_samples_per_sub_domain),
            ]
        )
    if args.split_by_sub_domain:
        cmd.append("--split_by_sub_domain")
    run_command(cmd, cwd=BASE_DIR)


def run_result(args):
    output = args.result_file
    if output is None:
        output = os.path.join(args.save_dir, "result.csv")
    cmd = [
        sys.executable,
        "result.py",
        "--results_dir",
        args.save_dir,
        "--output",
        output,
    ]
    if args.split_by_sub_domain:
        cmd.append("--aggregate_by_parent")
    run_command(cmd, cwd=BASE_DIR)
    print(f"result written to {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LongBench v2 on Qwen3.5-35B-A3B with Ray Serve + vLLM."
    )
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--api_model",
        type=str,
        default=None,
        help="Model id exposed by the OpenAI-compatible server. Defaults to basename(model_path).",
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVE_PORT", "8000")))
    parser.add_argument("--dash_port", type=int, default=int(os.getenv("DASH_PORT", "8265")))
    parser.add_argument("--server_timeout", type=int, default=1800)
    parser.add_argument("--max_model_len", type=int, default=10_000_000)
    parser.add_argument("--hf_overrides", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(BASE_DIR / "data" / "data.json"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(BASE_DIR / "results" / DEFAULT_MODEL_NAME),
        help="Experiment results directory.",
    )
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--n_proc", type=int, default=64)
    parser.add_argument("--json_indent", type=int, default=4)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--rag", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--sub_domains", type=str, default=None)
    parser.add_argument("--lengths", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_sub_domain", type=int, default=None)
    parser.add_argument(
        "--split_by_sub_domain",
        action="store_true",
        help="Write one prediction JSONL per LongBench sub_domain.",
    )
    parser.add_argument("--no_start_server", action="store_true")
    parser.add_argument("--keep_server", action="store_true")
    parser.add_argument("--shutdown_existing", action="store_true", default=True)
    parser.add_argument("--no_shutdown_existing", dest="shutdown_existing", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.save_dir = str(Path(args.save_dir).expanduser().resolve())
    if args.result_file is not None:
        args.result_file = str(Path(args.result_file).expanduser().resolve())
    ensure_local_no_proxy(os.environ)
    os.environ.setdefault("MAX_INPUT_LEN", "100000000000")
    api_model = args.api_model or Path(args.model_path).name
    server_process = None

    try:
        if args.no_start_server:
            wait_for_server(args.port, api_model, args.server_timeout)
        else:
            server_process = start_server(args, api_model)
            wait_for_server(args.port, api_model, args.server_timeout, server_process)
        run_pred(args, api_model)
        run_result(args)
    finally:
        if not args.keep_server and not args.no_start_server:
            stop_server(server_process)


if __name__ == "__main__":
    main()
