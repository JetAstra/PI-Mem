import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
SERVE_SCRIPT = REPO_ROOT / "serve" / "llm0180.py"
DEFAULT_API_KEY = "token-abc123"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

QWEN35_BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
QWEN35_YARN_MODEL = os.getenv(
    "QWEN35_YARN_MODEL", str(REPO_ROOT / "models" / "Qwen3.5-35B-A3B-yarn")
)
QWEN35_MEMAGENT_MODEL = os.getenv(
    "QWEN35_MEMAGENT_MODEL",
    str(REPO_ROOT / "models" / "Qwen3.5-35B-A3B-MemAgent"),
)
QWEN35_PI_MEM_MODEL = os.getenv(
    "QWEN35_PI_MEM_MODEL", "JetLM/PI-Mem-35B-A3B"
)


@dataclass(frozen=True)
class EnvConfig:
    MAX_INPUT_LEN: int | None = None
    MAX_OUTPUT_LEN: int | None = None
    RECURRENT_MAX_CONTEXT_LEN: int | None = None
    RECURRENT_CHUNK_SIZE: int | None = None
    RECURRENT_MAX_NEW: int | None = None
    PARALLEL_MAX_PASSES: int | None = None
    PARALLEL_MERGE_MAX_TOKENS: int | None = None
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: int | None = None
    ENABLE_THINK: bool | None = None

    def apply(self, base_env: dict[str, str]) -> dict[str, str]:
        env = dict(base_env)
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                env[field.name] = str(value)
                print(f"set {field.name}={value}")
        return env


@dataclass(frozen=True)
class Config:
    name: str
    model_path: str
    method: str
    env: EnvConfig
    model_env_var: str
    tp: int = 2
    n_proc: int = 128
    serve_max_model_len: int | None = None
    serve_hf_overrides: str | None = None
    serve_enforce_eager: bool = False
    serve_disable_custom_all_reduce: bool = False

    @property
    def api_model(self) -> str:
        path = Path(self.model_path).expanduser()
        return path.name if path.is_dir() else self.model_path

    def validate_model_path(self) -> None:
        path = Path(self.model_path).expanduser()
        if path.is_absolute() and not path.exists():
            raise FileNotFoundError(
                f"Model directory does not exist: {path}. "
                f"Set {self.model_env_var} to the correct directory."
            )


VANILLA_ENV = EnvConfig(
    MAX_INPUT_LEN=4_000_000,
    MAX_OUTPUT_LEN=512,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
YARN_ENV = EnvConfig(
    MAX_INPUT_LEN=4_000_000,
    MAX_OUTPUT_LEN=4096,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
RAG_ENV = EnvConfig(
    MAX_INPUT_LEN=4_000_000,
    MAX_OUTPUT_LEN=4096,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
    ENABLE_THINK=False,
)
MEMAGENT_ENV = EnvConfig(
    RECURRENT_MAX_CONTEXT_LEN=100_000_000_000,
    RECURRENT_CHUNK_SIZE=15_000,
    RECURRENT_MAX_NEW=4096,
)
PI_MEM_ENV = EnvConfig(
    RECURRENT_MAX_CONTEXT_LEN=100_000_000_000,
    RECURRENT_CHUNK_SIZE=15_000,
    RECURRENT_MAX_NEW=4096,
    PARALLEL_MAX_PASSES=1,
    PARALLEL_MERGE_MAX_TOKENS=4096,
)

CONFIGS = [
    Config(
        name="vanilla",
        model_path=QWEN35_BASE_MODEL,
        method="openai",
        env=VANILLA_ENV,
        model_env_var="QWEN35_BASE_MODEL",
        serve_max_model_len=4_000_000,
    ),
    Config(
        name="yarn",
        model_path=QWEN35_YARN_MODEL,
        method="openai",
        env=YARN_ENV,
        model_env_var="QWEN35_YARN_MODEL",
        serve_max_model_len=4_000_000,
    ),
    Config(
        name="rag",
        model_path=QWEN35_BASE_MODEL,
        method="rag",
        env=RAG_ENV,
        model_env_var="QWEN35_BASE_MODEL",
        serve_max_model_len=4_000_000,
    ),
    Config(
        name="memagent-base",
        model_path=QWEN35_BASE_MODEL,
        method="recurrent-boxed",
        env=MEMAGENT_ENV,
        model_env_var="QWEN35_BASE_MODEL",
    ),
    Config(
        name="memagent-trained",
        model_path=QWEN35_MEMAGENT_MODEL,
        method="recurrent-boxed",
        env=MEMAGENT_ENV,
        model_env_var="QWEN35_MEMAGENT_MODEL",
    ),
    # Here we recommend v4 prompt for PI-Mem.
    Config(
        name="pi-mem-base",
        model_path=QWEN35_BASE_MODEL,
        method="parallel-boxed-v4",
        env=PI_MEM_ENV,
        model_env_var="QWEN35_BASE_MODEL",
    ),
    Config(
        name="pi-mem-trained",
        model_path=QWEN35_PI_MEM_MODEL,
        method="parallel-boxed-v4",
        env=PI_MEM_ENV,
        model_env_var="QWEN35_PI_MEM_MODEL",
    ),
]
CONFIG_MAP = {config.name: config for config in CONFIGS}


def ensure_local_no_proxy(env: dict[str, str]) -> dict[str, str]:
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    for key in ("NO_PROXY", "no_proxy"):
        values = [value.strip() for value in env.get(key, "").split(",") if value.strip()]
        for host in LOCAL_HOSTS:
            if host not in values:
                values.append(host)
        env[key] = ",".join(values)
    return env


def run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def shutdown_existing_serve(dash_port: int) -> None:
    subprocess.run(
        ["serve", "shutdown", "-a", f"http://localhost:{dash_port}"],
        input="yes\n",
        text=True,
        env=ensure_local_no_proxy(os.environ.copy()),
        check=False,
    )


def fetch_model_ids(port: int) -> list[str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["id"] for item in payload.get("data", [])]


def wait_for_server(
    port: int,
    api_model: str,
    timeout: int,
    process: subprocess.Popen | None = None,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}")
        try:
            model_ids = fetch_model_ids(port)
            if api_model in model_ids:
                print(f"server ready: {api_model}")
                return
            print(f"waiting for model id {api_model}; available={model_ids}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"waiting for server: {exc}")
        time.sleep(5)
    raise TimeoutError(f"Server was not ready within {timeout} seconds")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=30)


def start_server(
    config: Config, args: argparse.Namespace, env: dict[str, str]
) -> subprocess.Popen:
    config.validate_model_path()
    if args.shutdown_existing:
        shutdown_existing_serve(args.dash_port)
    command = [
        sys.executable,
        str(SERVE_SCRIPT),
        "--model",
        config.model_path,
        "--tp",
        str(config.tp),
        "--port",
        str(args.port),
        "--dash-port",
        str(args.dash_port),
    ]
    if config.serve_max_model_len is not None:
        command.extend(["--max-model-len", str(config.serve_max_model_len)])
    if config.serve_hf_overrides is not None:
        command.extend(["--hf-overrides", config.serve_hf_overrides])
    if config.serve_enforce_eager:
        command.append("--enforce-eager")
    if config.serve_disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    print("serving command:")
    print(" ".join(command))
    print(f"expect model id: {config.api_model}")
    return subprocess.Popen(command, env=env, preexec_fn=os.setsid)


def run_prediction(
    config: Config, args: argparse.Namespace, env: dict[str, str]
) -> None:
    output_dir = Path(args.save_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "pred.py",
        "--model",
        "Qwen3.5-35B-A3B",
        "--tokenizer",
        config.model_path,
        "--api_model",
        config.api_model,
        "--api",
        config.method,
        "--api_base",
        f"http://127.0.0.1:{args.port}/v1",
        "--api_key",
        args.api_key,
        "--data_path",
        args.data_path,
        "--save_dir",
        str(output_dir),
        "--save_file",
        config.name,
        "--n_proc",
        str(args.n_proc if args.n_proc is not None else config.n_proc),
        "--json_indent",
        str(args.json_indent),
        "--disable_thinking",
    ]
    for option in ("domains", "sub_domains", "lengths"):
        value = getattr(args, option)
        if value:
            command.extend([f"--{option}", value])
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    if args.max_samples_per_sub_domain is not None:
        command.extend(
            ["--max_samples_per_sub_domain", str(args.max_samples_per_sub_domain)]
        )
    if args.split_by_sub_domain:
        command.append("--split_by_sub_domain")
    if args.force:
        command.append("--force")
    run_command(command, cwd=BASE_DIR, env=env)


def run_config(config: Config, args: argparse.Namespace) -> None:
    env = ensure_local_no_proxy(config.env.apply(os.environ.copy()))
    env["SERVE_PORT"] = str(args.port)
    env["DASH_PORT"] = str(args.dash_port)
    server_process = None
    try:
        if args.no_start_server:
            wait_for_server(args.port, config.api_model, args.server_timeout)
        else:
            server_process = start_server(config, args, env)
            wait_for_server(
                args.port, config.api_model, args.server_timeout, server_process
            )
        run_prediction(config, args, env)
    finally:
        if not args.keep_server and not args.no_start_server:
            stop_server(server_process)


def select_configs(value: str) -> list[Config]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if value.strip() == "all":
        names = list(CONFIG_MAP)
    unknown = [name for name in names if name not in CONFIG_MAP]
    if unknown:
        raise ValueError(f"Unknown configs {unknown}; available: {list(CONFIG_MAP)}")
    if not names:
        raise ValueError("At least one config must be selected")
    return [CONFIG_MAP[name] for name in names]


def run_result(args: argparse.Namespace) -> None:
    output = args.result_file or str(Path(args.save_dir) / "result.csv")
    command = [
        sys.executable,
        "result.py",
        "--results_dir",
        args.save_dir,
        "--output",
        output,
    ]
    if args.split_by_sub_domain:
        command.append("--aggregate_by_parent")
    run_command(command, cwd=BASE_DIR, env=os.environ.copy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ordered Qwen3.5 configurations on LongBench v2."
    )
    parser.add_argument(
        "--configs",
        "--config",
        default="pi-mem-trained",
        help="Comma-separated config names in execution order, or 'all'.",
    )
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVE_PORT", "8000")))
    parser.add_argument(
        "--dash-port", type=int, default=int(os.getenv("DASH_PORT", "8265"))
    )
    parser.add_argument("--server-timeout", type=int, default=1800)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--data-path", default=str(BASE_DIR / "data" / "data.json"))
    parser.add_argument("--save-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--n-proc", type=int, default=None)
    parser.add_argument("--json-indent", type=int, default=4)
    parser.add_argument("--domains", default=None)
    parser.add_argument("--sub-domains", default=None)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-sub-domain", type=int, default=None)
    parser.add_argument("--split-by-sub-domain", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-result", action="store_true")
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--shutdown-existing", action="store_true", default=True)
    parser.add_argument(
        "--no-shutdown-existing", dest="shutdown_existing", action="store_false"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_configs:
        for config in CONFIGS:
            print(
                f"{config.name:17} method={config.method:16} "
                f"tp={config.tp} model={config.model_path}"
            )
        return
    configs = select_configs(args.configs)
    if (args.no_start_server or args.keep_server) and len(configs) != 1:
        raise ValueError("--no-start-server/--keep-server require exactly one config")
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.save_dir = str(Path(args.save_dir).expanduser().resolve())
    if args.result_file is not None:
        args.result_file = str(Path(args.result_file).expanduser().resolve())
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    for config in configs:
        print("=" * 80)
        print(f"Running config: {config.name}")
        print("=" * 80)
        run_config(config, args)
    if not args.skip_result:
        run_result(args)


if __name__ == "__main__":
    main()
