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
DEFAULT_DATA_ROOT = "hf://datasets/JetLM/PI-Mem-Data/hotpotqa_eval"
DEFAULT_MEMORY_DATA_ROOT = BASE_DIR.parent / "memory_data"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

HQA_LENGTHS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
HQA_OVER_1M_LENGTHS = {12800, 25600}
OOD_TASKS = [
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
OOD_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

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
    answer_prefix_for_vt: bool = False

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


@dataclass(frozen=True)
class EvalTask:
    group: str
    length: int
    subset: str | None = None

    @property
    def data_filename(self) -> str:
        if self.group == "hqa":
            return f"eval_{self.length}.json"
        return f"eval_{self.subset}_{self.length}.json"


DIRECT_ENV = EnvConfig(
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
    PARALLEL_MAX_PASSES=3,
    PARALLEL_MERGE_MAX_TOKENS=4096,
)

CONFIGS = [
    Config(
        name="vanilla",
        model_path=QWEN35_BASE_MODEL,
        method="openai",
        env=DIRECT_ENV,
        model_env_var="QWEN35_BASE_MODEL",
        serve_max_model_len=4_000_000,
    ),
    Config(
        name="yarn",
        model_path=QWEN35_YARN_MODEL,
        method="openai",
        env=DIRECT_ENV,
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
    Config(
        name="pi-mem-base",
        model_path=QWEN35_BASE_MODEL,
        method="parallel-boxed",
        env=PI_MEM_ENV,
        model_env_var="QWEN35_BASE_MODEL",
    ),
    Config(
        name="pi-mem-trained",
        model_path=QWEN35_PI_MEM_MODEL,
        method="parallel-boxed",
        env=PI_MEM_ENV,
        model_env_var="QWEN35_PI_MEM_MODEL",
    ),
]


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


def build_task_command(
    config: Config, task: EvalTask, args: argparse.Namespace
) -> list[str]:
    if task.group == "hqa":
        script = "ruler_hqa_over1m.py" if task.length in HQA_OVER_1M_LENGTHS else "ruler_hqa.py"
        save_dir = Path(args.results_dir) / f"ruler_hqa_{task.length}"
        command = [sys.executable, script, "--length", str(task.length)]
    else:
        script = "ruler_general.py"
        save_dir = Path(args.results_dir) / f"ruler_{task.subset}_{task.length}"
        command = [
            sys.executable,
            script,
            "--split",
            str(task.subset),
            "--length",
            str(task.length),
        ]
        if config.answer_prefix_for_vt and task.subset == "vt":
            command.append("--use_answer_prefix")
    command.extend(
        [
            "--save_dir",
            str(save_dir),
            "--save_file",
            config.name,
            "--model",
            config.api_model,
            "--tokenizer",
            config.model_path,
            "--api",
            config.method,
            "--num_samples",
            str(args.num_samples),
            "--n_proc",
            str(args.n_proc if args.n_proc is not None else config.n_proc),
        ]
    )
    if args.force:
        command.append("--force")
    return command


def run_config(config: Config, tasks: list[EvalTask], args: argparse.Namespace) -> None:
    env = ensure_local_no_proxy(config.env.apply(os.environ.copy()))
    env["SERVE_PORT"] = str(args.port)
    env["DASH_PORT"] = str(args.dash_port)
    env["DATAROOT"] = args.data_root
    env["MEMORY_DATA_ROOT"] = args.memory_data_root
    server_process = None
    try:
        if args.no_start_server:
            wait_for_server(args.port, config.api_model, args.server_timeout)
        else:
            server_process = start_server(config, args, env)
            wait_for_server(
                args.port, config.api_model, args.server_timeout, server_process
            )
        for task in tasks:
            command = build_task_command(config, task, args)
            print(" ".join(command))
            subprocess.run(command, cwd=BASE_DIR, env=env, check=True)
    finally:
        if not args.keep_server and not args.no_start_server:
            stop_server(server_process)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def select_configs(value: str, configs: list[Config]) -> list[Config]:
    config_map = {config.name: config for config in configs}
    names = list(config_map) if value.strip() == "all" else parse_csv(value)
    unknown = [name for name in names if name not in config_map]
    if unknown:
        raise ValueError(f"Unknown configs {unknown}; available: {list(config_map)}")
    if not names:
        raise ValueError("At least one config must be selected")
    return [config_map[name] for name in names]


def select_tasks(args: argparse.Namespace) -> list[EvalTask]:
    groups = parse_csv(args.tasks)
    unknown_groups = [group for group in groups if group not in {"hqa", "ood"}]
    if unknown_groups:
        raise ValueError(f"Unknown task groups: {unknown_groups}; available: hqa,ood")
    hqa_lengths = parse_int_csv(args.hqa_lengths)
    ood_subsets = parse_csv(args.ood_tasks)
    ood_lengths = parse_int_csv(args.ood_lengths)
    invalid_hqa = [length for length in hqa_lengths if length not in HQA_LENGTHS]
    invalid_ood_tasks = [task for task in ood_subsets if task not in OOD_TASKS]
    invalid_ood_lengths = [length for length in ood_lengths if length not in OOD_LENGTHS]
    if invalid_hqa or invalid_ood_tasks or invalid_ood_lengths:
        raise ValueError(
            "Invalid task selection: "
            f"hqa_lengths={invalid_hqa}, ood_tasks={invalid_ood_tasks}, "
            f"ood_lengths={invalid_ood_lengths}"
        )
    tasks = []
    for group in groups:
        if group == "hqa":
            tasks.extend(EvalTask("hqa", length) for length in hqa_lengths)
        else:
            tasks.extend(
                EvalTask("ood", length, subset)
                for subset in ood_subsets
                for length in ood_lengths
            )
    return tasks


def validate_local_data(tasks: list[EvalTask], args: argparse.Namespace) -> None:
    if args.no_data_check:
        return
    if not args.data_root.startswith("hf://"):
        data_root = Path(args.data_root).expanduser()
        missing = [task.data_filename for task in tasks if not (data_root / task.data_filename).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} evaluation files under {data_root}: {preview}"
            )
    memory_root = Path(args.memory_data_root).expanduser()
    required = set()
    if any(task.group == "hqa" and task.length in HQA_OVER_1M_LENGTHS for task in tasks):
        required.update({"hotpotqa_dev.json", "processing.py"})
    if any(task.group == "ood" and task.subset == "qa_1" for task in tasks):
        required.add("squad.json")
    missing = [name for name in sorted(required) if not (memory_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing support files under {memory_root}: {', '.join(missing)}. "
            "Set MEMORY_DATA_ROOT/--memory-data-root to their local directory."
        )


def create_parser(description: str, default_results_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--configs",
        "--config",
        default="pi-mem-trained",
        help="Comma-separated config names in execution order, or 'all'.",
    )
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument(
        "--tasks",
        default="hqa,ood",
        help="Comma-separated task groups in execution order: hqa,ood.",
    )
    parser.add_argument(
        "--hqa-lengths", default=",".join(str(length) for length in HQA_LENGTHS)
    )
    parser.add_argument("--ood-tasks", default=",".join(OOD_TASKS))
    parser.add_argument(
        "--ood-lengths", default=",".join(str(length) for length in OOD_LENGTHS)
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-proc", type=int, default=None)
    parser.add_argument("--results-dir", default=str(default_results_dir))
    parser.add_argument(
        "--data-root", default=os.getenv("DATA_ROOT", DEFAULT_DATA_ROOT)
    )
    parser.add_argument(
        "--memory-data-root",
        default=os.getenv("MEMORY_DATA_ROOT", str(DEFAULT_MEMORY_DATA_ROOT)),
    )
    parser.add_argument("--no-data-check", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVE_PORT", "8000")))
    parser.add_argument(
        "--dash-port", type=int, default=int(os.getenv("DASH_PORT", "8265"))
    )
    parser.add_argument("--server-timeout", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--shutdown-existing", action="store_true", default=True)
    parser.add_argument(
        "--no-shutdown-existing", dest="shutdown_existing", action="store_false"
    )
    return parser


def run_main(
    configs: list[Config], description: str, default_results_dir: Path
) -> None:
    args = create_parser(description, default_results_dir).parse_args()
    if args.list_configs:
        for config in configs:
            print(
                f"{config.name:17} method={config.method:16} "
                f"tp={config.tp} model={config.model_path}"
            )
        return
    selected_configs = select_configs(args.configs, configs)
    tasks = select_tasks(args)
    if not tasks:
        raise ValueError("At least one evaluation task must be selected")
    if (args.no_start_server or args.keep_server) and len(selected_configs) != 1:
        raise ValueError("--no-start-server/--keep-server require exactly one config")
    if not args.data_root.startswith("hf://"):
        args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.memory_data_root = str(Path(args.memory_data_root).expanduser().resolve())
    args.results_dir = str(Path(args.results_dir).expanduser().resolve())
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    validate_local_data(tasks, args)
    print(f"data root: {args.data_root}")
    print(f"task groups: {args.tasks}; total tasks per config: {len(tasks)}")
    for config in selected_configs:
        print("=" * 80)
        print(f"Running config: {config.name}")
        print("=" * 80)
        run_config(config, tasks, args)


def main() -> None:
    run_main(
        CONFIGS,
        "Run ordered Qwen3.5 configurations on full HQA and RULER OOD.",
        BASE_DIR / "results_qwen35",
    )


if __name__ == "__main__":
    main()
