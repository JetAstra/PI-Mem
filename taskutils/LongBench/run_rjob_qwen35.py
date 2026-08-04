import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from run_qwen35_vllm import (
    ensure_local_no_proxy,
    run_command,
    shutdown_existing_serve,
    stop_server,
    wait_for_server,
)


sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
SERVE_SCRIPT = REPO_ROOT / "serve" / "llm0180.py"
MODEL_NAME = "Qwen3.5-35B-A3B"
QWEN25_MODEL_NAME = "Qwen2.5-7B-Instruct"
DEFAULT_API_KEY = "token-abc123"

QWEN35_BASE = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/"
    "models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)
QWEN35_VANILLA_LONG = (
    "/mnt/shared-storage-user/liudawei/home/verl/models/Qwen3.5-35B-A3B"
)
QWEN35_VANILLA_YARN = (
    "/mnt/shared-storage-user/liudawei/home/verl/models/Qwen3.5-35B-A3B-yarn"
)
QWEN35_RECURRENT_RL_STEP80 = (
    "/mnt/shared-storage-user/liudawei/home/verl-new/checkpoints/ParallelAgent/"
    "Qwen3_5-35B-A3B-megatron-memagent-rjob/global_step_80/actor/huggingface"
)
QWEN35_PARALLEL_RL_STEP80 = (
    "/mnt/shared-storage-user/liudawei/home/verl-new/checkpoints/ParallelAgent/"
    "Qwen3_5-35B-A3B-v2-megatron-rjob/global_step_80/actor/huggingface"
)
QWEN25_VANILLA = "/mnt/shared-storage-user/liudawei/home/verl/models/Qwen2.5-7B-Instruct"
QWEN25_YARN = "/mnt/shared-storage-user/liudawei/home/verl/models/Qwen2.5-7B-Instruct-yarn"
QWEN25_RECURRENT_RL = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/"
    "models--BytedTsinghua-SIA--RL-MemoryAgent-7B"
)
QWEN25_PARALLEL_RL_STEP240 = (
    "/mnt/shared-storage-user/liudawei/work_dirs_ckpt/verl/"
    "Qwen2.5-7B-8GPU-2nodes-parallel-rjob/ckpt/global_step_240/actor/huggingface"
)

@dataclass
class ENV:
    MAX_INPUT_LEN: int | None = None
    MAX_OUTPUT_LEN: int | None = None
    RECURRENT_MAX_CONTEXT_LEN: int | None = None
    RECURRENT_CHUNK_SIZE: int | None = None
    RECURRENT_MAX_NEW: int | None = None
    PARALLEL_MAX_PASSES: int | None = None
    PARALLEL_MERGE_MAX_TOKENS: int | None = None
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: int | None = None
    ENABLE_THINK: bool | None = None

    def apply(self, base_env):
        env = dict(base_env)
        for key, value in self.__dict__.items():
            if value is not None:
                env[key] = str(value)
                print(f"set {key}={value}")
        return env


class Config:
    def __init__(
        self,
        name,
        ckpt,
        method,
        env,
        tp=2,
        concur=128,
        serve_max_model_len=None,
        serve_hf_overrides=None,
        serve_enforce_eager=False,
        serve_disable_custom_all_reduce=False,
        model_name=MODEL_NAME,
    ):
        self.name = name
        self.ckpt = ckpt
        self.method = method
        self.env = env
        self.tp = tp
        self.concur = concur
        self.serve_max_model_len = serve_max_model_len
        self.serve_hf_overrides = serve_hf_overrides
        self.serve_enforce_eager = serve_enforce_eager
        self.serve_disable_custom_all_reduce = serve_disable_custom_all_reduce
        self.model_name = model_name
        self.api_model = Path(ckpt).name if Path(ckpt).is_dir() else ckpt
        self.server_process = None

    def build_env(self, args):
        env = ensure_local_no_proxy(os.environ.copy())
        env["SERVE_PORT"] = str(args.port)
        env["DASH_PORT"] = str(args.dash_port)
        return self.env.apply(env)

    def output_dir(self, args):
        return os.path.join(args.save_dir, self.name)

    def serve(self, args, env):
        if args.shutdown_existing:
            shutdown_existing_serve(args.dash_port)

        cmd = [
            sys.executable,
            str(SERVE_SCRIPT),
            "--model",
            self.ckpt,
            "--tp",
            str(self.tp),
            "--port",
            str(args.port),
            "--dash-port",
            str(args.dash_port),
        ]
        if self.serve_max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.serve_max_model_len)])
        if self.serve_hf_overrides is not None:
            cmd.extend(["--hf-overrides", self.serve_hf_overrides])
        if self.serve_enforce_eager:
            cmd.append("--enforce-eager")
        if self.serve_disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")

        print("serving command:")
        print(" ".join(cmd))
        print(f"expect model id: {self.api_model}")
        self.server_process = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
        wait_for_server(args.port, self.api_model, args.server_timeout, self.server_process)

    def run_pred(self, args, env):
        api_base = f"http://127.0.0.1:{args.port}/v1"
        output_dir = self.output_dir(args)
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            sys.executable,
            "pred.py",
            "--model",
            self.model_name,
            "--tokenizer",
            self.ckpt,
            "--api_model",
            self.api_model,
            "--api",
            self.method,
            "--api_base",
            api_base,
            "--api_key",
            args.api_key,
            "--data_path",
            args.data_path,
            "--save_dir",
            output_dir,
            "--save_file",
            self.name,
            "--n_proc",
            str(self.concur),
            "--json_indent",
            str(args.json_indent),
            "--disable_thinking",
        ]
        if args.force:
            cmd.append("--force")
        if self.method == "openai":
            if args.cot:
                cmd.append("--cot")
            if args.no_context:
                cmd.append("--no_context")
            if args.rag > 0:
                cmd.extend(["--rag", str(args.rag)])
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
                ["--max_samples_per_sub_domain", str(args.max_samples_per_sub_domain)]
            )
        if args.split_by_sub_domain:
            cmd.append("--split_by_sub_domain")
        run_command(cmd, cwd=BASE_DIR, env=env)
        return output_dir

    def run(self, args):
        env = self.build_env(args)
        try:
            if args.no_start_server:
                wait_for_server(args.port, self.api_model, args.server_timeout)
            else:
                self.serve(args, env)
            return self.run_pred(args, env)
        finally:
            if not args.keep_server and not args.no_start_server:
                stop_server(self.server_process)


OPENAI_ENV = ENV(
    MAX_INPUT_LEN=100000000000,
    MAX_OUTPUT_LEN=512,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
OPENAI_YARN_ENV = ENV(
    MAX_INPUT_LEN=100000000000,
    MAX_OUTPUT_LEN=4096,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
OPENAI_QWEN25_ENV = ENV(
    MAX_INPUT_LEN=3_699_000,
    MAX_OUTPUT_LEN=1024,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
OPENAI_QWEN25_YARN_ENV = ENV(
    MAX_INPUT_LEN=272_144,
    MAX_OUTPUT_LEN=1024,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
RAG_QWEN35_ENV = ENV(
    MAX_INPUT_LEN=4_000_000,
    MAX_OUTPUT_LEN=4096,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
    ENABLE_THINK=False,
)
RAG_QWEN25_ENV = ENV(
    MAX_INPUT_LEN=3_699_000,
    MAX_OUTPUT_LEN=1024,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
    ENABLE_THINK=False,
)
AGENT_ENV = ENV(
    RECURRENT_MAX_CONTEXT_LEN=100000000000,
    RECURRENT_CHUNK_SIZE=15000,
    RECURRENT_MAX_NEW=4096,
)
PARALLEL_AGENT_ENV = ENV(
    RECURRENT_MAX_CONTEXT_LEN=100000000000,
    RECURRENT_CHUNK_SIZE=15000,
    RECURRENT_MAX_NEW=4096,
    PARALLEL_MAX_PASSES=1,
    PARALLEL_MERGE_MAX_TOKENS=4096,
)
QWEN25_AGENT_ENV = ENV(
    RECURRENT_MAX_CONTEXT_LEN=100000000000,
    RECURRENT_CHUNK_SIZE=5000,
    RECURRENT_MAX_NEW=1024,
)
QWEN25_PARALLEL_AGENT_ENV = ENV(
    RECURRENT_MAX_CONTEXT_LEN=100000000000,
    RECURRENT_CHUNK_SIZE=5000,
    RECURRENT_MAX_NEW=1024,
    PARALLEL_MAX_PASSES=3,
    PARALLEL_MERGE_MAX_TOKENS=1024,
)

CONFIGS = [
    Config(
        name="qwen35-vanilla",
        ckpt=QWEN35_VANILLA_LONG,
        method="openai",
        env=OPENAI_ENV,
        serve_max_model_len=5_000_000,
    ),
    Config(
        name="qwen35-vanilla-yarn8.0",
        ckpt=QWEN35_VANILLA_YARN,
        method="openai",
        env=OPENAI_YARN_ENV,
        serve_max_model_len=5_000_000,
    ),
    Config(
        name="qwen35-rag",
        ckpt=QWEN35_VANILLA_LONG,
        method="rag",
        env=RAG_QWEN35_ENV,
        tp=2,
        concur=128,
        serve_max_model_len=4_000_000,
    ),
    Config(
        name="qwen35-recurrent-base",
        ckpt=QWEN35_BASE,
        method="recurrent-boxed",
        env=AGENT_ENV,
    ),
    Config(
        name="qwen35-recurrent-rl-step80",
        ckpt=QWEN35_RECURRENT_RL_STEP80,
        method="recurrent-boxed",
        env=AGENT_ENV,
    ),
    Config(
        name="qwen35-parallel-base",
        ckpt=QWEN35_BASE,
        method="parallel-boxed",
        tp=2,
        concur=128,
        env=PARALLEL_AGENT_ENV,
    ),
    Config(
        name="qwen35-parallel-base-v4",
        ckpt=QWEN35_BASE,
        method="parallel-boxed-v4",
        tp=2,
        concur=128,
        env=PARALLEL_AGENT_ENV,
    ),
    Config(
        name="qwen35-parallel-rl-step80-v4",
        ckpt=QWEN35_PARALLEL_RL_STEP80,
        method="parallel-boxed-v4",
        tp=2,
        concur=80,
        env=PARALLEL_AGENT_ENV,
    ),
    Config(
        name="qwen25-vanilla",
        ckpt=QWEN25_VANILLA,
        method="openai",
        env=OPENAI_QWEN25_ENV,
        tp=2,
        concur=128,
        serve_max_model_len=3_700_000,
        serve_enforce_eager=True,
        serve_disable_custom_all_reduce=True,
        model_name=QWEN25_MODEL_NAME,
    ),
    Config(
        name="qwen25-rag",
        ckpt=QWEN25_VANILLA,
        method="rag",
        env=RAG_QWEN25_ENV,
        tp=2,
        concur=128,
        serve_max_model_len=3_700_000,
        serve_enforce_eager=True,
        serve_disable_custom_all_reduce=True,
        model_name=QWEN25_MODEL_NAME,
    ),
    Config(
        name="qwen25-vanilla-yarn4.0",
        ckpt=QWEN25_YARN,
        method="openai",
        env=OPENAI_QWEN25_YARN_ENV,
        tp=2,
        concur=128,
        serve_max_model_len=282_144,
        serve_enforce_eager=True,
        serve_disable_custom_all_reduce=True,
        model_name=QWEN25_MODEL_NAME,
    ),
    Config(
        name="qwen25-recurrent-rl",
        ckpt=QWEN25_RECURRENT_RL,
        method="recurrent-boxed",
        env=QWEN25_AGENT_ENV,
        tp=1,
        concur=128,
        model_name=QWEN25_MODEL_NAME,
    ),
    Config(
        name="qwen25-parallel-rl",
        ckpt=QWEN25_PARALLEL_RL_STEP240,
        method="parallel-boxed",
        env=QWEN25_PARALLEL_AGENT_ENV,
        tp=1,
        concur=128,
        model_name=QWEN25_MODEL_NAME,
    ),
    Config(
        name="qwen25-parallel-rl-v4",
        ckpt=QWEN25_PARALLEL_RL_STEP240,
        method="parallel-boxed-v4",
        env=QWEN25_PARALLEL_AGENT_ENV,
        tp=1,
        concur=128,
        model_name=QWEN25_MODEL_NAME,
    ),
]
CONFIG_MAP = {config.name: config for config in CONFIGS}


def select_configs(config_arg):
    if config_arg == "all":
        return CONFIGS
    names = [name.strip() for name in config_arg.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIG_MAP]
    if unknown:
        raise ValueError(
            f"Unknown config(s): {unknown}. Available: {sorted(CONFIG_MAP)}"
        )
    return [CONFIG_MAP[name] for name in names]


def run_result(args):
    output = args.result_file or os.path.join(args.save_dir, "result.csv")
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
    print(f"result written to {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LongBench v2 Qwen3.5-35B evaluation configs."
    )
    parser.add_argument(
        "--config",
        "--configs",
        dest="configs",
        type=str,
        default="all",
        help="Config name, comma-separated names, or all.",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVE_PORT", "8000")))
    parser.add_argument("--dash_port", type=int, default=int(os.getenv("DASH_PORT", "8265")))
    parser.add_argument("--server_timeout", type=int, default=1800)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(BASE_DIR / "data" / "data.json"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(BASE_DIR / "results"),
        help="Root results directory. Each config writes to a subdirectory.",
    )
    parser.add_argument("--result_file", type=str, default=None)
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
    parser.add_argument("--json_indent", type=int, default=4)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--rag", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_result", action="store_true")
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
    configs = select_configs(args.configs)
    if args.no_start_server and len(configs) != 1:
        raise ValueError("--no_start_server only supports one config at a time.")
    os.makedirs(args.save_dir, exist_ok=True)
    for config in configs:
        print("=" * 80)
        print(f"Running config: {config.name}")
        print("=" * 80)
        config.run(args)
        if not args.skip_result:
            run_result(args)


if __name__ == "__main__":
    main()
