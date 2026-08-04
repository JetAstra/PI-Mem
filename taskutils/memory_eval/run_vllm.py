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
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.stdout.reconfigure(line_buffering=True)
MODELROOT = os.getenv("MODELROOT", "/mnt/hdfs/hongli/model")


@dataclass
class ENV:
    # config for direct generation
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    # Config for memory agent
    RECURRENT_MAX_CONTEXT_LEN: int = None
    RECURRENT_CHUNK_SIZE: int = None
    RECURRENT_MAX_NEW: int = None

    def setenv(self):
        if not hasattr(self, "_environ"):
            self._environ = {}
        for key, value in self.__dict__.items():
            if key == "_environ" or value is None:
                continue
            self._environ[key] = os.environ.get(key)
            os.environ[key] = str(value)
            print(f"set {key}={value}")

    def unsetenv(self):
        for key, old_value in getattr(self, "_environ", {}).items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        self._environ = {}


# for ruler hqa, we just control the number of distractive wiki items instead the context length
# 50~7K tokens, 100~14K tokens and so on.
RULER_HQA_TESTS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_HQA_TESTS_OVER_1M = [12800, 25600]
# for other ruler task, we use the standard synthetic scripts for convenient and control the context length.
RULER_TASKS = [
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
RULER_PROMPT_LENGTH = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
RULER_GENERRAL_TESTS = [(task, length) for task in RULER_TASKS for length in RULER_PROMPT_LENGTH]

def resolve_model_name(model_path: str, served_model_name: str | None) -> str:
    if served_model_name:
        return served_model_name
    candidate = Path(model_path)
    return candidate.name if candidate.exists() or "/" in model_path else model_path


class Config:
    SERVE_TAG = "__serve"

    def __init__(
        self,
        name,
        ckpt,
        tokenizer,
        tp,
        method,
        env,
        concur=1024,
        num_samples=10,
        results_dir="results",
        host="127.0.0.1",
        port=8000,
        server_timeout=1800,
        no_start_server=False,
        max_model_len=140000,
        max_num_batched_tokens=65536,
        gpu_memory_utilization=0.85,
        model_impl="auto",
        attention_backend=None,
        disable_custom_all_reduce=False,
        enforce_eager=False,
        served_model_name=None,
        parallel_max_passes=3,
        parallel_merge_max_tokens=8000,
        quiet_server=True,
    ):
        self.name = name
        self.ckpt = ckpt
        self.tokenizer = tokenizer
        self.model = resolve_model_name(self.ckpt, served_model_name)
        self.method = method
        self.tp = tp
        self.env = env
        self.concur = concur
        self.num_samples = num_samples
        self.results_dir = results_dir

        self.host = host
        self.port = port
        self.server_timeout = server_timeout
        self.no_start_server = no_start_server

        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model_impl = model_impl
        self.attention_backend = attention_backend
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.enforce_eager = enforce_eager

        self.parallel_max_passes = parallel_max_passes
        self.parallel_merge_max_tokens = parallel_merge_max_tokens
        self.quiet_server = quiet_server

        self.test_process = {}
        self._serve_help_text = None

    def _set_runtime_env(self):
        if not hasattr(self, "_runtime_environ"):
            self._runtime_environ = {}
        runtime_vars = {
            "SERVE_HOST": str(self.host),
            "SERVE_PORT": str(self.port),
            "PARALLEL_MAX_PASSES": str(self.parallel_max_passes),
            "PARALLEL_MERGE_MAX_TOKENS": str(self.parallel_merge_max_tokens),
            "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
        }
        if self.quiet_server and os.getenv("VLLM_LOGGING_LEVEL") is None:
            runtime_vars["VLLM_LOGGING_LEVEL"] = "WARNING"
        for key, value in runtime_vars.items():
            self._runtime_environ[key] = os.environ.get(key)
            os.environ[key] = value
            print(f"set {key}={value}")

    def _unset_runtime_env(self):
        for key, old_value in getattr(self, "_runtime_environ", {}).items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        self._runtime_environ = {}

    def _build_server_cmd(self):
        cmd = [
            "vllm",
            "serve",
            self.ckpt,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tp),
            "--served-model-name",
            self.model,
            "--trust-remote-code",
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--model-impl",
            self.model_impl
        ]
        if self.attention_backend:
            cmd.extend(["--attention-backend", self.attention_backend])
        if self.disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.quiet_server:
            if self._supports_serve_flag("--disable-log-stats"):
                cmd.append("--disable-log-stats")
            if self._supports_serve_flag("--disable-uvicorn-access-log"):
                cmd.append("--disable-uvicorn-access-log")
            if self._supports_serve_flag("--uvicorn-log-level"):
                cmd.extend(["--uvicorn-log-level", "warning"])
            if self._supports_serve_flag("--no-enable-log-requests"):
                cmd.append("--no-enable-log-requests")
            elif self._supports_serve_flag("--disable-log-requests"):
                cmd.append("--disable-log-requests")
            if self._supports_serve_flag("--max-log-len"):
                cmd.extend(["--max-log-len", "0"])
        return cmd

    def _get_serve_help_text(self):
        if self._serve_help_text is not None:
            return self._serve_help_text
        try:
            proc = subprocess.run(
                ["vllm", "serve", "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self._serve_help_text = (proc.stdout or "") + (proc.stderr or "")
        except Exception:
            self._serve_help_text = ""
        return self._serve_help_text

    def _supports_serve_flag(self, flag):
        return flag in self._get_serve_help_text()

    def _get_models(self):
        url = f"http://{self.host}:{self.port}/v1/models"
        try:
            with urlopen(url, timeout=10) as resp:
                import json

                return json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, OSError, ValueError):
            return None

    def _wait_for_server(self):
        deadline = time.time() + self.server_timeout
        while time.time() < deadline:
            data = self._get_models()
            if data and any(item.get("id") == self.model for item in data.get("data", [])):
                print(f"[server] ready: {self.model} on {self.host}:{self.port}")
                return
            print("[server] waiting for vLLM ...")
            time.sleep(5)
        raise TimeoutError(
            f"Timed out waiting for model {self.model} on {self.host}:{self.port}"
        )

    def serve(self, wait=True):
        if not wait:
            return

        if self.no_start_server:
            self._wait_for_server()
            return

        cmd = self._build_server_cmd()
        print("serving command:")
        print(" ".join(cmd))
        serve_p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.test_process[self.SERVE_TAG] = serve_p
        self._wait_for_server()

    def _build_eval_cmd(self, test):
        if test in RULER_HQA_TESTS:
            base_cmd = [sys.executable, "ruler_hqa.py"]
            base_cmd += [
                "--model",
                self.model,
                "--length",
                str(test),
                "--save_dir",
                os.path.join(self.results_dir, f"ruler_hqa_{test}"),
                "--save_file",
                self.name,
                "--tokenizer",
                self.tokenizer,
                "--api",
                self.method,
                "--num_samples",
                str(self.num_samples),
                "--n_proc",
                str(self.concur),
            ]
        elif test in RULER_GENERRAL_TESTS:
            split, length = test
            base_cmd = [sys.executable, "ruler_general.py"]
            base_cmd += [
                "--model",
                self.model,
                "--split",
                split,
                "--length",
                str(length),
                "--save_dir",
                os.path.join(self.results_dir, f"ruler_{split}_{length}"),
                "--save_file",
                self.name,
                "--tokenizer",
                self.tokenizer,
                "--api",
                self.method,
                "--num_samples",
                str(self.num_samples),
                "--n_proc",
                str(self.concur),
            ]
        elif test in RULER_HQA_TESTS_OVER_1M:
            base_cmd = [sys.executable, "ruler_hqa_over1m.py"]
            base_cmd += [
                "--model",
                self.model,
                "--length",
                str(test),
                "--save_dir",
                os.path.join(self.results_dir, f"ruler_hqa_{test}"),
                "--save_file",
                self.name,
                "--tokenizer",
                self.tokenizer,
                "--api",
                self.method,
                "--num_samples",
                str(self.num_samples),
                "--n_proc",
                str(self.concur),
            ]
        else:
            raise ValueError(f"Not Implemented Task {test}, please check")

        return base_cmd

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.env.setenv()
        self._set_runtime_env()
        self.serve(serve)

        try:
            for test in tests:
                cmd = self._build_eval_cmd(test)
                if force:
                    cmd.append("--force")
                print("eval command:")
                print(" ".join(cmd))
                p = subprocess.Popen(cmd)
                self.test_process[test] = p
                p.wait()
                if p.returncode != 0:
                    raise subprocess.CalledProcessError(p.returncode, cmd)
        finally:
            self.env.unsetenv()
            self._unset_runtime_env()
            if serve and self.SERVE_TAG in self.test_process:
                os.killpg(os.getpgid(self.test_process[self.SERVE_TAG].pid), signal.SIGINT)
                try:
                    self.test_process[self.SERVE_TAG].wait(30)
                except Exception:
                    self.test_process[self.SERVE_TAG].kill()

        print("all tests finished")

    def __del__(self):
        for key, process in self.test_process.items():
            if process.poll() is not None:
                continue
            if key == self.SERVE_TAG:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            else:
                process.kill()


def run_ruler_hqa(config: Config, include_over1m: bool = True, force: bool = False):
    tests = list(RULER_HQA_TESTS)
    if include_over1m:
        tests.extend(RULER_HQA_TESTS_OVER_1M)
    config.run(tests, serve=True, force=force)


def run_ood_tasks(config: Config, force: bool = False):
    subset = [
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
    lengths = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
    tests = [(s, l) for s in subset for l in lengths if not (s == "qa_1" and l > 262144)]
    config.run(tests, serve=True, force=force)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run taskutils/memory_eval with vLLM OpenAI-compatible server (no Ray)."
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=f"{MODELROOT}/Qwen3.5-35B-A3B")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--served_model_name", type=str, default=None)
    parser.add_argument("--api", type=str, default="parallel", choices=["openai", "completion", "parallel"])
    parser.add_argument("--suite", type=str, default="all", choices=["hqa", "ood", "all"])
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--n_proc", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="results_vllm")

    parser.add_argument("--host", type=str, default=os.getenv("SERVE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVE_PORT", "8000")))
    parser.add_argument("--server_timeout", type=int, default=1800)
    parser.add_argument("--no_start_server", action="store_true")

    parser.add_argument("--max_model_len", type=int, default=140000)
    parser.add_argument("--max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--model_impl",
        choices=["auto", "vllm", "transformers"],
        default="auto",
    )
    parser.add_argument("--attention_backend", type=str, default=None)
    parser.add_argument("--disable_custom_all_reduce", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")

    parser.add_argument("--max_input_len", type=int, default=140000)
    parser.add_argument("--max_output_len", type=int, default=10000)
    parser.add_argument("--parallel_max_context_len", type=int, default=140000)
    parser.add_argument("--parallel_chunk_size", type=int, default=16384)
    parser.add_argument("--parallel_max_new", type=int, default=10240)
    parser.add_argument("--parallel_max_passes", type=int, default=3)
    parser.add_argument("--parallel_merge_max_tokens", type=int, default=10240)
    parser.add_argument("--quiet_server", dest="quiet_server", action="store_true")
    parser.add_argument("--no_quiet_server", dest="quiet_server", action="store_false")
    parser.set_defaults(quiet_server=True)

    parser.add_argument("--skip_over1m", action="store_true")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = args.model_path
    tokenizer = args.tokenizer or ckpt
    run_name = args.name if args.name else f"{resolve_model_name(ckpt, args.served_model_name)}-{args.api}"

    env = ENV(
        MAX_INPUT_LEN=args.max_input_len,
        MAX_OUTPUT_LEN=args.max_output_len,
        RECURRENT_MAX_CONTEXT_LEN=args.parallel_max_context_len,
        RECURRENT_CHUNK_SIZE=args.parallel_chunk_size,
        RECURRENT_MAX_NEW=args.parallel_max_new,
    )

    config = Config(
        name=run_name,
        ckpt=ckpt,
        tokenizer=tokenizer,
        tp=args.tp,
        method=args.api,
        env=env,
        concur=args.n_proc,
        num_samples=args.num_samples,
        results_dir=args.results_dir,
        host=args.host,
        port=args.port,
        server_timeout=args.server_timeout,
        no_start_server=args.no_start_server,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        model_impl=args.model_impl,
        attention_backend=args.attention_backend,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enforce_eager=args.enforce_eager,
        served_model_name=args.served_model_name,
        parallel_max_passes=args.parallel_max_passes,
        parallel_merge_max_tokens=args.parallel_merge_max_tokens,
        quiet_server=args.quiet_server,
    )

    print(f"{config.host=}, {config.port=}")
    if args.suite in ("ood", "all"):
        run_ood_tasks(config, force=args.force)
    if args.suite in ("hqa", "all"):
        run_ruler_hqa(config, include_over1m=not args.skip_over1m, force=args.force)


if __name__ == "__main__":
    main()


""" # how to run
conda activate qwenlongl1_5
cd taskutils/memory_eval/ && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python run_vllm.py --model_path /mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-7B-Instruct \
    --tp 2 \
    --num_samples 64 \
    --results_dir results_qwen2.5-7B \
    --suite all \
    --n_proc 32 \
    --max_num_batched_tokens 256000 \
    --max_model_len 131072 \
    --max_input_len 2000000 \
    --max_output_len 1024 \
    --parallel_max_context_len 2000000 \
    --parallel_chunk_size 5000 \
    --parallel_max_new 1024 \
    --parallel_max_passes 3 \
    --parallel_merge_max_tokens 1024 \
    --skip_over1m

conda activate verl-071
cd taskutils/memory_eval/ && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python run_vllm.py --model_path /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307 \
    --tp 2 \
    --num_samples 64 \
    --results_dir results_qwen3.5-35B-parallel-boxed/qwen3.5-35B-baseline-boxed \
    --suite all \
    --n_proc 32 \
    --max_num_batched_tokens 256000 \
    --max_model_len 131072 \
    --max_input_len 2000000 \
    --max_output_len 4096 \
    --parallel_max_context_len 2000000 \
    --parallel_chunk_size 15000 \
    --parallel_max_new 4096 \
    --parallel_max_passes 3 \
    --parallel_merge_max_tokens 4096 \
    --skip_over1m

python run_vllm.py --model_path /mnt/shared-storage-user/liudawei/home/verl-new/checkpoints/ParallelAgent/Qwen3_5-35B-A3B-megatron-rjob/global_step_2/actor/huggingface \
    --tp 2 \
    --num_samples 64 \
    --results_dir results_qwen3.5-35B-parallel-boxed/qwen3.5-35B-debug \
    --suite all \
    --n_proc 32 \
    --max_num_batched_tokens 256000 \
    --max_model_len 131072 \
    --max_input_len 2000000 \
    --max_output_len 4096 \
    --parallel_max_context_len 2000000 \
    --parallel_chunk_size 15000 \
    --parallel_max_new 4096 \
    --parallel_max_passes 3 \
    --parallel_merge_max_tokens 4096 \
    --skip_over1m
"""
