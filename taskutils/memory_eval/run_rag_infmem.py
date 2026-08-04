import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DASH_PORT = os.getenv("DASH_PORT", "8265")
SERVE_PORT = os.getenv("SERVE_PORT", "8000")
SCRIPT_DIR = Path(__file__).resolve().parent

QWEN25_7B_PATH = "/mnt/shared-storage-user/liudawei/home/verl/models/Qwen2.5-7B-Instruct"
QWEN35_35B_PATH = "/mnt/shared-storage-user/liudawei/home/verl/models/Qwen3.5-35B-A3B"

RULER_HQA_TESTS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_HQA_TESTS_OVER_1M = [12800, 25600]
RULER_HQA_ALL_TESTS = RULER_HQA_TESTS + RULER_HQA_TESTS_OVER_1M

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
RULER_PROMPT_LENGTHS = [
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]
RULER_GENERAL_TESTS = [
    (task, length) for task in RULER_TASKS for length in RULER_PROMPT_LENGTHS
]


@dataclass
class ENV:
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    RECURRENT_MAX_CONTEXT_LEN: int | None = None
    RECURRENT_CHUNK_SIZE: int | None = None
    RECURRENT_MAX_NEW: int | None = None
    PARALLEL_MAX_PASSES: int | None = None
    PARALLEL_MERGE_MAX_TOKENS: int | None = None
    ENABLE_THINK: bool = False
    EARLY_STOP: int | None = None
    BM25_IMPL: str | None = None

    def setenv(self):
        self._previous_environ = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or value is None:
                continue
            self._previous_environ[key] = os.environ.get(key)
            os.environ[key] = str(value)
            print(f"set {key}={value}")

    def unsetenv(self):
        previous = getattr(self, "_previous_environ", {})
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._previous_environ = {}


class Config:
    SERVE_TAG = "__serve"

    def __init__(
        self,
        name,
        ckpt,
        tp,
        method,
        env,
        concur=128,
        num_samples=64,
        results_dir="results_rag_infmem",
        serve_max_model_len=None,
        serve_hf_overrides=None,
        serve_gpu_memory_utilization=None,
        serve_enforce_eager=False,
        serve_disable_custom_all_reduce=False,
    ):
        self.name = name
        self.ckpt = ckpt
        self.model = Path(self.ckpt).name if Path(self.ckpt).is_dir() else self.ckpt
        self.method = method
        self.tp = tp
        self.env = env
        self.concur = concur
        self.num_samples = num_samples
        self.results_dir = results_dir
        self.serve_max_model_len = serve_max_model_len
        self.serve_hf_overrides = serve_hf_overrides
        self.serve_gpu_memory_utilization = serve_gpu_memory_utilization
        self.serve_enforce_eager = serve_enforce_eager
        self.serve_disable_custom_all_reduce = serve_disable_custom_all_reduce
        self.test_process = {}

    @staticmethod
    def _shutdown_ray_serve():
        print("shutting down existing Ray/Serve, if any...")
        subprocess.run(
            ["serve", "shutdown", "-a", f"http://localhost:{DASH_PORT}"],
            input="y\n",
            text=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["ray", "stop", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            p = subprocess.run(
                [
                    "curl",
                    "--noproxy",
                    "*",
                    "-m",
                    "2",
                    f"http://127.0.0.1:{SERVE_PORT}/v1/models",
                ],
                capture_output=True,
            )
            if p.returncode != 0:
                return
            time.sleep(2)
        print("warning: /v1/models still responds after shutdown")

    @staticmethod
    def _stop_process_group(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            process.kill()

    def serve(self, wait=True):
        serve_name = "serve/llm0180.py"
        serve_script = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../..",
                serve_name,
            )
        )
        cmd_parts = ["python", serve_script, "--model", self.ckpt, "--tp", str(self.tp)]
        if self.serve_max_model_len is not None:
            cmd_parts.extend(["--max-model-len", str(self.serve_max_model_len)])
        if self.serve_hf_overrides is not None:
            cmd_parts.extend(["--hf-overrides", str(self.serve_hf_overrides)])
        if self.serve_gpu_memory_utilization is not None:
            cmd_parts.extend(
                ["--gpu-memory-utilization", str(self.serve_gpu_memory_utilization)]
            )
        if self.serve_enforce_eager:
            cmd_parts.append("--enforce-eager")
        if self.serve_disable_custom_all_reduce:
            cmd_parts.append("--disable-custom-all-reduce")

        print("serving command:")
        print(" ".join(shlex.quote(x) for x in cmd_parts))
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        if wait:
            self._shutdown_ray_serve()
            serve_p = subprocess.Popen(cmd_parts, preexec_fn=os.setsid)
            self.test_process[self.SERVE_TAG] = serve_p
            while True:
                if serve_p.poll() is not None:
                    raise RuntimeError(
                        f"serve process exited before readiness: {serve_p.returncode}"
                    )
                print("try to connect...")
                p = subprocess.run(
                    [
                        "curl",
                        "--noproxy",
                        "*",
                        "-m",
                        "100000000",
                        f"http://127.0.0.1:{SERVE_PORT}/v1/models",
                    ],
                    capture_output=True,
                )
                stdout = p.stdout.decode(errors="replace")
                if p.returncode != 0:
                    print("waiting...")
                    time.sleep(5)
                elif rf'"id":"{self.model}"' not in stdout:
                    print("model not found, maybe shutting down previous server...")
                    time.sleep(5)
                else:
                    print("connected")
                    break
        else:
            p = subprocess.run(
                [
                    "curl",
                    "--noproxy",
                    "*",
                    "-m",
                    "10",
                    f"http://127.0.0.1:{SERVE_PORT}/v1/models",
                ],
                capture_output=True,
            )
            if p.returncode != 0:
                raise RuntimeError("server not started")
        print(p.stdout)

    def _build_task_command(self, test):
        common = [
            "--model",
            self.model,
            "--tokenizer",
            self.ckpt,
            "--api",
            self.method,
            "--num_samples",
            str(self.num_samples),
            "--n_proc",
            str(self.concur),
        ]
        if test in RULER_HQA_TESTS:
            return [
                "python",
                "ruler_hqa.py",
                "--length",
                str(test),
                "--save_dir",
                f"{self.results_dir.removesuffix('/')}/ruler_hqa_{test}",
                "--save_file",
                self.name,
                *common,
            ]
        if test in RULER_HQA_TESTS_OVER_1M:
            return [
                "python",
                "ruler_hqa_over1m.py",
                "--length",
                str(test),
                "--save_dir",
                f"{self.results_dir.removesuffix('/')}/ruler_hqa_{test}",
                "--save_file",
                self.name,
                *common,
            ]
        if test in RULER_GENERAL_TESTS:
            split, length = test
            return [
                "python",
                "ruler_general.py",
                "--split",
                split,
                "--length",
                str(length),
                "--save_dir",
                f"{self.results_dir.removesuffix('/')}/ruler_{split}_{length}",
                "--save_file",
                self.name,
                *common,
            ]
        raise NotImplementedError(f"Not implemented task: {test}")

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        try:
            self.env.setenv()
            self.serve(serve)
            for test in tests:
                cmd_parts = self._build_task_command(test)
                if force:
                    cmd_parts.append("--force")
                print("task command:")
                print(" ".join(shlex.quote(x) for x in cmd_parts))
                p = subprocess.Popen(cmd_parts)
                self.test_process[test] = p
                p.wait()
                if p.returncode != 0:
                    raise RuntimeError(f"task failed with return code {p.returncode}: {test}")
        finally:
            self.env.unsetenv()
            if serve and self.SERVE_TAG in self.test_process:
                p = self.test_process[self.SERVE_TAG]
                self._stop_process_group(p)
                self._shutdown_ray_serve()
        print("all tests finished")

    def __del__(self):
        for key, process in getattr(self, "test_process", {}).items():
            try:
                if key == self.SERVE_TAG:
                    self._stop_process_group(process)
                elif process.poll() is None:
                    process.kill()
            except Exception:
                pass


def _num_samples():
    return int(os.getenv("NUM_SAMPLES", "64"))


def _n_proc():
    return int(os.getenv("N_PROC", "128"))


CONFIGS_BY_NAME = {
    "qwen25_rag": Config(
        name="Qwen2.5-7B-rag",
        ckpt=QWEN25_7B_PATH,
        tp=2,
        method="rag",
        concur=_n_proc(),
        env=ENV(
            MAX_INPUT_LEN=3_699_000,
            MAX_OUTPUT_LEN=1024,
            ENABLE_THINK=False,
        ),
        num_samples=_num_samples(),
        serve_max_model_len=3_700_000,
        serve_enforce_eager=True,
        serve_disable_custom_all_reduce=True,
        results_dir="results_rag_infmem/qwen2.5-7B-rag",
    ),
    "qwen25_infmem": Config(
        name="Qwen2.5-7B-infmem",
        ckpt=QWEN25_7B_PATH,
        tp=2,
        method="infmem",
        concur=_n_proc(),
        env=ENV(
            MAX_INPUT_LEN=3_699_000,
            MAX_OUTPUT_LEN=1024,
            RECURRENT_MAX_CONTEXT_LEN=100000000000,
            RECURRENT_CHUNK_SIZE=5000,
            RECURRENT_MAX_NEW=1024,
            ENABLE_THINK=False,
            EARLY_STOP=3,
            BM25_IMPL="enhanced",
        ),
        num_samples=_num_samples(),
        serve_max_model_len=3_700_000,
        serve_enforce_eager=True,
        serve_disable_custom_all_reduce=True,
        results_dir="results_rag_infmem/qwen2.5-7B-infmem",
    ),
    "qwen35_rag": Config(
        name="Qwen3.5-35B-rag",
        ckpt=QWEN35_35B_PATH,
        tp=2,
        method="rag",
        concur=_n_proc(),
        env=ENV(
            MAX_INPUT_LEN=4_000_000,
            MAX_OUTPUT_LEN=4096,
            ENABLE_THINK=False,
        ),
        num_samples=_num_samples(),
        serve_max_model_len=4_000_000,
        results_dir="results_rag_infmem/qwen3.5-35B-rag",
    ),
    "qwen35_infmem": Config(
        name="Qwen3.5-35B-infmem",
        ckpt=QWEN35_35B_PATH,
        tp=2,
        method="infmem",
        concur=_n_proc(),
        env=ENV(
            MAX_INPUT_LEN=4_000_000,
            MAX_OUTPUT_LEN=4096,
            RECURRENT_MAX_CONTEXT_LEN=100000000000,
            RECURRENT_CHUNK_SIZE=15000,
            RECURRENT_MAX_NEW=4096,
            ENABLE_THINK=False,
            EARLY_STOP=3,
            BM25_IMPL="enhanced",
        ),
        num_samples=_num_samples(),
        serve_max_model_len=4_000_000,
        results_dir="results_rag_infmem/qwen3.5-35B-infmem",
    ),
}


def selected_config_names():
    config_filter = os.getenv("CONFIG_FILTER", "").strip()
    if not config_filter:
        return list(CONFIGS_BY_NAME)
    names = [name.strip() for name in config_filter.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown CONFIG_FILTER entries: {unknown}")
    return names


def selected_configs():
    names = selected_config_names()
    return [CONFIGS_BY_NAME[name] for name in names]


def task_mode():
    return os.getenv("TASK_MODE", "full").strip().lower()


def ood_tests():
    if task_mode() == "smoke":
        return [("qa_1", 8192)]
    return RULER_GENERAL_TESTS


def hqa_tests():
    if task_mode() == "smoke":
        return [50]
    return RULER_HQA_ALL_TESTS


def parse_task_spec():
    spec = os.getenv("TASK_SPEC", "").strip()
    if not spec:
        return None

    tests = []
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) == 2 and parts[0].lower() == "hqa":
            tests.append(int(parts[1]))
        elif len(parts) == 2:
            tests.append((parts[0], int(parts[1])))
        elif len(parts) == 3 and parts[0].lower() == "ruler":
            tests.append((parts[1], int(parts[2])))
        else:
            raise ValueError(
                "TASK_SPEC entries must be `hqa:<length>`, "
                "`<ruler_task>:<length>`, or `ruler:<task>:<length>`; "
                f"got {entry!r}"
            )

    if not tests:
        raise ValueError("TASK_SPEC did not contain any tasks")

    invalid = []
    for test in tests:
        if isinstance(test, int):
            if test not in RULER_HQA_ALL_TESTS:
                invalid.append(test)
        elif test not in RULER_GENERAL_TESTS:
            invalid.append(test)
    if invalid:
        raise ValueError(f"TASK_SPEC contains unsupported tasks: {invalid}")
    return tests


def selected_tests():
    return parse_task_spec() or (ood_tests() + hqa_tests())


def run_ood_tasks():
    tests = ood_tests()
    for config in selected_configs():
        config.run(tests, serve=True, force=False)


def run_ruler_hqa():
    tests = hqa_tests()
    for config in selected_configs():
        config.run(tests, serve=True, force=False)


def run_all_tasks():
    tests = selected_tests()
    for config in selected_configs():
        config.run(tests, serve=True, force=False)


def _false_env(name):
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _safe_path_part(value):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._-") or "run"


def _run_label():
    explicit = os.getenv("RUN_LABEL", "").strip()
    if explicit:
        return _safe_path_part(explicit)
    task_spec = os.getenv("TASK_SPEC", "").strip()
    if task_spec:
        return _safe_path_part(task_spec)
    return _safe_path_part(task_mode())


def _log_root():
    raw = os.getenv("RAG_INFMEM_LOG_ROOT", "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else SCRIPT_DIR / path
    return SCRIPT_DIR / "results_rag_infmem/logs"


def _run_all_tasks_split_logs():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    label = _run_label()
    log_root = _log_root()
    script = str(Path(__file__).resolve())

    for config_name in selected_config_names():
        log_dir = log_root / config_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{label}_{timestamp}.log"

        env = os.environ.copy()
        env["CONFIG_FILTER"] = config_name
        env["RUN_RAG_INFMEM_INLINE"] = "1"
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [sys.executable, script]
        print(f"[{config_name}] log: {log_file}", flush=True)
        with log_file.open("w", buffering=1) as f:
            f.write("command: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            f.write(
                "env: "
                f"TASK_MODE={task_mode()} "
                f"TASK_SPEC={os.getenv('TASK_SPEC', '')!r} "
                f"NUM_SAMPLES={_num_samples()} "
                f"N_PROC={_n_proc()} "
                f"CONFIG_FILTER={config_name}\n"
            )
            f.flush()
            p = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            returncode = p.wait()

        print(f"[{config_name}] finished with return code {returncode}", flush=True)
        if returncode != 0:
            raise RuntimeError(f"{config_name} failed; see log: {log_file}")


if __name__ == "__main__":
    nvidiasmi = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
    print(nvidiasmi)
    print(
        f"{SERVE_PORT=}, {DASH_PORT=}, {task_mode()=}, "
        f"NUM_SAMPLES={_num_samples()}, N_PROC={_n_proc()}, "
        f"CONFIG_FILTER={os.getenv('CONFIG_FILTER', '')!r}, "
        f"TASK_SPEC={os.getenv('TASK_SPEC', '')!r}, "
        f"SPLIT_CONFIG_LOGS={os.getenv('SPLIT_CONFIG_LOGS', '1')!r}"
    )
    if os.getenv("RUN_RAG_INFMEM_INLINE", "") == "1" or _false_env(
        "SPLIT_CONFIG_LOGS"
    ):
        run_all_tasks()
    else:
        _run_all_tasks_split_logs()
