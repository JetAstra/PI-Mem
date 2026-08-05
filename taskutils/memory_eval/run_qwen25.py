import os

from run_qwen35 import Config, EnvConfig, REPO_ROOT, BASE_DIR, run_main


QWEN25_BASE_MODEL = os.getenv("QWEN25_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
QWEN25_YARN_MODEL = os.getenv(
    "QWEN25_YARN_MODEL", str(REPO_ROOT / "models" / "Qwen2.5-7B-Instruct-yarn")
)
QWEN25_MEMAGENT_MODEL = os.getenv(
    "QWEN25_MEMAGENT_MODEL", "BytedTsinghua-SIA/RL-MemoryAgent-7B"
)
QWEN25_PI_MEM_MODEL = os.getenv("QWEN25_PI_MEM_MODEL", "JetLM/PI-Mem-7B")

DIRECT_ENV = EnvConfig(
    MAX_INPUT_LEN=3_699_000,
    MAX_OUTPUT_LEN=1024,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
YARN_ENV = EnvConfig(
    MAX_INPUT_LEN=272_144,
    MAX_OUTPUT_LEN=1024,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
)
RAG_ENV = EnvConfig(
    MAX_INPUT_LEN=3_699_000,
    MAX_OUTPUT_LEN=1024,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1,
    ENABLE_THINK=False,
)
MEMAGENT_ENV = EnvConfig(
    RECURRENT_MAX_CONTEXT_LEN=100_000_000_000,
    RECURRENT_CHUNK_SIZE=5000,
    RECURRENT_MAX_NEW=1024,
)
PI_MEM_ENV = EnvConfig(
    RECURRENT_MAX_CONTEXT_LEN=100_000_000_000,
    RECURRENT_CHUNK_SIZE=5000,
    RECURRENT_MAX_NEW=1024,
    PARALLEL_MAX_PASSES=3,
    PARALLEL_MERGE_MAX_TOKENS=1024,
)

COMMON_SERVE_ARGS = {
    "tp": 2,
    "serve_enforce_eager": True,
    "serve_disable_custom_all_reduce": True,
}

CONFIGS = [
    Config(
        name="vanilla",
        model_path=QWEN25_BASE_MODEL,
        method="openai",
        env=DIRECT_ENV,
        model_env_var="QWEN25_BASE_MODEL",
        serve_max_model_len=3_700_000,
        **COMMON_SERVE_ARGS,
    ),
    Config(
        name="yarn",
        model_path=QWEN25_YARN_MODEL,
        method="openai",
        env=YARN_ENV,
        model_env_var="QWEN25_YARN_MODEL",
        serve_max_model_len=282_144,
        **COMMON_SERVE_ARGS,
    ),
    Config(
        name="rag",
        model_path=QWEN25_BASE_MODEL,
        method="rag",
        env=RAG_ENV,
        model_env_var="QWEN25_BASE_MODEL",
        serve_max_model_len=3_700_000,
        **COMMON_SERVE_ARGS,
    ),
    Config(
        name="memagent-base",
        model_path=QWEN25_BASE_MODEL,
        method="recurrent-boxed",
        env=MEMAGENT_ENV,
        model_env_var="QWEN25_BASE_MODEL",
        n_proc=256,
        **COMMON_SERVE_ARGS,
    ),
    Config(
        name="memagent-trained",
        model_path=QWEN25_MEMAGENT_MODEL,
        method="recurrent-boxed",
        env=MEMAGENT_ENV,
        model_env_var="QWEN25_MEMAGENT_MODEL",
        n_proc=256,
        **COMMON_SERVE_ARGS,
    ),
    Config(
        name="pi-mem-base",
        model_path=QWEN25_BASE_MODEL,
        method="parallel-boxed",
        env=PI_MEM_ENV,
        model_env_var="QWEN25_BASE_MODEL",
        n_proc=256,
        answer_prefix_for_vt=True,
        **COMMON_SERVE_ARGS,
    ),
    Config(
        name="pi-mem-trained",
        model_path=QWEN25_PI_MEM_MODEL,
        method="parallel-boxed",
        env=PI_MEM_ENV,
        model_env_var="QWEN25_PI_MEM_MODEL",
        answer_prefix_for_vt=True,
        **COMMON_SERVE_ARGS,
    ),
]


def main() -> None:
    run_main(
        CONFIGS,
        "Run ordered Qwen2.5 configurations on full HQA and RULER OOD.",
        BASE_DIR / "results_qwen25",
    )


if __name__ == "__main__":
    main()
