"""Run training-free Qwen3.5 parallel-workflow ablations."""

import os

from run_rjob_qwen35 import Config, ENV


ABLATE_CONFIG = os.environ.get("ABLATE_CONFIG", "").strip()
RESULTS_ROOT = os.environ.get(
    "ABLATE_RESULTS_ROOT",
    "results_qwen3.5-35B-parallel-training-free-ablate",
).rstrip("/")

TRAINING_FREE_CKPT = (
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/"
    "models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)

ABLATIONS = {
    # Chunk-size ablations: memory size fixed at 4096.
    "chunk-size-5000": (5000, 4096),
    "chunk-size-25000": (25000, 4096),
    # Memory-size ablations: chunk size fixed at 15000.
    "memory-size-2048": (15000, 2048),
    "memory-size-8192": (15000, 8192),
    "control": (15000, 4096),
}

HQA_TASKS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]


def build_config() -> Config:
    try:
        chunk_size, memory_size = ABLATIONS[ABLATE_CONFIG]
    except KeyError:
        choices = ", ".join(ABLATIONS)
        raise ValueError(
            f"Unknown ABLATE_CONFIG={ABLATE_CONFIG!r}. Valid choices: {choices}"
        ) from None

    return Config(
        name=f"Qwen3.5-35B-training-free-{ABLATE_CONFIG}",
        ckpt=TRAINING_FREE_CKPT,
        tp=2,
        method="parallel-boxed",
        concur=128,
        env=ENV(
            RECURRENT_MAX_CONTEXT_LEN=100000000000,
            RECURRENT_CHUNK_SIZE=chunk_size,
            # Chunk extraction and final answer output stay capped at 4096.
            RECURRENT_MAX_NEW=4096,
            PARALLEL_MAX_PASSES=3,
            # The consolidated global-memory output cap is the memory size.
            PARALLEL_MERGE_MAX_TOKENS=memory_size,
        ),
        num_samples=64,
        results_dir=f"{RESULTS_ROOT}/{ABLATE_CONFIG}",
    )


if __name__ == "__main__":
    config = build_config()
    print(
        f"Running {ABLATE_CONFIG}: "
        f"chunk_size={config.env.RECURRENT_CHUNK_SIZE}, "
        f"memory_size={config.env.PARALLEL_MERGE_MAX_TOKENS}, "
        f"final_max_tokens={config.env.RECURRENT_MAX_NEW}, "
        f"results_dir={config.results_dir}"
    )
    config.run(HQA_TASKS, serve=True, force=False)
