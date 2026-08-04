import os


def _str_to_bool(value, default=False):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


DATAROOT = os.getenv(
    "DATAROOT",
    "/mnt/shared-storage-user/liudawei/home/verl-new/hotpotqa",
)
MEMORY_DATA_ROOT = os.getenv(
    "MEMORY_DATA_ROOT",
    "/mnt/shared-storage-user/liudawei/home/verl-new/taskutils/memory_data",
)
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "120000"))
MAX_OUTPUT_LEN = int(os.getenv("MAX_OUTPUT_LEN", "10000"))
URL = f"http://{os.getenv('SERVE_HOST', '127.0.0.1')}:{os.getenv('SERVE_PORT', '8000')}/v1"
API_KEY = "123-abc"
RECURRENT_MAX_CONTEXT_LEN = int(os.getenv("RECURRENT_MAX_CONTEXT_LEN", "120000"))
RECURRENT_CHUNK_SIZE = int(os.getenv("RECURRENT_CHUNK_SIZE", "65536"))
RECURRENT_MAX_NEW = int(os.getenv("RECURRENT_MAX_NEW", "10240"))
PARALLEL_MAX_PASSES = int(os.getenv("PARALLEL_MAX_PASSES", "3"))
PARALLEL_MERGE_MAX_TOKENS = int(os.getenv("PARALLEL_MERGE_MAX_TOKENS", "8000"))
ENABLE_THINK = _str_to_bool(os.getenv("ENABLE_THINK"), False)
EARLY_STOP = int(os.getenv("EARLY_STOP", "3"))
BM25_IMPL = os.getenv("BM25_IMPL", "enhanced")
from pprint import pprint

pprint({k: v for k, v in globals().items() if k in os.environ})
