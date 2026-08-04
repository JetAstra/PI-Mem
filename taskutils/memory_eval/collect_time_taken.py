#!/usr/bin/env python3
"""Collect full 64-sample inference Time taken entries from eval logs."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("taskutils/memory_eval/results_qwen2.5-7B-parallel-CORRECT/step_240")

NAMESPACE_RE = re.compile(r"^Namespace\((.*?)\)", re.MULTILINE)
KEY_VALUE_RE = re.compile(
    r"(\w+)=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[^,)]*)"
)
TIME_RE = re.compile(r"Time taken:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse *.txt logs in a directory and write a CSV containing only "
            "full, non-resume 64-sample inference timings."
        )
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing training_*.txt logs. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <log_dir>/time_taken_summary.csv",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=64,
        help="Only collect runs whose Namespace and sampling logs use this sample count.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1.0,
        help="Drop completed runs below this duration; catches accidental tiny resume timings.",
    )
    parser.add_argument(
        "--skipped-output",
        type=Path,
        default=None,
        help="Optional CSV path for skipped Namespace blocks and skip reasons.",
    )
    return parser.parse_args()


def parse_namespace(namespace_body: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, raw_value in KEY_VALUE_RE.findall(namespace_body):
        value = raw_value.strip()
        if value == "True":
            values[key] = True
        elif value == "False":
            values[key] = False
        elif value == "None":
            values[key] = None
        elif len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
            values[key] = ast.literal_eval(value)
        else:
            values[key] = parse_number(value)
    return values


def parse_number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def infer_task(namespace: dict[str, Any]) -> str:
    split = namespace.get("split")
    if split:
        return str(split)

    save_dir = str(namespace.get("save_dir") or "")
    length = namespace.get("length")
    name = Path(save_dir).name
    if name.startswith("ruler_"):
        name = name[len("ruler_") :]
    if length is not None and name.endswith(f"_{length}"):
        name = name[: -(len(str(length)) + 1)]
    return name


def extract_checkpoint(tokenizer: Any) -> str:
    tokenizer_str = str(tokenizer or "")
    match = re.search(r"(global_step_\d+)", tokenizer_str)
    if match:
        return match.group(1)
    return Path(tokenizer_str).name if tokenizer_str else ""


def first_completed_inference_time(
    block: str,
    block_start_offset: int,
    expected_samples: int,
) -> tuple[float, int] | None:
    completed_pattern = re.compile(
        rf"(?<!\d){re.escape(str(expected_samples))}/{re.escape(str(expected_samples))}(?!\d)"
    )
    for match in TIME_RE.finditer(block):
        before_time = block[: match.start()]
        if completed_pattern.search(before_time):
            return float(match.group(1)), block_start_offset + match.start()
    return None


def skip_reason(namespace: dict[str, Any], block: str, expected_samples: int) -> str | None:
    if namespace.get("num_samples") != expected_samples:
        return "num_samples_not_expected"

    random_sampling = extract_int(r"Random sampling:\s*(\d+)", block)
    if random_sampling != expected_samples:
        return "random_sampling_not_expected"

    sampling_data_len = extract_int(r"sampling data len\s+(\d+)", block)
    if sampling_data_len != expected_samples:
        return "sampling_data_len_not_expected"

    return None


def sort_length(value: Any) -> tuple[int, Any]:
    value_str = str(value)
    if value_str.isdigit():
        return (0, int(value_str))
    return (1, value_str)


def collect(
    log_dir: Path,
    expected_samples: int,
    min_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for log_path in sorted(log_dir.glob("*.txt")):
        text = log_path.read_text(errors="replace")
        namespace_matches = list(NAMESPACE_RE.finditer(text))
        counters["namespace_blocks"] += len(namespace_matches)

        for index, match in enumerate(namespace_matches):
            block_end = (
                namespace_matches[index + 1].start()
                if index + 1 < len(namespace_matches)
                else len(text)
            )
            block = text[match.start() : block_end]
            namespace = parse_namespace(match.group(1))
            namespace_line = line_number(text, match.start())
            task = infer_task(namespace)

            base = {
                "log_file": log_path.name,
                "namespace_line": namespace_line,
                "task": task,
                "length": namespace.get("length", ""),
                "save_dir": namespace.get("save_dir", ""),
                "save_file": namespace.get("save_file", ""),
                "model": namespace.get("model", ""),
                "tokenizer": namespace.get("tokenizer", ""),
                "checkpoint": extract_checkpoint(namespace.get("tokenizer")),
                "n_proc": namespace.get("n_proc", ""),
                "api": namespace.get("api", ""),
                "num_samples": namespace.get("num_samples", ""),
                "random_sampling": extract_int(r"Random sampling:\s*(\d+)", block),
                "original_data_len": extract_int(r"original data len\s+(\d+)", block),
                "sampling_data_len": extract_int(r"sampling data len\s+(\d+)", block),
            }

            reason = skip_reason(namespace, block, expected_samples)
            if reason is not None:
                counters[f"skipped_{reason}"] += 1
                skipped.append({**base, "skip_reason": reason})
                continue

            timing = first_completed_inference_time(block, match.start(), expected_samples)
            if timing is None:
                reason = "no_completed_expected_tqdm_before_time"
                counters[f"skipped_{reason}"] += 1
                skipped.append({**base, "skip_reason": reason})
                continue

            time_taken_seconds, time_offset = timing
            if time_taken_seconds < min_seconds:
                reason = "time_below_min_seconds"
                counters[f"skipped_{reason}"] += 1
                skipped.append({**base, "skip_reason": reason})
                continue

            rows.append(
                {
                    **base,
                    "time_line": line_number(text, time_offset),
                    "time_taken_seconds": time_taken_seconds,
                    "time_taken_minutes": time_taken_seconds / 60,
                }
            )
            counters["collected"] += 1

    rows.sort(
        key=lambda row: (
            str(row["checkpoint"]),
            str(row["model"]),
            str(row["tokenizer"]),
            str(row["save_dir"]),
            sort_length(row["length"]),
            str(row["log_file"]),
            int(row["namespace_line"]),
        )
    )
    return rows, skipped, counters


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {log_dir}")

    output = args.output or log_dir / "time_taken_summary.csv"
    rows, skipped, counters = collect(log_dir, args.expected_samples, args.min_seconds)

    fieldnames = [
        "task",
        "length",
        "time_taken_seconds",
        "time_taken_minutes",
        "save_dir",
        "checkpoint",
        "model",
        "tokenizer",
        "save_file",
        "api",
        "n_proc",
        "num_samples",
        "random_sampling",
        "original_data_len",
        "sampling_data_len",
        "log_file",
        "namespace_line",
        "time_line",
    ]
    write_csv(output, rows, fieldnames)

    if args.skipped_output is not None:
        skipped_fieldnames = [
            "skip_reason",
            "task",
            "length",
            "save_dir",
            "checkpoint",
            "model",
            "tokenizer",
            "save_file",
            "api",
            "n_proc",
            "num_samples",
            "random_sampling",
            "original_data_len",
            "sampling_data_len",
            "log_file",
            "namespace_line",
        ]
        write_csv(args.skipped_output, skipped, skipped_fieldnames)

    print(f"Wrote {len(rows)} rows to {output}")
    for key in sorted(counters):
        print(f"{key}: {counters[key]}")


if __name__ == "__main__":
    main()
