#!/usr/bin/env python3
"""Aggregate Qwen3.5 RULER scores and latency logs for selected checkpoints."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collect_time_taken import collect


@dataclass(frozen=True)
class Checkpoint:
    ckpt_id: str
    ckpt_dir: str
    aliases: tuple[str, ...] = ()


CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        "vanilla_baseline",
        "taskutils/memory_eval/results_qwen3.5-35B-vanilla/qwen3.5-35B-baseline",
    ),
    Checkpoint(
        "vanilla_baseline_yarn4",
        "taskutils/memory_eval/results_qwen3.5-35B-vanilla/qwen3.5-35B-baseline-yarn4.0",
    ),
    Checkpoint(
        "recurrent_baseline_boxed",
        "taskutils/memory_eval/results_qwen3.5-35B-recurrent-boxed/qwen3.5-35B-baseline-boxed",
    ),
    Checkpoint(
        "recurrent_step80",
        "taskutils/memory_eval/results_qwen3.5-35B-recurrent-boxed/step_80",
    ),
    Checkpoint(
        "parallel_baseline_boxed",
        "taskutils/memory_eval/results_qwen3.5-35B-v2-parallel-boxed/qwen3.5-35B-baseline-boxed",
        aliases=(
            "taskutils/memory_eval/results_qwen3.5-35B-parallel-boxed/qwen3.5-35B-baseline-boxed",
        ),
    ),
    Checkpoint(
        "parallel_step80",
        "taskutils/memory_eval/results_qwen3.5-35B-v2-parallel-boxed/step_80",
        aliases=(
            "taskutils/memory_eval/results_qwen3.5-35B-v2-parallel-boxed-512k/step_80",
        ),
    ),
)

SEARCH_ROOTS: tuple[Path, ...] = (
    Path("taskutils/memory_eval/results_qwen3.5-35B-vanilla"),
    Path("taskutils/memory_eval/results_qwen3.5-35B-recurrent-boxed"),
    Path("taskutils/memory_eval/results_qwen3.5-35B-v2-parallel-boxed"),
    Path("taskutils/memory_eval/results_qwen3.5-35B-v2-parallel-boxed-512k"),
    Path("taskutils/memory_eval/logs"),
)

OUTPUT_SCORE = Path("plots/qwen3_5-ruler-score-raw.csv")
OUTPUT_LATENCY_64 = Path("plots/qwen3_5-ruler-latency-samples64-raw.csv")
OUTPUT_LATENCY_128 = Path("plots/qwen3_5-ruler-latency-samples128-raw.csv")

SCORE_FIELDS = [
    "ckpt_id",
    "ckpt_dir",
    "source_csv",
    "row_index",
    "dataset",
    "metric",
    "score",
    "score_column",
]

LATENCY_FIELDS = [
    "ckpt_id",
    "ckpt_dir",
    "dataset",
    "task",
    "length",
    "samples",
    "time_taken_seconds",
    "time_taken_minutes",
    "mapping_note",
    "source_save_dir",
    "normalized_save_dir",
    "source_log_dir",
    "log_file",
    "namespace_line",
    "time_line",
    "save_file",
    "model",
    "tokenizer",
    "checkpoint",
    "api",
    "n_proc",
    "num_samples",
    "random_sampling",
    "original_data_len",
    "sampling_data_len",
    "source_priority",
]


def normalize_path(value: str | Path) -> str:
    path = str(value).replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    marker = "taskutils/memory_eval/"
    if marker in path:
        path = path[path.index(marker) :]
    elif path.startswith("results_"):
        path = f"{marker}{path}"
    return path.rstrip("/")


def save_dir_prefixes(ckpt: Checkpoint) -> tuple[str, ...]:
    prefixes = [ckpt.ckpt_dir, *ckpt.aliases]
    return tuple(normalize_path(prefix) for prefix in prefixes)


def starts_with_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def map_checkpoint(
    save_dir: str,
    source_log_dir: str,
) -> tuple[Checkpoint | None, str]:
    normalized_save_dir = normalize_path(save_dir)
    normalized_source_dir = normalize_path(source_log_dir)

    for ckpt in CHECKPOINTS:
        for prefix in save_dir_prefixes(ckpt):
            if starts_with_prefix(normalized_save_dir, prefix):
                note = "exact_save_dir"
                if prefix != normalize_path(ckpt.ckpt_dir):
                    note = f"alias_save_dir:{prefix}"
                return ckpt, note

    yarn_source = normalize_path(
        "taskutils/memory_eval/results_qwen3.5-35B-vanilla/qwen3.5-35B-baseline-yarn4.0"
    )
    legacy_yarn_prefix = normalize_path(
        "taskutils/memory_eval/results_qwen3.5-35B-vanilla/qwen3.5-35B-baseline-boxed"
    )
    if starts_with_prefix(normalized_source_dir, yarn_source) and starts_with_prefix(
        normalized_save_dir, legacy_yarn_prefix
    ):
        ckpt = next(item for item in CHECKPOINTS if item.ckpt_id == "vanilla_baseline_yarn4")
        return ckpt, f"yarn4_source_legacy_save_dir:{legacy_yarn_prefix}"

    return None, ""


def source_priority(ckpt: Checkpoint, source_log_dir: str) -> int:
    normalized_source_dir = normalize_path(source_log_dir)
    target = normalize_path(ckpt.ckpt_dir)
    if starts_with_prefix(normalized_source_dir, target):
        return 30
    if normalized_source_dir.startswith("taskutils/memory_eval/logs"):
        return 20
    target_parent = str(Path(target).parent).replace("\\", "/")
    if starts_with_prefix(normalized_source_dir, target_parent):
        return 10
    return 0


def collect_score_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ckpt in CHECKPOINTS:
        csv_path = Path(ckpt.ckpt_dir) / "aggregated_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing score CSV: {csv_path}")

        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or len(reader.fieldnames) < 3:
                raise ValueError(f"Unexpected score CSV columns in {csv_path}")
            score_column = reader.fieldnames[2]

            for row_index, row in enumerate(reader):
                dataset = row.get("Dataset", "")
                if not dataset.startswith("ruler_"):
                    continue
                rows.append(
                    {
                        "ckpt_id": ckpt.ckpt_id,
                        "ckpt_dir": ckpt.ckpt_dir,
                        "source_csv": str(csv_path),
                        "row_index": row_index,
                        "dataset": dataset,
                        "metric": row.get("Metric", ""),
                        "score": row.get(score_column, ""),
                        "score_column": score_column,
                    }
                )
    return rows


def log_dirs() -> list[Path]:
    dirs: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for log_path in root.rglob("*.txt"):
            dirs.add(log_path.parent)
    return sorted(dirs)


def dataset_from_save_dir(save_dir: str) -> str:
    return Path(normalize_path(save_dir)).name


def sort_length(value: Any) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def latency_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    ckpt_index = {ckpt.ckpt_id: index for index, ckpt in enumerate(CHECKPOINTS)}
    return (
        ckpt_index.get(str(row["ckpt_id"]), 999),
        str(row["task"]),
        sort_length(row["length"]),
        str(row["dataset"]),
        int(row["samples"]),
    )


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["source_priority"]),
        str(row["source_log_dir"]),
        str(row["log_file"]),
        int(row["namespace_line"]),
    )


def collect_latency_rows(expected_samples: int) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, int], dict[str, Any]] = {}

    for log_dir in log_dirs():
        collected, _, _ = collect(log_dir, expected_samples=expected_samples, min_seconds=1.0)
        source_log_dir = normalize_path(log_dir)

        for row in collected:
            ckpt, mapping_note = map_checkpoint(str(row.get("save_dir", "")), source_log_dir)
            if ckpt is None:
                continue

            normalized_save_dir = normalize_path(str(row.get("save_dir", "")))
            output_row = {
                "ckpt_id": ckpt.ckpt_id,
                "ckpt_dir": ckpt.ckpt_dir,
                "dataset": dataset_from_save_dir(normalized_save_dir),
                "task": row.get("task", ""),
                "length": row.get("length", ""),
                "samples": expected_samples,
                "time_taken_seconds": row.get("time_taken_seconds", ""),
                "time_taken_minutes": row.get("time_taken_minutes", ""),
                "mapping_note": mapping_note,
                "source_save_dir": row.get("save_dir", ""),
                "normalized_save_dir": normalized_save_dir,
                "source_log_dir": source_log_dir,
                "log_file": row.get("log_file", ""),
                "namespace_line": row.get("namespace_line", ""),
                "time_line": row.get("time_line", ""),
                "save_file": row.get("save_file", ""),
                "model": row.get("model", ""),
                "tokenizer": row.get("tokenizer", ""),
                "checkpoint": row.get("checkpoint", ""),
                "api": row.get("api", ""),
                "n_proc": row.get("n_proc", ""),
                "num_samples": row.get("num_samples", ""),
                "random_sampling": row.get("random_sampling", ""),
                "original_data_len": row.get("original_data_len", ""),
                "sampling_data_len": row.get("sampling_data_len", ""),
                "source_priority": source_priority(ckpt, source_log_dir),
            }
            key = (ckpt.ckpt_id, str(output_row["dataset"]), expected_samples)
            existing = deduped.get(key)
            if existing is None or candidate_sort_key(output_row) > candidate_sort_key(existing):
                deduped[key] = output_row

    rows = list(deduped.values())
    rows.sort(key=latency_sort_key)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_counts(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"{label}: {len(rows)} rows")
    for ckpt in CHECKPOINTS:
        count = sum(1 for row in rows if row["ckpt_id"] == ckpt.ckpt_id)
        print(f"  {ckpt.ckpt_id}: {count}")


def main() -> None:
    score_rows = collect_score_rows()
    latency64_rows = collect_latency_rows(expected_samples=64)
    latency128_rows = collect_latency_rows(expected_samples=128)

    write_csv(OUTPUT_SCORE, score_rows, SCORE_FIELDS)
    write_csv(OUTPUT_LATENCY_64, latency64_rows, LATENCY_FIELDS)
    write_csv(OUTPUT_LATENCY_128, latency128_rows, LATENCY_FIELDS)

    print(f"Wrote {OUTPUT_SCORE}")
    print(f"Wrote {OUTPUT_LATENCY_64}")
    print(f"Wrote {OUTPUT_LATENCY_128}")
    print_counts("score", score_rows)
    print_counts("latency samples64", latency64_rows)
    print_counts("latency samples128", latency128_rows)


if __name__ == "__main__":
    main()
