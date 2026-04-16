#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import wandb

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")


def parse_step_line(line: str) -> Tuple[int, Dict[str, float]]:
    clean = ANSI_RE.sub("", line).strip()
    m = STEP_RE.search(clean)
    if not m:
        raise ValueError("missing step")

    step = int(m.group(1))
    start = clean.find(f"step:{step}")
    if start < 0:
        raise ValueError("step token not found after regex")

    payload = clean[start:]
    chunks = [c.strip() for c in payload.split(" - ") if c.strip()]
    if not chunks or not chunks[0].startswith("step:"):
        raise ValueError("invalid step prefix")

    metrics: Dict[str, float] = {}
    for chunk in chunks[1:]:
        if ":" not in chunk:
            raise ValueError(f"invalid metric chunk: {chunk}")
        key, val = chunk.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"empty metric key in chunk: {chunk}")
        try:
            num = float(val)
        except ValueError as e:
            raise ValueError(f"invalid metric value for {key}: {val}") from e
        metrics[key] = num

    if not metrics:
        raise ValueError("no metrics parsed")

    return step, metrics


def load_rows(log_path: Path) -> List[Tuple[int, Dict[str, float]]]:
    rows: List[Tuple[int, Dict[str, float]]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if "step:" not in line:
                continue
            try:
                step, metrics = parse_step_line(line)
            except ValueError:
                continue
            rows.append((step, metrics))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Import memagent step logs to wandb offline.")
    p.add_argument("--log-file", required=True, type=Path)
    p.add_argument("--project", default="memagent-log-import")
    p.add_argument("--entity", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-dir", default="wandb")
    args = p.parse_args()

    if not args.log_file.exists():
        raise FileNotFoundError(f"log file not found: {args.log_file}")

    rows = load_rows(args.log_file)
    if not rows:
        raise RuntimeError("no valid step rows parsed from log")

    rows.sort(key=lambda x: x[0])
    schema = sorted(rows[0][1].keys())

    valid_rows: List[Tuple[int, Dict[str, float]]] = []
    dropped = 0
    first_bad = None
    for step, metrics in rows:
        keys = sorted(metrics.keys())
        if keys != schema:
            dropped += 1
            if first_bad is None:
                missing = sorted(set(schema) - set(keys))
                extra = sorted(set(keys) - set(schema))
                first_bad = (step, missing, extra)
            continue
        valid_rows.append((step, metrics))

    if not valid_rows:
        raise RuntimeError("all rows were dropped by strict schema check")

    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_SILENT", "true")

    run_name = args.run_name or f"{args.log_file.stem}-offline"
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        dir=args.wandb_dir,
        mode="offline",
        config={
            "source_log": str(args.log_file),
            "strict_schema": True,
            "schema_size": len(schema),
        },
    )

    for step, metrics in valid_rows:
        run.log(metrics, step=step)

    run.summary["parsed_rows"] = len(rows)
    run.summary["logged_rows"] = len(valid_rows)
    run.summary["dropped_rows_schema_mismatch"] = dropped
    run.summary["first_step"] = valid_rows[0][0]
    run.summary["last_step"] = valid_rows[-1][0]
    run.finish()

    print(f"Parsed rows: {len(rows)}")
    print(f"Logged rows: {len(valid_rows)}")
    print(f"Dropped rows (schema mismatch): {dropped}")
    print(f"Schema keys: {len(schema)}")
    print(f"Step range: {valid_rows[0][0]}..{valid_rows[-1][0]}")
    if first_bad is not None:
        step, missing, extra = first_bad
        print(f"First bad step: {step}")
        print(f"Missing keys: {missing}")
        print(f"Extra keys: {extra}")
    print(f"Wandb run name: {run_name}")
    print(f"Wandb base dir: {Path(args.wandb_dir).resolve()}")


if __name__ == "__main__":
    main()

'''
python tools/import_memagent_log_to_wandb.py \
    --log-file /mnt/shared-storage-user/liudawei/songhaixu/checkpoints/Qwen2.5-7B-8GPU-1nodes-MemAgent-debug/memagent_log_20260413_070556.txt \
    --project memagent-log-import \
    --run-name  memagent_log_2.5-7B \
    --wandb-dir /mnt/shared-storage-user/dllm-share/liudawei/verl/wandb_offline
'''