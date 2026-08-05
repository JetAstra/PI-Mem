import argparse
import glob
import json
import os
import re

import pandas as pd


def _iter_json_objects(file_path):
    """Yield JSON objects from JSONL or concatenated pretty JSON files."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return

    decoder = json.JSONDecoder()
    idx = 0
    n = len(content)
    while idx < n:
        while idx < n and content[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, next_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            next_obj = content.find("{", idx + 1)
            if next_obj == -1:
                break
            idx = next_obj
            continue

        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        idx = next_idx


def parse_parallel_passes_file(file_path, on_missing="warning"):
    """Read parallel passes count and compute min/max/mean.

    Priority:
    1) top-level parallel_passes_used
    2) fallback to len(parallel_trace.passes) when field (1) is missing
    """
    values = []
    total_entries = 0
    missing_or_invalid = 0

    for entry in _iter_json_objects(file_path):
        total_entries += 1
        if "parallel_passes_used" in entry:
            value = entry.get("parallel_passes_used")
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                missing_or_invalid += 1
            continue

        # Backward compatibility for older jsonl without top-level parallel_passes_used.
        trace = entry.get("parallel_trace")
        if trace is None:
            trace = entry.get("trace")
        passes = trace.get("passes") if isinstance(trace, dict) else None
        if isinstance(passes, list):
            values.append(float(len(passes)))
        else:
            missing_or_invalid += 1

    if missing_or_invalid > 0:
        msg = (
            f"{file_path}: {missing_or_invalid}/{total_entries} entries missing or invalid "
            "'parallel_passes_used' and fallback 'parallel_trace.passes'"
        )
        if on_missing == "error":
            raise ValueError(msg)
        if on_missing == "warning":
            print(f"WARNING: {msg}")

    if not values:
        return None

    return {
        "parallel_passes_min": round(min(values), 4),
        "parallel_passes_max": round(max(values), 4),
        "parallel_passes_mean": round(sum(values) / len(values), 4),
        "valid_count": len(values),
        "missing_count": missing_or_invalid,
    }


def collect_stats(base_dir, relpath, on_missing="warning"):
    rows = []
    pattern = os.path.join(base_dir, *relpath)
    for file_path in sorted(glob.glob(pattern, recursive=True)):
        relative_path = os.path.relpath(file_path, base_dir)
        dataset_name = relative_path.split(os.sep)[0]
        method_name = os.path.basename(file_path).replace(".jsonl", "")

        stats = parse_parallel_passes_file(file_path, on_missing=on_missing)
        if stats is None:
            continue

        rows.append(
            {
                "Dataset": dataset_name,
                "Method": method_name,
                **stats,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

    dataset_order = sorted(df["Dataset"].unique(), key=natural_sort_key)
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values(["Dataset", "Method"]).reset_index(drop=True)
    df["Dataset"] = df["Dataset"].astype(str)
    return df


if __name__ == "__main__":
    # python taskutils/memory_eval/visualize_parallel_passes.py --on-missing error
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--on-missing",
        choices=["warning", "error", "ignore"],
        default="warning",
        help="How to handle missing top-level parallel_passes_used.",
    )
    args = parser.parse_args()

    base_dir = "<relative path>"
    relpath = ["ruler*", "*.jsonl"]

    df = collect_stats(base_dir, relpath, on_missing=args.on_missing)
    if df.empty:
        print("No valid records found.")
    else:
        output_full = os.path.join(base_dir, "aggregated_parallel_passes.csv")
        df.to_csv(output_full, index=False)

        print("--- Parallel Passes (min/max/mean) ---")
        print(df)
        print(f"\nSaved full stats to: {output_full}")
