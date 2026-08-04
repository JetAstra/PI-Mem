import argparse
import csv
import json
import os
from collections import defaultdict


HEADER = ["Model", "Overall", "Easy", "Hard", "Short", "Medium", "Long"]


def load_predictions(filename):
    try:
        data = json.load(open(filename, encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        return data
    except Exception:
        content = open(filename, encoding="utf-8").read().strip()
        if not content:
            return []
        decoder = json.JSONDecoder()
        items = []
        idx = 0
        while idx < len(content):
            while idx < len(content) and content[idx].isspace():
                idx += 1
            if idx >= len(content):
                break
            try:
                obj, idx = decoder.raw_decode(content, idx)
            except json.JSONDecodeError as exc:
                print(f"Warning: ignoring invalid trailing JSON in {filename}: {exc}")
                break
            items.append(obj)
        return items


def iter_prediction_files(results_dir):
    for root, _, files in os.walk(results_dir):
        for file in sorted(files):
            if not file.endswith((".json", ".jsonl")):
                continue
            filename = os.path.join(root, file)
            rel_name = os.path.relpath(filename, results_dir)
            name = os.path.splitext(rel_name)[0].replace(os.sep, "/")
            yield name, filename


def compute_row(name, pred_data):
    compensated = False

    def percentage(correct, total):
        if total == 0:
            return "N/A"
        return str(round(100 * correct / total, 1))

    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred["judge"])
        if compensated and pred["pred"] is None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred["length"] == "short":
            short += 1
            short_acc += acc
        elif pred["length"] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    return [
        name,
        percentage(easy_acc + hard_acc, len(pred_data)),
        percentage(easy_acc, easy),
        percentage(hard_acc, hard),
        percentage(short_acc, short),
        percentage(medium_acc, medium),
        percentage(long_acc, long),
    ]


def parent_result_name(name):
    parts = name.split("/")
    if len(parts) <= 1:
        return None
    return parts[0]


def write_rows(rows, output, output_format):
    if output_format is None:
        output_format = "csv" if output.lower().endswith(".csv") else "tsv"
    if output_format == "csv":
        with open(output, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerows(rows)
        return
    with open(output, "w", encoding="utf-8") as fout:
        fout.write("\n".join("\t".join(row) for row in rows))


def evaluate(results_dir, output, aggregate_by_parent=False, output_format=None):
    files = list(iter_prediction_files(results_dir))
    rows = [HEADER]
    parent_groups = defaultdict(list)
    file_rows = []

    for name, filename in files:
        pred_data = load_predictions(filename)
        if not pred_data:
            continue
        parent = parent_result_name(name)
        if parent is not None:
            parent_groups[parent].extend(pred_data)
        file_rows.append((name, compute_row(name, pred_data)))

    emitted_file_rows = set()
    aggregate_rows = []
    if aggregate_by_parent:
        for parent in sorted(parent_groups):
            child_rows = [item for item in file_rows if item[0].startswith(parent + "/")]
            if len(child_rows) <= 1:
                continue
            for name, row in child_rows:
                rows.append(row)
                emitted_file_rows.add(name)
            aggregate_rows.append(compute_row(parent, parent_groups[parent]))

    for name, row in file_rows:
        if name not in emitted_file_rows:
            rows.append(row)

    rows.extend(aggregate_rows)
    write_rows(rows, output, output_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="result.csv")
    parser.add_argument(
        "--aggregate_by_parent",
        action="store_true",
        help="Also add one aggregate row per immediate child directory.",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default=None,
        help="Defaults to csv for .csv output, otherwise tsv.",
    )
    args = parser.parse_args()
    evaluate(args.results_dir, args.output, args.aggregate_by_parent, args.format)

"""Usage:

SAVE_DIR=taskutils/LongBench/results/all_configs
SAVE_DIR=taskutils/LongBench/results/parallel_ablate_prompt
SAVE_DIR=taskutils/LongBench/results/all_configs_vanilla_debug
python taskutils/LongBench/result.py \
      --results_dir "${SAVE_DIR}" \
      --output "${SAVE_DIR}/result.csv" \
      --aggregate_by_parent
"""
