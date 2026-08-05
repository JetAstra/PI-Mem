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
import os
import glob
import json
import re
import pandas as pd
from collections import defaultdict


METRICS_KEY = ['judge_sub_em'] 


def _iter_json_objects(file_path):
    """Yield JSON objects from files containing either JSONL or pretty-printed concatenated JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
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
            # Try to recover by seeking the next probable object start.
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


def parse_jsonl_file(file_path):
    judge_values_raw = defaultdict(list)
    try:
        l = 0
        for entry in _iter_json_objects(file_path):
            l += 1
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key in METRICS_KEY:
                    # 收集原始的judge指标值
                    judge_values_raw[key].append(value)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    if l != 128:
        print(file_path, l)
    if l == 0:
        # 删除没有数据的文件
        os.remove(file_path)
        print(f"Removed empty file: {file_path}")
    calculated_metrics = {}
    for judge_key, values in judge_values_raw.items():
        if values:
            avg_value = round(sum(values) / len(values) * 100, 2)
            if judge_key == "judge":
                display_key = "judge"
            else:
                display_key = judge_key.replace("judge_", "")
            
            calculated_metrics[display_key] = avg_value
    return calculated_metrics


def collect_and_transform_data(base_dir, relpath):
    assert len(relpath) == 2, "relative path should looks like [f'{dataset_name}', '*.json']"
    data_for_df = []

    for file_path in glob.glob(os.path.join(base_dir, *relpath), recursive=True):
        relative_path = os.path.relpath(file_path, base_dir)
        parts = relative_path.split(os.sep)
        dataset_name = parts[0]
        method_name = os.path.basename(file_path).replace(".jsonl", "")

        metrics = parse_jsonl_file(file_path)
        for metric_name, value in metrics.items():
            data_for_df.append({
                "Dataset": dataset_name,
                "Metric": metric_name,
                "Method": method_name,
                "Value": value
            })
    df = pd.DataFrame(data_for_df)

    # Pivot the DataFrame to have 'Dataset', 'Metric' as index and 'Method' as columns
    pivot_df = df.pivot_table(index=['Dataset', 'Metric'], columns='Method', values='Value')

    # Prepare for custom sorting (max first, then other metrics alphabetically)
    all_rows = []
    max_only_rows_data = [] # To store data for the 'max only' DataFrame
    dataset_names = pivot_df.index.get_level_values('Dataset').unique()
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    dataset_names = sorted(dataset_names, key=natural_sort_key)
    for dataset in dataset_names:
        dataset_metrics_df = pivot_df.loc[dataset]
        other_metrics = sorted([m for m in dataset_metrics_df.index.unique()])
        for metric in other_metrics:
            row_dict = {'Dataset': dataset, 'Metric': metric}
            row_dict.update(dataset_metrics_df.loc[metric].to_dict())
            all_rows.append(row_dict)
    if all_rows:
        final_df = pd.DataFrame(all_rows).set_index(['Dataset', 'Metric'])
        final_df = final_df.reindex(columns=pivot_df.columns, fill_value=pd.NA)
    else:
        final_df = pd.DataFrame(columns=pivot_df.columns).set_index(['Dataset', 'Metric'])
    return final_df


def compute_length_group_avg(full_results_df):
    """
    按照Dataset名称的后缀长度（如32768, 65536, 131072）分组，
    计算每个长度分组下所有任务的平均值，返回一个新的DataFrame。
    """
    if full_results_df.empty:
        return pd.DataFrame()

    records = []
    for (dataset, metric), row in full_results_df.iterrows():
        # 提取后缀长度：dataset名称中最后一个_后面的数字
        match = re.search(r'_(\d+)$', dataset)
        if match:
            length = match.group(1)
        else:
            length = "unknown"
        record = row.to_dict()
        record['Length'] = length
        record['Metric'] = metric
        records.append(record)

    df = pd.DataFrame(records)
    method_columns = [c for c in df.columns if c not in ('Length', 'Metric')]

    # 按 Length 和 Metric 分组求平均
    grouped = df.groupby(['Length', 'Metric'])[method_columns].mean().round(2)

    # 按长度数值排序
    def length_sort_key(idx):
        length_str, metric = idx
        try:
            return (int(length_str), metric)
        except ValueError:
            return (float('inf'), metric)

    grouped = grouped.loc[sorted(grouped.index, key=length_sort_key)]

    # 同时计算全局平均（所有长度的总平均）
    overall_avg = df.groupby(['Metric'])[method_columns].mean().round(2)
    overall_avg.index = pd.MultiIndex.from_tuples(
        [('AVG_ALL', metric) for metric in overall_avg.index],
        names=['Length', 'Metric']
    )

    # 合并
    result = pd.concat([grouped, overall_avg])
    result.index.names = ['Length', 'Metric']

    return result


# --- Main Logic ---
if __name__ == "__main__":
    # python taskutils/memory_eval/visualize.py
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150) 
    pd.set_option('display.colheader_justify', 'left')

    base_dir = "<relative path>"
    # relpath = ['ruler_hqa*', '*.jsonl']
    relpath = ['ruler*', '*.jsonl']

    full_results_df = collect_and_transform_data(base_dir, relpath)
    # 将full_results_df保存为CSV文件
    output_csv_path = os.path.join(base_dir, "aggregated_results.csv")
    full_results_df.to_csv(output_csv_path)
    print("--- Result ---")
    print(full_results_df)

    # 按后缀长度分组平均
    length_avg_df = compute_length_group_avg(full_results_df)
    if not length_avg_df.empty:
        print("\n--- Average by Length ---")
        print(length_avg_df)
        # 追加到CSV文件后面
        with open(output_csv_path, 'a') as f:
            f.write("\n\n# Average by Length Group\n")
        length_avg_df.to_csv(output_csv_path, mode='a')
        print(f"\nLength-grouped averages appended to: {output_csv_path}")

    print("\n" + "="*80 + "\n") # 分隔符
