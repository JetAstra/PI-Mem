LongBench v2 没有像 HQA 图里那样崩，主要不是因为 YaRN 在超长区没问题，而是这个 benchmark 的统计方式和实际样本分布把问题“稀释”了。

我查了 `taskutils/LongBench/data/qwen35_token_length_stats.json` 和逐样本长度：

- LongBench v2 一共 503 条。
- prompt token `p90 = 545K`，`p95 = 1.15M`，`p99 = 3.56M`，max `4.52M`。
- 超过 `1M` 的只有 33 条。
- 这次日志跑的是 `qwen35-vanilla-yarn6.0`，不是 yarn4.0。按 `262144 * 6 ≈ 1.57M` 算，真正进入 yarn6.0 明显外推区的只有 18 条。
- 超过 `3M` 的只有 8 条，其中 7 条是 `Code repo QA`。

更关键的是，LongBench v2 的 `Long` 不是“全都 1M+”。`long` 类别 108 条，median 只有 `478K`。所以 result.csv 里的 Long 分数其实不是你 HQA 图里那种 `1.8M/3.6M` 分组曲线。

我也对齐了 vanilla / yarn6.0 的逐样本 judge：

```text
overall:
vanilla 50.7
yarn6.0 48.5

>1.57M 的 18 条:
vanilla 6/18
yarn6.0 6/18
```

在这 18 条真正超长样本里，只有 2 条结果不同：1 条 yarn 变错，1 条 yarn 变对，其余 16 条一样。因此 LongBench v2 上没有出现 HQA 那种清晰下坠。

原因可以概括成：

1. **HQA 是受控长度压力测试**  
   你画的 HQA 直接在 `1.8M/3.6M` 这种长度点上比较，而且任务更像长上下文精确检索。YaRN 超过有效扩展区后，位置编码误差会直接反映到答案。

2. **LongBench v2 是混合任务 + 四选一**  
   它是 A/B/C/D multiple-choice，很多题即使没有精确定位到远端证据，也可能靠局部信息、先验、选项排除拿分。这个指标对“超长精确检索失败”不如 HQA/RULER 敏感。

3. **极长样本数量太少**  
   超过 yarn6.0 名义区间的只有 18/503，占比 3.6%。即使这些全错，overall 也只会掉几个点。

4. **超长样本本来 vanilla 也不强**  
   在 3M-4M 桶里，vanilla 只有 `1/7`，yarn 是 `2/7`。也就是说很多 case 对 vanilla/yarn 都已经很难，无法形成明显的“yarn 相比 vanilla 大幅退化”。

5. **这次是 yarn6.0，不是 HQA 图里的 yarn4.0**  
   yarn4.0 的 nominal 区间大约 `1.05M`，HQA 的 `3.6M` 已经远超很多；LongBench 这里 yarn6.0 的 nominal 区间到 `1.57M`，实际落入严重外推区的样本更少。

关于 latency：`run_vanilla_20260622_093339.log` 里 yarn6.0 pred 用时 `4346s`，vanilla 反而 `54797s`。这个 LongBench wall time 不能直接当作干净的 per-length latency 对比，因为它是 503 条混合长度、`n_proc=128`、4 replicas、最大并发约 `1.5x/request` 的异步跑法，总时间会被少数超长 straggler 强烈支配。HQA 图那种按长度分组的 latency 更适合解释“长度增长导致的耗时变化”。