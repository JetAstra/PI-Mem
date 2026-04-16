# 必须迁移（recurrent 训练主链）

1. ray_trainer.py (line 468)
    recurrent 开关初始化：RRegister/recurrent_config，async/sync 两套加载。
    recurrent 数据集接管：dataset_cls(...) + get_bactch_keys()。
    生成链路改成 generation_manager.run_llm_loop(...)，并维护 final_mask/sample_index。
    reward 对齐改成 final_batch(...).union(original_batch)。
    recurrent 下禁用/限制：REMAX、RM、balance_batch。
    recurrent advantage 改成 compute_1D_grpo_advantage + sample_index 广播。
    actor update 前 graceful_padding，写入 no_padding_mask，末尾再去 padding。
2. dp_actor.py (line 237)
    update_policy 支持 no_padding_mask，先 indexing_proto 去掉 padding 样本。
    mini/micro batch 切分支持 td_split（应对可变 batch）。
梯度累计缩放按 token/seq 权重修正（不是固定按 batch 长度）。
3. fsdp_workers.py (line 223)
    actor 的 train_batch_size/ppo_mini_batch_size 归一化逻辑。
    generate_sequences 透传 pad_to 和 generation_kwargs。
    update_actor 调用 self.actor.update_policy(...) 的路径要保持兼容 recurrent 的 no_padding_mask。
4. vllm_rollout_spmd.py (line 206)
    generate_sequences(prompts, pad_to=None, **kwargs)。
    kwargs.update(...) 在 greedy/validate 分支生效。
    validate 下 n=1，pad 长度用 pad_to 或当前 sampling params。
5. ppo_trainer.yaml (line 234)
    recurrent: 总配置块（enable、memory/trival/tool_gsm8k、path/async_path/config）。
    训练时与 recurrent 相关的算法开关（如 adv_estimator、grpo_use_adv）也要和新版本语义对齐。


---

#### kill vscode

```sh
ps aux | grep python.debugpy

ps aux | grep /mnt/shared-storage-user/liudawei/envs/dllm/bin/python3.10 | grep -v grep | awk '{print $2}'

ps aux | grep python.debugpy | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep '/mnt/shared-storage-user/dllm-share/songhaixu/miniforge3/envs/qwenlongl1_5' | grep -v grep | awk '{print $2}' | xargs kill -9

```
---