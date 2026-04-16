vllm serve "/mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-32B-Instruct/" \
    --host 0.0.0.0 \
    --port 23547 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 64



curl http://100.97.88.235:23547/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/shared-storage-user/dllm-share/Models/Qwen2_2.5/Qwen2.5-32B-Instruct/",
    "prompt": [
      "你好，请用一句话介绍你自己。",
      "请把下面这句话翻译成英文：今天天气很好。",
      "用三点总结强化学习的核心思想。",
      "请把下面这句话翻译成英文：今天天气很好。"
    ],
    "max_tokens": 1280,
    "temperature": 0.7
  }'
