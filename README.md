<div align="center">

# PI-Mem: Pushing Long-Context Reasoning to 3.6M Tokens with Parallel-Iterative Memory

<p>
  <a href="https://huggingface.co/collections/JetLM/pi-mem"><img alt="Models" src="https://img.shields.io/badge/Models-dea60b?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/JetLM/PI-Mem-Data"><img alt="Datasets" src="https://img.shields.io/badge/Datasets-dea60b?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://github.com/JetAstra/PI-Mem"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-24292F?style=for-the-badge&logo=github&logoColor=white"></a>
  <img alt="Paper" src="https://img.shields.io/badge/Paper-D14D4D?style=for-the-badge&logo=arxiv&logoColor=white">
</p>

</div>

---

<p align="center">
  <img src="./assets/teaser.png" width="80%" alt="Comparison between recurrent memory and PI-Mem">
</p>

<p align="center"><em>
Recurrent memory processes chunks sequentially and may overwrite early evidence with later noise. PI-Mem reads chunks in parallel against a shared memory, preserving relevant evidence while shortening the serial inference path.
</em></p>

## Overview

**PI-Mem (Parallel-Iterative Memory)** replaces sequential updates with a bounded **read–select–merge** workflow. In each turn, it reads all chunks in parallel against a shared memory, selects new or complementary evidence, and merges it into a compact state for the next turn. The workflow exits when no new evidence is found or the maximum number of turns is reached.

<p align="center">
  <img src="./assets/method.png" width="100%" alt="The PI-Mem read-select-merge workflow">
</p>

<p align="center"><em>
Each PI-Mem turn performs parallel reading, evidence selection, and compact memory merging. Final answering uses the question and the consolidated global memory rather than the original ultra-long context.
</em></p>

## Installation

PI-Mem is built on the [`main` branch of verl](https://github.com/verl-project/verl/tree/main), using [commit `e1ff774`](https://github.com/verl-project/verl/commit/e1ff774f74565b44a567d02014454543e7361628) as its upstream base. Because the software stack includes several CUDA-compiled components, a portable one-command installation cannot reliably cover every GPU and system configuration. We recommend reproducing the tested stack below and using [`docker/Dockerfile.stable.vllm`](./docker/Dockerfile.stable.vllm) as the ordered build reference.

### Reference environment

| Component                          | Tested version                       |
| ---------------------------------- | ------------------------------------ |
| Operating system                   | Ubuntu 24.04                         |
| Python                             | 3.12                                 |
| CUDA                               | >= 12.8                              |                          |
| PyTorch / TorchVision / TorchAudio | 2.10.0 / 0.25.0 / 2.10.0 (CUDA 12.x) |
| vLLM                               | 0.18.0                               |
| Transformers                       | 5.3.0                                |
| FlashAttention                     | 2.8.3                                |
| Ray                                | 2.55.1                               |
| Megatron Core                      | 0.16.0                               |
| Transformer Engine                 | 2.12                                 |

### Setup guidance

1. Start from a CUDA 12.8 or newer development environment with Python 3.12 and install the matching PyTorch packages.
2. Install CUDA extensions—such as Apex, Transformer Engine, and FlashAttention—against the same CUDA and PyTorch ABI, following the order in the reference Dockerfile.
3. Install vLLM, Transformers, Ray, and the remaining Python dependencies using the versions recorded in [`requirements.txt`](./requirements.txt).
4. Install this repository in editable mode after its dependencies are available:

```bash
git clone https://github.com/JetAstra/PI-Mem.git
cd PI-Mem
pip install -e . --no-deps
```

> [!IMPORTANT]
> `requirements.txt` is an exact snapshot of our development environment, not a directly portable lockfile. Some entries point to machine-local wheels or source trees through `file://` paths and must be replaced with builds appropriate for the target system. Megatron Core, Transformer Engine, and Apex are needed only for the corresponding training backends; evaluation with the provided vLLM launchers does not require the full training stack.

## Evaluation

Run evaluations from the repository root using the corresponding launcher:

| Benchmark         | Launch script                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| RULER HQA and OOD | [Qwen3.5-35B-A3B](./taskutils/memory_eval/eval_qwen35.sh) · [Qwen2.5-7B](./taskutils/memory_eval/eval_qwen25.sh) |
| LongBench v2      | [Qwen3.5-35B-A3B](./taskutils/LongBench/eval_longbench.sh)                                                       |

The RULER launchers expose the configurations, task subsets, and context lengths at the top of each script, so they can be adjusted without changing the Python entry points.

Evaluation data are available under [`hotpotqa_eval/`](https://huggingface.co/datasets/JetLM/PI-Mem-Data/tree/main/hotpotqa_eval) in `JetLM/PI-Mem-Data`. The same repository provides reference traces for [`PI-Mem-35B-A3B`](https://huggingface.co/datasets/JetLM/PI-Mem-Data/tree/main/PI-Mem-35B-A3B-trace) and [`PI-Mem-7B`](https://huggingface.co/datasets/JetLM/PI-Mem-Data/tree/main/PI-Mem-7B-trace). After evaluation, set `base_dir` in [`taskutils/memory_eval/visualize.py`](./taskutils/memory_eval/visualize.py) to the result directory and run it to generate `aggregated_results.csv`, including averages grouped by context length.

## Training

The [`main`](https://github.com/JetAstra/PI-Mem/tree/main) branch contains the Qwen3.5 training implementation. PI-Mem-7B was trained with an earlier verl codebase; use the dedicated [`qwen2.5`](https://github.com/JetAstra/PI-Mem/tree/qwen2.5) branch to reproduce that experiment.

For Qwen3.5, download [`hotpotqa_train/hotpotqa_train_doc1000.parquet`](https://huggingface.co/datasets/JetLM/PI-Mem-Data/blob/main/hotpotqa_train/hotpotqa_train_doc1000.parquet) from `JetLM/PI-Mem-Data`, then update the model, data, environment, and checkpoint paths in [`examples/parallel_trainer/run_qwen3_5_35b_megatron_debug.sh`](./examples/parallel_trainer/run_qwen3_5_35b_megatron_debug.sh). The released PI-Mem-35B-A3B checkpoint was trained for **80 rollout steps**; `trainer.total_epochs` in the example script is only a placeholder and does not represent the reported training duration.

> [!NOTE]
> Qwen3.5 may encounter a Megatron tensor-parallel shape error when `TP > 2`. If this occurs, apply the fix from [NVIDIA/Megatron-LM#3529](https://github.com/NVIDIA/Megatron-LM/pull/3529/changes) to [`megatron/core/transformer/attention.py`](https://github.com/NVIDIA/Megatron-LM/pull/3529/changes#diff-cfcaf53b88d3893379f5522e1e2a6d0b15eba6158e964fb1892415de482aadf8).

## Acknowledgements

This repository is built on top of [verl](https://github.com/verl-project/verl) and [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent). We thank their authors and contributors for open-sourcing their work.

## Citation

```bibtex
@misc{liu2026pimem,
  title={PI-Mem: Pushing Long-Context Reasoning to 3.6M Tokens with Parallel-Iterative Memory},
  author={Dawei Liu and Haixu Song and Shuang Cheng and Shijie Wang and Haozheng Hou and Kaifeng Liu and Ermo Hua and Zhonghang Yuan and Zhijie Zhong and Yuchen Fan and Biqing Qi and Bowen Zhou},
  year={2026},
  eprint={2608.03048},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.03048}
}
```
