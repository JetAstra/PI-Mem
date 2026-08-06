<div align="center">

# PI-Mem: Pushing Long-Context Reasoning to 3.6M Tokens with Parallel-Iterative Memory

<p>
  <a href="https://huggingface.co/collections/JetLM/pi-mem"><img alt="Models" src="https://img.shields.io/badge/Models-dea60b?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/JetLM/PI-Mem-Data"><img alt="Datasets" src="https://img.shields.io/badge/Datasets-dea60b?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://github.com/JetAstra/PI-Mem"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-24292F?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2608.03048"><img alt="Paper" src="https://img.shields.io/badge/Paper-D14D4D?style=for-the-badge&logo=arxiv&logoColor=white"></a>
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

This branch is based on [verl v0.4.1](https://github.com/verl-project/verl/tree/v0.4.1) and uses the FSDP backend. Follow the official [verl v0.4.1 installation guide](https://verl.readthedocs.io/en/v0.4.1/start/install.html) to prepare the environment. Hence, `Megatron-LM` and `TransformerEngine` are **NOT** required.

## Training

We adopt the same dataset [`hotpotqa_train_32k.parquet`](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/blob/main/hotpotqa_train_32k.parquet) as MemAgent for training. The training configuration is provided in [`run_parallel_7B_debug.sh`](./run_parallel_7B_debug.sh). Before launching, update the Conda environment, model path, data paths, checkpoint directory, and distributed settings near the top of the script for your system.


The released PI-Mem-7B checkpoint corresponds to **rollout step 240**.

For evaluation, please refer to the [`main`](https://github.com/JetAstra/PI-Mem/tree/main) branch.

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