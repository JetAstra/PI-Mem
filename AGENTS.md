# Workspace Runtime Rules

## Conda activation (required)
For commands in this workspace, initialize Conda and activate this exact env first, **only when you execute complex Python code**:

```bash
source /mnt/shared-storage-user/liudawei/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/dllm-share/songhaixu/miniforge3/envs/qwenlongl1_5
```

## Environment immutability (strict)
- Do not install, upgrade, or uninstall any package in this environment.
- Do not run package mutation commands such as `pip install`, `pip uninstall`, `pip install -U`, `conda install`, `conda update`, `conda remove`, or similar.
- If a missing dependency blocks work, stop and ask the user instead of changing the environment.
