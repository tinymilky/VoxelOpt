# INSTALL.md

## Agent Runbook for Codex and Claude Code

<p align="center">
  <a href="https://help.openai.com/en/articles/11096431-openai-codex-cli-getting-started"><img alt="OpenAI Codex" src="https://img.shields.io/badge/OpenAI%20Codex-terminal%20agent-111111?logo=openai&logoColor=white"></a>
  <a href="https://docs.anthropic.com/en/docs/claude-code/overview"><img alt="Claude Code" src="https://img.shields.io/badge/Claude%20Code-terminal%20agent-d97757?logo=anthropic&logoColor=white"></a>
</p>

This file is written for terminal coding agents such as
[OpenAI Codex](https://help.openai.com/en/articles/11096431-openai-codex-cli-getting-started)
and [Anthropic Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview).
It provides the exact steps needed to reproduce the VoxelOpt abdomen CT
registration result without touching unrelated files.

## Ground Rules for Agents

- Work from the repository root.
- Do not commit `abdomenreg/`, `logs_abct/`, `.npy`, `.npz`, `.nii`, or
  `.nii.gz` files.
- Keep the public source surface focused on VoxelOpt:
  `get_unet_features.py`, `test_abdomen.py`, `abdomenreg_loader.py`,
  `costVolComplex.py`, and the metric/warping utilities.
- Prefer a one-pair smoke test before the full 42-pair evaluation.
- Use one GPU at a time with `--gpu_id`; this repo does not need multi-GPU
  launchers.

## 1. Create Environment

```bash
conda create -n voxelopt python=3.9 -y
conda activate voxelopt
```

Install PyTorch for the target CUDA version from the official PyTorch selector:

```text
https://pytorch.org/get-started/locally/
```

Then install Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. Place Data

Download the preprocessed abdominal CT registration data:

```text
https://www.dropbox.com/scl/fo/1ri37zp2awc1e218p0zjx/AHw9tXM-wowNqT8WzG6Uq5c?rlkey=ppgyoll7vzzg6hgdz8uzt9h7q&st=drein7eg&dl=0
```

The repository root should look like:

```text
VoxelOpt/
  abdomenreg/
    img/img0001.nii.gz ... img0030.nii.gz
    label/label0001.nii.gz ... label0030.nii.gz
  src/
    unet.pth
```

The default scripts use the test split subjects `0024` to `0030`.

## 3. Extract Feature Maps

```bash
python src/get_unet_features.py --data_path ./abdomenreg --split test --gpu_id 0 --overwrite
```

Expected generated files:

```text
abdomenreg/fea/img0024.npy
...
abdomenreg/fea/img0030.npy
```

## 4. Smoke Test

Run one ordered pair before launching the full evaluation:

```bash
python src/test_abdomen.py --data_path ./abdomenreg --gpu_id 0 --max_pairs 1
```

Expected behavior:

- It reports `Dataset size: 42`.
- It writes `logs_abct/results_ks1_half1_ada1_foundation_n1.csv`.
- The first pair should complete without CUDA out-of-memory.

## 5. Full Table 1 Reproduction

```bash
python src/test_abdomen.py --data_path ./abdomenreg --gpu_id 0
```

Expected output CSV:

```text
logs_abct/results_ks1_half1_ada1_foundation.csv
```

Expected metrics are approximately:

```text
Dice:  58.5%
HD95:  18.5
SDLogJ: 0.21
```

## 6. Useful Agent Prompts

For Codex or Claude Code:

```text
Read README.md and INSTALL.md. Verify the VoxelOpt reproduction setup without
committing generated data. First run the one-pair smoke test, then summarize
whether the full 42-pair command is ready.
```

```text
Run feature extraction for abdomenreg test subjects and then run the VoxelOpt
Table 1 evaluation on GPU 0. Do not modify source files unless a command fails;
if it fails, inspect the traceback and make the smallest fix.
```

## 7. Cleanup Before Commit

Generated outputs should stay untracked:

```bash
git status --short
```

If present, these should not be committed:

```text
abdomenreg/
logs_abct/
__pycache__/
*.npy
*.npz
*.nii.gz
```
