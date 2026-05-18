# Qwen Paper Translation SFT

This repository is for full-parameter supervised fine-tuning of `Qwen/Qwen3-32B`
for English-to-Chinese translation of CS/AI academic papers.

The target machine is 8 x RTX PRO 6000 96GB. The default direction is BF16
multi-GPU training with DeepSpeed ZeRO-3 or FSDP, not naive DDP.

## Environment

This project uses `uv` and Python 3.12.

Create or update the training, evaluation, and development environment:

```bash
uv sync --extra quality
```

Run commands through the project environment:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

PyTorch is configured to install CUDA 12.8 wheels from the official PyTorch
index. Newer NVIDIA drivers, including CUDA 13-capable drivers, can run these
wheels through driver backward compatibility.

Optional inference packages need separate handling because `vllm`, `sglang`,
and COMET often pin incompatible `torch`, `transformers`, and `protobuf`
versions.

`vllm` is available as a project extra:

```bash
uv sync --extra vllm
```

For SGLang, create a separate environment outside this project lock:

```bash
uv venv .venv-sglang --python 3.12
uv pip install --python .venv-sglang sglang[all]
```

## Repository Layout

```text
configs/              Training and distributed runtime configs.
scripts/              CLI scripts for data preparation, training, evaluation.
src/nlp_project/      Importable project code.
data/raw/             Local raw datasets. Ignored by git.
data/processed/       Local processed JSONL/Parquet datasets. Ignored by git.
data/glossary/        Small tracked terminology resources.
outputs/checkpoints/  Local model checkpoints. Ignored by git.
outputs/eval/         Local evaluation outputs. Ignored by git.
outputs/logs/         Local run logs. Ignored by git.
```

## First Milestone

The first engineering milestone is an end-to-end tiny sample run:

1. Prepare a tiny bilingual dataset.
2. Convert it to chat SFT format.
3. Load the Qwen3 tokenizer.
4. Run a short SFT smoke test.
5. Generate translations and compute SacreBLEU.

See `AGENTS.md` for project requirements and implementation guidance.

## Data Preparation

After downloading raw datasets into `data/raw/`, prepare the initial Stage 1
tiny/dev splits:

```bash
uv run python scripts/prepare_stage1_data.py
uv run python scripts/validate_sft_data.py
```

This writes ignored local artifacts:

```text
data/processed/stage1/tiny_train.jsonl
data/processed/stage1/validation.jsonl
data/processed/stage1/test.jsonl
data/processed/stage1/sft/*.jsonl
outputs/eval/data_profile/*.json
```

The intermediate files use English as `source` and Chinese as `target`. The SFT
files wrap each example in the non-thinking academic translation chat prompt.
