# Qwen Paper Translation SFT

This repository contains a local workflow for full-parameter supervised
fine-tuning of `Qwen/Qwen3-32B` for English-to-Chinese translation of CS/AI
academic papers.

The target machine is 8 x RTX PRO 6000 96GB. The standard training path is BF16
multi-GPU training with DeepSpeed ZeRO-3.

## Environment

This project uses `uv` and Python 3.12.

```bash
uv sync --extra quality
uv run python --version
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

PyTorch is configured to install CUDA 12.8 wheels from the official PyTorch
index. Optional inference packages such as `vllm` and SGLang may need separate
environments because they often pin conflicting dependency versions.

## Repository Layout

```text
configs/deepspeed/    DeepSpeed runtime configs.
configs/training/     Formal Stage 1 and Stage 2 training configs.
scripts/data/         Dataset download, preparation, and SFT validation CLIs.
scripts/training/     SFT training CLI.
scripts/evaluation/   Translation evaluation CLI.
scripts/models/       Local model inspection CLI.
src/nlp_project/      Importable project code split by responsibility.
assets/models/        Local Hugging Face model files. Ignored by git.
data/raw/             Local raw datasets. Ignored by git.
data/processed/       Processed JSONL/SFT datasets. Ignored by git.
data/glossary/        Tracked terminology resources.
runs/checkpoints/     Local model checkpoints. Ignored by git.
runs/eval/            Local evaluation outputs. Ignored by git.
runs/logs/            Local training and validation logs. Ignored by git.
```

## Model Files

Place the base model under:

```text
assets/models/Qwen3-32B/
```

Verify local model files before training:

```bash
uv run python scripts/models/inspect_local_model.py assets/models/Qwen3-32B --load-tokenizer
```

## Data Preparation

Download or place raw datasets under `data/raw/`, then build Stage 1 processed
and SFT splits:

```bash
uv run python scripts/data/prepare_stage1_data.py
uv run python scripts/data/validate_sft_data.py
```

Default outputs:

```text
data/processed/stage1/train.jsonl
data/processed/stage1/validation.jsonl
data/processed/stage1/test.jsonl
data/processed/stage1/sft/train.jsonl
data/processed/stage1/sft/validation.jsonl
data/processed/stage1/sft/test.jsonl
runs/eval/data_profile/*.json
```

Build Stage 2 arXiv CS/AI paper segments before teacher translation:

```bash
uv run python scripts/data/stage2_collect_papers.py \
  --target-papers 2000 \
  --min-citations 21 \
  --openalex-mailto you@example.com
uv run python scripts/data/stage2_fetch_html.py
uv run python scripts/data/stage2_extract_segments.py
```

Default Stage 2 segment-only outputs:

```text
data/raw/stage2/arxiv_candidates.jsonl
data/raw/stage2/openalex_enriched.jsonl
data/raw/stage2/accepted_papers.jsonl
data/raw/stage2/html/*.html
data/processed/stage2/documents/*.json
data/processed/stage2/segments/all.jsonl
```

This Stage 2 segment pipeline stops before teacher-model translation and SFT
chat conversion.

## Training

Stage 1 adapts the base model to general and scientific English-to-Chinese
translation:

```bash
uv run torchrun --nproc_per_node=8 scripts/training/train_sft.py \
  --config configs/training/qwen3_32b_stage1_full.yaml
```

Stage 2 continues from the Stage 1 checkpoint on CS/AI paper-style data:

```bash
uv run torchrun --nproc_per_node=8 scripts/training/train_sft.py \
  --config configs/training/qwen3_32b_stage2_full.yaml
```

Training configs write checkpoints and logs under `runs/`.

## Evaluation

Evaluate on the Stage 1 test split:

```bash
uv run python scripts/evaluation/evaluate_translation.py \
  --input data/processed/stage1/test.jsonl \
  --output-dir runs/eval/qwen3_32b_stage1_full \
  --limit 100
```

The current evaluator computes SacreBLEU and saves samples. It is intentionally
small; model-backed generation can be added behind the same output format:

```text
runs/eval/<run_name>/metrics.json
runs/eval/<run_name>/samples.jsonl
```

## Useful Checks

```bash
uv run ruff check .
uv run pytest
uv run python scripts/data/validate_sft_data.py
```
