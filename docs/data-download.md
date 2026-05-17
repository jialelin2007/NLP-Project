# Raw Dataset Download

The raw datasets for this project are stored under `data/raw/` and ignored by
git. Download them on a machine that can reach Hugging Face, then copy the
resulting directories back to this repository if needed.

## Prerequisites

Create the project environment first:

```bash
uv sync --extra quality
```

Optional but recommended for faster large-file transfer:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

If the datasets require authentication in your environment, log in first:

```bash
uv run hf auth login
```

## Download Commands

Run all requested downloads:

```bash
bash scripts/download_raw_datasets.sh
```

Or run the commands manually:

```bash
mkdir -p data/raw/quickmt-valid.zh-en
uv run hf download quickmt/quickmt-valid.zh-en \
  --repo-type dataset \
  --local-dir data/raw/quickmt-valid.zh-en \
  --max-workers 4

mkdir -p data/raw/neuclir-csl
uv run hf download neuclir/csl \
  --repo-type dataset \
  --local-dir data/raw/neuclir-csl \
  --include "data/csl.jsonl.gz" "data/csl.gt.063023.jsonl.gz" "README.md" \
  --max-workers 4

mkdir -p data/raw/quickmt-train.zh-en
uv run hf download quickmt/quickmt-train.zh-en \
  --repo-type dataset \
  --local-dir data/raw/quickmt-train.zh-en \
  --max-workers 4
```

The quickmt training dataset is large. If the transfer is unstable, rerun the
same command; `hf download` reuses the local cache and existing files.

## Expected Local Layout

```text
data/raw/
  quickmt-valid.zh-en/
  neuclir-csl/
    data/csl.jsonl.gz
    data/csl.gt.063023.jsonl.gz
  quickmt-train.zh-en/
```

## Verification

After download, check files and sizes:

```bash
find data/raw/quickmt-valid.zh-en data/raw/neuclir-csl data/raw/quickmt-train.zh-en \
  -maxdepth 3 -type f -printf "%p\t%s bytes\n" | sort
```

Check that the CSL gzip files can be read:

```bash
gzip -t data/raw/neuclir-csl/data/csl.jsonl.gz
gzip -t data/raw/neuclir-csl/data/csl.gt.063023.jsonl.gz
```

Preview the first CSL rows:

```bash
gzip -cd data/raw/neuclir-csl/data/csl.jsonl.gz | head -n 2
gzip -cd data/raw/neuclir-csl/data/csl.gt.063023.jsonl.gz | head -n 2
```

