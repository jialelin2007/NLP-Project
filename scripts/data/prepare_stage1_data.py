#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.processing import (  # noqa: E402
    build_sft_record,
    is_valid_translation_pair,
    iter_jsonl_gz,
    make_csl_example,
    make_quickmt_example,
    read_quickmt_parquet,
    split_by_stable_hash,
    write_jsonl,
)

ENGINEERING_CSL_CATEGORIES = {"工学", "理学", "医学", "管理学", "农学"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 translation SFT data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/stage1"))
    parser.add_argument("--profile-dir", type=Path, default=Path("runs/eval/data_profile"))
    parser.add_argument("--quickmt-train-size", type=int, default=1000)
    parser.add_argument("--csl-train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--quickmt-train-shards", type=int, default=1)
    return parser.parse_args()


def collect_quickmt_examples(
    parquet_files: list[Path], *, split: str, limit: int, start_index: int = 0
) -> list[dict]:
    examples = []
    seen_pairs: set[tuple[str, str]] = set()
    index = start_index
    for parquet_file in parquet_files:
        for record in read_quickmt_parquet(parquet_file):
            example = make_quickmt_example(record, split=split, index=index)
            index += 1
            pair = (example["source"], example["target"])
            if pair in seen_pairs:
                continue
            if not is_valid_translation_pair(example["source"], example["target"]):
                continue
            seen_pairs.add(pair)
            examples.append(example)
            if len(examples) >= limit:
                return examples
    return examples


def assign_split_by_offset(index: int) -> str:
    bucket = index % 10
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def iter_csl_examples(raw_dir: Path, *, split: str) -> Iterator[dict]:
    zh_path = raw_dir / "neuclir-csl" / "data" / "csl.jsonl.gz"
    en_path = raw_dir / "neuclir-csl" / "data" / "csl.gt.063023.jsonl.gz"
    for zh_record, en_record in zip(iter_jsonl_gz(zh_path), iter_jsonl_gz(en_path), strict=True):
        yield make_csl_example(zh_record, en_record, split=split)


def collect_csl_examples(raw_dir: Path, *, split: str, limit: int) -> list[dict]:
    examples = []
    seen_ids: set[str] = set()
    for example in iter_csl_examples(raw_dir, split=split):
        if split_by_stable_hash(example["id"]) != split:
            continue
        if example["id"] in seen_ids:
            continue
        if example["metadata"]["category"] not in ENGINEERING_CSL_CATEGORIES:
            continue
        if not is_valid_translation_pair(example["source"], example["target"]):
            continue
        seen_ids.add(example["id"])
        examples.append(example)
        if len(examples) >= limit:
            break
    return examples


def write_profile(profile_dir: Path, name: str, examples: list[dict]) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    domains = Counter(example["domain"] for example in examples)
    datasets = Counter(example["metadata"]["source_dataset"] for example in examples)
    source_lengths = [len(example["source"]) for example in examples]
    target_lengths = [len(example["target"]) for example in examples]
    summary = {
        "name": name,
        "num_examples": len(examples),
        "domains": dict(domains),
        "source_datasets": dict(datasets),
        "source_chars": summarize_lengths(source_lengths),
        "target_chars": summarize_lengths(target_lengths),
        "sample_ids": [example["id"] for example in examples[:10]],
    }
    (profile_dir / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize_lengths(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def write_split(output_dir: Path, profile_dir: Path, name: str, examples: list[dict]) -> None:
    intermediate_path = output_dir / f"{name}.jsonl"
    sft_path = output_dir / "sft" / f"{name}.jsonl"
    write_jsonl(intermediate_path, examples)
    write_jsonl(sft_path, (build_sft_record(example) for example in examples))
    write_profile(profile_dir, name, examples)


def main() -> None:
    args = parse_args()
    valid_files = sorted((args.raw_dir / "quickmt-valid.zh-en" / "data").glob("*.parquet"))
    train_files = sorted((args.raw_dir / "quickmt-train.zh-en" / "data").glob("*.parquet"))[
        : args.quickmt_train_shards
    ]
    if not valid_files:
        raise FileNotFoundError("No quickmt-valid parquet files found.")
    if not train_files:
        raise FileNotFoundError("No quickmt-train parquet files found.")

    train = [
        *collect_quickmt_examples(
            train_files, split="train", limit=args.quickmt_train_size, start_index=0
        ),
        *collect_csl_examples(args.raw_dir, split="train", limit=args.csl_train_size),
    ]
    quickmt_validation = []
    quickmt_test = []
    valid_index = 0
    for parquet_file in valid_files:
        for record in read_quickmt_parquet(parquet_file):
            assigned_split = assign_split_by_offset(valid_index)
            if assigned_split not in {"validation", "test"}:
                valid_index += 1
                continue
            example = make_quickmt_example(record, split=assigned_split, index=valid_index)
            valid_index += 1
            if not is_valid_translation_pair(example["source"], example["target"]):
                continue
            if assigned_split == "validation" and len(quickmt_validation) < args.validation_size:
                quickmt_validation.append(example)
            if assigned_split == "test" and len(quickmt_test) < args.test_size:
                quickmt_test.append(example)
            if (
                len(quickmt_validation) >= args.validation_size
                and len(quickmt_test) >= args.test_size
            ):
                break
        if len(quickmt_validation) >= args.validation_size and len(quickmt_test) >= args.test_size:
            break

    validation = [
        *quickmt_validation,
        *collect_csl_examples(args.raw_dir, split="validation", limit=args.validation_size),
    ]
    test = [
        *quickmt_test,
        *collect_csl_examples(args.raw_dir, split="test", limit=args.test_size),
    ]

    write_split(args.output_dir, args.profile_dir, "train", train)
    write_split(args.output_dir, args.profile_dir, "validation", validation)
    write_split(args.output_dir, args.profile_dir, "test", test)

    print(
        json.dumps(
            {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
                "output_dir": str(args.output_dir),
                "profile_dir": str(args.profile_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
