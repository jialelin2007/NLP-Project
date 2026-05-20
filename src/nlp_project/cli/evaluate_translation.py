from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlp_project.evaluation.metrics import compute_sacrebleu, make_sample_record, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run translation evaluation.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/stage1/test.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/eval/copy_source_baseline")
    )
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = []
    predictions = []
    references = []

    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= args.limit:
                break
            example = json.loads(line)
            prediction = example["source"]
            predictions.append(prediction)
            references.append(example["target"])
            samples.append(
                make_sample_record(
                    example_id=example["id"],
                    source=example["source"],
                    reference=example["target"],
                    prediction=prediction,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    write_json(
        args.output_dir / "metrics.json",
        {"sacrebleu": compute_sacrebleu(predictions, references), "num_samples": len(samples)},
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
