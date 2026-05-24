#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
DATA_DIR = REPORT / "data"
TABLE_DIR = REPORT / "tables"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def profile_translation_jsonl(path: Path) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    sections: Counter[str] = Counter()
    papers: set[str] = set()
    source_lengths: list[int] = []
    target_lengths: list[int] = []

    for record in iter_jsonl(path):
        domains[str(record.get("domain"))] += 1
        splits[str(record.get("split"))] += 1
        metadata = record.get("metadata") or {}
        sources[str(metadata.get("source_dataset"))] += 1
        sections[str(metadata.get("section"))] += 1
        if metadata.get("paper_id"):
            papers.add(str(metadata["paper_id"]))
        source = record.get("source") or ""
        target = record.get("target") or ""
        if not source and record.get("messages"):
            source = (record["messages"][1].get("content") or "").split("\n\n", 1)[-1]
            target = record["messages"][2].get("content") or ""
        source_lengths.append(len(source))
        target_lengths.append(len(target))

    def stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"mean": 0.0, "median": 0, "p90": 0, "max": 0}
        ordered = sorted(values)
        return {
            "mean": round(statistics.mean(values), 1),
            "median": ordered[len(ordered) // 2],
            "p90": ordered[int(len(ordered) * 0.9)],
            "max": max(values),
        }

    return {
        "path": str(path.relative_to(ROOT)),
        "count": len(source_lengths),
        "domains": dict(domains),
        "sources": dict(sources),
        "splits": dict(splits),
        "sections": dict(sections),
        "paper_count": len(papers),
        "source_chars": stats(source_lengths),
        "target_chars": stats(target_lengths),
    }


def collect_eval_points() -> list[dict[str, Any]]:
    points: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((ROOT / "runs/checkpoints").glob("**/trainer_state.json")):
        state = read_json(path)
        run = path.relative_to(ROOT / "runs/checkpoints").parts[0]
        for row in state.get("log_history", []):
            if "eval_loss" not in row:
                continue
            step = int(row["step"])
            points[(run, step)] = {
                "run": run,
                "step": step,
                "eval_loss": float(row["eval_loss"]),
                "eval_token_accuracy": float(row.get("eval_mean_token_accuracy", 0.0)),
                "epoch": float(row.get("epoch", 0.0)),
            }
    return sorted(points.values(), key=lambda item: (item["run"], item["step"]))


def make_eval_plot_points(eval_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in eval_points:
        if item["run"] == "qwen3_32b_stage1_full" and item["step"] <= 1000:
            rows.append({**item, "series": "Stage1 full"})
        elif item["run"] == "qwen3_32b_stage1_bestval_from1000" and item["step"] > 1000:
            rows.append({**item, "series": "From1000 small LR"})
    return rows


def collect_train_points() -> list[dict[str, Any]]:
    points: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((ROOT / "runs/checkpoints").glob("**/trainer_state.json")):
        state = read_json(path)
        run = path.relative_to(ROOT / "runs/checkpoints").parts[0]
        for row in state.get("log_history", []):
            if "loss" not in row or int(row.get("step", -1)) % 50 != 0:
                continue
            step = int(row["step"])
            points[(run, step)] = {
                "run": run,
                "step": step,
                "train_loss": float(row["loss"]),
                "train_token_accuracy": float(row.get("mean_token_accuracy", 0.0)),
                "learning_rate": float(row.get("learning_rate", 0.0)),
            }
    return sorted(points.values(), key=lambda item: (item["run"], item["step"]))


def make_train_plot_points(train_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in train_points:
        if item["run"] == "qwen3_32b_stage1_full" and item["step"] <= 1000:
            rows.append({**item, "series": "Stage1 full"})
        elif item["run"] == "qwen3_32b_stage1_bestval_from1000" and item["step"] > 1000:
            rows.append({**item, "series": "From1000 small LR"})
    return rows


def collect_wandb_summaries() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((ROOT / "wandb").glob("run-*/files/wandb-summary.json")):
        summary = read_json(path)
        run_dir = path.parents[1].name
        config_path = path.parent / "config.yaml"
        name = run_dir.split("-", 2)[-1]
        if config_path.is_file():
            text = config_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.startswith("run_name:") or line.startswith("wandb_run_name:"):
                    name = line.split(":", 1)[1].strip()
                    break
        rows.append(
            {
                "run_dir": run_dir,
                "run_id": run_dir.split("-")[-1],
                "name": name,
                "global_step": int(summary.get("train/global_step", 0)),
                "train_loss": summary.get("train/loss", ""),
                "eval_loss": summary.get("eval/loss", ""),
                "eval_token_accuracy": summary.get("eval/mean_token_accuracy", ""),
                "runtime_hours": round(float(summary.get("_runtime", 0)) / 3600, 2),
            }
        )
    return rows


def collect_stage2_paper_stats() -> dict[str, Any]:
    path = ROOT / "data/raw/stage2/accepted_papers.jsonl"
    citations: list[int] = []
    categories: Counter[str] = Counter()
    years: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        citation = int(record.get("cited_by_count") or 0)
        citations.append(citation)
        for category in record.get("categories") or []:
            categories[category.replace("openalex:", "")] += 1
        year = str(record.get("submitted_at") or "")[:4]
        if year:
            years[year] += 1
        if len(samples) < 6:
            samples.append(
                {
                    "arxiv_id": record.get("arxiv_id"),
                    "citations": citation,
                    "title": record.get("title"),
                }
            )
    return {
        "count": len(citations),
        "citation_min": min(citations) if citations else 0,
        "citation_median": statistics.median(citations) if citations else 0,
        "citation_mean": round(statistics.mean(citations), 1) if citations else 0,
        "citation_max": max(citations) if citations else 0,
        "top_categories": categories.most_common(10),
        "years": dict(sorted(years.items())),
        "samples": samples,
    }


def collect_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    stage1_path = ROOT / "data/processed/stage1/train.jsonl"
    for record in iter_jsonl(stage1_path):
        examples["stage1_quickmt"] = record
        break
    for record in iter_jsonl(stage1_path):
        if record.get("domain") == "scientific":
            examples["stage1_csl"] = record
            break
    stage2_path = ROOT / "data/processed/stage2/segments/all.jsonl"
    for record in iter_jsonl(stage2_path):
        examples["stage2_segment"] = record
        break
    teacher_path = ROOT / "runs/eval/teacher_model_reasoning_comparison_stage2_prompt/translations.jsonl"
    for record in iter_jsonl(teacher_path):
        examples["teacher_translation"] = record
        break
    return examples


def write_simple_table(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = [
        r"\begin{tabular}{%s}" % ("l" * len(headers)),
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables(summary: dict[str, Any]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stage1 = summary["profiles"]["stage1_train"]
    stage1_val = summary["profiles"]["stage1_validation"]
    stage1_test = summary["profiles"]["stage1_test"]
    stage2 = summary["profiles"]["stage2_segments"]
    write_simple_table(
        TABLE_DIR / "dataset_summary.tex",
        ["Dataset", "Examples", "Source / domain", "Papers", "Avg. source chars", "Avg. target chars"],
        [
            ["Stage 1 train", stage1["count"], "1M QuickMT + 158.6K CSL", stage1["paper_count"], stage1["source_chars"]["mean"], stage1["target_chars"]["mean"]],
            ["Stage 1 validation", stage1_val["count"], "683 QuickMT + 10K CSL", stage1_val["paper_count"], stage1_val["source_chars"]["mean"], stage1_val["target_chars"]["mean"]],
            ["Stage 1 test", stage1_test["count"], "667 QuickMT + 10K CSL", stage1_test["paper_count"], stage1_test["source_chars"]["mean"], stage1_test["target_chars"]["mean"]],
            ["Stage 2 segments", stage2["count"], "arXiv CS/AI papers", stage2["paper_count"], stage2["source_chars"]["mean"], "-"],
        ],
    )
    write_simple_table(
        TABLE_DIR / "training_summary.tex",
        ["Experiment", "Source", "Steps", "Train loss", "Eval loss", "Eval token acc.", "Status"],
        [
            ["Stage1 warmup", "wandb rf5tj6pa", 35, 1.177, "-", "-", "interrupted warmup"],
            ["Stage1 full", "checkpoint-1000", 1000, 0.8365, 0.6761, 0.8225, "best eval point"],
            ["Stage1 full", "wandb 49q2uxvx", 1480, 0.7440, 0.6990, 0.8199, "degraded eval"],
            ["From1000 small LR", "checkpoint-1500", 1500, 0.7443, 0.6862, 0.8215, "best-model load OOM"],
        ],
    )
    stage2_stats = summary["stage2_papers"]
    write_simple_table(
        TABLE_DIR / "stage2_collection.tex",
        ["Metric", "Value"],
        [
            ["Accepted OpenAlex/arXiv papers", stage2_stats["count"]],
            ["Minimum citation count", stage2_stats["citation_min"]],
            ["Median citation count", stage2_stats["citation_median"]],
            ["Mean citation count", stage2_stats["citation_mean"]],
            ["Maximum citation count", stage2_stats["citation_max"]],
            ["Extracted English segments", stage2["count"]],
            ["Papers covered by segments", stage2["paper_count"]],
        ],
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    profiles = {
        "stage1_train": profile_translation_jsonl(ROOT / "data/processed/stage1/train.jsonl"),
        "stage1_validation": profile_translation_jsonl(ROOT / "data/processed/stage1/validation.jsonl"),
        "stage1_test": profile_translation_jsonl(ROOT / "data/processed/stage1/test.jsonl"),
        "stage2_segments": profile_translation_jsonl(ROOT / "data/processed/stage2/segments/all.jsonl"),
    }
    eval_points = collect_eval_points()
    train_points = collect_train_points()
    eval_plot_points = make_eval_plot_points(eval_points)
    train_plot_points = make_train_plot_points(train_points)
    wandb_summaries = collect_wandb_summaries()
    stage2_papers = collect_stage2_paper_stats()
    examples = collect_examples()

    write_csv(
        DATA_DIR / "eval_loss.csv",
        eval_points,
        ["run", "step", "eval_loss", "eval_token_accuracy", "epoch"],
    )
    write_csv(
        DATA_DIR / "train_points.csv",
        train_points,
        ["run", "step", "train_loss", "train_token_accuracy", "learning_rate"],
    )
    write_csv(
        DATA_DIR / "eval_loss_plot.csv",
        eval_plot_points,
        ["series", "run", "step", "eval_loss", "eval_token_accuracy", "epoch"],
    )
    write_csv(
        DATA_DIR / "eval_stage1_full.csv",
        [row for row in eval_plot_points if row["series"] == "Stage1 full"],
        ["series", "run", "step", "eval_loss", "eval_token_accuracy", "epoch"],
    )
    write_csv(
        DATA_DIR / "eval_from1000.csv",
        [row for row in eval_plot_points if row["series"] == "From1000 small LR"],
        ["series", "run", "step", "eval_loss", "eval_token_accuracy", "epoch"],
    )
    write_csv(
        DATA_DIR / "train_points_plot.csv",
        train_plot_points,
        ["series", "run", "step", "train_loss", "train_token_accuracy", "learning_rate"],
    )
    write_csv(
        DATA_DIR / "train_stage1_full.csv",
        [row for row in train_plot_points if row["series"] == "Stage1 full"],
        ["series", "run", "step", "train_loss", "train_token_accuracy", "learning_rate"],
    )
    write_csv(
        DATA_DIR / "train_from1000.csv",
        [row for row in train_plot_points if row["series"] == "From1000 small LR"],
        ["series", "run", "step", "train_loss", "train_token_accuracy", "learning_rate"],
    )
    write_csv(
        DATA_DIR / "wandb_summaries.csv",
        wandb_summaries,
        ["run_dir", "run_id", "name", "global_step", "train_loss", "eval_loss", "eval_token_accuracy", "runtime_hours"],
    )
    write_csv(
        DATA_DIR / "stage2_years.csv",
        [{"year": year, "count": count} for year, count in stage2_papers["years"].items()],
        ["year", "count"],
    )
    write_csv(
        DATA_DIR / "stage2_categories.csv",
        [{"category": category, "count": count} for category, count in stage2_papers["top_categories"]],
        ["category", "count"],
    )

    summary = {
        "profiles": profiles,
        "wandb_summaries": wandb_summaries,
        "stage2_papers": stage2_papers,
        "examples": examples,
        "counts": {
            "stage2_sft_train": count_jsonl(ROOT / "data/processed/stage2/sft/train.jsonl"),
            "stage2_sft_validation": count_jsonl(ROOT / "data/processed/stage2/sft/validation.jsonl"),
            "stage2_sft_test": count_jsonl(ROOT / "data/processed/stage2/sft/test.jsonl"),
        },
    }
    (DATA_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tables(summary)
    print(DATA_DIR / "summary.json")


if __name__ == "__main__":
    main()
