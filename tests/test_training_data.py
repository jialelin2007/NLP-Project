import json
from pathlib import Path

from nlp_project.training.data import load_sft_message_datasets


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_load_sft_message_datasets_ignores_metadata_shape_differences(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"

    _write_jsonl(
        train_file,
        [
            {
                "id": "a",
                "domain": "general",
                "split": "train",
                "metadata": {
                    "source_dataset": "quickmt",
                    "score": 0.9,
                    "paper_id": None,
                    "section": None,
                },
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            },
            {
                "id": "b",
                "domain": "scientific",
                "split": "train",
                "metadata": {
                    "source_dataset": "csl",
                    "score": None,
                    "paper_id": "p1",
                    "section": "title_abstract",
                    "keywords": ["k"],
                    "keywords_en": ["k"],
                    "category": "工学",
                    "category_eng": "Engineering",
                    "discipline": "计算机科学",
                    "discipline_eng": "Computer Science",
                },
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            },
        ],
    )
    _write_jsonl(
        validation_file,
        [
            {
                "id": "c",
                "domain": "scientific",
                "split": "validation",
                "metadata": {
                    "source_dataset": "csl",
                    "score": None,
                    "paper_id": "p2",
                    "section": "title_abstract",
                    "keywords": ["k"],
                    "keywords_en": ["k"],
                    "category": "理学",
                    "category_eng": "Science",
                    "discipline": "数学",
                    "discipline_eng": "Mathematics",
                },
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        ],
    )

    datasets = load_sft_message_datasets(train_file, validation_file)

    assert datasets["train"].column_names == ["messages"]
    assert datasets["validation"].column_names == ["messages"]
    assert datasets["train"][0]["messages"][2]["content"] == "a"
    assert datasets["validation"][0]["messages"][0]["role"] == "system"
