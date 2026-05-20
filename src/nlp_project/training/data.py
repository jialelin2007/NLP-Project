from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict


def iter_message_records(path: str | Path) -> Iterator[dict[str, list[dict[str, str]]]]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield {"messages": record["messages"]}


def load_sft_message_datasets(
    train_file: Path, validation_file: Path
) -> DatasetDict:
    train_dataset = Dataset.from_generator(
        iter_message_records,
        gen_kwargs={"path": str(train_file)},
    )
    validation_dataset = Dataset.from_generator(
        iter_message_records,
        gen_kwargs={"path": str(validation_file)},
    )
    return DatasetDict({"train": train_dataset, "validation": validation_dataset})
