from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_name_or_path: str | Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
