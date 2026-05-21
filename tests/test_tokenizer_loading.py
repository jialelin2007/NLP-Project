from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nlp_project.models.tokenizer import load_tokenizer


def test_load_tokenizer_enables_mistral_regex_fix() -> None:
    with patch("nlp_project.models.tokenizer.AutoTokenizer.from_pretrained") as from_pretrained:
        load_tokenizer(Path("checkpoint-1000"))

    from_pretrained.assert_called_once_with(
        Path("checkpoint-1000"),
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
