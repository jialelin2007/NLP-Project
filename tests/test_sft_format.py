from nlp_project.data.sft_format import SFTValidationError, validate_sft_record


def test_validate_sft_record_accepts_three_message_translation_record() -> None:
    record = {
        "id": "x1",
        "domain": "scientific",
        "split": "train",
        "metadata": {"source_dataset": "unit"},
        "messages": [
            {"role": "system", "content": "Translate without explanations."},
            {"role": "user", "content": "Translate:\n\nAn algorithm converges."},
            {"role": "assistant", "content": "算法会收敛。"},
        ],
    }

    validate_sft_record(record)


def test_validate_sft_record_rejects_think_tags() -> None:
    record = {
        "id": "x1",
        "domain": "scientific",
        "split": "train",
        "metadata": {"source_dataset": "unit"},
        "messages": [
            {"role": "system", "content": "Translate."},
            {"role": "user", "content": "Translate:\n\nText"},
            {"role": "assistant", "content": "<think>reasoning</think>译文"},
        ],
    }

    try:
        validate_sft_record(record)
    except SFTValidationError as exc:
        assert "think" in str(exc).lower()
    else:
        raise AssertionError("expected SFTValidationError")
