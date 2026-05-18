import json

from nlp_project.data_processing import (
    build_sft_record,
    is_valid_translation_pair,
    make_csl_example,
    make_quickmt_example,
    split_by_stable_hash,
)


def test_make_quickmt_example_uses_english_source_and_chinese_target() -> None:
    record = {"en": "Neural networks are widely used.", "zh": "神经网络被广泛使用。", "sco": 0.9}

    example = make_quickmt_example(record, split="train", index=7)

    assert example["id"] == "quickmt_train_0000000007"
    assert example["source"] == "Neural networks are widely used."
    assert example["target"] == "神经网络被广泛使用。"
    assert example["domain"] == "general"
    assert example["split"] == "train"
    assert example["metadata"]["source_dataset"] == "quickmt"
    assert example["metadata"]["score"] == 0.9


def test_make_csl_example_joins_title_and_abstract_by_doc_id() -> None:
    zh_record = {
        "doc_id": "csl-1",
        "title": "神经机器翻译研究",
        "abstract": "本文研究神经机器翻译模型。",
        "keywords": ["机器翻译"],
        "category": "工学",
        "category_eng": "Engineering",
        "discipline": "计算机科学",
        "discipline_eng": "Computer Science",
    }
    en_record = {
        "doc_id": "csl-1",
        "title": "Research on neural machine translation",
        "abstract": "This paper studies neural machine translation models.",
        "keywords": ["machine translation"],
        "category": "Engineering",
        "discipline": "Computer Science",
    }

    example = make_csl_example(zh_record, en_record, split="validation")

    assert example["id"] == "csl-1"
    assert example["source"] == (
        "Research on neural machine translation\n\n"
        "This paper studies neural machine translation models."
    )
    assert example["target"] == "神经机器翻译研究\n\n本文研究神经机器翻译模型。"
    assert example["domain"] == "scientific"
    assert example["split"] == "validation"
    assert example["metadata"]["source_dataset"] == "csl"
    assert example["metadata"]["category"] == "工学"
    assert example["metadata"]["category_eng"] == "Engineering"
    assert example["metadata"]["discipline_eng"] == "Computer Science"


def test_make_csl_example_rejects_mismatched_doc_id() -> None:
    zh_record = {"doc_id": "csl-1", "title": "中文", "abstract": "摘要"}
    en_record = {"doc_id": "csl-2", "title": "English", "abstract": "Abstract"}

    try:
        make_csl_example(zh_record, en_record, split="train")
    except ValueError as exc:
        assert "doc_id mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_is_valid_translation_pair_filters_empty_short_ratio_and_self_talk() -> None:
    assert is_valid_translation_pair("A useful source sentence.", "一个有用的中文译文。")
    assert not is_valid_translation_pair("", "一个有用的中文译文。")
    assert not is_valid_translation_pair("A useful source sentence.", "")
    assert not is_valid_translation_pair("Too short.", "短")
    assert not is_valid_translation_pair("A useful source sentence.", "<think>隐藏推理</think>")
    assert not is_valid_translation_pair("x " * 200, "短译文")


def test_build_sft_record_uses_non_thinking_translation_prompt() -> None:
    example = {
        "id": "example-1",
        "source": "Transformers are effective.",
        "target": "Transformer 很有效。",
        "domain": "scientific",
        "split": "train",
        "metadata": {"source_dataset": "unit"},
    }

    record = build_sft_record(example)

    assert record["id"] == "example-1"
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
    assert "Do not add explanations" in record["messages"][0]["content"]
    assert "<think>" not in json.dumps(record, ensure_ascii=False)
    assert record["messages"][1]["content"].endswith("Transformers are effective.")
    assert record["messages"][2]["content"] == "Transformer 很有效。"


def test_split_by_stable_hash_is_deterministic_and_disjoint() -> None:
    train = {value for value in range(100) if split_by_stable_hash(f"id-{value}") == "train"}
    validation = {
        value for value in range(100) if split_by_stable_hash(f"id-{value}") == "validation"
    }
    test = {value for value in range(100) if split_by_stable_hash(f"id-{value}") == "test"}

    assert train
    assert validation
    assert test
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert split_by_stable_hash("same-id") == split_by_stable_hash("same-id")
