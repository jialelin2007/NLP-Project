from nlp_project.evaluation import compute_sacrebleu, make_sample_record


def test_compute_sacrebleu_returns_score() -> None:
    score = compute_sacrebleu(["算法会收敛。"], ["算法会收敛。"])
    assert score > 99


def test_make_sample_record_preserves_source_reference_prediction() -> None:
    record = make_sample_record(
        example_id="x1",
        source="An algorithm converges.",
        reference="算法会收敛。",
        prediction="算法收敛。",
    )
    assert record["id"] == "x1"
    assert record["source"] == "An algorithm converges."
    assert record["reference"] == "算法会收敛。"
    assert record["prediction"] == "算法收敛。"
