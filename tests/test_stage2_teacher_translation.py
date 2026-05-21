from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx

from nlp_project.data.sft_format import validate_sft_record
from nlp_project.data.stage2_translation import (
    ResponsesTeacherClient,
    TeacherResponseError,
    classify_teacher_error,
    make_teacher_sft_record,
    select_shard,
)


def _load_translate_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/data/stage2_translate_teacher.py"
    spec = importlib.util.spec_from_file_location("stage2_translate_teacher_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _segment(segment_id: str = "2401.12345v1_abstract_0001", split: str = "train") -> dict:
    return {
        "id": segment_id,
        "source": "Transformers are effective for machine translation.",
        "domain": "cs_ai_paper",
        "split": split,
        "metadata": {
            "source_dataset": "arxiv",
            "paper_id": "2401.12345v1",
            "section": "abstract",
            "title": "Paper title",
            "source_url": "https://arxiv.org/html/2401.12345",
        },
    }


def test_make_teacher_sft_record_preserves_source_target_and_metadata() -> None:
    record = make_teacher_sft_record(
        _segment(),
        target="Transformer 对机器翻译很有效。",
        teacher_model="gpt-5.4",
    )

    validate_sft_record(record)
    assert record["id"] == "2401.12345v1_abstract_0001"
    assert record["source"] == "Transformers are effective for machine translation."
    assert record["target"] == "Transformer 对机器翻译很有效。"
    assert record["metadata"]["paper_id"] == "2401.12345v1"
    assert record["metadata"]["teacher_model"] == "gpt-5.4"
    assert record["messages"][2]["content"] == "Transformer 对机器翻译很有效。"


def test_select_shard_is_stable_by_paper_id() -> None:
    segment = _segment()

    first = select_shard(segment, num_shards=8)
    second = select_shard({**segment, "id": "different_segment"}, num_shards=8)

    assert first == second
    assert 0 <= first < 8


def test_responses_teacher_client_posts_responses_payload_and_extracts_text() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Transformer 对机器翻译很有效。",
                            }
                        ]
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = ResponsesTeacherClient(
        base_url="https://api.vip1129.cc/v1",
        api_key="secret",
        model="gpt-5.4",
        http_client=http_client,
    )

    result = client.translate("Transformers are effective for machine translation.")

    assert result == "Transformer 对机器翻译很有效。"
    assert requests[0]["model"] == "gpt-5.4"
    assert requests[0]["input"][0]["role"] == "system"
    assert requests[0]["input"][1]["content"].endswith(
        "Transformers are effective for machine translation."
    )


def test_responses_teacher_client_marks_non_json_response_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway overload</html>")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = ResponsesTeacherClient(
        base_url="https://api.vip1129.cc/v1",
        api_key="secret",
        model="gpt-5.4",
        http_client=http_client,
    )

    try:
        client.translate("Transformers are effective for machine translation.")
    except TeacherResponseError as exc:
        assert exc.classification == "retryable"
        assert "not valid JSON" in str(exc)
        assert "gateway overload" in str(exc)
    else:
        raise AssertionError("expected TeacherResponseError")


def test_classify_teacher_error_distinguishes_retryable_and_permanent_statuses() -> None:
    request = httpx.Request("POST", "https://api.example.test/v1/responses")

    assert (
        classify_teacher_error(
            httpx.HTTPStatusError(
                "rate", request=request, response=httpx.Response(429, request=request)
            )
        )
        == "retryable"
    )
    assert (
        classify_teacher_error(
            httpx.HTTPStatusError(
                "server", request=request, response=httpx.Response(500, request=request)
            )
        )
        == "retryable"
    )
    assert (
        classify_teacher_error(
            httpx.HTTPStatusError(
                "missing", request=request, response=httpx.Response(404, request=request)
            )
        )
        == "permanent"
    )
    assert (
        classify_teacher_error(
            httpx.HTTPStatusError(
                "bad", request=request, response=httpx.Response(406, request=request)
            )
        )
        == "permanent"
    )
    assert classify_teacher_error(httpx.ReadTimeout("timeout")) == "retryable"

    assert classify_teacher_error(TeacherResponseError("bad json", "retryable")) == "retryable"


def test_stage2_translate_teacher_writes_split_outputs_and_skips_completed(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_translate_script_module()
    input_path = tmp_path / "segments" / "all.jsonl"
    output_dir = tmp_path / "sft"
    input_path.parent.mkdir(parents=True)
    records = [_segment("seg-1", split="train"), _segment("seg-2", split="validation")]
    input_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    calls: list[str] = []

    class FakeTeacherClient:
        model = "gpt-5.4"

        def translate(self, source: str) -> str:
            calls.append(source)
            return "Transformer 对机器翻译很有效。"

        def close(self) -> None:
            pass

    args = module.parse_args(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--api-key",
            "secret",
            "--max-workers",
            "2",
            "--request-sleep",
            "0",
        ]
    )
    monkeypatch.setattr(module, "build_teacher_client", lambda args: FakeTeacherClient())

    summary = module.translate_stage2_segments(args)
    rerun_summary = module.translate_stage2_segments(args)

    assert summary["translated"] == 2
    assert rerun_summary["skipped_completed"] == 2
    assert len(calls) == 2
    train_records = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_records = [
        json.loads(line)
        for line in (output_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train_records[0]["id"] == "seg-1"
    assert validation_records[0]["id"] == "seg-2"
    validate_sft_record(train_records[0])


def test_stage2_translate_teacher_retries_429_and_records_permanent_errors(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_translate_script_module()
    input_path = tmp_path / "segments" / "all.jsonl"
    output_dir = tmp_path / "sft"
    input_path.parent.mkdir(parents=True)
    records = [_segment("retry-seg"), _segment("permanent-seg")]
    records[1]["source"] = "This source should be marked as permanently unavailable."
    input_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    request = httpx.Request("POST", "https://api.example.test/v1/responses")

    class FakeTeacherClient:
        model = "gpt-5.4"

        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def translate(self, source: str) -> str:
            self.calls[source] = self.calls.get(source, 0) + 1
            if "Transformers" in source and len(self.calls) == 1 and self.calls[source] == 1:
                raise httpx.HTTPStatusError(
                    "rate",
                    request=request,
                    response=httpx.Response(429, request=request),
                )
            if len(self.calls) > 1:
                raise httpx.HTTPStatusError(
                    "missing",
                    request=request,
                    response=httpx.Response(404, request=request),
                )
            return "Transformer 对机器翻译很有效。"

        def close(self) -> None:
            pass

    fake_client = FakeTeacherClient()
    args = module.parse_args(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--api-key",
            "secret",
            "--max-workers",
            "1",
            "--request-sleep",
            "0",
            "--max-retries",
            "2",
            "--retry-base-sleep",
            "0",
        ]
    )
    monkeypatch.setattr(module, "build_teacher_client", lambda args: fake_client)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    summary = module.translate_stage2_segments(args)

    assert summary["translated"] == 1
    assert summary["failed"] == 1
    errors = [
        json.loads(line)
        for line in (output_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert errors[0]["id"] == "permanent-seg"
    assert errors[0]["classification"] == "permanent"


def test_request_throttle_spaces_concurrent_requests(monkeypatch) -> None:
    module = _load_translate_script_module()
    clock = {"now": 10.0}
    sleeps: list[float] = []

    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(module.time, "sleep", fake_sleep)
    throttle = module.RequestThrottle(0.5)

    throttle.wait()
    throttle.wait()
    throttle.wait()

    assert sleeps == [0.5, 0.5]
