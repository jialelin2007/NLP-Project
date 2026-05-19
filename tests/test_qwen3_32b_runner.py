from pathlib import Path


def test_qwen3_32b_runner_uses_project_torchrun() -> None:
    source = Path("scripts/run_qwen3_32b_smoke.sh").read_text(encoding="utf-8")

    assert "uv run torchrun --nproc_per_node=8" in source
    assert "scripts/inspect_model.py" in source
    assert "scripts/validate_sft_data.py" in source
