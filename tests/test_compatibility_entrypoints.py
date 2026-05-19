import importlib
from pathlib import Path


def test_canonical_functional_modules_are_importable() -> None:
    module_names = [
        "nlp_project.data.processing",
        "nlp_project.data.sft_format",
        "nlp_project.evaluation.metrics",
        "nlp_project.models.inventory",
        "nlp_project.training.config",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name)


def test_legacy_modules_remain_importable() -> None:
    module_names = [
        "nlp_project.data_processing",
        "nlp_project.sft_format",
        "nlp_project.evaluation",
        "nlp_project.model_inventory",
        "nlp_project.training_config",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name)


def test_legacy_and_canonical_script_entrypoints_exist() -> None:
    script_paths = [
        "scripts/prepare_stage1_data.py",
        "scripts/validate_sft_data.py",
        "scripts/evaluate.py",
        "scripts/inspect_model.py",
        "scripts/train_sft.py",
        "scripts/run_smoke_test.sh",
        "scripts/run_qwen3_32b_smoke.sh",
        "scripts/data/prepare_stage1_data.py",
        "scripts/data/validate_sft_data.py",
        "scripts/evaluation/evaluate_translation.py",
        "scripts/models/inspect_local_model.py",
        "scripts/training/train_sft.py",
        "scripts/training/run_smoke_test.sh",
        "scripts/training/run_qwen3_32b_smoke.sh",
    ]

    for script_path in script_paths:
        assert Path(script_path).is_file(), script_path


def test_canonical_config_paths_exist() -> None:
    config_paths = [
        "configs/deepspeed/zero3_bf16.json",
        "configs/training/qwen3_32b_stage1_smoke.yaml",
        "configs/training/qwen3_32b_stage1_8gpu_smoke.yaml",
    ]

    for config_path in config_paths:
        assert Path(config_path).is_file(), config_path
