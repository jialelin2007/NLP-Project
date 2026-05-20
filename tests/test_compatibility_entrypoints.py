import importlib
import importlib.util
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


def test_legacy_module_aliases_are_removed() -> None:
    module_names = [
        "nlp_project.data_processing",
        "nlp_project.sft_format",
        "nlp_project.model_inventory",
        "nlp_project.training_config",
    ]

    for module_name in module_names:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_only_canonical_script_entrypoints_exist() -> None:
    legacy_script_paths = [
        "scripts/prepare_stage1_data.py",
        "scripts/validate_sft_data.py",
        "scripts/evaluate.py",
        "scripts/inspect_model.py",
        "scripts/train_sft.py",
        "scripts/run_smoke_test.sh",
        "scripts/run_qwen3_32b_smoke.sh",
        "scripts/download_raw_datasets.sh",
    ]
    script_paths = [
        "scripts/data/download_raw_datasets.sh",
        "scripts/data/prepare_stage1_data.py",
        "scripts/data/validate_sft_data.py",
        "scripts/evaluation/evaluate_translation.py",
        "scripts/models/inspect_local_model.py",
        "scripts/training/train_sft.py",
        "scripts/training/run_smoke_test.sh",
        "scripts/training/run_qwen3_32b_smoke.sh",
    ]

    for script_path in legacy_script_paths:
        assert not Path(script_path).exists(), script_path
    for script_path in script_paths:
        assert Path(script_path).is_file(), script_path


def test_only_canonical_config_paths_exist() -> None:
    legacy_config_paths = [
        "configs/ds_zero3_bf16.json",
        "configs/qwen3_32b_full_sft_stage1_smoke.yaml",
        "configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml",
    ]
    config_paths = [
        "configs/deepspeed/zero3_bf16.json",
        "configs/training/qwen3_32b_stage1_smoke.yaml",
        "configs/training/qwen3_32b_stage1_8gpu_smoke.yaml",
    ]

    for config_path in legacy_config_paths:
        assert not Path(config_path).exists(), config_path
    for config_path in config_paths:
        assert Path(config_path).is_file(), config_path
