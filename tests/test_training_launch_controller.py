# -*- coding: utf-8 -*-
"""训练位置与启动计划分叉的直接可运行测试。"""

import _bootstrap  # noqa: F401  保持测试路径与现有测试一致

import ast
from pathlib import Path
import sys

from core.remote_training.launch import (  # noqa: E402
    LocalLaunchPlan,
    RemoteExecutionUnavailable,
    RemoteLaunchPlan,
    RemoteTargetValidationError,
    TrainingLaunchController,
)
from core.remote_training.profile_store import RemoteTrainingProfileStoreError  # noqa: E402
from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from test_remote_training_profiles import _ed25519_key  # noqa: E402


PROFILE_ID = "1" * 32


def make_profile():
    return RemoteTrainingProfile(
        name="实验室 A100",
        host="train-lab",
        port=22,
        username="trainer",
        remote_root="/srv/ezyolo/trainer",
        host_public_key=_ed25519_key(),
        id=PROFILE_ID,
    )


def runtime_config(*, task="detect"):
    return {
        "version": "YOLOv10",
        "model_prefix": "yolov10",
        "model_size": "n",
        "task": task,
        "epochs": 100,
        "batch_size": 8,
        "img_size": 640,
        "device": "mps",
    }


class Store:
    def __init__(self, profiles):
        self.profiles = profiles

    def list(self):
        return list(self.profiles)


class BrokenStore:
    def list(self):
        raise RemoteTrainingProfileStoreError("corrupted")


def assert_rejected(callable_object, *args, expected=RemoteTargetValidationError, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except expected:
        return
    raise AssertionError("预期输入被拒绝")


def test_local_plan_preserves_device_and_existing_task_support():
    controller = TrainingLaunchController(Store([make_profile()]))
    config = runtime_config(task="classify")
    plan = controller.resolve(project_id=12, runtime_config=config, target="local")
    assert isinstance(plan, LocalLaunchPlan)
    assert plan.task_type == "classify"
    assert plan.runtime_config["device"] == "mps"
    assert dict(plan.runtime_config) == config


def test_remote_plan_allows_detect_segment_and_never_contains_device():
    controller = TrainingLaunchController(Store([make_profile()]))
    for task in ("detect", "segment"):
        plan = controller.resolve(
            project_id=12,
            runtime_config=runtime_config(task=task),
            target=f"remote:{PROFILE_ID}",
        )
        assert isinstance(plan, RemoteLaunchPlan)
        assert plan.profile.id == PROFILE_ID
        assert plan.model_symbol == "yolov10n"
        assert "device" not in plan.runtime_config
        assert plan.task_type == task


def test_remote_plan_rejects_unsupported_task_without_local_fallback():
    controller = TrainingLaunchController(Store([make_profile()]))
    for task in ("classify", "pose", "world"):
        assert_rejected(
            controller.resolve,
            project_id=12,
            runtime_config=runtime_config(task=task),
            target=f"remote:{PROFILE_ID}",
        )


def test_remote_plan_rejects_missing_or_corrupt_profile():
    missing = TrainingLaunchController(Store([]))
    assert_rejected(
        missing.resolve,
        project_id=12,
        runtime_config=runtime_config(),
        target=f"remote:{PROFILE_ID}",
    )
    corrupt = TrainingLaunchController(BrokenStore())
    assert_rejected(
        corrupt.resolve,
        project_id=12,
        runtime_config=runtime_config(),
        target=f"remote:{PROFILE_ID}",
    )


def test_remote_target_format_and_model_symbol_are_validated():
    controller = TrainingLaunchController(Store([make_profile()]))
    assert_rejected(
        controller.resolve,
        project_id=12,
        runtime_config=runtime_config(),
        target="remote:not-a-profile",
    )
    bad = runtime_config()
    bad["model_prefix"] = "model;rm"
    assert_rejected(
        controller.resolve,
        project_id=12,
        runtime_config=bad,
        target=f"remote:{PROFILE_ID}",
    )


def test_executor_not_registered_blocks_remote_without_downgrading_to_local():
    controller = TrainingLaunchController(Store([make_profile()]))
    assert_rejected(
        controller.resolve_for_execution,
        project_id=12,
        runtime_config=runtime_config(),
        target=f"remote:{PROFILE_ID}",
        expected=RemoteExecutionUnavailable,
    )
    local = controller.resolve_for_execution(
        project_id=12,
        runtime_config=runtime_config(),
        target="local",
    )
    assert isinstance(local, LocalLaunchPlan)


def test_launch_module_stays_outside_training_transport_and_frameworks():
    source = (
        Path(__file__).parent.parent / "core" / "remote_training" / "launch.py"
    ).read_text(encoding="utf-8")
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    for forbidden in ("ultralytics", "subprocess", "paramiko", "PyQt"):
        assert not any(
            module == forbidden or module.startswith(forbidden + ".")
            for module in imports
        )


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
