# -*- coding: utf-8 -*-
"""远程训练与既有本机训练的机械边界测试。"""

import _bootstrap  # noqa: F401

import ast
import inspect
from pathlib import Path
import sys

from gui.remote_training_thread import RemoteTrainingThread  # noqa: E402


APP_ROOT = Path(__file__).parent.parent


def _class_source(path: Path, class_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"找不到类 {class_name}")


def test_original_local_training_thread_has_no_remote_execution_dependency():
    source = _class_source(APP_ROOT / "gui" / "pages" / "train_page.py", "TrainingThread")
    for forbidden in (
        "RemoteTrainingThread",
        "RemoteDatasetPlanner",
        "RemoteCommandBuilder",
        "SshRsyncBackend",
        "RemoteTrainingProfileStore",
        "ssh",
        "rsync",
    ):
        assert forbidden not in source


def test_remote_thread_requires_an_explicit_result_verifier_and_never_uses_local_trainer():
    signature = inspect.signature(RemoteTrainingThread.__init__)
    assert signature.parameters["result_verifier"].default is inspect.Parameter.empty
    source = (APP_ROOT / "gui" / "remote_training_thread.py").read_text(encoding="utf-8")
    assert "from gui.pages.train_page import TrainingThread" not in source
    assert "prepare_data_yaml" not in source


def test_train_page_has_no_shell_or_subprocess_command_construction():
    source = (APP_ROOT / "gui" / "pages" / "train_page.py").read_text(encoding="utf-8")
    for forbidden in ("subprocess.", "shell=True", "RemoteCommandBuilder(", "HostTrustStore("):
        assert forbidden not in source


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
