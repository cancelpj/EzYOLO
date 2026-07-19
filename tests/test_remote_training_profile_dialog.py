# -*- coding: utf-8 -*-
"""远程服务器档案编辑对话框的离屏测试。"""

import _bootstrap  # noqa: F401

import sys

from PyQt6.QtWidgets import QLineEdit, QPlainTextEdit  # noqa: E402

from gui.pages.remote_training_profile_dialog import (  # noqa: E402
    RemoteTrainingProfileDialog,
)
from test_remote_training_profiles import _ed25519_key  # noqa: E402


_app = _bootstrap.app()


def test_dialog_collects_only_safe_profile_fields_and_preserves_id_on_edit():
    dialog = RemoteTrainingProfileDialog()
    dialog.name_input.setText("实验室 A100")
    dialog.host_input.setText("train-lab")
    dialog.username_input.setText("trainer")
    dialog.remote_root_input.setText("/srv/ezyolo/trainer")
    dialog.host_key_input.setPlainText(_ed25519_key())
    dialog._save()
    saved = dialog.saved_profile
    assert saved is not None
    assert saved.username == "trainer"
    assert saved.remote_root == "/srv/ezyolo/trainer"

    edit = RemoteTrainingProfileDialog(saved)
    edit.name_input.setText("实验室 A100（共享）")
    edit._save()
    assert edit.saved_profile is not None
    assert edit.saved_profile.id == saved.id


def test_dialog_has_no_auth_secret_input_surface():
    dialog = RemoteTrainingProfileDialog()
    object_names = {
        widget.objectName()
        for widget in dialog.findChildren((QLineEdit, QPlainTextEdit))
        if widget.objectName().startswith("remote_profile_")
    }
    assert object_names == {
        "remote_profile_name_input",
        "remote_profile_host_input",
        "remote_profile_username_input",
        "remote_profile_root_input",
        "remote_profile_host_key_input",
    }


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
