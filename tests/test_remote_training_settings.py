# -*- coding: utf-8 -*-
"""设置页远程训练档案入口的离屏测试。"""

import _bootstrap  # noqa: F401

import sys

from PyQt6.QtWidgets import QLabel  # noqa: E402

from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from gui.pages.settings_page import SettingsPage  # noqa: E402
from test_remote_training_profiles import _ed25519_key  # noqa: E402


_app = _bootstrap.app()


def _profile():
    return RemoteTrainingProfile(
        name="实验室 A100",
        host="train-lab",
        port=22,
        username="trainer",
        remote_root="/srv/ezyolo/trainer",
        host_public_key=_ed25519_key(),
        id="6" * 32,
    )


def test_remote_profiles_are_visible_and_reset_does_not_delete_them():
    page = SettingsPage()
    profile = _profile()
    page.remote_profile_store.save([profile])
    page.refresh_remote_profiles()

    assert page.selected_remote_profile() == profile
    assert page.remote_profile_combo.currentData() == profile.id
    assert page.btn_test_remote_profile.isEnabled()
    assert any("不保存密码" in label.text() for label in page.findChildren(QLabel))

    page.reset_settings()
    assert page.remote_profile_store.list() == [profile]


def test_profile_test_request_only_emits_selected_profile_without_network_io():
    page = SettingsPage()
    profile = _profile()
    page.remote_profile_store.save([profile])
    page.refresh_remote_profiles()
    requested = []
    page.remote_profile_test_requested.connect(requested.append)

    page.request_remote_profile_test()

    assert requested == [profile]
    assert not page.btn_test_remote_profile.isEnabled()
    assert "只读连接预检" in page.remote_profile_status.text()


def test_profile_switch_does_not_reenable_connection_test_while_preflight_is_running():
    page = SettingsPage()
    first = _profile()
    second = RemoteTrainingProfile(
        name="备用服务器",
        host="train-lab-2",
        port=2202,
        username="trainer",
        remote_root="/srv/ezyolo/backup",
        host_public_key=_ed25519_key(),
        id="7" * 32,
    )
    page.remote_profile_store.save([first, second])
    page.refresh_remote_profiles(first.id)

    page.request_remote_profile_test()
    page.remote_profile_combo.setCurrentIndex(1)

    assert not page.btn_test_remote_profile.isEnabled()
    page.set_remote_profile_test_status("预检完成", success=True)
    assert page.btn_test_remote_profile.isEnabled()


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
