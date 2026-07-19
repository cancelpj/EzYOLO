# -*- coding: utf-8 -*-
"""训练页本地/远程目标选择的离屏回归测试。"""

import _bootstrap  # noqa: F401

import sys

from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from gui.pages.train_page import TrainPage  # noqa: E402
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
        id="4" * 32,
    )


def test_remote_target_disables_local_device_and_summary_names_server_policy():
    page = TrainPage()
    profile = _profile()
    page.remote_profile_store.save([profile])
    page.refresh_remote_targets()

    index = page.training_target.findData(f"remote:{profile.id}")
    assert index >= 0
    page.training_target.setCurrentIndex(index)
    _app.processEvents()

    assert not page.device.isEnabled()
    assert profile.remote_root in page.remote_target_hint.text()
    assert "服务器按自己的策略" in page.remote_target_hint.text()
    assert "远程 · 实验室 A100" in page.summary_label.text()
    assert "由服务器策略选择" in page.summary_label.text()


def test_deleted_remote_target_explicitly_returns_to_local_without_network_io():
    page = TrainPage()
    profile = _profile()
    page.remote_profile_store.save([profile])
    page.refresh_remote_targets()
    page.training_target.setCurrentIndex(
        page.training_target.findData(f"remote:{profile.id}")
    )
    _app.processEvents()

    page.remote_profile_store.save([])
    page.refresh_remote_targets()

    assert page.training_target.currentData() == "local"
    assert page.device.isEnabled()
    assert "已不存在" in page.remote_target_hint.text()


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
