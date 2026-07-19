# -*- coding: utf-8 -*-
"""训练配置栏展开/收起的离屏回归测试。"""

import _bootstrap  # noqa: F401  必须先隔离 Qt、数据库和用户设置

import sys
from unittest.mock import patch

from PyQt6.QtCore import Qt  # noqa: E402

from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from gui.pages.train_page import (  # noqa: E402
    CONFIG_PANEL_COLLAPSED_WIDTH,
    CONFIG_PANEL_EXPANDED_WIDTH,
    TrainPage,
)
from test_remote_training_profiles import _ed25519_key  # noqa: E402


_app = _bootstrap.app()


class _RunningThread:
    def isRunning(self):
        return True


class _StoppedThread:
    def isRunning(self):
        return False


def _show(page):
    page.resize(960, 720)
    page.show()
    for _ in range(3):
        _app.processEvents()


def _running_config():
    return {
        "epochs": 12,
        "model_size": "n",
    }


def _profile():
    return RemoteTrainingProfile(
        name="实验室 A100",
        host="train-lab",
        port=22,
        username="trainer",
        remote_root="/srv/ezyolo/trainer",
        host_public_key=_ed25519_key(),
        id="5" * 32,
    )


def test_untrained_config_is_expanded_and_only_shows_start_action():
    page = TrainPage()
    _show(page)

    assert page.config_panel.width() == CONFIG_PANEL_EXPANDED_WIDTH
    assert page.config_scroll.isVisible()
    assert page.scroll_content.isEnabled()
    assert page.btn_collapse_config.isVisible()
    assert not page.btn_expand_config.isVisible()
    assert page.btn_start.isVisible()
    assert page.btn_start.minimumHeight() == 40
    assert not page.btn_stop.isVisible()
    assert not page.btn_goto_result.isVisible()
    assert not page.btn_train_again.isVisible()
    assert not page.runtime_summary_label.isVisible()
    assert not page.status_label.isVisible()

    page.close()


def test_training_auto_collapses_once_and_manual_expand_is_not_overwritten_by_epoch_updates():
    page = TrainPage()
    _show(page)
    page.training_thread = _RunningThread()

    page._begin_training_ui(_running_config(), remote=False)
    _app.processEvents()

    assert page.config_panel.width() == CONFIG_PANEL_COLLAPSED_WIDTH
    assert not page.config_scroll.isVisible()
    assert page.btn_expand_config.isVisible()
    assert not page.btn_collapse_config.isVisible()
    assert not page.scroll_content.isEnabled(), "训练中展开配置也只能查看"
    # 收起后运行摘要必须紧跟在展开箭头下方，不能因隐藏的 scroll stretch
    # 留下一大片空白；否则「收起」看上去像只是把内容清空了。
    run_top = page.run_panel.mapTo(page.config_panel, page.run_panel.rect().topLeft()).y()
    assert run_top < 80
    assert page.runtime_summary_label.isVisible()
    assert "模型：" in page.runtime_summary_label.text()
    assert "本地训练" in page.runtime_summary_label.text()
    assert page.status_label.isVisible()
    assert page.progress_bar.isVisible()
    assert page.btn_stop.isVisible()
    assert not page.btn_start.isVisible()

    page._set_config_collapsed(False)
    _app.processEvents()
    assert page.config_panel.width() == CONFIG_PANEL_EXPANDED_WIDTH
    assert page.config_scroll.isVisible()
    assert not page.scroll_content.isEnabled()

    page.on_epoch_started(1, 12)
    page.on_epoch_finished(1, {})
    _app.processEvents()
    assert page.config_panel.width() == CONFIG_PANEL_EXPANDED_WIDTH
    assert page.config_scroll.isVisible(), "后续训练信号不能覆盖用户手动展开"

    page.close()


def test_collapsed_runtime_summary_names_the_selected_remote_target():
    page = TrainPage()
    profile = _profile()
    page.remote_profile_store.save([profile])
    page.refresh_remote_targets()
    page.training_target.setCurrentIndex(
        page.training_target.findData(f"remote:{profile.id}")
    )
    page._set_config_collapsed(True)
    _show(page)

    assert "远程 · 实验室 A100" in page.runtime_summary_label.text()

    page.close()


def test_collapsed_blocker_wraps_in_the_rail_and_done_actions_use_full_width_slots():
    page = TrainPage()
    _show(page)

    page.refresh_readiness()
    page._set_config_collapsed(True)
    _app.processEvents()

    assert page.config_panel.width() == CONFIG_PANEL_COLLAPSED_WIDTH
    assert page.blocker is not None
    assert page.blocker["reason"] == page.status_label.text()
    assert page.status_label.isVisible()
    assert page.status_label.wordWrap()
    assert page.status_label.height() >= page.status_label.heightForWidth(page.status_label.width())
    assert page.config_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    page.show_done_panel()
    _app.processEvents()
    assert not page.btn_start.isVisible()
    assert not page.btn_stop.isVisible()
    assert page.btn_goto_result.isVisible()
    assert page.btn_train_again.isVisible()
    assert page.btn_goto_result.minimumHeight() == 40
    assert page.btn_train_again.minimumHeight() == 36
    assert page.btn_goto_result.width() == page.btn_train_again.width()

    page.btn_train_again.click()
    _app.processEvents()
    assert page.config_panel.width() == CONFIG_PANEL_EXPANDED_WIDTH
    assert page.config_scroll.isVisible()
    assert page.scroll_content.isEnabled()

    page.close()


def test_success_keeps_manual_choice_while_failure_resets_to_expanded():
    page = TrainPage()
    _show(page)
    page.training_thread = _RunningThread()
    page._begin_training_ui(_running_config(), remote=False)
    page._set_config_collapsed(False)
    page.training_thread = _StoppedThread()

    with patch("gui.pages.train_page.QMessageBox.information"):
        page.on_training_finished(True, "done")
    _app.processEvents()

    assert page.config_panel.width() == CONFIG_PANEL_EXPANDED_WIDTH
    assert page.btn_goto_result.isVisible()

    page.reset_ui_state()
    page.training_thread = _RunningThread()
    page._begin_training_ui(_running_config(), remote=False)
    page.training_thread = _StoppedThread()
    with patch("gui.pages.train_page.QMessageBox.warning"):
        page.on_training_finished(False, "failed")
    _app.processEvents()

    assert page.config_panel.width() == CONFIG_PANEL_EXPANDED_WIDTH
    assert page.config_scroll.isVisible()
    assert page.scroll_content.isEnabled()

    page.close()


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
