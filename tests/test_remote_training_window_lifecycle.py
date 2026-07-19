# -*- coding: utf-8 -*-
"""远程训练 QThread 的窗口关闭保护测试。"""

import _bootstrap  # noqa: F401  必须先隔离 Qt、数据库和用户设置

from gui.main_window import MainWindow  # noqa: E402
from gui.pages.train_page import TrainPage  # noqa: E402


_app = _bootstrap.app()


class _CloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class _ActiveRemoteThread:
    def __init__(self):
        self.cancel_requests = 0

    def isRunning(self):
        return True

    def request_cancel(self):
        self.cancel_requests += 1


class _ActiveLocalThread:
    def __init__(self):
        self.stop_requests = 0

    def isRunning(self):
        return True

    def stop(self):
        self.stop_requests += 1


class _ActivePreflightThread:
    def isRunning(self):
        return True


def test_train_page_close_requests_remote_cancel_without_waiting_for_thread():
    page = TrainPage()
    thread = _ActiveRemoteThread()
    page.training_thread = thread
    page._active_training_is_remote = True

    assert page.request_close() is False
    assert thread.cancel_requests == 1
    assert page.stop_requested is True
    assert "暂时不能关闭" in page.status_label.text()


def test_train_page_close_requests_local_stop_without_waiting_for_thread():
    page = TrainPage()
    thread = _ActiveLocalThread()
    page.training_thread = thread
    page._active_training_is_remote = False

    assert page.request_close() is False
    assert thread.stop_requests == 1
    assert page.stop_requested is True
    assert "暂时不能关闭" in page.status_label.text()


def test_main_window_refuses_close_while_training_is_still_safe_stopping():
    window = MainWindow()
    window.train_page.request_close = lambda: False
    event = _CloseEvent()

    window.closeEvent(event)

    assert event.ignored is True
    assert event.accepted is False
    assert "训练仍在安全收尾" in window.notice.text.text()


def test_main_window_refuses_close_while_read_only_preflight_thread_is_running():
    window = MainWindow()
    window.train_page.request_close = lambda: True
    window._remote_profile_test_thread = _ActivePreflightThread()
    event = _CloseEvent()

    window.closeEvent(event)

    assert event.ignored is True
    assert event.accepted is False
    assert "连接预检仍在进行" in window.notice.text.text()


def test_main_window_closes_when_no_training_or_preflight_thread_is_running():
    window = MainWindow()
    window.train_page.request_close = lambda: True
    event = _CloseEvent()

    window.closeEvent(event)

    assert event.accepted is True
    assert event.ignored is False


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
