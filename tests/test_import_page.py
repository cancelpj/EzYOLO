# -*- coding: utf-8 -*-
"""导入页交互连续性测试。

覆盖的真实问题：
    1. 点了导入之后，一个事件循环内必须看到明确状态（不是干等一个空进度条）。
    2. 导入任务状态和缩略图加载状态是两套东西，互相不能把对方隐藏掉——
       切到别的页面再回来（refresh_project_images）、缩略图加载完成，
       都不该让还在跑的导入状态条消失。
    3. 图片 / 文件夹导入的实际工作必须在后台线程，不能卡住调用者（GUI 主线程）。
    4. 任务结束（成功/取消/失败）后必须恢复被禁用的按钮。
    5. 取消按钮要真的能喊停后台线程。

运行：
    python -m pytest tests/test_import_page.py -q
    或
    python tests/test_import_page.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys
import threading
import time
import tempfile
from pathlib import Path
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QImage, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QFileDialog, QLabel, QStyleFactory

import models.database as database_module
from gui.pages.import_page import ImportPage, short_task_label
from gui.main_window import MainWindow
from gui.workflow import PAGE_SETTINGS, STEP_IMPORT
from core.import_manager import ImportManager

_app = _bootstrap.app()
db = _bootstrap.db

_TMP_DIR = Path(tempfile.mkdtemp(prefix="ezyolo-import-page-"))


def _silent_dialogs():
    """吞掉应用内弹窗（确认框 / 提示框），测试不该被模态框卡住。

    页面现在用的是 gui.widgets.app_dialog 里那几个函数，不再是 QMessageBox，
    所以要挡的是这几个名字。确认框一律当成「用户点了确认」。
    """
    stack = ExitStack()
    stack.enter_context(patch("gui.pages.import_page.show_info"))
    stack.enter_context(patch("gui.pages.import_page.show_warning"))
    stack.enter_context(patch("gui.pages.import_page.confirm", return_value=True))
    stack.enter_context(patch("gui.pages.import_page.confirm_destructive", return_value=True))
    return stack


def _video_plan(frame_interval=5, mode="interval", sample_count=None):
    """替掉抽帧设置框：直接返回一个抽帧方案，不弹窗。"""
    return patch(
        "gui.pages.import_page.ask_video_extract_plan",
        return_value={
            'mode': mode,
            'frame_interval': frame_interval,
            'sample_count': sample_count,
        },
    )


def _make_project() -> int:
    return _bootstrap.create_temp_project(name="导入页测试项目", project_type="detect", classes=[])


def _make_annotated_project() -> int:
    """一个已经有标注的项目：导入标注时才会问「要不要覆盖」。"""
    project_id = _make_project()
    image_path = _make_images(_TMP_DIR / f"annotated_{project_id}", 1)[0]
    image_id = db.add_image(
        project_id=project_id, filename=Path(image_path).name,
        storage_path=image_path, width=32, height=32,
    )
    db.add_annotation(
        image_id=image_id, project_id=project_id, class_id=0, class_name="人",
        annotation_type="rectangle", data={'x': 1, 'y': 1, 'width': 5, 'height': 5},
    )
    return project_id


def _make_images(folder: Path, count: int) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = folder / f"img_{i:03d}.jpg"
        image = QImage(32, 32, QImage.Format.Format_RGB888)
        image.fill(QColor(10 * i % 255, 20, 30))
        image.save(str(path))
        paths.append(str(path))
    return paths


def _make_video(path: Path, frame_count: int = 30, size=(64, 48)) -> Path:
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(frame_count):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:] = (i * 5 % 255, 0, 0)
        writer.write(frame)
    writer.release()
    return path


def _pump_until(predicate, timeout=10.0):
    """跑事件循环直到条件成立或超时，返回是否成立。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app.processEvents()
        if predicate():
            return True
    return False


def _wait_import_done(page):
    ok = _pump_until(lambda: not page._import_busy)
    assert ok, "导入任务没有在超时时间内收尾"


def test_video_import_shows_busy_state_within_same_call():
    """确认导入后，一个事件循环内（同一次调用里）就要看到明确状态。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    video_path = _make_video(_TMP_DIR / "busy.mp4", frame_count=40)

    with _silent_dialogs(), \
         _video_plan(5):
        page.process_video_import(str(video_path), group_id=None)

        # 不经过任何 processEvents，直接检查：必须已经是「导入中」状态
        assert page._import_busy is True
        assert page.import_status_frame.isHidden() is False
        assert page.import_progress_bar.maximum() == 0, "总帧数还不知道时应该是忙碌态（indeterminate）"
        assert "打开视频" in page.import_status_label.text()
        assert page.btn_cancel_import.isHidden() is False

        # 会再起一个导入的、和会毁掉当前项目的，导入中都不能点。
        # 后两个现在住在「管理」菜单里，禁的是菜单项本身
        for control in (
            page.btn_import_folder, page.btn_import_images,
            page.btn_import_video, page.btn_import_annotations,
            page.action_delete_project, page.action_clear,
        ):
            assert not control.isEnabled(), f"{control.text()} 导入中应该被禁用"

        _wait_import_done(page)

    assert len(page.images) > 0
    for control in (page.btn_import_folder, page.btn_import_video, page.action_delete_project):
        assert control.isEnabled(), "导入结束后应该恢复可用"


def test_images_import_shows_determinate_progress_immediately():
    """已知总量（选好的图片列表）应该立刻是确定进度，不是转圈忙碌态。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    file_paths = _make_images(_TMP_DIR / "determinate", 5)

    with _silent_dialogs():
        page.process_image_import(file_paths, group_id=None)

        assert page._import_busy is True
        assert page.import_progress_bar.maximum() == 100, "已知总量应该切到确定进度"
        assert "0/5" in page.import_status_label.text()

        _wait_import_done(page)

    assert len(page.images) == 5


def test_page_switch_refresh_does_not_hide_active_import():
    """切到别的页面再回来时会调 refresh_project_images，不该把导入状态藏起来。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    page._start_import_ui("正在准备导入: demo.mp4")
    assert page.import_status_frame.isHidden() is False

    # 主窗口从别的页面切回导入页时调用的正是这个方法
    page.refresh_project_images()

    assert page.import_status_frame.isHidden() is False, "切页刷新不该隐藏正在进行的导入状态"
    assert page._import_busy is True

    # 清理，避免影响后面的测试
    page._end_import_ui()


def test_real_main_window_settings_round_trip_keeps_import_status():
    """复现用户路径：导入中点设置，再点回第 1 步，状态和文字仍在。"""
    project_id = _make_project()
    window = MainWindow()
    window.load_projects(select_id=project_id)
    window.show()
    _app.processEvents()
    page = window.import_page

    page._start_import_ui("正在打开视频: demo.mp4")

    window.switch_page(PAGE_SETTINGS)
    window.switch_page(STEP_IMPORT)
    _app.processEvents()

    assert page._import_busy is True
    assert page.import_status_frame.isVisible(), "从设置返回后导入状态条消失了"
    assert "demo.mp4" in page.import_status_label.text(), "返回后丢失了当前任务说明"

    page._end_import_ui()
    window.close()
    db.delete_project(project_id)


def test_thumbnail_load_finish_does_not_hide_active_import():
    """缩略图加载完成（on_load_finished）不该把导入状态条带下去。"""
    project_id = _make_project()
    image_paths = _make_images(_TMP_DIR / "thumb_independent", 3)
    for path in image_paths:
        db.add_image(project_id, Path(path).name, path, width=32, height=32)

    page = ImportPage()
    page.set_project(project_id)
    _pump_until(lambda: page.load_worker is None)  # 等 set_project 触发的初始缩略图加载先跑完

    page._start_import_ui("正在导入其它内容…")
    assert page.import_status_frame.isHidden() is False

    # 触发一次独立的缩略图刷新（不是通过 load_project_images，模拟纯缩略图加载场景）
    page.force_refresh_images()
    ok = _pump_until(lambda: page.load_worker is None)
    assert ok, "缩略图加载线程没有在超时时间内结束"

    assert page.import_status_frame.isHidden() is False, "缩略图加载完成不该隐藏导入状态"
    assert page._import_busy is True

    page._end_import_ui()


def test_cancel_button_stops_active_thread():
    """点取消按钮要真的调用到后台线程的 cancel()。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    calls = []

    class _FakeThread:
        def cancel(self):
            calls.append("cancelled")

    page._active_import_thread = _FakeThread()
    page._start_import_ui("正在导入…")

    page._cancel_active_import()

    assert calls == ["cancelled"]
    assert not page.btn_cancel_import.isEnabled()
    assert "取消" in page.import_status_label.text()

    page._end_import_ui()


def test_folder_and_images_import_do_not_block_caller_thread():
    """process_folder_import / process_image_import 必须立刻返回，实际工作在后台线程。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    folder = _TMP_DIR / "slow_folder"
    _make_images(folder, 3)

    real_import_folder = ImportManager.import_folder

    def slow_import_folder(self, *args, **kwargs):
        time.sleep(0.5)
        return real_import_folder(self, *args, **kwargs)

    with _silent_dialogs(), \
         patch.object(ImportManager, "import_folder", slow_import_folder):
        start = time.monotonic()
        page.process_folder_import(str(folder), group_id=None)
        elapsed = time.monotonic() - start

        assert elapsed < 0.2, f"process_folder_import 阻塞了调用者线程: {elapsed:.3f}s"

        _wait_import_done(page)

    assert len(page.images) == 3


def test_unknown_total_frames_video_import_completes_without_crash():
    """总帧数未知的视频也要能正常导入完，不因为除零而崩掉（GUI 层串联真实场景）。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    video_path = _make_video(_TMP_DIR / "unknown_gui.mp4", frame_count=20)

    import cv2
    real_video_capture = cv2.VideoCapture

    class _FakeCap:
        def __init__(self, path):
            self._real = real_video_capture(path)

        def isOpened(self):
            return self._real.isOpened()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 0
            return self._real.get(prop)

        def read(self):
            return self._real.read()

        def release(self):
            self._real.release()

    with _silent_dialogs(), \
         _video_plan(5), \
         patch("core.import_manager.cv2.VideoCapture", lambda p: _FakeCap(p)):
        page.process_video_import(str(video_path), group_id=None)
        assert page.import_progress_bar.maximum() == 0
        _wait_import_done(page)

    assert len(page.images) > 0


def test_project_switch_keeps_old_thread_alive_until_it_actually_finishes():
    """切换项目时取消旧导入：QThread 对象必须活到它真正退出为止，
    不能在还在运行时就丢掉引用（否则触发 "QThread: Destroyed while thread
    is still running"）。旧线程收尾后要真正释放，且不能污染新项目的数据。
    """
    project_a = _make_project()
    project_b = _make_project()
    page = ImportPage()
    page.set_project(project_a)

    folder = _TMP_DIR / "switch_slow_worker"
    _make_images(folder, 5)

    real_import_folder = ImportManager.import_folder

    def slow_import_folder(self, *args, **kwargs):
        # 真实导入很快就跑完，额外多睡一会儿，确保切项目那一刻线程仍在运行
        result = real_import_folder(self, *args, **kwargs)
        time.sleep(0.3)
        return result

    with _silent_dialogs(), \
         patch.object(ImportManager, "import_folder", slow_import_folder):
        page.process_folder_import(str(folder), group_id=None)
        old_thread = page._active_import_thread
        assert old_thread is not None

        page.set_project(project_b)

        # 切换后：不再是「当前活跃」导入，但对象必须还活着、还在跑，
        # 不能被提前 GC / deleteLater
        assert page._active_import_thread is None
        assert old_thread in page._retired_import_threads, "旧线程应该被继续持有，直到它真正退出"
        assert old_thread.isRunning() is True, "此刻旧线程应该还没跑完（被 mock 多睡了 0.3s）"
        assert page._import_busy is False

        ok = _pump_until(lambda: old_thread not in page._retired_import_threads, timeout=5.0)
        assert ok, "旧线程没有在超时时间内被正确收尾"

    assert old_thread.isRunning() is False, "旧线程收尾后必须真的已经退出"
    assert page.current_project_id == project_b
    assert page.images == [], "旧项目的导入结果不能污染新项目"


def test_normal_cancel_and_failure_paths_all_release_thread_reference():
    """正常完成 / 失败 / 取消三条路径，导入收尾后都不能残留线程引用（不泄漏）。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    # 路径一：正常完成
    file_paths = _make_images(_TMP_DIR / "release_normal", 3)
    with _silent_dialogs():
        page.process_image_import(file_paths, group_id=None)
        _wait_import_done(page)
        ok = _pump_until(lambda: page._active_import_thread is None and not page._retired_import_threads)
        assert ok, "正常完成后线程引用没有被释放"
    assert page._active_import_thread is None
    assert page._retired_import_threads == []

    # 路径二：失败（后台抛异常）
    with _silent_dialogs(), \
         patch.object(ImportManager, "import_images", side_effect=RuntimeError("boom")):
        page.process_image_import(file_paths, group_id=None)
        _wait_import_done(page)
        ok = _pump_until(lambda: page._active_import_thread is None and not page._retired_import_threads)
        assert ok, "失败后线程引用没有被释放"
    assert page._active_import_thread is None
    assert page._retired_import_threads == []

    # 路径三：取消
    video_path = _make_video(_TMP_DIR / "release_cancel.mp4", frame_count=40)
    with _silent_dialogs(), \
         _video_plan(1):
        page.process_video_import(str(video_path), group_id=None)
        page._cancel_active_import()
        _wait_import_done(page)
        ok = _pump_until(lambda: page._active_import_thread is None and not page._retired_import_threads)
        assert ok, "取消后线程引用没有被释放"
    assert page._active_import_thread is None
    assert page._retired_import_threads == []


def test_cancelled_import_progress_bar_is_not_shown_as_full_completion():
    """取消导入后，页面进度条和 summary 不应该被显示成「100% 已完成」。"""
    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    video_path = _make_video(_TMP_DIR / "cancel_ui_progress.mp4", frame_count=40)

    import cv2
    real_video_capture = cv2.VideoCapture

    class _SlowCap:
        """跟真实 VideoCapture 行为一致，只是每帧多睡一点，留出取消的窗口。"""

        def __init__(self, path):
            self._real = real_video_capture(path)

        def isOpened(self):
            return self._real.isOpened()

        def get(self, prop):
            return self._real.get(prop)

        def read(self):
            time.sleep(0.03)
            return self._real.read()

        def release(self):
            self._real.release()

    with _silent_dialogs(), \
         _video_plan(1), \
         patch("core.import_manager.cv2.VideoCapture", lambda p: _SlowCap(p)):
        page.process_video_import(str(video_path), group_id=None)

        ok = _pump_until(lambda: page.import_progress_bar.value() > 0, timeout=5.0)
        assert ok, "没能观察到真实进度，取消窗口没抓住"

        page._cancel_active_import()
        _wait_import_done(page)

    assert "已取消" in page.import_status_label.text()
    assert page.import_progress_bar.value() < 100, "取消后不应该把进度条显示成 100% 完成"
    assert len(page.images) < 40, "取消应该在导完全部帧之前生效"


def test_new_project_uses_the_in_app_text_dialog_not_the_system_one():
    """新建项目不能再弹系统的 QInputDialog：那个框跟原来那个丑抽帧框是同一个模子。"""
    import gui.pages.import_page as import_page_module

    assert not hasattr(import_page_module, "QInputDialog"), \
        "导入页不该再依赖系统输入框"

    page = ImportPage()
    created = []
    page.projects_changed.connect(created.append)

    asked = {}

    def fake_ask_text(_parent, title, _label, **kwargs):
        asked['title'] = title
        asked['confirm_text'] = kwargs.get('confirm_text')
        return "新的安全帽项目"

    with patch("gui.pages.import_page.ask_text", fake_ask_text), \
         patch("gui.pages.import_page.ask_task_type", return_value="detect"):
        page.create_new_project()

    assert asked['title'] == "新建项目"
    assert asked['confirm_text'] == "创建项目", "确认按钮要说清楚它会干什么"

    assert len(created) == 1, "应该建出一个项目并通知主窗口"
    project = db.get_project(created[0])
    assert project['name'] == "新的安全帽项目"

    db.delete_project(created[0])


def test_new_project_is_not_created_when_the_name_dialog_is_cancelled():
    """取消起名字（ask_text 返回 None）就什么都不建。"""
    page = ImportPage()
    created = []
    page.projects_changed.connect(created.append)

    with patch("gui.pages.import_page.ask_text", return_value=None), \
         patch("gui.pages.import_page.ask_task_type", return_value="detect") as task_type:
        page.create_new_project()

    assert created == []
    assert not task_type.called, "名字都没起，不该继续问任务类型"


def test_annotation_import_confirmations_use_honest_button_labels():
    """导入标注这一路上的两个二选一，按钮要照实说，不能都叫「取消」。

    「不覆盖」实际是「保留现有标注」并继续导入，写「取消」会让人以为
    整个导入都放弃了；「不另选图像文件夹」是「不用」，也不是取消。
    """
    project_id = _make_annotated_project()
    page = ImportPage()
    page.set_project(project_id)

    labels_dir = _TMP_DIR / "yolo_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    calls = []
    finished = []

    def record(_parent, title, _message, **kwargs):
        calls.append((title, kwargs.get('confirm_text'), kwargs.get('cancel_text')))
        return False  # 两个都选「安全的那一个」

    # 等的是「标注导入真的收尾了」（收尾时会报一句结果），不是「loading_overlay 没了」：
    # 那个遮罩也会被缩略图加载线程的 on_load_finished 顺手删掉，它先到的话，这里就会
    # 在标注导入线程还没收尾时退出 patch，收尾时的那句提示就成了一个真模态框，
    # 挂在后面某个测试的事件泵里。
    with _silent_dialogs(), \
         patch("gui.pages.import_page.show_info", lambda *a, **k: finished.append(a)), \
         patch("gui.pages.import_page.confirm", record), \
         patch("gui.pages.import_page.confirm_destructive", record), \
         patch("gui.pages.import_page.ask_import_group", return_value=(True, None)), \
         patch.object(QFileDialog, "getExistingDirectory", return_value=str(labels_dir)):
        page.import_yolo_annotations(group_id=None)
        ok = _pump_until(lambda: bool(finished), timeout=10.0)
        assert ok, "标注导入没有在超时时间内收尾"
        page.import_thread.wait()

    titles = {title: (confirm_text, cancel_text) for title, confirm_text, cancel_text in calls}

    assert "图像文件夹" in titles, calls
    assert titles["图像文件夹"] == ("去选择", "不用"), titles["图像文件夹"]

    assert "覆盖已有标注" in titles, calls
    assert titles["覆盖已有标注"] == ("覆盖", "保留现有标注"), titles["覆盖已有标注"]

    for _title, _confirm_text, cancel_text in calls:
        assert cancel_text != "取消", f"「{_title}」的安全按钮不该笼统叫「取消」"

    db.delete_project(project_id)


# ==================== 工具栏胶囊 / 管理菜单的状态 ====================

def _seed_images(project_id, count=3):
    for i in range(count):
        db.add_image(project_id, f"seed_{i}.jpg", f"/tmp/seed_{i}.jpg", width=32, height=32)


def test_task_chip_shows_the_short_chinese_label_only():
    """工具栏胶囊只写中文：「任务：目标检测」，不再拖一个 detect 的尾巴。

    共用的任务类型对话框仍然显示带英文的完整标签——那里 detect / segment
    要和 YOLO 的术语对得上，是有用的；在胶囊上它只是把宽度撑长。
    """
    from gui.widgets.task_type_dialog import task_type_label

    assert short_task_label('detect') == "目标检测"
    assert short_task_label('segment') == "实例分割"
    assert short_task_label(None) == "未设置"

    # 对话框那份标签没被改动
    assert task_type_label('detect') == "目标检测 detect"

    project_id = _make_project()
    page = ImportPage()
    page.set_project(project_id)

    assert page.btn_task_type.text() == "任务：目标检测", page.btn_task_type.text()

    db.delete_project(project_id)


def test_selection_actions_are_disabled_until_something_is_selected():
    """没选图片时「移动分组 / 删除选中」就该是灰的。

    以前它们一直亮着，点下去只弹一句「还没选图片」——用一个弹窗代替了
    本来一眼就该看出来的状态。
    """
    project_id = _make_project()
    _seed_images(project_id, 3)

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    assert not page.action_move_group.isEnabled(), "没选图片，移动分组不该能点"
    assert not page.action_delete_selected.isEnabled(), "没选图片，删除选中不该能点"

    page.image_list.item(0).setSelected(True)
    _app.processEvents()

    assert page.action_move_group.isEnabled(), "选了图片就该能移动分组"
    assert page.action_delete_selected.isEnabled(), "选了图片就该能删除选中"

    page.image_list.clearSelection()
    _app.processEvents()

    assert not page.action_move_group.isEnabled(), "取消选中之后要变回灰的"
    assert not page.action_delete_selected.isEnabled()

    page.stop_image_loading()
    db.delete_project(project_id)


def test_menu_recomputes_state_when_it_opens():
    """菜单弹出前重算一次：选中状态可能是在菜单关着的时候变的。"""
    project_id = _make_project()
    _seed_images(project_id, 2)

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    # 绕过信号，直接把选中状态做出来——模拟「菜单不知道的变化」
    page.action_move_group.setEnabled(False)
    page.image_list.item(0).setSelected(True)
    page.action_move_group.setEnabled(False)

    page.manage_menu.aboutToShow.emit()

    assert page.action_move_group.isEnabled(), "菜单打开时应该重算可用性"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_clear_all_is_disabled_when_there_is_nothing_to_clear():
    """项目里没有图片时「清空全部图片」是灰的；有图片才亮。"""
    project_id = _make_project()

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    assert not page.action_clear.isEnabled(), "没有图片就没有东西可清空"
    # 项目本身还是删得掉的
    assert page.action_delete_project.isEnabled()

    _seed_images(project_id, 2)
    page.load_project_images()
    _app.processEvents()

    assert page.action_clear.isEnabled(), "有图片了就该能清空"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_import_busy_blocks_destruction_but_still_allows_moving_selected_images():
    """导入中：清空 / 删除项目一律关掉；已选中图片的移动、删除照常可用。

    正在往项目里写图片的时候不能把项目端了；但对已有图片的操作没有理由禁掉。
    """
    project_id = _make_project()
    _seed_images(project_id, 3)

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    page.image_list.item(0).setSelected(True)
    _app.processEvents()

    page._start_import_ui("正在导入…")

    assert not page.action_clear.isEnabled(), "导入中不能清空图片"
    assert not page.action_delete_project.isEnabled(), "导入中不能删除项目"
    assert page.action_move_group.isEnabled(), "导入中仍然可以移动已选中的图片"
    assert page.action_delete_selected.isEnabled(), "导入中仍然可以删除已选中的图片"

    page._end_import_ui()
    _app.processEvents()

    assert page.action_clear.isEnabled(), "导入结束后要恢复"
    assert page.action_delete_project.isEnabled()

    page.stop_image_loading()
    db.delete_project(project_id)


# ==================== 缩略图上的标注框预览 ====================
# 用户要的是：不进标注页，光看导入页就知道哪张图标了什么位置。
# 下面盯四件事——框画出来了、开关立刻生效且不动数据、600 张不做 N+1、
# 标注改了缩略图要跟着变（而且只变改过的那几张）。

_RED = '#FF0000'
_GREEN = '#00FF00'

_BOX_CLASSES = [
    {'id': 0, 'name': '人', 'color': _RED},
    {'id': 1, 'name': '车', 'color': _GREEN},
]


def _make_box_project() -> int:
    return _bootstrap.create_temp_project(
        name="标注框预览项目", project_type="detect", classes=_BOX_CLASSES,
    )


def _seed_image_with_boxes(project_id, folder: Path, name: str,
                           size=(40, 20), boxes=()) -> tuple:
    """造一张真图片 + 若干 bbox，返回 (image_id, storage_path, [标注 id])。

    尺寸默认用 40x20 的横图：非等比缩放到 160x160 时 x/y 比例不同，
    框要是算错了，这种图最先露馅。
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    image = QImage(size[0], size[1], QImage.Format.Format_RGB888)
    image.fill(QColor(250, 250, 250))
    image.save(str(path))

    image_id = db.add_image(
        project_id, name, str(path), width=size[0], height=size[1],
    )
    annotation_ids = [
        db.add_annotation(
            image_id=image_id, project_id=project_id, class_id=class_id,
            class_name=f"类{class_id}", annotation_type='bbox', data=box,
        )
        for class_id, box in boxes
    ]
    return image_id, str(path), annotation_ids


def _settle_thumbnails(page: ImportPage, timeout=30.0):
    """等到缩略图这条线彻底安静下来：读盘线程结束、框查回来、分块重画画完。

    框的查询在后台线程里、叠框又是分块排队做的，所以「页面刷完了」不再是同一次
    调用里的事——测试要断言最终画面，就必须等这三件事都落地。
    """
    ok = _pump_until(
        lambda: (page.load_worker is None
                 and not page._previews_pending
                 and not page._icon_refresh_pending),
        timeout=timeout,
    )
    assert ok, "缩略图（读盘 / 查框 / 分块重画）没有在超时时间内全部结束"


def _loaded_page(project_id) -> ImportPage:
    """建页面并等缩略图后台线程真的跑完。"""
    page = ImportPage()
    page.set_project(project_id)
    _settle_thumbnails(page)
    return page


def _icon_image(page: ImportPage, row: int = 0,
                mode: QIcon.Mode = QIcon.Mode.Normal) -> QImage:
    icon = page.image_list.item(row).icon()
    return icon.pixmap(QSize(160, 160), mode, QIcon.State.Off).toImage()


def _base_image(page: ImportPage, storage_path: str) -> QImage:
    """缓存里那张「没有框」的底图。"""
    return page.thumbnail_cache[storage_path].toImage()


def _rows_by_image_id(page: ImportPage) -> dict:
    return {
        page.image_list.item(i).data(Qt.ItemDataRole.UserRole): i
        for i in range(page.image_list.count())
    }


def _color_count(image: QImage, color: str) -> int:
    target = QColor(color).rgb()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) == target
    )


@contextmanager
def _traced_sql(statements: list):
    """把这段时间里真正执行过的 SQL 全部记下来，用来证明「没有 N+1」。"""
    real_get_connection = database_module.Database.get_connection

    @contextmanager
    def traced(self):
        with real_get_connection(self) as conn:
            conn.set_trace_callback(statements.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    with patch.object(database_module.Database, "get_connection", traced):
        yield statements


def test_thumbnails_draw_existing_boxes_by_default():
    """默认就画框：缩略图跟底图不一样，而且用的是项目类别的颜色。"""
    project_id = _make_box_project()
    _image_id, path, _ = _seed_image_with_boxes(
        project_id, _TMP_DIR / "boxes_default", "one_box.jpg",
        size=(40, 20), boxes=[(0, {'x': 10, 'y': 5, 'width': 20, 'height': 10})],
    )

    page = _loaded_page(project_id)

    assert page.chk_show_boxes.isChecked(), "「显示标注框」默认就该是开着的"

    base = _base_image(page, path)
    shown = _icon_image(page)

    assert shown != base, "缩略图上没有画出标注框"
    assert _color_count(base, _RED) == 0, "底图本身不该带框——那是缓存要复用的原图"
    assert _color_count(shown, _RED) > 0, "框没有用项目类别配的红色"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_multiple_classes_keep_their_own_colours_on_one_thumbnail():
    """一张图里两个类别：两种颜色，同类恒定同色。"""
    project_id = _make_box_project()
    _seed_image_with_boxes(
        project_id, _TMP_DIR / "boxes_multi", "two_boxes.jpg",
        size=(40, 20),
        boxes=[
            (0, {'x': 2, 'y': 2, 'width': 14, 'height': 8}),
            (1, {'x': 22, 'y': 8, 'width': 14, 'height': 10}),
        ],
    )

    page = _loaded_page(project_id)
    shown = _icon_image(page)

    assert _color_count(shown, _RED) > 0, "类别 0 的红框没画出来"
    assert _color_count(shown, _GREEN) > 0, "类别 1 的绿框没画出来"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_unannotated_image_gets_no_empty_box():
    """没标注的图不画空框——缩略图就该是原图本身。"""
    project_id = _make_box_project()
    paths = _make_images(_TMP_DIR / "boxes_none", 1)
    db.add_image(project_id, Path(paths[0]).name, paths[0], width=32, height=32)

    page = _loaded_page(project_id)

    assert _icon_image(page) == _base_image(page, paths[0]), "没标注的图被画上了东西"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_toggle_switches_thumbnails_without_touching_annotations():
    """切开关只换图：不写数据库，也不再去读一次磁盘。

    换图本身是分块排队做的（见 test_toggling_boxes_on_600_thumbnails_never_blocks_the_ui），
    所以这里等它把队排完再看画面；但「不写库、不读盘」这两条一个字都不松。
    """
    project_id = _make_box_project()
    image_id, path, _ = _seed_image_with_boxes(
        project_id, _TMP_DIR / "boxes_toggle", "toggle.jpg",
        size=(40, 20), boxes=[(0, {'x': 10, 'y': 5, 'width': 20, 'height': 10})],
    )

    page = _loaded_page(project_id)

    base = _base_image(page, path)
    with_boxes = _icon_image(page)
    assert with_boxes != base

    annotations_before = db.get_image_annotations(image_id)
    version_before = db.get_project_annotation_versions(project_id)

    with patch("gui.pages.import_page.cv2.imread") as imread:
        page.chk_show_boxes.setChecked(False)
        assert page.show_annotation_boxes is False
        _settle_thumbnails(page)

        assert _icon_image(page) == base, "关掉开关后缩略图上还有框"

        # 再开回来
        page.chk_show_boxes.setChecked(True)
        _settle_thumbnails(page)

        assert _icon_image(page) == with_boxes, "开回来之后框没有回来"

        assert imread.call_count == 0, "切开关不该重新读磁盘——底图缓存里就有"

    assert page.load_worker is None, "切开关不该起后台加载线程"

    # 标注一个字节都不许动
    assert db.get_image_annotations(image_id) == annotations_before, "开关改动了标注数据"
    assert db.get_project_annotation_versions(project_id) == version_before

    page.stop_image_loading()
    db.delete_project(project_id)


def test_box_previews_are_read_in_one_batched_query():
    """600 张图不能查 600 次库：整批一次查完。"""
    project_id = _make_box_project()
    folder = _TMP_DIR / "boxes_batch"
    for i in range(5):
        _seed_image_with_boxes(
            project_id, folder, f"batch_{i}.jpg",
            size=(40, 20), boxes=[(0, {'x': 4, 'y': 4, 'width': 10, 'height': 8})],
        )

    statements = []
    with _traced_sql(statements):
        previews = db.get_project_bbox_previews(project_id)

    selects = [s for s in statements if s.strip().upper().startswith("SELECT")]
    assert len(selects) == 1, f"读框应该只查一次，实际查了 {len(selects)} 次：{selects}"
    assert len(previews) == 5, "5 张图的框都要按 image_id 分好组回来"
    assert all(len(boxes) == 1 for boxes in previews.values())

    statements.clear()
    with _traced_sql(statements):
        versions = db.get_project_annotation_versions(project_id)

    selects = [s for s in statements if s.strip().upper().startswith("SELECT")]
    assert len(selects) == 1, f"读版本号也只该查一次，实际 {len(selects)} 次"
    assert len(versions) == 5

    db.delete_project(project_id)


def test_loading_the_page_does_not_query_annotations_once_per_image():
    """页面加载时对 annotations 表的查询次数跟图片数量无关（N+1 回归防线）。"""
    project_id = _make_box_project()
    folder = _TMP_DIR / "boxes_no_n_plus_one"
    for i in range(6):
        _seed_image_with_boxes(
            project_id, folder, f"n1_{i}.jpg",
            size=(40, 20), boxes=[(0, {'x': 4, 'y': 4, 'width': 10, 'height': 8})],
        )

    page = ImportPage()

    statements = []
    with _traced_sql(statements):
        page.set_project(project_id)
        # 查框现在在后台线程里，必须等它回来再数——不然数到的是「还没查」
        assert _pump_until(lambda: not page._previews_pending), "框没有在超时时间内查回来"

    annotation_queries = [s for s in statements if 'FROM annotations' in s]
    assert len(annotation_queries) == 2, (
        f"6 张图应该只查 2 次 annotations（框 + 版本号），实际 {len(annotation_queries)} 次；"
        "逐图查就是 N+1"
    )

    _settle_thumbnails(page)
    page.stop_image_loading()
    db.delete_project(project_id)


def test_annotation_version_changes_when_a_box_moves_but_the_count_does_not():
    """框数没变、框挪了位置：版本号必须变，否则缩略图会一直停在旧框上。"""
    project_id = _make_box_project()
    image_id, _path, annotation_ids = _seed_image_with_boxes(
        project_id, _TMP_DIR / "boxes_version", "version.jpg",
        size=(40, 20), boxes=[(0, {'x': 1, 'y': 1, 'width': 5, 'height': 5})],
    )

    before = db.get_project_annotation_versions(project_id)[image_id]

    db.update_annotation(annotation_ids[0], data={'x': 20, 'y': 10, 'width': 15, 'height': 8})

    after = db.get_project_annotation_versions(project_id)[image_id]

    assert before[0] == after[0] == 1, "这次改动没有增减框，框数应该一样"
    assert before != after, "框内容变了但版本号没变——缓存永远不会失效"

    # 换类别（颜色会变）同样要能被看见
    changed_class = db.get_project_annotation_versions(project_id)[image_id]
    db.update_annotation(annotation_ids[0], class_id=1, class_name="车")
    assert db.get_project_annotation_versions(project_id)[image_id] != changed_class

    db.delete_project(project_id)


def test_editing_one_box_refreshes_only_that_thumbnail():
    """改一张图的标注：只有那一张重画，别的图连缓存都不碰，也不重新读盘。

    这条同时钉死「变更签名不能只看 status」——两张图从头到尾都是 annotated，
    只比 status 的话这里什么都不会刷新。
    """
    project_id = _make_box_project()
    folder = _TMP_DIR / "boxes_partial_refresh"
    image_a, _path_a, ann_a = _seed_image_with_boxes(
        project_id, folder, "edit_me.jpg",
        size=(40, 20), boxes=[(0, {'x': 2, 'y': 2, 'width': 10, 'height': 8})],
    )
    image_b, _path_b, _ann_b = _seed_image_with_boxes(
        project_id, folder, "leave_me.jpg",
        size=(40, 20), boxes=[(0, {'x': 2, 'y': 2, 'width': 10, 'height': 8})],
    )

    page = _loaded_page(project_id)

    row_of = _rows_by_image_id(page)
    a_before = _icon_image(page, row_of[image_a])
    cached_a_before = page._overlay_cache[image_a][1]
    cached_b_before = page._overlay_cache[image_b][1]

    statuses_before = {img['id']: img['status'] for img in db.get_project_images(project_id)}

    # 在标注页把 A 的框拖到别处——图片状态还是 annotated，只有框动了
    db.update_annotation(ann_a[0], data={'x': 24, 'y': 9, 'width': 14, 'height': 10})

    statuses_after = {img['id']: img['status'] for img in db.get_project_images(project_id)}
    assert statuses_before == statuses_after == {image_a: 'annotated', image_b: 'annotated'}, \
        "前提没成立：这次改动本来就不该改变图片状态"

    # 主窗口从标注页切回导入页时调的就是这个
    page.refresh_project_images()
    _settle_thumbnails(page)

    assert page.load_worker is None, "重画框不该再去读一次磁盘"

    row_of = _rows_by_image_id(page)
    a_after = _icon_image(page, row_of[image_a])

    assert a_after != a_before, "A 的框改了，缩略图却没跟着变"
    assert page._overlay_cache[image_a][1] is not cached_a_before, "A 应该被重画"
    assert page._overlay_cache[image_b][1] is cached_b_before, \
        "B 的标注没动，不该被重画——标注变化只失效受影响的图片"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_recolouring_a_class_recolours_its_boxes():
    """把类别的颜色改了：标注一个字节没动，但缩略图上的框得跟着换色。

    「同一类别始终同色」是跨页面的约定——标注页看到红框，回到导入页不能还是旧颜色。
    """
    project_id = _make_box_project()
    _seed_image_with_boxes(
        project_id, _TMP_DIR / "boxes_recolour", "recolour.jpg",
        size=(40, 20), boxes=[(0, {'x': 10, 'y': 5, 'width': 20, 'height': 10})],
    )

    page = _loaded_page(project_id)
    assert _color_count(_icon_image(page), _RED) > 0, "前提：现在是红框"

    # 在标注页把「人」从红改成绿；标注本身没动
    db.update_project(project_id, classes=[
        {'id': 0, 'name': '人', 'color': _GREEN},
        {'id': 1, 'name': '车', 'color': _RED},
    ])

    page.refresh_project_images()
    _settle_thumbnails(page)

    shown = _icon_image(page)
    assert _color_count(shown, _GREEN) > 0, "类别改成绿色了，框却没换色"
    assert _color_count(shown, _RED) == 0, "旧的红框还留在缩略图上"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_show_boxes_toggle_never_overlaps_the_filter_label():
    """「显示标注框」不能压在「筛选」上。

    这个 bug 只在**完整样式表**下出现：样式表把 QCheckBox::indicator 从 14px 撑到
    17px，可样式是级联下来的，工具栏布局早在那之前就按「没上样式」的 88px 量好了
    槽宽并定死了「筛选」的位置；勾选框随后在那个槽里居中撑开，右边溢出的部分正好
    压到「筛选」上。所以这条测试必须先 setStyleSheet，否则什么都测不出来
    ——之前就是漏了这一步才没拦住。

    勾选框现在装在一个定宽的壳里，比较位置要换算到工具栏坐标系。
    """
    from gui.styles import get_full_stylesheet

    project_id = _make_box_project()

    for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
        page = ImportPage()
        page.setStyleSheet(get_full_stylesheet('light'))  # 缺了这行等于没测
        page.set_project(project_id)
        page.resize(width, height)
        page.show()
        _app.processEvents()

        chk = page.chk_show_boxes
        filter_label = next(
            w for w in page.toolbar.findChildren(QLabel) if w.text() == "筛选"
        )

        chk_rect = chk.rect().translated(
            chk.mapTo(page.toolbar, chk.rect().topLeft())
        )

        assert chk_rect.right() < filter_label.geometry().left(), (
            f"{width}x{height}: 「显示标注框」压到了「筛选」上 "
            f"(右边 {chk_rect.right()} >= 筛选左边 {filter_label.geometry().left()})"
        )

        # 上面这条几何断言只有在真实平台（cocoa）上才抓得到——离屏平台的原生
        # indicator 尺寸恰好和样式表一致，压根不会重叠。所以再钉一条与平台无关的：
        # 勾选框必须待在一个定宽的壳里，而且壳要装得下它上完样式后的真实宽度。
        # 壳一旦被拿掉（勾选框直接进工具栏布局），这条立刻翻。
        holder = chk.parentWidget()
        assert holder is not page.toolbar, \
            "勾选框必须装在自己的定宽壳里，不能直接进工具栏布局"
        assert holder.minimumWidth() == holder.maximumWidth(), \
            "壳必须是定宽的：宽度不定死，布局就会拿样式级联之前的旧尺寸排位置"
        assert holder.width() >= chk.sizeHint().width(), (
            f"{width}x{height}: 壳 {holder.width()}px 装不下上完样式的勾选框 "
            f"{chk.sizeHint().width()}px"
        )

        # 顺带守住：不能为了不重叠就把右边的控件挤出工具栏
        toolbar_layout = page.toolbar.layout()
        last = toolbar_layout.itemAt(toolbar_layout.count() - 1).widget()
        inner_right = page.toolbar.width() - toolbar_layout.contentsMargins().right()
        assert last.geometry().right() <= inner_right, (
            f"{width}x{height}: 最右边的控件被挤出工具栏了"
        )

        page.stop_image_loading()
        page.close()

    db.delete_project(project_id)


def test_selected_thumbnail_is_not_tinted_blue():
    """选中态只用卡片边框，不给图片蒙一层蓝——那层蓝正好把框压得看不清。"""
    project_id = _make_box_project()
    _seed_image_with_boxes(
        project_id, _TMP_DIR / "boxes_selected", "selected.jpg",
        size=(40, 20), boxes=[(0, {'x': 10, 'y': 5, 'width': 20, 'height': 10})],
    )

    page = _loaded_page(project_id)

    normal = _icon_image(page, 0, QIcon.Mode.Normal)
    selected = _icon_image(page, 0, QIcon.Mode.Selected)
    active = _icon_image(page, 0, QIcon.Mode.Active)

    assert selected == normal, "选中态的缩略图被 QIcon 刷上了一层蓝"
    assert active == normal, "hover 态也不该改变缩略图"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_importing_annotations_makes_boxes_appear_without_a_manual_refresh():
    """标注是导进来的（YOLO/COCO/VOC）时，缩略图也要跟着出现框。"""
    project_id = _make_box_project()
    paths = _make_images(_TMP_DIR / "boxes_after_import", 1)
    image_id = db.add_image(project_id, Path(paths[0]).name, paths[0], width=32, height=32)

    page = _loaded_page(project_id)
    assert _icon_image(page) == _base_image(page, paths[0]), "前提：现在还没有框"

    db.add_annotation(
        image_id=image_id, project_id=project_id, class_id=0, class_name="人",
        annotation_type='bbox', data={'x': 4, 'y': 4, 'width': 20, 'height': 20},
    )

    page.refresh_project_images()
    _settle_thumbnails(page)

    assert _color_count(_icon_image(page), _RED) > 0, "新导入的标注没有出现在缩略图上"

    page.stop_image_loading()
    db.delete_project(project_id)


# ==================== 600 张图：刷新不能把界面按住 ====================
# 用户的项目动辄几百张图。三件事必须同时成立：
#   查框在后台线程（不是 GUI 线程），叠框分块排队（不是一次循环 600 张），
#   过期的结果和过期的队列一律作废（不能拿旧数据盖掉新状态）。

_BIG_COUNT = 600

_KEEP_STYLES = []  # QWidget.setStyle 不接管所有权：样式被回收就等于没测


def _seed_boxed_images(project_id, count: int):
    """真往库里写 count 张图，每张一个 bbox——不是 mock 出来的假数据。"""
    for i in range(count):
        image_id = db.add_image(
            project_id, f"big_{i:04d}.jpg",
            str(_TMP_DIR / "big" / f"{project_id}_{i:04d}.jpg"),
            width=40, height=20,
        )
        db.add_annotation(
            image_id=image_id, project_id=project_id, class_id=0, class_name="类0",
            annotation_type='bbox', data={'x': 8, 'y': 4, 'width': 20, 'height': 12},
        )


def _page_with_cached_thumbnails(project_id) -> ImportPage:
    """底图预先放进缓存：这里测的是「底图都有了之后怎么刷」，不掺读盘那条线。"""
    page = ImportPage()
    for image in db.get_project_images(project_id):
        pixmap = QPixmap(160, 160)
        pixmap.fill(QColor(240, 240, 240))
        page.thumbnail_cache[image['storage_path']] = pixmap
    return page


def _seed_light_image(project_id, folder: Path, name: str, size=(40, 20)) -> str:
    """一张浅色的、没有标注的图：用来看「文字」本身，不被图片/框的颜色干扰。"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    image = QImage(size[0], size[1], QImage.Format.Format_RGB888)
    image.fill(QColor(250, 250, 250))
    image.save(str(path))
    db.add_image(project_id, name, str(path), width=size[0], height=size[1])
    return str(path)


def test_annotation_previews_are_read_off_the_gui_thread():
    """600 张图的框 + 版本号：一次读快照读回来，而且是在后台线程里读的。

    这两条查询压在 GUI 线程上时，打开一个大项目就是「界面先僵一下」。查完之前
    列表必须已经建好、能滚动——所以查询发出去的那一刻，主线程手上还没有任何框。

    后台只准走 get_project_bbox_preview_snapshot：分两次调那两个老 API 会各拿一个
    读快照，中途有人挪框就会读到「旧框 + 新版本号」（见 tests/test_bbox_preview_snapshot.py）。
    """
    project_id = _make_box_project()
    _seed_boxed_images(project_id, _BIG_COUNT)

    page = _page_with_cached_thumbnails(project_id)

    reader_threads = []
    split_reads = []
    real_snapshot = database_module.Database.get_project_bbox_preview_snapshot

    def traced_snapshot(self, pid):
        reader_threads.append(threading.current_thread().ident)
        return real_snapshot(self, pid)

    def traced_split_read(name, real):
        # 只记账不抛错：在后台线程里抛异常只会让信号发不出来，测试卡在超时上，
        # 看不出真正的原因
        def traced(self, pid):
            split_reads.append(name)
            return real(self, pid)
        return traced

    statements = []
    with _traced_sql(statements), \
         patch.object(database_module.Database, "get_project_bbox_preview_snapshot", traced_snapshot), \
         patch.object(database_module.Database, "get_project_bbox_previews",
                      traced_split_read("get_project_bbox_previews",
                                        database_module.Database.get_project_bbox_previews)), \
         patch.object(database_module.Database, "get_project_annotation_versions",
                      traced_split_read("get_project_annotation_versions",
                                        database_module.Database.get_project_annotation_versions)):
        page.set_project(project_id)

        # set_project 返回的这一刻：600 个格子已经在了，框还在后台查
        assert page.image_list.count() == _BIG_COUNT
        assert page.load_worker is None, "底图全在缓存里，不该起读盘线程"
        assert page._previews_pending is True
        assert page._annotation_versions == {}, "框是在 GUI 线程里同步查完的"

        assert _pump_until(lambda: not page._previews_pending, timeout=30.0), \
            "框没有在超时时间内查回来"

    main_ident = threading.main_thread().ident
    assert reader_threads, "根本没去查框"
    assert all(ident != main_ident for ident in reader_threads), \
        "框 / 版本号是在 GUI 线程里查的"
    assert not split_reads, f"后台没走原子快照，而是分开读了：{split_reads}"

    annotation_selects = [s for s in statements if 'FROM annotations' in s]
    assert len(annotation_selects) == 2, (
        f"600 张图应该只查 2 次 annotations（框 + 版本号），实际 {len(annotation_selects)} 次；"
        "逐图查就是 N+1"
    )
    assert len(page._annotation_versions) == _BIG_COUNT

    _settle_thumbnails(page)
    assert _color_count(_icon_image(page, _BIG_COUNT - 1), _RED) > 0, \
        "框查回来之后没有叠到缩略图上"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_toggling_boxes_on_600_thumbnails_never_blocks_the_ui():
    """切「显示标注框」必须立刻返回：600 张分块重画，中间把事件循环还回去。"""
    project_id = _make_box_project()
    _seed_boxed_images(project_id, _BIG_COUNT)

    page = _page_with_cached_thumbnails(project_id)
    page.set_project(project_id)
    _settle_thumbnails(page)

    assert page.image_list.count() == _BIG_COUNT
    last_with_boxes = _icon_image(page, _BIG_COUNT - 1)
    assert _color_count(last_with_boxes, _RED) > 0, "前提：600 张现在都带框"

    heartbeats = []
    page.chk_show_boxes.setChecked(False)
    # 0ms 心跳：它排在分块重画的后面，只要重画肯把控制权还回来，它就能跑起来
    QTimer.singleShot(0, lambda: heartbeats.append(page._icon_refresh_pending))

    assert page._icon_refresh_pending is True, \
        "切开关在这一次调用里就把 600 张画完了——这段时间界面是僵的"
    assert _icon_image(page, _BIG_COUNT - 1) == last_with_boxes, \
        "最后一张已经被重画：说明开关里同步循环了全部 600 项"

    assert _pump_until(lambda: not page._icon_refresh_pending, timeout=30.0), \
        "分块重画没有在超时时间内跑完"

    assert heartbeats == [True], \
        "0ms 心跳没能在重画还没做完的时候插进来——事件循环被一口气占住了"

    for row in (0, _BIG_COUNT // 2, _BIG_COUNT - 1):
        assert _color_count(_icon_image(page, row), _RED) == 0, f"第 {row} 张的框没去掉"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_a_stale_refresh_queue_cannot_overwrite_the_newer_state():
    """刷到一半又切了一次开关：旧队列整队作废，不能把它那批旧图补回来。"""
    project_id = _make_box_project()
    _seed_boxed_images(project_id, _BIG_COUNT)

    page = _page_with_cached_thumbnails(project_id)
    page.set_project(project_id)
    _settle_thumbnails(page)

    page.chk_show_boxes.setChecked(False)   # 队列 A：把框去掉
    _app.processEvents()                    # 先让 A 画掉一块
    assert page._icon_refresh_pending is True
    assert page._icon_refresh_cursor > 0, "A 一块都没画，这条测试就没意义了"
    stale_generation = page._icon_refresh_generation

    page.chk_show_boxes.setChecked(True)    # 队列 B：框加回来，A 当场作废
    assert page._icon_refresh_generation != stale_generation

    # A 剩下的块就算真被调起来，也必须自己退出，一个格子都不许画
    page._refresh_icon_chunk(stale_generation)

    assert _pump_until(lambda: not page._icon_refresh_pending, timeout=30.0)

    for row in (0, 11, _BIG_COUNT // 2, _BIG_COUNT - 1):
        assert _color_count(_icon_image(page, row), _RED) > 0, \
            f"第 {row} 张的框被过期的刷新队列抹掉了"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_stale_preview_results_never_land_on_the_current_project():
    """晚到的查询结果（旧项目 / 旧世代）一律丢掉，不能盖掉当前项目的状态。"""
    project_a = _make_box_project()
    project_b = _make_box_project()
    _seed_image_with_boxes(
        project_b, _TMP_DIR / "stale_previews", "b.jpg",
        size=(40, 20), boxes=[(0, {'x': 10, 'y': 5, 'width': 20, 'height': 10})],
    )

    page = _loaded_page(project_b)
    versions_before = dict(page._annotation_versions)
    previews_before = dict(page._bbox_previews)
    shown_before = _icon_image(page)

    # 旧项目的结果晚到了
    page._on_annotation_previews_ready({
        'project_id': project_a,
        'generation': page._preview_generation,
        'bbox_previews': {999: [{'class_id': 1, 'data': {'x': 0, 'y': 0, 'width': 9, 'height': 9}}]},
        'annotation_versions': {999: (1, 1, 'stale')},
        'class_colors': {0: _GREEN},
    })
    # 同一个项目，但世代已经被后来的一次刷新作废了
    page._on_annotation_previews_ready({
        'project_id': project_b,
        'generation': page._preview_generation - 1,
        'bbox_previews': {},
        'annotation_versions': {},
        'class_colors': {},
    })

    assert page._annotation_versions == versions_before, "过期结果盖掉了当前项目的标注版本"
    assert page._bbox_previews == previews_before, "过期结果盖掉了当前项目的框"
    assert page._class_colors.get(0) == _RED, "过期结果把类别配色也改了"

    _settle_thumbnails(page)
    assert _icon_image(page) == shown_before, "过期结果改变了缩略图"

    page.stop_image_loading()
    db.delete_project(project_a)
    db.delete_project(project_b)


def test_thumbnail_loading_does_not_dismiss_the_annotation_import_overlay():
    """「正在导入标注…」的遮罩只能由标注导入自己收尾，缩略图线程不许代劳。

    导入 YOLO / COCO / VOC 标注的过程里会重建缩略图列表，缩略图加载线程先跑完，
    以前就顺手把这个遮罩删了——用户看到的是提示刚亮起来就没了，而标注其实还在导；
    这个遮罩也是唯一挡住重复点击的东西。
    """
    from gui.widgets.loading_dialog import LoadingOverlay

    project_id = _make_box_project()
    paths = _make_images(_TMP_DIR / "overlay_owner", 2)
    for path in paths:
        db.add_image(project_id, Path(path).name, path, width=32, height=32)

    page = _loaded_page(project_id)

    # 标注导入起手做的就是这两句（见 import_yolo_annotations）
    page.loading_overlay = LoadingOverlay(page, "正在导入YOLO标注...")
    page.loading_overlay.show_loading()

    # 让缩略图真的再读一遍盘：底图还在缓存里的话根本不会起线程，也就测不到
    page.thumbnail_cache.clear()
    page.force_refresh_images()
    assert page.load_worker is not None, "前提没成立：缩略图加载线程没起来"

    _settle_thumbnails(page)

    assert hasattr(page, 'loading_overlay'), "缩略图加载完成把标注导入的遮罩删掉了"
    assert not page.loading_overlay.isHidden(), "遮罩还在，却已经被缩略图线程隐藏了"

    # 反过来也要守住：标注导入真的完成时，遮罩必须被收掉，不能漏收
    with _silent_dialogs():
        page.on_annotation_import_finished(True, "导入成功", 3, 0)
    _settle_thumbnails(page)

    assert not hasattr(page, 'loading_overlay'), "标注导入完成后遮罩没有被收掉"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_selected_item_text_stays_dark_under_fusion():
    """Fusion 下选中一张图，格子里的文件名不能变成白字、直接消失。

    不显式写 color 时，选中项的文字用的是调色板里的 HighlightedText——Fusion 下
    那是白色；而选中项的底色仍然是浅色（我们不给它铺蓝底，铺了会把标注框压得看
    不清），于是白字落在浅底上，整行字就没了。选中与否只由蓝色边框表达。
    """
    from gui.styles import COLORS

    project_id = _make_box_project()
    _seed_light_image(project_id, _TMP_DIR / "fusion_selected", "fusion.jpg")

    page = _loaded_page(project_id)

    fusion = QStyleFactory.create("Fusion")
    _KEEP_STYLES.append(fusion)
    page.image_list.setStyle(fusion)
    page.image_list.resize(260, 260)
    page.image_list.show()
    _app.processEvents()

    page.image_list.item(0).setSelected(True)
    _app.processEvents()

    canvas = QPixmap(page.image_list.size())
    canvas.fill(QColor('#FFFFFF'))
    page.image_list.render(canvas)
    shot = canvas.toImage()

    # 这张图、它的格子、它的边框全是浅色，所以深色像素只可能来自文字本身
    dark_pixels = sum(
        1
        for y in range(shot.height())
        for x in range(shot.width())
        if QColor(shot.pixel(x, y)).lightness() < 100
    )
    assert dark_pixels > 0, "Fusion 下选中项的文字是白的，落在浅色格子上等于没有"

    assert f"color: {COLORS['text_primary']}" in page.image_list.styleSheet(), \
        "选中态的文字颜色必须写死成正文色，不能听凭调色板给一个白色"

    # 选中了也不给图片蒙一层蓝
    assert _icon_image(page, 0, QIcon.Mode.Selected) == _icon_image(page, 0, QIcon.Mode.Normal)

    page.image_list.close()
    page.stop_image_loading()
    db.delete_project(project_id)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(globals()))
