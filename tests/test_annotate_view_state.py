# -*- coding: utf-8 -*-
"""标注页的响应式状态：信息栏 / 翻页按钮 / 图片栏 / 类别按钮 / 画布视图锁。

这些断言都只检查用户能直接看到或操作到的结果：
  顶部 / 底部  —— 顶部只留图片、工具和标注方式，状态完整沉到底部
  翻页按钮      —— 上一张和下一张同尺寸，不裁字
  收起图片栏  —— 让出来的宽度必须真的给到画布，展开要能还回去
  等宽按钮    —— 「添加类别」和「改为选中类别」同排，1:1，不许一个宽一个窄
  视图锁      —— 锁上之后切图保持缩放和位置，没锁就照旧适配窗口

    python tests/test_annotate_view_state.py
    python -m pytest tests/test_annotate_view_state.py -q
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QImage, QColor  # noqa: E402
from PyQt6.QtWidgets import QProgressBar, QSizePolicy  # noqa: E402

from gui.workflow import STEP_ANNOTATE  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402

_app = _bootstrap.app()
db = _bootstrap.db

_IMAGE_DIR = Path(tempfile.mkdtemp(prefix="ezyolo-viewstate-"))

# 两张同尺寸 + 一张尺寸差得离谱的，用来分别验证「保持」和「不跑出画布」
IMAGE_SIZES = [(320, 240), (320, 240), (4000, 300)]


def seed_project() -> int:
    project_id = _bootstrap.create_temp_project(
        name="视图状态测试项目",
        project_type="detect",
        classes=[
            {'id': 0, 'name': '猫', 'color': '#FF3B30'},
            {'id': 1, 'name': '用于验证省略提示的超长类别名称', 'color': '#007AFF'},
        ],
    )
    for index, (width, height) in enumerate(IMAGE_SIZES):
        path = _IMAGE_DIR / f"view{index}.jpg"
        image = QImage(width, height, QImage.Format.Format_RGB888)
        image.fill(QColor(120, 120, 130))
        image.save(str(path))
        db.add_image(project_id, path.name, str(path), width=width, height=height)
    return project_id


def open_annotate_page():
    """打开一个装着三张图的标注页。"""
    project_id = seed_project()
    from gui import main_window as mw
    mw.db = db

    window = MainWindow()
    window.load_projects(select_id=project_id)
    window.resize(1100, 720)
    window.show()
    window.switch_page(STEP_ANNOTATE)
    settle()

    return window, project_id


def settle():
    for _ in range(3):
        _app.processEvents()


def close(window, project_id):
    window.close()
    db.delete_project(project_id)


def select_image(page, index: int):
    page.image_list.setCurrentRow(index)
    page.on_image_selected(page.image_list.item(index))
    settle()


# ---------- 图片栏收起 ----------

def test_collapsing_image_list_gives_the_width_to_the_canvas():
    """收起：左栏藏起来，宽度进了中栏；展开：三栏宽度回到原样。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page

    before = page.splitter.sizes()
    center_before = page.center_panel.width()

    page.btn_collapse_image_list.click()
    settle()

    assert not page.left_panel.isVisible(), "收起后左侧图片栏还在占位"
    assert page.btn_expand_image_list.isVisible(), "收起后工具栏上没有展开入口"
    assert page.center_panel.width() > center_before, (
        f"左栏让出的宽度没给到画布：{center_before} -> {page.center_panel.width()}"
    )

    page.btn_expand_image_list.click()
    settle()

    assert page.left_panel.isVisible()
    assert not page.btn_expand_image_list.isVisible()
    after = page.splitter.sizes()
    # 窗口约束下允许几个像素的出入，但不能整栏跑偏
    assert all(abs(a - b) <= 4 for a, b in zip(after, before)), \
        f"展开后三栏宽度没还回去：{before} -> {after}"

    close(window, project_id)


def test_collapsed_state_is_persisted():
    """收起状态写进 QSettings，下次进来还是收着的。"""
    from PyQt6.QtCore import QSettings

    window, project_id = open_annotate_page()
    page = window.annotate_page

    page.btn_collapse_image_list.click()
    settle()

    settings = QSettings("EzYOLO", "Settings")
    assert str(settings.value("annotate_image_list_collapsed")).lower() in ('true', '1')

    page.btn_expand_image_list.click()
    settle()
    assert str(settings.value("annotate_image_list_collapsed")).lower() in ('false', '0')

    close(window, project_id)


# ---------- 类别区两个按钮 ----------

def test_class_buttons_are_exactly_equal_width():
    """「添加类别」和「改为选中类别」同排等宽，1100 宽窗口下都不切字。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page

    add = page.btn_add_class
    apply_ = page.btn_apply_attr

    assert abs(add.width() - apply_.width()) <= 1, \
        f"两个按钮不等宽：添加类别 {add.width()} vs 改为选中类别 {apply_.width()}"
    assert add.y() == apply_.y(), "两个按钮不在同一排"

    for button in (add, apply_):
        needed = button.fontMetrics().horizontalAdvance(button.text())
        assert button.width() >= needed, \
            f"{button.text()} 被切字：需要 {needed}px，只有 {button.width()}px"

    close(window, project_id)


# ---------- 顶部 / 底部信息 ----------

def test_context_bar_keeps_image_tools_and_annotation_mode():
    """顶部不再有进度和类别，图片、工具和标注方式各有固定职责。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page

    captions = [
        label.text() for label in page.context_bar.findChildren(type(page.image_name_label))
        if label.objectName() == "caption"
    ]
    assert captions == ["当前图片", "工具", "标注方式"], f"顶部列不对：{captions}"
    assert not page.context_bar.findChildren(QProgressBar), "顶部不该再有标注进度条"
    assert page.image_name_label.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Preferred
    assert page.image_name_label.minimumWidth() >= 180
    assert page.image_name_label.maximumWidth() <= 320

    close(window, project_id)


def test_status_bar_shows_and_refreshes_progress_category_and_batch_state():
    """底栏实时显示位置、标注进度、类别、本图标注、工具和批处理状态。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page

    select_image(page, 0)
    assert page.status_image.text() == "当前: 1/3"
    assert page.status_progress.text() == "标注: 0/3"
    assert page.status_annotation.text() == "本图标注: 0"
    assert page.status_tool.text() == "工具: 矩形"
    assert page.current_class_chip.text().startswith("类别: ■ "), "当前类别缺少文字和颜色方块线索"
    assert page.btn_shortcut_help.isVisible(), "状态栏缺少快捷键入口"
    for label in (
        page.status_image,
        page.status_progress,
        page.status_annotation,
        page.status_tool,
    ):
        assert label.width() >= label.fontMetrics().horizontalAdvance(label.text()), (
            f"1100px 宽度下状态栏文字被裁切：{label.text()}"
        )

    page.images[0]['status'] = 'annotated'
    page.update_status_bar()
    assert page.status_progress.text() == "标注: 1/3", "图片标注状态变化后底栏进度没有刷新"

    page.current_class_id = -1
    page._update_current_class_chip()
    assert page.current_class_chip.text() == "类别: 未选择"

    page.class_list.setCurrentRow(1)
    page.on_class_selected()
    settle()
    assert "超长类别名称" in page.current_class_chip.toolTip(), "长类别名没有完整 tooltip"
    assert page.current_class_chip.text().startswith("类别: ■ ")

    def assert_status_widgets_do_not_overlap():
        layout = page.status_bar.layout()
        widgets = [
            layout.itemAt(index).widget()
            for index in range(layout.count())
            if layout.itemAt(index).widget() is not None
            and layout.itemAt(index).widget().isVisible()
        ]
        for previous, following in zip(widgets, widgets[1:]):
            assert previous.geometry().right() < following.geometry().left(), (
                "状态栏控件重叠："
                f"{previous.text()!r} {previous.geometry()} 与 "
                f"{following.text()!r} {following.geometry()}"
            )

    assert_status_widgets_do_not_overlap()
    window.resize(1470, 832)
    settle()
    assert_status_widgets_do_not_overlap()

    # 批量任务跑着的时候，进度和取消入口就该一直在——哪怕期间标了个框、切了张图，
    # 那些都会调 update_status_bar。以前这里是无条件清空，用户看着像是任务没了。
    page.on_batch_inference_progress(1, 2, 3, "正在处理的超长文件名.jpg")
    assert page.status_batch.text().startswith("批量标注: 2/3"), "批处理状态没有显示在独立底栏字段"
    assert page.btn_cancel_batch.isVisible(), "批量任务跑着的时候必须有取消入口"

    page.update_status_bar()
    assert page.status_batch.text().startswith("批量标注: 2/3"), \
        "常规状态刷新把正在跑的批量任务进度抹掉了"
    assert page.btn_cancel_batch.isVisible(), "常规状态刷新把取消入口抹掉了"
    assert_status_widgets_do_not_overlap()

    # 任务结束之后才清空
    page._finish_batch_ui()
    assert page.status_batch.text() == "", "任务结束后批处理临时状态没有清空"
    assert not page.btn_cancel_batch.isVisible(), "任务结束后取消入口还留着"

    close(window, project_id)


# ---------- 翻页按钮 ----------

def test_navigation_buttons_have_equal_size_without_cutting_text():
    """上一张和下一张保持左右位置及样式，但尺寸严格一致。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page

    previous = page.btn_prev
    next_ = page.btn_next
    assert previous.size() == next_.size(), (
        f"翻页按钮尺寸不一致：上一张 {previous.size()}，下一张 {next_.size()}"
    )
    assert previous.x() < next_.x(), "翻页按钮左右位置反了"
    assert next_.objectName() == "primary", "下一张主按钮样式丢失"
    for button in (previous, next_):
        assert button.width() >= button.sizeHint().width(), f"{button.text()} 被切字"

    close(window, project_id)


# ---------- 画布视图锁 ----------

def test_lock_button_is_hidden_without_an_image_and_sits_at_top_right():
    """没图片时锁按钮不出现；有图片时贴着画布右上角 8px。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page
    canvas = page.canvas

    assert not canvas.lock_button.isVisible(), "还没选图片就冒出一把锁"

    select_image(page, 0)
    assert canvas.lock_button.isVisible()
    assert canvas.lock_button.size().width() == 28
    assert canvas.lock_button.size().height() == 28

    def assert_geometry():
        button = canvas.lock_button
        assert button.y() == 8, f"锁离画布上边 {button.y()}px，应该是 8px"
        gap = canvas.width() - (button.x() + button.width())
        assert gap == 8, f"锁离画布右边 {gap}px，应该是 8px"

    assert_geometry()

    # 窗口变了，锁还得贴着右上角
    window.resize(1360, 800)
    settle()
    assert_geometry()

    close(window, project_id)


def test_locked_view_keeps_scale_and_offset_across_same_sized_images():
    """锁上之后切到同尺寸的图：缩放和位置原样保留，不再重新适配。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page
    canvas = page.canvas

    select_image(page, 0)

    canvas.lock_button.setChecked(True)
    settle()
    assert canvas.view_locked

    # 先手动缩放并平移，制造一个「用户调过」的视图
    canvas.image_scale = 0.75
    canvas.image_offset = QPoint(30, 20)

    select_image(page, 1)  # 同为 320x240

    assert abs(canvas.image_scale - 0.75) < 1e-6, \
        f"锁定后切图缩放变了：{canvas.image_scale}"
    assert abs(canvas.image_offset.x() - 30) <= 1 and abs(canvas.image_offset.y() - 20) <= 1, \
        f"锁定后切图位置变了：{canvas.image_offset}"

    close(window, project_id)


def test_unlocked_switch_resets_the_view():
    """没锁：切图照旧回到「适配窗口并居中」。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page
    canvas = page.canvas

    select_image(page, 0)
    assert not canvas.view_locked

    canvas.image_scale = 0.5
    canvas.image_offset = QPoint(5, 5)

    select_image(page, 1)

    expected_scale = canvas.image_scale
    expected_offset = QPoint(canvas.image_offset)
    canvas.reset_view()
    assert abs(canvas.image_scale - expected_scale) < 1e-6, "切图后没有重置视图"
    assert canvas.image_offset == expected_offset, "切图后没有重新居中"

    close(window, project_id)


def test_locked_switch_to_a_wildly_different_size_keeps_the_image_on_canvas():
    """锁定 + 新图尺寸差得离谱：不许抛错，也不许整张图跑出画布。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page
    canvas = page.canvas

    select_image(page, 0)  # 320x240
    canvas.lock_button.setChecked(True)
    settle()

    canvas.image_scale = 2.0
    canvas.image_offset = QPoint(-100, -60)

    select_image(page, 2)  # 4000x300

    scaled_width = canvas.current_image.width() * canvas.image_scale
    scaled_height = canvas.current_image.height() * canvas.image_scale
    left = canvas.image_offset.x()
    top = canvas.image_offset.y()

    assert left < canvas.width() and left + scaled_width > 0, \
        f"图片横向跑出了画布：offset.x={left}, 宽 {scaled_width}, 画布 {canvas.width()}"
    assert top < canvas.height() and top + scaled_height > 0, \
        f"图片纵向跑出了画布：offset.y={top}, 高 {scaled_height}, 画布 {canvas.height()}"
    assert abs(canvas.image_scale - 2.0) < 1e-6, "锁定时缩放不该被改掉"

    close(window, project_id)


def test_reset_view_shortcut_does_not_unlock():
    """按 R 只重置当前这一张的视图，锁还是锁着的。"""
    window, project_id = open_annotate_page()
    page = window.annotate_page
    canvas = page.canvas

    select_image(page, 0)
    canvas.lock_button.setChecked(True)
    settle()

    canvas.image_scale = 0.4
    canvas.reset_view()

    assert canvas.view_locked, "重置视图把锁也解开了"
    assert canvas.lock_button.isChecked()

    close(window, project_id)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(dict(globals())))
