# -*- coding: utf-8 -*-
"""ContextHelp：轻量行内帮助组件。

默认收起时只显示右侧的小入口，展开后是无编号的短提示；风险信息单独成行。
展开收起不能在窗口里留空白，长中文在不同宽度和字号下也不能溢出或被裁切。

    python tests/test_context_help.py
    python -m pytest tests/test_context_help.py -q
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy

from gui.styles import COLORS
from gui.widgets.context_help import ContextHelp

_app = _bootstrap.app()

STEPS = ["选择需要导入的图片", "确认标注类型", "点击开始导入"]

LONG_STEPS = [
    "在导入页选择包含待标注图片的文件夹，支持同时选择多个子目录，"
    "系统会自动识别其中的图片格式并跳过无法解析的文件，请耐心等待扫描完成",
    "确认每一张图片对应的标注类型与所属分组，避免训练集和测试集互相混淆导致模型评估结果失真",
]

SLACK = 2


def _settle(rounds=3):
    for _ in range(rounds):
        _app.processEvents()


def _step_labels(help_widget):
    return [
        label for label in help_widget.content.findChildren(QLabel)
        if label.objectName() == "contextHelpTipText"
    ]


def _tip_rows(help_widget):
    return [
        row for row in help_widget.content.findChildren(QFrame)
        if row.objectName() in ("contextHelpTipRow", "contextHelpRiskRow")
    ]


def test_default_state_is_collapsed():
    help_widget = ContextHelp(STEPS)
    assert not help_widget.toggle.isChecked()
    assert not help_widget.content.isVisible()
    help_widget.close()


def test_click_expands_then_collapses_height_restores():
    help_widget = ContextHelp(STEPS)
    help_widget.show()
    _settle()

    collapsed_height = help_widget.sizeHint().height()

    QTest.mouseClick(help_widget.toggle, Qt.MouseButton.LeftButton)
    _settle()
    assert help_widget.toggle.isChecked()
    assert help_widget.content.isVisible()
    expanded_height = help_widget.sizeHint().height()
    assert expanded_height > collapsed_height, "展开后高度应该比收起时高"

    QTest.mouseClick(help_widget.toggle, Qt.MouseButton.LeftButton)
    _settle()
    assert not help_widget.toggle.isChecked()
    assert not help_widget.content.isVisible()
    restored_height = help_widget.sizeHint().height()
    assert restored_height == collapsed_height, (
        f"关闭后应该完全收回高度：收起前 {collapsed_height}px，关闭后却是 {restored_height}px"
    )

    help_widget.close()


def test_enter_key_toggles():
    help_widget = ContextHelp(STEPS)
    help_widget.show()
    help_widget.toggle.setFocus()
    _settle()

    QTest.keyClick(help_widget.toggle, Qt.Key.Key_Return)
    _settle()
    assert help_widget.toggle.isChecked(), "Enter 应该展开"

    QTest.keyClick(help_widget.toggle, Qt.Key.Key_Return)
    _settle()
    assert not help_widget.toggle.isChecked(), "再按一次 Enter 应该收起"

    help_widget.close()


def test_space_key_toggles():
    help_widget = ContextHelp(STEPS)
    help_widget.show()
    help_widget.toggle.setFocus()
    _settle()

    QTest.keyClick(help_widget.toggle, Qt.Key.Key_Space)
    _settle()
    assert help_widget.toggle.isChecked(), "Space 应该展开"

    QTest.keyClick(help_widget.toggle, Qt.Key.Key_Space)
    _settle()
    assert not help_widget.toggle.isChecked(), "再按一次 Space 应该收起"

    help_widget.close()


def test_toggle_is_focusable():
    help_widget = ContextHelp(STEPS)
    assert help_widget.toggle.focusPolicy() != Qt.FocusPolicy.NoFocus
    help_widget.close()


def test_toggle_row_hit_area_is_at_least_32px_tall():
    help_widget = ContextHelp(STEPS)
    help_widget.show()
    _settle()
    assert help_widget.toggle.sizeHint().height() >= 32, (
        f"整行开关的可点击高度只有 {help_widget.toggle.sizeHint().height()}px，不足 32px"
    )
    help_widget.close()


def test_collapsed_entry_is_compact_borderless_and_right_aligned():
    help_widget = ContextHelp(STEPS, title="导入提示")
    help_widget.resize(800, help_widget.sizeHint().height())
    help_widget.show()
    _settle()

    assert help_widget.toggle.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Maximum
    assert help_widget.toggle.width() < 180, (
        f"收起入口不该像整行卡片一样宽，实际 {help_widget.toggle.width()}px"
    )
    assert help_widget.toggle.geometry().right() >= help_widget.header.width() - SLACK, (
        "轻量帮助入口应该靠右，而不是占据页面左侧的主内容位置"
    )
    assert "background-color: transparent" in help_widget.styleSheet()
    assert "border: none" in help_widget.styleSheet()

    help_widget.close()


def test_context_title_is_specific_not_a_global_tutorial_label():
    help_widget = ContextHelp(STEPS, title="训练前检查")
    assert help_widget._label.text() == "训练前检查"
    assert help_widget._label.text() != "使用说明"
    assert help_widget.toggle.accessibleName() == "训练前检查"
    help_widget.close()


def test_chevron_icon_is_16px_collapsed_and_expanded():
    help_widget = ContextHelp(STEPS)
    help_widget.show()
    _settle()

    def logical_size(pixmap):
        ratio = pixmap.devicePixelRatio() or 1.0
        return round(pixmap.width() / ratio), round(pixmap.height() / ratio)

    collapsed_pixmap = help_widget._chevron.pixmap()
    assert logical_size(collapsed_pixmap) == (16, 16)

    QTest.mouseClick(help_widget.toggle, Qt.MouseButton.LeftButton)
    _settle()
    expanded_pixmap = help_widget._chevron.pixmap()
    assert logical_size(expanded_pixmap) == (16, 16)

    help_widget.close()


def test_tips_keep_their_text_without_tutorial_numbers():
    help_widget = ContextHelp(STEPS)
    help_widget.set_expanded(True)
    _settle()

    labels = _step_labels(help_widget)
    assert len(labels) == len(STEPS)
    for label, text in zip(labels, STEPS):
        assert label.text() == text
        assert not label.text()[0].isdigit(), "帮助内容不该再像 1、2、3 的新手教程"

    help_widget.close()


def test_risk_steps_get_warning_style_others_do_not():
    help_widget = ContextHelp(STEPS, risk_steps=[2])
    help_widget.set_expanded(True)
    _settle()

    labels = _step_labels(help_widget)
    risky, safe = labels[1], labels[0]
    rows = _tip_rows(help_widget)

    assert COLORS['warning'] in risky.styleSheet(), "风险步骤应该用警示色文字"
    assert COLORS['warning'] not in safe.styleSheet(), "非风险步骤不该套警示样式"
    assert rows[1].objectName() == "contextHelpRiskRow", "风险提示应该单独成行"
    assert rows[0].objectName() == "contextHelpTipRow", "普通提示不该整行套警告底色"
    assert "background-color: transparent" in rows[0].styleSheet(), (
        "普通提示行必须显式透明，否则会继承全局 QFrame 的白色卡片样式"
    )
    assert "border: none" in rows[0].styleSheet()

    help_widget.close()


def test_keyboard_focus_does_not_turn_collapsed_entry_into_a_filled_card():
    help_widget = ContextHelp(STEPS, title="导入提示")
    focus_rule = help_widget.toggle.styleSheet().split(
        "QPushButton#contextHelpToggle:focus", 1
    )[1].split("}", 1)[0]

    assert "background-color: transparent" in focus_rule
    assert "border-color:" in focus_rule, "键盘焦点仍需有清楚但克制的描边"

    help_widget.close()


def test_set_steps_replaces_content_without_leftovers():
    help_widget = ContextHelp(STEPS)
    help_widget.set_expanded(True)
    _settle()
    assert len(_step_labels(help_widget)) == len(STEPS)

    new_steps = ["新的第一步", "新的第二步"]
    help_widget.set_steps(new_steps)
    _settle()

    labels = _step_labels(help_widget)
    assert len(labels) == len(new_steps)
    assert labels[0].text() == "新的第一步"
    assert labels[1].text() == "新的第二步"

    help_widget.close()


def test_new_instance_does_not_inherit_expanded_state():
    first = ContextHelp(STEPS)
    first.set_expanded(True)
    _settle()
    assert first.toggle.isChecked()

    second = ContextHelp(STEPS)
    assert not second.toggle.isChecked(), "新实例不该继承上一个实例的展开状态"

    first.close()
    second.close()


def _label_overflows_or_clips(label, container_width) -> str:
    """长中文换行标签：横向不许溢出容器，纵向不许把换行后的文字切掉。"""
    if label.width() > container_width + SLACK:
        return f"标签宽度 {label.width()}px 超出容器 {container_width}px"

    needed_height = label.heightForWidth(label.width()) if label.width() > 0 else 0
    if needed_height > label.height() + SLACK:
        return f"需要 {needed_height}px 才能放下换行文字，只有 {label.height()}px"
    return ""


def test_long_chinese_steps_do_not_clip_or_overflow_across_sizes_and_font_scales():
    base_point_size = _app.font().pointSizeF() or 13.0
    scales = (1.0, 1.25, 1.5, 2.0)
    widths = (1100, 1280)

    problems = []
    for scale in scales:
        font = _app.font()
        font.setPointSizeF(base_point_size * scale)

        for width in widths:
            help_widget = ContextHelp(LONG_STEPS, risk_steps=[2])
            help_widget.setFont(font)
            for child in help_widget.findChildren(QLabel):
                child.setFont(font)
            help_widget.set_expanded(True)
            help_widget.resize(width, help_widget.sizeHint().height())
            help_widget.show()
            _settle(5)
            # wordWrap 标签的换行是按当前宽度算的，字体一变高度也要重新布局一次
            help_widget.resize(width, help_widget.sizeHint().height())
            _settle(5)

            container_width = help_widget.content.width()
            for label in _step_labels(help_widget):
                assert label.wordWrap(), "步骤文字必须能换行，否则长句子只能横向溢出"
                problem = _label_overflows_or_clips(label, container_width)
                if problem:
                    problems.append(f"[scale={scale} width={width}] {problem}: {label.text()[:20]!r}")

            horizontal_scrollbar = getattr(help_widget, "horizontalScrollBar", None)
            if horizontal_scrollbar is not None and horizontal_scrollbar() is not None:
                assert horizontal_scrollbar().maximum() == 0

            help_widget.close()

    assert not problems, "长中文在某些尺寸/字号下溢出或被裁切:\n" + "\n".join(problems)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(globals()))
