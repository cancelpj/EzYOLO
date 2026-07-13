# -*- coding: utf-8 -*-
"""菜单按钮必须使用项目自己的向下箭头，不能退回 Qt 原生指示器。

    python tests/test_menu_arrow_styles.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import re
import sys
from pathlib import Path

from PyQt6.QtWidgets import (  # noqa: E402
    QComboBox, QDoubleSpinBox, QPushButton, QSpinBox, QStyle,
    QStyleOptionComboBox, QStyleOptionSpinBox, QToolButton, QWidget,
)

from gui.main_window import MainWindow  # noqa: E402
from gui.pages.auto_label_dialog import AutoLabelDialog  # noqa: E402
from gui.styles import COLORS, generate_stylesheet  # noqa: E402

_app = _bootstrap.app()
_ASSETS_DIR = Path(__file__).parent.parent / "gui" / "assets"


def _make_window():
    from gui import main_window as main_window_module
    main_window_module.db = _bootstrap.db
    return MainWindow()


def test_menu_chevron_assets_and_global_qss_are_explicit():
    expected_colors = {
        "chevron_down.svg": "#007AFF",
        "chevron_down_disabled.svg": "#AEAEB2",
        "chevron_down_white.svg": "#FFFFFF",
        "chevron_down_red.svg": "#D70015",
    }
    for name, color in expected_colors.items():
        asset = _ASSETS_DIR / name
        assert asset.is_file(), name
        svg = asset.read_text(encoding="utf-8")
        assert 'width="12" height="12"' in svg
        assert 'stroke-width="1.6"' in svg
        assert f'stroke="{color}"' in svg

    stylesheet = generate_stylesheet(COLORS)
    assert 'QPushButton[menuIndicator="true"]' in stylesheet
    assert 'padding-right: 34px;' in stylesheet
    assert 'QPushButton#primary[menuIndicator="true"]::menu-indicator' in stylesheet
    assert 'chevron_down.svg' in stylesheet
    assert 'chevron_down_white.svg' in stylesheet
    assert 'chevron_down_disabled.svg' in stylesheet
    assert not re.search(r"^\s*QPushButton::menu-indicator", stylesheet, re.MULTILINE)


def test_global_qss_toolbutton_combo_spin_arrow_rules():
    """统一后的箭头规则：菜单箭头 16px/34px 专用区，下拉框 14px/34px，
    数字步进箭头 12px、上下各自点击区 18px。
    """
    stylesheet = generate_stylesheet(COLORS)

    assert 'QPushButton[menuIndicator="true"], QToolButton[menuIndicator="true"]' in stylesheet
    assert (
        'QPushButton[menuIndicator="true"]::menu-indicator,\n'
        'QToolButton[menuIndicator="true"]::menu-indicator {'
    ) in stylesheet
    assert 'QToolButton[menuIndicator="true"]::menu-indicator:disabled' in stylesheet

    combo_block = stylesheet[stylesheet.index('QComboBox {'):stylesheet.index('QComboBox QAbstractItemView')]
    assert 'padding-right: 34px;' in combo_block
    assert 'width: 34px;' in combo_block
    assert 'width: 14px;\n    height: 14px;' in combo_block

    spin_block = stylesheet[stylesheet.index('QSpinBox::up-button'):stylesheet.index('QTextEdit {')]
    assert 'subcontrol-origin: border;' in spin_block
    assert 'height: 18px;' in spin_block
    assert spin_block.count('width: 12px;\n    height: 12px;') == 2


def test_manage_button_uses_icon_arrow_not_text_triangle():
    window = _make_window()
    try:
        btn = window.import_page.btn_manage
        assert isinstance(btn, QToolButton)
        assert btn.text() == "管理"
        assert "▸" not in btn.text() and "▾" not in btn.text()
        assert btn.property("menuIndicator") is True
    finally:
        window.close()


def test_combo_and_spin_hit_areas_meet_minimums():
    container = QWidget()
    container.setStyleSheet(generate_stylesheet(COLORS))
    container.resize(300, 200)

    combo = QComboBox(container)
    combo.addItems(["a", "b"])
    combo.resize(200, 30)
    combo.show()
    _app.processEvents()
    combo_option = QStyleOptionComboBox()
    combo_option.initFrom(combo)
    assert combo.style().subControlRect(
        QStyle.ComplexControl.CC_ComboBox, combo_option,
        QStyle.SubControl.SC_ComboBoxArrow, combo,
    ).width() >= 34

    spin = QSpinBox(container)
    spin.resize(120, 36)
    spin.show()

    dspin = QDoubleSpinBox(container)
    dspin.resize(120, 36)
    dspin.show()

    container.show()
    _app.processEvents()

    for box in (spin, dspin):
        option = QStyleOptionSpinBox()
        option.initFrom(box)
        up_rect = box.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxUp, box
        )
        down_rect = box.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxDown, box
        )
        assert up_rect.height() >= 18, up_rect.height()
        assert down_rect.height() >= 18, down_rect.height()

    container.close()


def test_all_menu_pushbuttons_are_marked_and_sam_refreshes_dynamically():
    window = _make_window()
    try:
        annotate = window.annotate_page
        train = window.train_page
        result = window.result_page

        expected = {
            annotate.btn_auto_label,
            annotate.btn_llm_label,
            train.btn_template_menu,
            result.btn_export_model,
        }
        assert all(button.menu() is not None for button in expected)
        assert all(button.property("menuIndicator") is True for button in expected)

        original_method = AutoLabelDialog.__dict__["get_saved_sam_config"]
        AutoLabelDialog.get_saved_sam_config = classmethod(
            lambda cls: {"sam_type": "SAM2", "usage_mode": "memory"}
        )
        try:
            annotate.apply_sam_button_mode()
            assert annotate.btn_sam.menu() is not None
            assert annotate.btn_sam.property("menuIndicator") is True
        finally:
            AutoLabelDialog.get_saved_sam_config = original_method

        AutoLabelDialog.get_saved_sam_config = classmethod(
            lambda cls: {"sam_type": "SAM", "usage_mode": "normal"}
        )
        try:
            annotate.apply_sam_button_mode()
            assert annotate.btn_sam.menu() is None
            assert annotate.btn_sam.property("menuIndicator") is False
        finally:
            AutoLabelDialog.get_saved_sam_config = original_method

        for button in window.findChildren(QPushButton):
            if button.menu() is not None:
                assert button.property("menuIndicator") is True, button.text()
    finally:
        window.close()


def test_annotate_tool_menu_arrows_have_roles_and_space():
    window = _make_window()
    try:
        annotate = window.annotate_page
        draw_style = annotate.btn_draw_tool.styleSheet()
        delete_style = annotate.btn_delete.styleSheet()

        assert annotate.btn_draw_tool.menu() is not None
        assert annotate.btn_delete.menu() is not None
        assert 'QToolButton::menu-arrow' in draw_style
        assert 'QToolButton::menu-indicator' in draw_style
        assert 'chevron_down.svg' in draw_style
        assert 'chevron_down_disabled.svg' in draw_style
        assert 'width: 34px;' in draw_style
        assert 'width: 16px;' in draw_style
        assert 'background-color: #FFFFFF;' in draw_style

        assert 'QToolButton::menu-arrow' in delete_style
        assert 'QToolButton::menu-indicator' in delete_style
        assert 'chevron_down_red.svg' in delete_style
        assert 'chevron_down_disabled.svg' in delete_style
        assert 'width: 34px;' in delete_style
        assert 'width: 16px;' in delete_style

        for button in (annotate.btn_draw_tool, annotate.btn_delete):
            assert button.width() >= button.minimumSizeHint().width()
    finally:
        window.close()


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(globals()))
