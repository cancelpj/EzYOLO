# -*- coding: utf-8 -*-
"""悬浮提示（QToolTip）深色样式 + 加载卡片阴影：只管这两处外观，不碰别的控件。

    python tests/test_tooltip_shadow_style.py
    python -m pytest tests/test_tooltip_shadow_style.py -q
"""

import _bootstrap  # noqa: F401  必须第一个导入

import re
import sys
from unittest.mock import patch

from gui.styles import COLORS, generate_stylesheet  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.pages.auto_label_dialog import AutoLabelDialog  # noqa: E402

_app = _bootstrap.app()
db = _bootstrap.db


def _make_window() -> MainWindow:
    from gui import main_window as mw
    mw.db = db
    return MainWindow()


def _tooltip_block(stylesheet: str) -> str:
    match = re.search(r"QToolTip\s*\{([^}]*)\}", stylesheet)
    assert match, "样式表里找不到 QToolTip 规则"
    return match.group(1)


def test_tooltip_is_dark_with_white_text_and_8_10_padding():
    stylesheet = generate_stylesheet(COLORS)
    block = _tooltip_block(stylesheet)

    assert "background-color: #3A3A3C;" in block
    assert "color: #FFFFFF;" in block
    assert "border: 1px solid rgba(255, 255, 255, 0.12);" in block
    assert "border-radius: 6px;" in block
    assert "padding: 8px 10px;" in block


def test_tooltip_qss_does_not_leak_into_menu_or_label():
    stylesheet = generate_stylesheet(COLORS)

    # QToolTip 规则只用单独选择器，不跟 QMenu/QLabel 合写成逗号列表
    assert not re.search(r"QToolTip\s*,", stylesheet)
    assert not re.search(r",\s*QToolTip\b", stylesheet)

    menu_block = re.search(r"QMenu\s*\{([^}]*)\}", stylesheet).group(1)
    label_block = re.search(r"QLabel\s*\{([^}]*)\}", stylesheet).group(1)
    assert "#3A3A3C" not in menu_block
    assert "#3A3A3C" not in label_block


TOOLTIP_MAX_CHARS = 40


def _assert_short_tooltip(text: str, name: str):
    assert text, f"{name} 的 tooltip 是空的"
    assert "\n" not in text, f"{name} 的 tooltip 不该换行: {text!r}"
    assert len(text) <= TOOLTIP_MAX_CHARS, f"{name} 的 tooltip 超过 {TOOLTIP_MAX_CHARS} 字: {text!r}"


def test_annotate_page_short_tooltips():
    window = _make_window()
    page = window.annotate_page

    _assert_short_tooltip(page.task_combo.toolTip(), "task_combo")
    assert page.task_combo.toolTip() == "标注方式：决定画框/多边形/关键点/整图分类"

    _assert_short_tooltip(page.btn_auto_label.toolTip(), "btn_auto_label")
    assert page.btn_auto_label.toolTip() == "用已训练模型自动标注"

    _assert_short_tooltip(page.btn_llm_label.toolTip(), "btn_llm_label")
    assert page.btn_llm_label.toolTip() == "用多模态大模型识别目标"

    memory_config = {
        "sam_type": "SAM2",
        "model_file": "sam2_b.pt",
        "device": "cpu",
        "imgsz": 1024,
        "conf": 0.4,
        "iou": 0.9,
        "retina_masks": True,
        "usage_mode": "memory",
    }
    with patch.object(AutoLabelDialog, "get_saved_sam_config", return_value=memory_config):
        page.apply_sam_button_mode()
        _assert_short_tooltip(page.btn_sam.toolTip(), "btn_sam(记忆)")
        assert page.btn_sam.toolTip() == "SAM 记忆标注：先教一次，之后自动标同类"

    normal_config = {
        "sam_type": "SAM",
        "model_file": "sam_b.pt",
        "device": "cpu",
        "imgsz": 1024,
        "conf": 0.4,
        "iou": 0.9,
        "retina_masks": True,
        "usage_mode": "normal",
    }
    with patch.object(AutoLabelDialog, "get_saved_sam_config", return_value=normal_config):
        page.apply_sam_button_mode()
        _assert_short_tooltip(page.btn_sam.toolTip(), "btn_sam(普通)")
        assert page.btn_sam.toolTip() == "点目标自动分割轮廓"


def test_loading_dialog_shadow_still_present():
    from gui.widgets.loading_dialog import LoadingDialog, LoadingOverlay

    dialog = LoadingDialog(None, title="加载中", message="请稍候...")
    card = dialog.findChild(object, "loading_card")
    assert card is not None
    effect = card.graphicsEffect()
    assert effect is not None
    assert effect.blurRadius() == 28

    host = window_widget = _make_window()
    overlay = LoadingOverlay(host, message="加载中...")
    overlay_card = overlay.findChild(object, "loading_card")
    assert overlay_card is not None
    overlay_effect = overlay_card.graphicsEffect()
    assert overlay_effect is not None


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(globals()))
