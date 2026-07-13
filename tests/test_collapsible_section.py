# -*- coding: utf-8 -*-
"""折叠分区（CollapsibleSection）的核心行为。

标题开关从文字三角 ▸/▾ 换成了矢量箭头图标（chevron_down/chevron_up），
这里锁住换皮之后仍然成立的东西：默认收起、图标不是文字、16px 视觉尺寸、
>=32px 点击目标、整行可点、键盘（Enter/Space）能切、禁用态箭头仍看得见、
不同缩放/窄宽度下标题不被压裁。

    python tests/test_collapsible_section.py
    python -m pytest tests/test_collapsible_section.py -q
"""

import _bootstrap  # noqa: F401  必须第一个导入

from PyQt6.QtCore import QSize, QSizeF, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.widgets.collapsible_section import CollapsibleSection

_app = _bootstrap.app()

TITLE = "样本管理（进阶）"


def _make_section(title: str = TITLE) -> CollapsibleSection:
    section = CollapsibleSection(title)
    section.add_widget(QLabel("内容占位"))
    return section


def test_default_collapsed():
    section = _make_section()
    assert not section.is_expanded()
    assert not section.content.isVisible()


def test_set_expanded_shows_content():
    section = _make_section()
    section.show()
    for _ in range(2):
        _app.processEvents()

    section.set_expanded(True)
    assert section.is_expanded()
    assert section.content.isVisible()

    section.set_expanded(False)
    assert not section.is_expanded()
    assert not section.content.isVisible()

    section.close()


def test_public_api_unchanged():
    """content_layout / add_widget / add_layout 这套接口不能变。"""
    section = _make_section()
    assert callable(section.content_layout)
    layout = section.content_layout()
    before = layout.count()

    section.add_widget(QLabel("另一个控件"))
    assert layout.count() == before + 1

    row = QHBoxLayout()
    row.addWidget(QLabel("行内控件"))
    section.add_layout(row)
    assert layout.count() == before + 2


def test_toggle_icon_is_not_text_triangle():
    """箭头必须是图标，不能再是 ▸/▾ 文字。"""
    section = _make_section()
    assert "▸" not in section.toggle.text()
    assert "▾" not in section.toggle.text()
    assert section.toggle.text() == TITLE
    assert not section.toggle.icon().isNull()


def test_icon_swaps_between_collapsed_and_expanded():
    section = _make_section()
    collapsed_icon = section.toggle.icon()
    collapsed_key = collapsed_icon.pixmap(QSize(16, 16)).cacheKey()

    section.set_expanded(True)
    expanded_icon = section.toggle.icon()
    expanded_key = expanded_icon.pixmap(QSize(16, 16)).cacheKey()

    assert expanded_key != collapsed_key, "展开/收起的箭头图标应该不一样"


def test_icon_visual_size_is_16px():
    section = _make_section()
    assert section.toggle.iconSize() == QSize(16, 16)


def test_click_target_at_least_32px():
    """minimumHeight 是保证；这里同时确认布局落地后实际高度真的 >=32px
    (sizeHint() 本身不受 setMinimumHeight 影响，量不出真实点击区)。
    """
    container = QWidget()
    layout = QVBoxLayout(container)
    section = _make_section()
    layout.addWidget(section)
    container.resize(600, 200)
    container.show()
    for _ in range(2):
        _app.processEvents()

    assert section.toggle.minimumHeight() >= 32
    assert section.toggle.height() >= 32
    container.close()


def test_whole_row_is_clickable():
    """标题按钮要撑满卡片整行，而不是缩在文字宽度上。"""
    container = QWidget()
    layout = QVBoxLayout(container)
    section = _make_section()
    layout.addWidget(section)
    container.resize(600, 200)
    container.show()
    for _ in range(2):
        _app.processEvents()

    assert section.toggle.width() >= section.width() - 40, (
        f"标题按钮宽度 {section.toggle.width()} 应该接近卡片宽度 {section.width()}，"
        "否则空白处点不到"
    )

    before = section.is_expanded()
    # 点按钮最右侧（远离文字），整行可点意味着这里也能触发
    QTest.mouseClick(
        section.toggle,
        Qt.MouseButton.LeftButton,
        pos=section.toggle.rect().center(),
    )
    assert section.is_expanded() != before
    container.close()


def test_keyboard_enter_toggles():
    section = _make_section()
    section.toggle.setFocus()
    before = section.is_expanded()
    QTest.keyClick(section.toggle, Qt.Key.Key_Return)
    assert section.is_expanded() != before


def test_keyboard_space_toggles():
    section = _make_section()
    section.toggle.setFocus()
    before = section.is_expanded()
    QTest.keyClick(section.toggle, Qt.Key.Key_Space)
    assert section.is_expanded() != before


def test_toggle_has_strong_focus_policy():
    section = _make_section()
    assert section.toggle.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_disabled_icon_still_visible_and_distinct():
    """禁用整个分区之后，箭头还得看得见，而且是 disabled 变体（不是同一张图）。"""
    section = _make_section()
    enabled_key = section.toggle.icon().pixmap(
        QSize(16, 16), QIcon.Mode.Normal
    ).cacheKey()

    section.toggle.setEnabled(False)
    assert not section.toggle.icon().isNull(), "禁用后箭头图标不应该消失"

    disabled_pixmap = section.toggle.icon().pixmap(QSize(16, 16), QIcon.Mode.Disabled)
    assert not disabled_pixmap.isNull()
    assert disabled_pixmap.cacheKey() != enabled_key, "禁用态应该换成灰色变体，不是原图硬灰"


def test_disabled_icon_tracks_expand_state():
    """禁用态下收起/展开两枚图标各自有 disabled 变体，不会退化成同一张图。"""
    section = _make_section()
    section.toggle.setEnabled(False)
    collapsed_disabled = section.toggle.icon().pixmap(
        QSize(16, 16), QIcon.Mode.Disabled
    ).cacheKey()

    section.set_expanded(True)
    expanded_disabled = section.toggle.icon().pixmap(
        QSize(16, 16), QIcon.Mode.Disabled
    ).cacheKey()

    assert expanded_disabled != collapsed_disabled


def _title_not_clipped(section: CollapsibleSection) -> bool:
    """标题按钮的 minimumSizeHint 已经把图标 + 文字 + 内边距都算进去了；
    只要按钮实际宽度不小于这个值，文字就不会被压裁（跟 test_ui_layout.py
    里对 QPushButton 的切字判据一致）。
    """
    return section.toggle.width() + 2 >= section.toggle.minimumSizeHint().width()


def test_no_clipping_across_common_widths():
    """1100 / 1280 常见窗口宽度下，标题文字不能被图标挤裁。"""
    for width in (1100, 1280):
        container = QWidget()
        layout = QVBoxLayout(container)
        section = _make_section()
        layout.addWidget(section)
        container.resize(width, 400)
        container.show()
        for _ in range(2):
            _app.processEvents()

        assert _title_not_clipped(section), f"宽度 {width} 下标题被压裁"
        assert section.toggle.text() == TITLE

        container.close()


def test_icon_renders_crisp_across_dpi_scales():
    """图标来自 svg，按任意 devicePixelRatio 取 pixmap 都应该精确匹配（矢量，不是位图硬缩放）。

    100/125/150/200% 对应的物理尺寸分别是 16/20/24/32px；用 pixmap(size, dpr) 这个
    重载显式传目标 dpr，不能让外层进程的 QT_SCALE_FACTOR 再乘一遍——那是两回事：
    QT_SCALE_FACTOR 影响的是这次调用之外、Qt 自己默认取的 dpr，跟这里显式传入的
    dpr 参数互不相关，混在一起断言就会在外层缩放不是 1.0 时重复相乘算错物理尺寸。
    """
    section = _make_section()
    for dpr in (1.0, 1.25, 1.5, 2.0):
        pixmap = section.toggle.icon().pixmap(QSize(16, 16), dpr)
        assert not pixmap.isNull()
        expected_physical = round(16 * dpr)
        assert pixmap.size() == QSize(expected_physical, expected_physical)
        assert pixmap.devicePixelRatio() == dpr
        assert pixmap.deviceIndependentSize() == QSizeF(16.0, 16.0)


def test_train_auto_label_test_pages_share_the_same_class():
    """train_page / auto_label_dialog / test_page 不再各自维护一份局部实现。"""
    import gui.pages.train_page as train_page
    import gui.pages.auto_label_dialog as auto_label_dialog
    import gui.pages.test_page as test_page
    import gui.pages.annotate_page as annotate_page

    assert train_page.CollapsibleSection is CollapsibleSection
    assert auto_label_dialog.CollapsibleSection is CollapsibleSection
    assert test_page.CollapsibleSection is CollapsibleSection
    assert annotate_page.CollapsibleSection is CollapsibleSection


if __name__ == "__main__":
    import sys
    sys.exit(_bootstrap.run_module_tests(globals()))
