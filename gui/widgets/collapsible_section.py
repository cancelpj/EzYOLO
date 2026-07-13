# -*- coding: utf-8 -*-
"""默认收起的分区。

给「用得着、但不是每次都用」的功能准备的：标注页右栏的样本管理和数据导出
都属于这一类——常驻展开只会把类别和 AI 入口挤到屏幕外面，可它们又不该被藏进
菜单里找不着。收起来只留一行标题，点一下才展开。

没有动画，也不记住展开状态：每次进页面都从收起开始，行为可预期。

标题行的箭头用项目里现成的 chevron_down / chevron_up 矢量图标（而不是
文字三角 ▸/▾）：收起态指向内容会出现的方向（down），展开态收拢向上（up）；
两枚图标各自带 disabled 变体，禁用整个分区时箭头跟着变灰，而不是凭空消失。
"""

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget

from gui.styles import COLORS, RADIUS, RADIUS_SM

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_ICON_SIZE = QSize(16, 16)


def _chevron(normal_name: str, disabled_name: str) -> QIcon:
    """一枚箭头图标的两种画法：正常态用原图，disabled 态换灰色版。

    两个文件都喂给 QIcon.Mode.Normal/Disabled，Qt 会在控件被 setEnabled(False)
    时自己按 mode 切换，不用手动在禁用时换图。
    """
    icon = QIcon()
    icon.addFile(str(_ASSETS_DIR / normal_name), _ICON_SIZE, QIcon.Mode.Normal, QIcon.State.Off)
    icon.addFile(str(_ASSETS_DIR / disabled_name), _ICON_SIZE, QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


class _ToggleButton(QPushButton):
    """标题开关按钮。

    Space 由 QAbstractButton 原生处理；Enter/Return 默认只在 autoDefault
    生效（父容器是 QDialog）时才触发点击，这里补上，保证不管挂在什么容器
    下键盘都能切换。
    """

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
            return
        super().keyPressEvent(event)


class CollapsibleSection(QFrame):
    """一张卡片：标题一行，内容默认收起。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._icon_collapsed = _chevron("chevron_down.svg", "chevron_down_disabled.svg")
        self._icon_expanded = _chevron("chevron_up.svg", "chevron_up_disabled.svg")

        self.setObjectName("card")
        self.setStyleSheet(f"""
            QFrame#card {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS}px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # 标题即开关。整行都能点，不用去瞄那个小箭头。
        self.toggle = _ToggleButton(title)
        self.toggle.setObjectName("ghost")
        self.toggle.setCheckable(True)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle.setMinimumHeight(32)
        self.toggle.setIconSize(_ICON_SIZE)
        self.toggle.setStyleSheet(f"""
            QPushButton#ghost {{
                text-align: left;
                padding: 4px 8px;
                border-radius: {RADIUS_SM}px;
                color: {COLORS['text_primary']};
                font-weight: 600;
            }}
            QPushButton#ghost:hover {{
                background-color: {COLORS['hover']};
            }}
            QPushButton#ghost:checked {{
                background-color: transparent;
            }}
        """)
        self.toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self.toggle)

        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 2)
        content_layout.setSpacing(8)
        self.content.setVisible(False)
        layout.addWidget(self.content)

        self._content_layout = content_layout
        self._on_toggled(False)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self.toggle.setIcon(self._icon_expanded if checked else self._icon_collapsed)

    # ==================== 对外 ====================

    def content_layout(self) -> QVBoxLayout:
        """往里放东西用这个。"""
        return self._content_layout

    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        self._content_layout.addLayout(layout)

    def set_expanded(self, expanded: bool):
        self.toggle.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()
