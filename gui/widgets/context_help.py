# -*- coding: utf-8 -*-
"""轻量的行内帮助入口。

收起时只在右侧显示一个场景化的小入口；展开后才出现少量提示。组件不记状态——
每次打开页面都从收起开始。
"""

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.styles import COLORS, RADIUS_SM

_TOGGLE_KEYS = (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)


class _ToggleButton(QPushButton):
    """QPushButton 默认只有 Space 会触发点击；Enter 要显式接管。"""

    def sizeHint(self):
        """按钮里是图标 + 文字 + 箭头，尺寸提示必须把子布局算进去。"""
        hint = super().sizeHint()
        if self.layout() is not None:
            content = self.layout().sizeHint()
            hint.setWidth(max(hint.width(), content.width() + 16))
            hint.setHeight(max(32, content.height() + 10))
        return hint

    def minimumSizeHint(self):
        return self.sizeHint()

    def keyPressEvent(self, event):
        if event.key() in _TOGGLE_KEYS:
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)


_ASSETS_DIR = Path(__file__).parent.parent / "assets"


def _svg_pixmap(name: str, size: int):
    return QIcon(str(_ASSETS_DIR / name)).pixmap(QSize(size, size))


def _icon_label(name: str, size: int) -> QLabel:
    label = QLabel()
    label.setPixmap(_svg_pixmap(name, size))
    label.setFixedSize(size, size)
    # 图标只是装饰：点击要穿透到下面的整行按钮上
    label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    return label


def _warning_soft_background() -> str:
    """从 COLORS['warning_fill'] 派生一个克制的浅底色，不额外造新配色。"""
    hex_color = COLORS['warning_fill'].lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, 0.12)"


class ContextHelp(QFrame):
    """可复用的轻量帮助：小入口可点/可聚焦，展开后是 2–3 条提示。"""

    def __init__(self, steps, parent=None, risk_steps=None, title="查看提示"):
        super().__init__(parent)
        self.setObjectName("contextHelp")
        self.setStyleSheet(f"""
            QFrame#contextHelp {{
                background-color: transparent;
                border: none;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.header = QWidget(self)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        header_layout.addStretch(1)

        self.toggle = _ToggleButton(self.header)
        self.toggle.setObjectName("contextHelpToggle")
        self.toggle.setCheckable(True)
        self.toggle.setAutoDefault(False)
        self.toggle.setDefault(False)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.toggle.setAccessibleName(title)
        self.toggle.setStyleSheet(f"""
            QPushButton#contextHelpToggle {{
                background-color: transparent;
                color: {COLORS['accent_text']};
                border: 1px solid transparent;
                border-radius: {RADIUS_SM}px;
                text-align: left;
                padding: 5px 8px;
            }}
            QPushButton#contextHelpToggle:hover {{
                background-color: {COLORS['selected']};
            }}
            QPushButton#contextHelpToggle:focus {{
                background-color: transparent;
                border-color: rgba(0, 122, 255, 0.28);
            }}
            QPushButton#contextHelpToggle:checked {{
                background-color: {COLORS['selected']};
            }}
        """)

        row = QHBoxLayout(self.toggle)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        row.addWidget(_icon_label("book.svg", 14))

        self._label = QLabel(title)
        self._label.setObjectName("contextHelpTitle")
        self._label.setStyleSheet(f"color: {COLORS['accent_text']}; font-weight: 600; border: none;")
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(self._label)

        row.addStretch(1)

        self._chevron = _icon_label("chevron_down.svg", 16)
        row.addWidget(self._chevron)

        header_layout.addWidget(self.toggle)
        outer.addWidget(self.header)

        self.content = QFrame(self)
        self.content.setObjectName("contextHelpContent")
        self.content.setStyleSheet("""
            QFrame#contextHelpContent {
                background-color: rgba(0, 122, 255, 0.055);
                border: 1px solid rgba(0, 122, 255, 0.14);
                border-radius: 8px;
            }
        """)
        self.content.setVisible(False)
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(12, 10, 12, 10)
        content_layout.setSpacing(6)
        outer.addWidget(self.content)

        self._content_layout = content_layout

        self.toggle.toggled.connect(self._on_toggled)
        self.set_steps(steps, risk_steps=risk_steps)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self._chevron.setPixmap(
            _svg_pixmap("chevron_up.svg" if checked else "chevron_down.svg", 16)
        )

    def set_steps(self, steps, risk_steps=None):
        """替换提示；risk_steps 是需要单独警示的提示序号（从 1 开始）。"""
        risk_indices = set(risk_steps or [])

        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # deleteLater() 的实际销毁要等事件循环处理 DeferredDelete，
                # 但 setParent(None) 立刻把它从控件树摘掉，findChildren 不会再看到它
                widget.setParent(None)
                widget.deleteLater()

        for index, text in enumerate(steps, start=1):
            row = QFrame(self.content)
            row.setObjectName("contextHelpRiskRow" if index in risk_indices else "contextHelpTipRow")
            if index not in risk_indices:
                # 全局 QFrame 是白底卡片；普通提示行必须显式清掉，否则会变成
                # 「浅蓝帮助面板里再套三张白卡片」。
                row.setStyleSheet(
                    "QFrame#contextHelpTipRow { background-color: transparent; border: none; }"
                )
            row_layout = QHBoxLayout(row)
            row_margin_x = 6 if index in risk_indices else 0
            row_margin_y = 4 if index in risk_indices else 0
            row_layout.setContentsMargins(
                row_margin_x, row_margin_y, row_margin_x, row_margin_y
            )
            row_layout.setSpacing(7)

            marker = QLabel("!" if index in risk_indices else "•")
            marker.setObjectName("contextHelpMarker")
            marker.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            marker.setFixedWidth(14)
            marker.setStyleSheet(
                f"color: {COLORS['warning'] if index in risk_indices else COLORS['primary']}; "
                "font-weight: 700; border: none; background: transparent;"
            )
            row_layout.addWidget(marker)

            label = QLabel(text)
            label.setObjectName("contextHelpTipText")
            label.setWordWrap(True)
            if index in risk_indices:
                row.setStyleSheet(
                    f"QFrame#contextHelpRiskRow {{ background-color: {_warning_soft_background()}; "
                    f"border: 1px solid rgba(255, 149, 0, 0.18); border-radius: {RADIUS_SM}px; }}"
                )
                label.setStyleSheet(
                    f"color: {COLORS['warning']}; border: none; background: transparent;"
                )
            else:
                label.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; border: none; background: transparent;"
                )
            row_layout.addWidget(label, 1)
            self._content_layout.addWidget(row)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_expanded(self, expanded: bool):
        self.toggle.setChecked(expanded)
