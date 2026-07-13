# -*- coding: utf-8 -*-
"""
主流程相关的通用控件

StepNavItem / StepNav —— 侧边栏的流程导航，每一步显示序号、名称和进展数字
PageHeader           —— 每页顶部：当前是第几步、一句说明、下一步去哪
StepGate             —— 前置条件不满足时挡在页面前面：缺什么 + 一个补上的入口
NoticeBar            —— 窗口内的一条轻量通知，可关闭，不打断操作
EmptyState           —— 页面内的空状态：一句话 + 主按钮
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics

from gui.styles import COLORS, CONTROL_HEIGHT_LG
from gui.workflow import (
    WORKFLOW_STEPS, STEP_BY_INDEX,
    DONE, CURRENT, READY, LOCKED,
)

# 每种状态用一个字符表示，不堆图标：做完了 / 正在做 / 可以做 / 还进不去
STATE_MARK = {
    DONE: '✓',
    CURRENT: '●',
    READY: '○',
    LOCKED: '○',
}

STATE_COLOR = {
    DONE: COLORS['success'],
    CURRENT: COLORS['primary'],
    READY: COLORS['text_secondary'],
    LOCKED: COLORS['text_disabled'],
}


class StepNavItem(QPushButton):
    """侧边栏中的一步：一个把两行字装在自己里面的按钮。

    按钮里放布局有个坑：QPushButton 的 sizeHint / minimumSizeHint 只按它自己那段
    文本算（这里是空的），根本不看子布局。父布局照这个值给高度，里面两行字就被
    压扁——13px 的中文一行要 18px，控件却只有 11px，名称和状态上下都被削掉。
    QSS 的 padding 也帮不上忙：它只影响按钮自绘的文本，子布局的 contentsRect
    一点没变（实测 contentsRect == rect）。

    所以尺寸只认一个来源——子布局：
      · 内边距写在布局的 contentsMargins 里，QSS 那边不再写 padding；
      · sizeHint / minimumSizeHint 直接返回布局算出来的尺寸，字体多大、
        缩放多少、平台是不是 Cocoa，高度都跟着字体度量走。

    字体用 QFont 直接设，不写在样式表里：样式表里的 font-size 要等控件 polish
    之后才反映到 fontMetrics()，构造期按它算宽度会拿到应用默认字体的度量。
    QFont 一设就生效，度量当场可信——下面 mark 的宽度就是这么算出来的。
    """

    NAME_PX = 13
    STATUS_PX = 11

    def __init__(self, step: dict, parent=None):
        super().__init__(parent)
        self.step = step
        self.setObjectName("step_item")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        # 视觉内边距（原来写在 QSS 的 padding 里，对子布局无效）+ 1px 边框
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self.mark = QLabel()
        self.mark.setFont(self._sized_font(self.NAME_PX))
        # 宽度按字形算，不写死：'✓' 这类符号在 Cocoa 上会落到别的字体里，
        # 实际字宽跟着系统走。写死 16px 的话，字形一宽就被竖着切掉半个。
        mark_metrics = QFontMetrics(self.mark.font())
        widest_mark = max(
            mark_metrics.horizontalAdvance(m) for m in STATE_MARK.values()
        )
        self.mark.setMinimumWidth(widest_mark + 4)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mark)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)

        self.name = QLabel(f"{step['num']}. {step['name']}")
        self.name.setFont(self._sized_font(self.NAME_PX, bold=True))
        self.name.setStyleSheet(
            self._label_style(COLORS['text_primary'], self.NAME_PX, bold=True)
        )
        text_col.addWidget(self.name)

        self.status = QLabel("")
        self.status.setFont(self._sized_font(self.STATUS_PX))
        self.status.setStyleSheet(
            self._label_style(COLORS['text_secondary'], self.STATUS_PX)
        )
        text_col.addWidget(self.status)

        layout.addLayout(text_col)
        layout.addStretch()

    def _sized_font(self, pixel_size: int, bold: bool = False) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(pixel_size)
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
        return font

    @staticmethod
    def _label_style(color: str, pixel_size: int, bold: bool = False) -> str:
        """让导航字号不受应用级主题样式覆盖。

        应用样式表给 QWidget 设了默认字号，而 Qt 样式表的优先级高于
        ``QFont.setPixelSize``。如果局部样式只写颜色，名称和状态会被覆盖成同一
        字号，高 DPI 下布局缓存的高度也会跟着失真。因此把字号、字重和状态颜色
        放在同一条局部样式里，亮色/暗色主题刷新都不会改变组件自己的尺寸契约。
        """
        weight = 600 if bold else 400
        return (
            f"color: {color}; "
            f"font-size: {pixel_size}px; "
            f"font-weight: {weight};"
        )

    def sizeHint(self):
        return self.layout().sizeHint()

    def minimumSizeHint(self):
        return self.layout().minimumSize()

    def apply_state(self, state: str, status_text: str):
        self.mark.setText(STATE_MARK.get(state, '○'))
        self.mark.setStyleSheet(
            self._label_style(
                STATE_COLOR.get(state, COLORS['text_secondary']),
                self.NAME_PX,
            )
        )

        if state == LOCKED:
            name_color = COLORS['text_disabled']
        else:
            name_color = COLORS['text_primary']
        self.name.setStyleSheet(
            self._label_style(name_color, self.NAME_PX, bold=True)
        )

        self.status.setText(status_text)
        self.status.setStyleSheet(
            self._label_style(COLORS['text_secondary'], self.STATUS_PX)
        )

        # 状态文字从无到有会让内容变高，得让父布局重新问一次尺寸
        self.updateGeometry()


class StepNav(QWidget):
    """侧边栏的流程导航。锁住的步骤仍可点开——点进去会看到为什么进不去、怎么补。"""

    step_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        for step in WORKFLOW_STEPS:
            item = StepNavItem(step)
            item.clicked.connect(
                lambda _checked, index=step['index']: self.step_clicked.emit(index)
            )
            layout.addWidget(item)
            self.items[step['index']] = item

    def update_states(self, states: dict, status_texts: dict, current_index: int):
        for index, item in self.items.items():
            item.apply_state(states.get(index, LOCKED), status_texts.get(index, ''))
            item.setChecked(index == current_index)


class PageHeader(QWidget):
    """页面顶栏：你在第几步、这一步是干什么的、下一步去哪。"""

    next_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page_header")
        self._next_index = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(16)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        self.title = QLabel()
        self.title.setObjectName("h1")
        title_row.addWidget(self.title)

        self.badge = QLabel()
        self.badge.setObjectName("step_badge")
        title_row.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch()

        text_col.addLayout(title_row)

        # 说明只有一句：允许换行，但不给它撑高页头的机会
        self.desc = QLabel()
        self.desc.setObjectName("subtitle")
        self.desc.setWordWrap(True)
        text_col.addWidget(self.desc)

        layout.addLayout(text_col, 1)

        self.next_btn = QPushButton()
        self.next_btn.setObjectName("primary")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setMinimumHeight(CONTROL_HEIGHT_LG)
        # 按钮不能被长标题挤扁，也不该抢走标题的宽度
        self.next_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.next_btn.clicked.connect(self._on_next)
        self.next_btn.hide()
        layout.addWidget(self.next_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def _on_next(self):
        if self._next_index is not None:
            self.next_clicked.emit(self._next_index)

    def set_step(self, index: int, title: str = None, desc: str = None):
        step = STEP_BY_INDEX.get(index)
        if step:
            self.badge.show()
            self.badge.setText(f"{step['num']} / {len(WORKFLOW_STEPS)}")
            self.title.setText(title or step['name'])
            self.desc.setText(desc or step['desc'])
        else:
            self.badge.hide()
            self.title.setText(title or "")
            self.desc.setText(desc or "")
        self.desc.setVisible(bool(self.desc.text()))

    def set_next_action(self, action):
        """action: {'text', 'index'} 或 None"""
        if not action:
            self._next_index = None
            self.next_btn.hide()
            return
        self._next_index = action['index']
        self.next_btn.setText(action['text'])
        self.next_btn.show()


class NoticeBar(QFrame):
    """窗口内的一条轻量通知：一句话 + 可选详情 + 关闭。

    用来替代启动时的弹窗——有事说一句，没事根本不出现，
    任何情况下都不打断用户，也不代替用户做决定。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("notice")
        self.setStyleSheet(
            f"QFrame#notice {{"
            f"  background-color: {COLORS['selected']};"
            f"  border: none;"
            f"  border-bottom: 1px solid {COLORS['border']};"
            f"  border-radius: 0px;"
            f"}}"
        )
        self.hide()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 8, 16, 8)
        layout.setSpacing(10)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px;")
        layout.addWidget(self.text, 1)

        self.close_btn = QPushButton("知道了")
        self.close_btn.setObjectName("ghost")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.clicked.connect(self.hide)
        layout.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def show_message(self, message: str, detail: str = ""):
        self.text.setText(message)
        self.text.setToolTip(detail)
        self.show()


class StepGate(QWidget):
    """前置条件不满足时显示：缺什么、去哪补。不解释概念。"""

    goto_requested = pyqtSignal(int)
    bypass_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._goto_index = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch()

        card = QFrame()
        card.setObjectName("card")
        # 上限不是定值：容器比它窄时卡片要能跟着缩，否则右边会被切掉
        card.setMaximumWidth(460)
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(8)

        self.title = QLabel()
        self.title.setObjectName("h2")
        self.title.setWordWrap(True)
        card_layout.addWidget(self.title)

        self.reason = QLabel()
        self.reason.setObjectName("subtitle")
        self.reason.setWordWrap(True)
        card_layout.addWidget(self.reason)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 8, 0, 0)
        btn_row.setSpacing(8)

        self.goto_btn = QPushButton()
        self.goto_btn.setObjectName("primary")
        self.goto_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.goto_btn.setMinimumHeight(CONTROL_HEIGHT_LG)
        self.goto_btn.clicked.connect(self._on_goto)
        btn_row.addWidget(self.goto_btn)

        self.bypass_btn = QPushButton()
        self.bypass_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bypass_btn.setMinimumHeight(CONTROL_HEIGHT_LG)
        self.bypass_btn.clicked.connect(self.bypass_requested.emit)
        self.bypass_btn.hide()
        btn_row.addWidget(self.bypass_btn)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(card)
        row.addStretch()
        outer.addLayout(row)
        outer.addStretch()

    def _on_goto(self):
        if self._goto_index is not None:
            self.goto_requested.emit(self._goto_index)

    def set_blocker(self, blocker: dict):
        self.title.setText(blocker['title'])
        self.reason.setText(blocker['reason'])
        self._goto_index = blocker['action_index']
        self.goto_btn.setText(blocker['action_text'])

        bypass_text = blocker.get('bypass_text')
        if bypass_text:
            self.bypass_btn.setText(bypass_text)
            self.bypass_btn.show()
        else:
            self.bypass_btn.hide()


def _wrapping_label(text: str, width: int = 480) -> QLabel:
    """居中、会自动换行、并且高度算得对的说明文字。

    width 是上限不是定值：setFixedWidth 会让 min == max，容器比它窄的时候
    标签缩不下去，文字就被横向切掉——窗口拉到 1100 宽时，测试页右边那句
    「结果会显示在这里，同时保存到 outputs 文件夹」中间会缺一截。
    改成 maximumWidth 之后布局能把它压窄，heightForWidth 再按换行后的实际
    行数给高度，横竖都不会切。
    """
    label = QLabel(text)
    label.setObjectName("subtitle")
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMaximumWidth(width)

    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
    policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
    policy.setHeightForWidth(True)
    label.setSizePolicy(policy)
    return label


class EmptyState(QFrame):
    """页面内的空状态：标题 + 一句说明 + 一个主按钮（可选若干次按钮）。"""

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)

        layout.addStretch()

        self.title = QLabel(title)
        self.title.setObjectName("h2")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.message = _wrapping_label(message)
        msg_row = QHBoxLayout()
        msg_row.addStretch()
        msg_row.addWidget(self.message)
        msg_row.addStretch()
        layout.addLayout(msg_row)

        self.btn_row = QHBoxLayout()
        self.btn_row.setContentsMargins(0, 8, 0, 0)
        self.btn_row.setSpacing(10)
        self.btn_row.addStretch()
        self.btn_row.addStretch()
        layout.addLayout(self.btn_row)

        layout.addStretch()

    def add_action(self, text: str, callback, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        if primary:
            btn.setObjectName("primary")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(34)
        btn.clicked.connect(callback)
        self.btn_row.insertWidget(self.btn_row.count() - 1, btn)
        return btn

    def set_text(self, title: str, message: str):
        self.title.setText(title)
        self.message.setText(message)
