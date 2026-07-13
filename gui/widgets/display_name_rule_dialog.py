# -*- coding: utf-8 -*-
"""图片显示名称规则设置对话框。

设置的是「界面上怎么叫这张图」的规则（存进 projects.display_name_rule），
不改文件名、不碰数据库 images 表、也不碰磁盘文件。三种模式怎么生成名字的
真正语义在 gui/display_names.py 里，这里只是把它接成一个能实时看预览的窗口。

这一版只是独立的对话框组件，还没有接到任何页面上——`selected_rule()` 拿到
规则之后要不要持久化、什么时候弹出，都由调用方决定。
"""

from typing import Dict, List, Optional, Sequence

from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout,
)

from gui.display_names import build_project_display_names, default_rule, validate_rule
from gui.styles import COLORS

# (mode, 单选项文案, 说明文案)
RULE_CHOICES = [
    ("original", "保留原名", "普通图片显示成原文件名；视频抽帧的图仍然显示成简短的“帧 223”。"),
    ("source", "按来源命名", "视频抽帧显示成“来源文件名_帧000223”；普通图片显示成“所在文件夹名_编号”。"),
    (
        "custom", "自定义前缀 + 编号",
        "自己填一个前缀，从指定编号开始，按项目里图片的顺序连续编号；"
        "前缀和编号之间会自动补一个下划线（前缀已经以下划线结尾就不重复），"
        "例如“阀门” + 起始 1 + 位数 5 → “阀门_00001”。",
    ),
]

PREVIEW_SAMPLE_SIZE = 3


class DisplayNameRuleDialog(QDialog):
    """三种模式选一个，实时看项目里前三张图会显示成什么名字。"""

    def __init__(
        self,
        parent=None,
        current_rule: Optional[Dict] = None,
        sample_images: Optional[Sequence[Dict]] = None,
        project_name: str = "",
        title: str = "图片显示名称规则",
    ):
        super().__init__(parent)
        # 预览样本要求调用方已经按项目图片的稳定顺序截好前几张，这里只管展示，
        # 不去访问数据库——保持这个对话框是纯 UI 组件，方便脱离真实项目测试。
        self.sample_images: List[Dict] = list(sample_images or [])[:PREVIEW_SAMPLE_SIZE]
        self.project_name = project_name
        self._selected_rule: Optional[Dict] = None

        self.setWindowTitle(title)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        heading = QLabel(title)
        heading.setObjectName("h2")
        layout.addWidget(heading)

        hint = QLabel("只改界面上显示的名字，不会重命名磁盘文件，也不影响标注和训练。")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.mode_group = QButtonGroup(self)
        self.radios: Dict[str, QRadioButton] = {}

        for mode, label, explain in RULE_CHOICES:
            radio = QRadioButton(label)
            self.mode_group.addButton(radio)
            layout.addWidget(radio)
            self.radios[mode] = radio

            note = QLabel(explain)
            note.setObjectName("caption")
            note.setWordWrap(True)
            note.setContentsMargins(24, 0, 0, 4)
            layout.addWidget(note)

        # 自定义模式的三个参数，缩进在“自定义前缀 + 编号”单选项下面
        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(24, 0, 0, 8)
        custom_row.addWidget(QLabel("前缀"))
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("例如 阀门 或 IMG_")
        custom_row.addWidget(self.prefix_input, 1)
        custom_row.addWidget(QLabel("起始编号"))
        self.start_input = QSpinBox()
        self.start_input.setRange(0, 999999)
        self.start_input.setValue(1)
        custom_row.addWidget(self.start_input)
        custom_row.addWidget(QLabel("位数"))
        self.digits_input = QSpinBox()
        self.digits_input.setRange(1, 10)
        self.digits_input.setValue(6)
        custom_row.addWidget(self.digits_input)
        layout.addLayout(custom_row)

        preview_box = QGroupBox(f"预览（前 {PREVIEW_SAMPLE_SIZE} 张）")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel()
        self.preview_label.setObjectName("caption")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_box)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {COLORS['error']};")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        layout.addSpacing(4)
        btn_row = QHBoxLayout()

        reset_btn = QPushButton("恢复默认")
        reset_btn.setObjectName("ghost")
        reset_btn.setToolTip("把选项恢复成“保留原名”；要点“应用”才会真正保存")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setObjectName("secondary")
        self.cancel_btn.setToolTip("放弃改动，不影响当前的显示名称规则")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("应用")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self.apply_btn)

        layout.addLayout(btn_row)

        # 默认焦点停在“取消”这个安全操作上，不是会产生副作用的“应用”，
        # 免得用户手滑按回车就把规则应用掉。
        self.cancel_btn.setDefault(True)
        self.cancel_btn.setFocus()

        self._apply_rule_to_inputs(self._initial_rule(current_rule))

        self.mode_group.buttonToggled.connect(lambda *_: self._refresh())
        self.prefix_input.textChanged.connect(self._refresh)
        self.start_input.valueChanged.connect(self._refresh)
        self.digits_input.valueChanged.connect(self._refresh)

        self._refresh()

    # ---- 规则 <-> 控件 ----

    @staticmethod
    def _initial_rule(current_rule: Optional[Dict]) -> Dict:
        if current_rule and validate_rule(current_rule) is None:
            return current_rule
        return default_rule()

    def _apply_rule_to_inputs(self, rule: Dict):
        mode = rule.get("mode", "original")
        self.radios.get(mode, self.radios["original"]).setChecked(True)
        if mode == "custom":
            self.prefix_input.setText(str(rule.get("prefix", "")))
            self.start_input.setValue(int(rule.get("start", 1)))
            self.digits_input.setValue(int(rule.get("digits", 6)))

    def _current_mode(self) -> str:
        for mode, radio in self.radios.items():
            if radio.isChecked():
                return mode
        return "original"

    def _current_rule(self) -> Dict:
        mode = self._current_mode()
        if mode == "custom":
            return {
                "mode": "custom",
                "prefix": self.prefix_input.text(),
                "start": self.start_input.value(),
                "digits": self.digits_input.value(),
            }
        return {"mode": mode}

    # ---- 交互 ----

    def _refresh(self, *_args):
        is_custom = self._current_mode() == "custom"
        self.prefix_input.setEnabled(is_custom)
        self.start_input.setEnabled(is_custom)
        self.digits_input.setEnabled(is_custom)

        rule = self._current_rule()
        error = validate_rule(rule)
        if error:
            self._show_error(error)
            return

        if not self.sample_images:
            self.preview_label.setText("项目还没有图片，暂无预览")
            self._show_error(None)
            return

        try:
            names = build_project_display_names(rule, self.sample_images, self.project_name)
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self.preview_label.setText("\n".join(names))
        self._show_error(None)

    def _show_error(self, message: Optional[str]):
        if message:
            self.error_label.setText(message)
            self.error_label.setVisible(True)
            self.preview_label.setText("规则不合法，暂时看不了预览")
            self.apply_btn.setEnabled(False)
        else:
            self.error_label.setVisible(False)
            self.apply_btn.setEnabled(True)

    def _on_reset(self):
        """只把控件恢复成“保留原名”并刷新预览；是否持久化由调用方在“应用”之后决定。"""
        self._apply_rule_to_inputs(default_rule())
        self._refresh()

    def _on_apply(self):
        rule = self._current_rule()
        error = validate_rule(rule)
        if error:
            self._show_error(error)
            return
        self._selected_rule = rule
        self.accept()

    # ---- 对外 ----

    def selected_rule(self) -> Optional[Dict]:
        """确认应用、且当时校验通过才有值；取消或还没点“应用”都是 None。"""
        return self._selected_rule
