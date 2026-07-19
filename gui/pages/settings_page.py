# -*- coding: utf-8 -*-
"""
设置页面

辅助页，不属于主流程，所以视觉上比流程页轻：没有大标题（页头已给），
只有三组内容 + 一条底部操作栏。

    常用设置      —— 自动保存、预训练模型路径
    AI 与自动标注 —— 自动标注用的是什么模型，就地打开配置
    标注快捷键    —— 键位，改完立即生效

保存规则写在界面上，不让用户猜：
    快捷键改完立即写入；路径和自动保存要点「保存设置」才写入。
"""

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox, QCheckBox,
    QScrollArea, QFrame, QFileDialog, QInputDialog, QComboBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QKeySequence

from gui.styles import COLORS, THEME_CHOICES, THEME_LIGHT, normalize_theme
from gui.widgets.elided_label import ElidedLabel
from gui.pages.remote_training_profile_dialog import RemoteTrainingProfileDialog
from core.remote_training.profile_store import (
    RemoteTrainingProfileStore,
    RemoteTrainingProfileStoreError,
)

APP_ROOT = Path(__file__).parent.parent.parent
DEFAULT_PRETRAINED_PATH = APP_ROOT / "pretrained"
SAM_CONFIG_FILE = APP_ROOT / "config" / "sam_config.json"
LLM_CONFIG_FILE = APP_ROOT / "config" / "llm_config.json"

# 设置键与界面显示名
THEME_SETTING_KEY = 'theme'

# (设置键, 显示名, 默认键位)
SHORTCUTS = [
    ("rect_tool_shortcut", "矩形工具", "W"),
    ("poly_tool_shortcut", "多边形工具", "P"),
    ("move_tool_shortcut", "移动工具", "V"),
    ("prev_image_shortcut", "上一张图片", "A"),
    ("next_image_shortcut", "下一张图片", "D"),
    ("delete_shortcut", "删除标注", "DELETE"),
    ("reset_view_shortcut", "重置视图", "R"),
]

HINT_IDLE = "路径和自动保存的改动，点「保存设置」后才写入。"


def normalize_shortcut(text: str) -> str:
    """把用户输入的键位整理成配置里存的那一个字符串，认不出来就返回空串。

    单个字母/数字照旧转成大写；DELETE、BACKSPACE、SPACE、UP 这类键名交给 Qt 判断，
    这样默认的 DELETE 被改掉之后还能再输回来。
    组合键（Ctrl+S）不收：读快捷键的地方是拿单个键位的文本去比对的。
    """
    key = " ".join(text.split()).upper()
    if not key or "+" in key:
        return ""
    if len(key) == 1:
        return key

    seq = QKeySequence.fromString(key)
    if seq.count() != 1 or seq[0].key() == Qt.Key.Key_unknown:
        return ""
    return key


def shortcut_key_code(text: str) -> Optional[int]:
    """把存下来的键位字符串解析成 Qt 键码，认不出来返回 None。"""
    key = normalize_shortcut(text)
    if not key:
        return None

    seq = QKeySequence.fromString(key)
    if seq.count() != 1:
        return None

    code = seq[0].key()
    return None if code == Qt.Key.Key_unknown else code


def event_matches_shortcut(event, text: str) -> bool:
    """这次按键是不是这个快捷键。

    比的是键码，不是 event.text()。DELETE / SPACE / 方向键根本没有可打印字符
    （event.text() 分别是 '\x7f'、' '、''），拿文本去比永远匹配不上——设置页
    会照样存下来并显示「已经生效」，实际上键是哑的。
    单键快捷键带上 Ctrl/Alt/Meta 就不算，免得和 Ctrl+Z 这类组合键抢。
    """
    code = shortcut_key_code(text)
    if code is None or event.key() != code:
        return False

    blocked = (
        Qt.KeyboardModifier.ControlModifier
        | Qt.KeyboardModifier.AltModifier
        | Qt.KeyboardModifier.MetaModifier
    )
    return not (event.modifiers() & blocked)


def read_config(path: Path) -> dict:
    """读配置文件，只用于显示当前状态；读不到就当没配置过。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class SettingsPage(QWidget):
    """设置页面"""

    # 主题变化信号
    theme_changed = pyqtSignal(str)  # 发送新的主题名称

    # 请求打开自动标注配置：''=默认页，'sam'=SAM 分割，'llm'=LLM 视觉
    # 设置页自己不认识 AutoLabelDialog，由主窗口接住并打开标注页那一个实例
    auto_label_config_requested = pyqtSignal(str)

    # 训练页只需要刷新目标下拉；连接测试由主窗口/远程线程执行，设置页不直接做 I/O。
    remote_profiles_changed = pyqtSignal()
    remote_profile_test_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.settings = QSettings("EzYOLO", "Settings")
        self.shortcut_buttons = {}
        self.remote_profile_store = RemoteTrainingProfileStore(self.settings)
        self._remote_profiles = []
        self._remote_profile_test_running = False
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 20, 24, 8)
        body_layout.setSpacing(14)

        # 设置项本身很窄，让分组框铺满一整屏只会把几行字摊成一片空白。
        # 固定一列内容宽度（宽屏时留白在右侧），窄屏时照常缩。
        column = QWidget()
        column.setMaximumWidth(760)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(14)
        column_layout.addWidget(self.create_common_group())
        column_layout.addWidget(self.create_remote_training_group())
        column_layout.addWidget(self.create_ai_group())
        column_layout.addWidget(self.create_shortcut_group())

        body_layout.addWidget(column)
        body_layout.addStretch()

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        divider = QFrame()
        divider.setObjectName("divider")
        outer.addWidget(divider)
        outer.addWidget(self.create_action_bar())

    # ==================== 分组 ====================

    @staticmethod
    def _caption(text: str) -> QLabel:
        """行首说明列，右对齐，贴 macOS 表单习惯。"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def create_common_group(self) -> QGroupBox:
        """常用设置：自动保存 + 预训练模型路径

        不用 QFormLayout：QMacStyle 下它的值列宽度按 sizeHint 走，长值不跟着
        分组框伸缩，路径这类内容会过早截断。显式网格 + 列拉伸没有这个问题。
        """
        group = QGroupBox("常用设置")

        grid = QGridLayout(group)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.auto_save_enabled = QCheckBox("标注过程中自动保存")
        self.auto_save_enabled.setChecked(
            self.settings.value("auto_save_enabled", True, type=bool)
        )
        self.auto_save_enabled.toggled.connect(self.on_auto_save_toggled)
        grid.addWidget(self._caption("自动保存:"), 0, 0)
        grid.addWidget(self.auto_save_enabled, 0, 1, 1, 2)

        self.theme_combo = QComboBox()
        for theme_key, label in THEME_CHOICES:
            self.theme_combo.addItem(label, theme_key)
        saved_theme = normalize_theme(self.settings.value(THEME_SETTING_KEY, THEME_LIGHT))
        index = self.theme_combo.findData(saved_theme)
        self.theme_combo.setCurrentIndex(index if index >= 0 else 0)
        self.theme_combo.setMaximumWidth(260)
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        grid.addWidget(self._caption("外观:"), 1, 0)
        grid.addWidget(self.theme_combo, 1, 1, 1, 2,
                       Qt.AlignmentFlag.AlignLeft)

        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setSuffix(" 分钟")
        self.auto_save_interval.setValue(int(self.settings.value("auto_save_interval", 5)))
        self.auto_save_interval.setEnabled(self.auto_save_enabled.isChecked())
        self.auto_save_interval.setMaximumWidth(260)
        self.auto_save_interval.valueChanged.connect(self.mark_dirty)
        grid.addWidget(self._caption("保存间隔:"), 2, 0)
        grid.addWidget(self.auto_save_interval, 2, 1, 1, 2,
                       Qt.AlignmentFlag.AlignLeft)

        self._pretrained_path_value = str(
            self.settings.value("pretrained_path", str(DEFAULT_PRETRAINED_PATH))
        )
        self.pretrained_path = ElidedLabel(mode=Qt.TextElideMode.ElideMiddle)
        self._refresh_pretrained_path_display()
        grid.addWidget(self._caption("预训练模型:"), 3, 0)
        grid.addWidget(self.pretrained_path, 3, 1)

        btn_browse = QPushButton("浏览…")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(
            lambda: self.browse_path("pretrained_path", "选择预训练模型目录")
        )
        grid.addWidget(btn_browse, 3, 2)

        return group

    def _refresh_pretrained_path_display(self):
        """路径可能很长：标签按可用宽度中部省略，完整路径和用途说明放 tooltip。"""
        self.pretrained_path.setText(self._pretrained_path_value)
        self.pretrained_path.setToolTip(
            f"{self._pretrained_path_value}\n训练时从这个目录读取 YOLO 预训练权重（.pt 文件）。"
        )

    def create_remote_training_group(self) -> QGroupBox:
        """远程训练服务器：只管理公开连接档案，绝不在设置中保存认证秘密。"""
        group = QGroupBox("远程训练服务器")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        hint = QLabel(
            "可选功能。登录认证由系统 OpenSSH / ssh-agent 管理；这里不保存密码、私钥或口令。"
        )
        hint.setObjectName("caption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.remote_profile_combo = QComboBox()
        self.remote_profile_combo.setObjectName("remote_training_profile_combo")
        self.remote_profile_combo.setToolTip("选择一个已保存的远程训练服务器档案。")
        self.remote_profile_combo.currentIndexChanged.connect(self._refresh_remote_profile_actions)
        row.addWidget(self.remote_profile_combo, 1)

        self.btn_add_remote_profile = QPushButton("添加…")
        self.btn_add_remote_profile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_remote_profile.clicked.connect(self.add_remote_profile)
        row.addWidget(self.btn_add_remote_profile)

        self.btn_edit_remote_profile = QPushButton("编辑…")
        self.btn_edit_remote_profile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_remote_profile.clicked.connect(self.edit_remote_profile)
        row.addWidget(self.btn_edit_remote_profile)

        self.btn_delete_remote_profile = QPushButton("移除")
        self.btn_delete_remote_profile.setObjectName("danger")
        self.btn_delete_remote_profile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_remote_profile.clicked.connect(self.delete_remote_profile)
        row.addWidget(self.btn_delete_remote_profile)
        layout.addLayout(row)

        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 0, 0, 0)
        self.btn_test_remote_profile = QPushButton("测试连接")
        self.btn_test_remote_profile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_test_remote_profile.setToolTip(
            "只读取服务器能力信息，不上传数据、不创建任务，也不会开始训练。"
        )
        self.btn_test_remote_profile.clicked.connect(self.request_remote_profile_test)
        test_row.addWidget(self.btn_test_remote_profile)
        test_row.addStretch()
        layout.addLayout(test_row)

        self.remote_profile_status = QLabel("")
        self.remote_profile_status.setObjectName("caption")
        self.remote_profile_status.setWordWrap(True)
        layout.addWidget(self.remote_profile_status)

        self.refresh_remote_profiles()
        return group

    @staticmethod
    def _remote_profile_text(profile) -> str:
        return f"{profile.name} · {profile.username}@{profile.host}:{profile.port}"

    def refresh_remote_profiles(self, selected_id: str | None = None) -> None:
        """从严格存储恢复档案；损坏数据 fail closed，不偷偷跳过未知字段。"""
        current_id = selected_id or self.remote_profile_combo.currentData()
        try:
            profiles = self.remote_profile_store.list()
        except RemoteTrainingProfileStoreError:
            self._remote_profiles = []
            self.remote_profile_combo.blockSignals(True)
            self.remote_profile_combo.clear()
            self.remote_profile_combo.addItem("远程服务器档案无法读取")
            self.remote_profile_combo.blockSignals(False)
            self.remote_profile_status.setText(
                "已保存的远程服务器档案格式不安全，未加载。请联系管理员或重新添加档案。"
            )
            self.remote_profile_status.setStyleSheet(f"color: {COLORS['error']};")
            self._refresh_remote_profile_actions()
            return

        self._remote_profiles = profiles
        self.remote_profile_combo.blockSignals(True)
        self.remote_profile_combo.clear()
        if not profiles:
            self.remote_profile_combo.addItem("还没有远程服务器")
        else:
            for profile in profiles:
                self.remote_profile_combo.addItem(
                    self._remote_profile_text(profile),
                    profile.id,
                )
            selected_index = self.remote_profile_combo.findData(current_id)
            self.remote_profile_combo.setCurrentIndex(
                selected_index if selected_index >= 0 else 0
            )
        self.remote_profile_combo.blockSignals(False)

        if profiles:
            self.remote_profile_status.setText(
                f"已保存 {len(profiles)} 台服务器。测试连接只做只读预检，不会上传数据或开始训练。"
            )
            self.remote_profile_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        else:
            self.remote_profile_status.setText("还没有服务器档案；训练页会继续只显示本地训练。")
            self.remote_profile_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self._refresh_remote_profile_actions()

    def selected_remote_profile(self):
        profile_id = self.remote_profile_combo.currentData()
        return next(
            (profile for profile in self._remote_profiles if profile.id == profile_id),
            None,
        )

    def _refresh_remote_profile_actions(self, _index: int | None = None) -> None:
        selected = self.selected_remote_profile()
        enabled = selected is not None
        self.btn_edit_remote_profile.setEnabled(enabled)
        self.btn_delete_remote_profile.setEnabled(enabled)
        self.btn_test_remote_profile.setEnabled(
            enabled and not self._remote_profile_test_running
        )
        if selected is not None:
            self.remote_profile_combo.setToolTip(
                f"{self._remote_profile_text(selected)}\n服务器目录：{selected.remote_root}"
            )

    def add_remote_profile(self) -> None:
        dialog = RemoteTrainingProfileDialog(parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.saved_profile is None:
            return
        try:
            self.remote_profile_store.save([*self._remote_profiles, dialog.saved_profile])
        except RemoteTrainingProfileStoreError:
            self.set_status("远程服务器档案未能保存。", 'warning')
            return
        self.refresh_remote_profiles(dialog.saved_profile.id)
        self.remote_profiles_changed.emit()
        self.set_status("远程服务器档案已添加。", 'success')

    def edit_remote_profile(self) -> None:
        selected = self.selected_remote_profile()
        if selected is None:
            return
        dialog = RemoteTrainingProfileDialog(selected, parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.saved_profile is None:
            return
        profiles = [
            dialog.saved_profile if profile.id == selected.id else profile
            for profile in self._remote_profiles
        ]
        try:
            self.remote_profile_store.save(profiles)
        except RemoteTrainingProfileStoreError:
            self.set_status("远程服务器档案未能保存。", 'warning')
            return
        self.refresh_remote_profiles(dialog.saved_profile.id)
        self.remote_profiles_changed.emit()
        self.set_status("远程服务器档案已更新。", 'success')

    def delete_remote_profile(self) -> None:
        selected = self.selected_remote_profile()
        if selected is None:
            return
        reply = QMessageBox.question(
            self,
            "移除远程服务器",
            f"确定移除“{selected.name}”吗？\n\n"
            "这只会删除本机保存的公开连接档案，不会删除服务器上的任何数据或任务。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self.remote_profile_store.delete(selected.id)
        except RemoteTrainingProfileStoreError:
            self.set_status("远程服务器档案未能移除。", 'warning')
            return
        self.refresh_remote_profiles()
        self.remote_profiles_changed.emit()
        self.set_status("远程服务器档案已移除。", 'success')

    def request_remote_profile_test(self) -> None:
        selected = self.selected_remote_profile()
        if selected is None:
            return
        self._remote_profile_test_running = True
        self._refresh_remote_profile_actions()
        self.remote_profile_status.setText("正在进行只读连接预检…")
        self.remote_profile_status.setStyleSheet(f"color: {COLORS['accent_text']};")
        self.remote_profile_test_requested.emit(selected)

    def set_remote_profile_test_status(self, text: str, *, success: bool | None) -> None:
        if success is not None:
            self._remote_profile_test_running = False
        self.remote_profile_status.setText(text)
        color = (
            COLORS['success']
            if success is True
            else COLORS['error']
            if success is False
            else COLORS['accent_text']
        )
        self.remote_profile_status.setStyleSheet(f"color: {color};")
        self._refresh_remote_profile_actions()

    def create_ai_group(self) -> QGroupBox:
        """AI 与自动标注：显示当前用的是什么，并且就地打开配置

        状态值单行显示、装不下省略并给 tooltip——QMacStyle 下 QFormLayout
        配换行标签会算错行高，文字折行后裁切、上下重叠，所以不走表单布局。

        原来这里只有一句「配置入口在数据标注 → 自动标注」：用户看得到状态，
        却要自己走到另一个页面、在工具栏里找那个按钮。现在按钮就在状态旁边，
        点了直接开同一个配置窗口（AutoLabelDialog），关掉后状态跟着刷新。
        """
        group = QGroupBox("AI 与自动标注")

        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.sam_status = ElidedLabel()
        grid.addWidget(self._caption("分割模型:"), 0, 0)
        grid.addWidget(self.sam_status, 0, 1)

        self.btn_config_sam = QPushButton("配置…")
        self.btn_config_sam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_config_sam.setToolTip("打开自动标注配置，直接跳到 SAM 分割那一页")
        self.btn_config_sam.clicked.connect(
            lambda: self.auto_label_config_requested.emit('sam')
        )
        grid.addWidget(self.btn_config_sam, 0, 2)

        self.llm_status = ElidedLabel()
        grid.addWidget(self._caption("视觉大模型:"), 1, 0)
        grid.addWidget(self.llm_status, 1, 1)

        self.btn_config_llm = QPushButton("配置…")
        self.btn_config_llm.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_config_llm.setToolTip("打开自动标注配置，直接跳到 LLM 视觉那一页（填 API Key）")
        self.btn_config_llm.clicked.connect(
            lambda: self.auto_label_config_requested.emit('llm')
        )
        grid.addWidget(self.btn_config_llm, 1, 2)

        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self.btn_open_auto_label = QPushButton("打开自动标注配置")
        self.btn_open_auto_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_auto_label.setToolTip(
            "YOLO 预标注、SAM 分割、LLM 视觉都在这个窗口里配"
        )
        self.btn_open_auto_label.clicked.connect(
            lambda: self.auto_label_config_requested.emit('')
        )
        btn_row.addWidget(self.btn_open_auto_label)

        hint = QLabel("和「数据标注 → 自动标注」打开的是同一个配置窗口。")
        hint.setObjectName("caption")
        hint.setWordWrap(True)
        btn_row.addWidget(hint, 1)

        layout.addLayout(btn_row)

        self.refresh_ai_status()
        return group

    def create_shortcut_group(self) -> QGroupBox:
        """标注快捷键：自己一组。

        原来它是「外观与其他」卡片里再套一个标题——卡片里嵌一个二级标题，
        层级读不出来。外观那一组只剩一行「主题：浅色主题」，既改不了也不用看，
        一并去掉；应用固定浅色这件事不需要一行常驻文字来说。
        """
        group = QGroupBox("标注快捷键")

        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        shortcut_hint = QLabel("点键位修改，立即生效。")
        shortcut_hint.setObjectName("caption")
        shortcut_hint.setWordWrap(True)
        layout.addWidget(shortcut_hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        for i, (setting_key, name, default) in enumerate(SHORTCUTS):
            row, col = divmod(i, 2)

            label = QLabel(name)
            grid.addWidget(label, row, col * 3)

            button = QPushButton(str(self.settings.value(setting_key, default)))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumWidth(88)
            button.setToolTip("点击修改快捷键")
            button.clicked.connect(
                lambda _checked, key=setting_key: self.set_shortcut(key)
            )
            grid.addWidget(button, row, col * 3 + 1)

            grid.setColumnStretch(col * 3 + 2, 1)
            self.shortcut_buttons[setting_key] = button

        layout.addLayout(grid)
        return group

    def create_action_bar(self) -> QWidget:
        """底部操作栏：状态说明 + 两个按钮"""
        bar = QWidget()

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 12, 24, 16)
        layout.setSpacing(10)

        self.status_label = QLabel(HINT_IDLE)
        self.status_label.setObjectName("caption")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label, 1)

        self.btn_reset = QPushButton("恢复默认")
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_settings)
        layout.addWidget(self.btn_reset)

        self.btn_save = QPushButton("保存设置")
        self.btn_save.setObjectName("primary")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.save_settings)
        layout.addWidget(self.btn_save)

        return bar

    # ==================== 状态 ====================

    def showEvent(self, event):
        """每次进设置页都重新读一次自动标注配置，显示的状态才是真的。"""
        super().showEvent(event)
        self.refresh_ai_status()

    def refresh_ai_status(self):
        """把自动标注当前用的模型显示出来（只读，不写配置）。"""
        sam = read_config(SAM_CONFIG_FILE)
        if sam:
            model_file = sam.get("model_file") or "未指定权重"
            self.sam_status.setText(f"{sam.get('sam_type', 'SAM')} · {model_file}")
            self.sam_status.setStyleSheet(f"color: {COLORS['text_primary']};")
        else:
            self.sam_status.setText("还没配置过，首次自动标注时用默认设置")
            self.sam_status.setStyleSheet(f"color: {COLORS['text_secondary']};")

        llm = read_config(LLM_CONFIG_FILE)
        model_name = llm.get("model_name") or ""
        if not model_name:
            self.llm_status.setText("还没配置过")
            self.llm_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        elif llm.get("api_key"):
            self.llm_status.setText(f"{model_name} · 已填写 API Key")
            self.llm_status.setStyleSheet(f"color: {COLORS['text_primary']};")
        else:
            self.llm_status.setText(f"{model_name} · 还没填 API Key，用它标注前要先填")
            self.llm_status.setStyleSheet(f"color: {COLORS['warning']};")

    def set_status(self, text: str, tone: str = 'muted'):
        """底部那行字：说清楚现在是「改了没存」还是「已经存了」。"""
        color = {
            'success': COLORS['success'],
            'warning': COLORS['warning'],
        }.get(tone, COLORS['text_secondary'])

        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 12px; color: {color};")

    def mark_dirty(self, *_args):
        self.set_status("有改动还没保存，点「保存设置」写入。", 'warning')

    def on_theme_changed(self, _index: int = None):
        """外观切换后立即生效并写入设置。"""
        theme_key = self.theme_combo.currentData() or THEME_LIGHT
        self.settings.setValue(THEME_SETTING_KEY, theme_key)
        self.theme_changed.emit(theme_key)
        self.set_status(
            f"已切换到{'暗色' if theme_key == 'dark' else '浅色'}外观。",
            'success',
        )

    def refresh_theme(self):
        """主题切换后刷新本页内联样式。"""
        self.refresh_ai_status()

    def on_auto_save_toggled(self, checked: bool):
        self.auto_save_interval.setEnabled(checked)
        self.mark_dirty()

    # ==================== 读写 ====================

    def set_shortcut(self, setting_key: str):
        """设置快捷键（立即写入）"""
        button = self.shortcut_buttons[setting_key]

        key, ok = QInputDialog.getText(
            self, "设置快捷键",
            "请输入新的快捷键：\n单个字母或数字（如 W、3），"
            "或键位名（如 DELETE、BACKSPACE、SPACE、UP）",
            text=button.text(),
        )
        if not (ok and key):
            return

        new_key = normalize_shortcut(key)
        if not new_key:
            if "+" in key:
                reason = "标注快捷键只收单个键，不支持组合键"
            else:
                reason = "不是认得的键位"
            self.set_status(f"「{key}」{reason}，快捷键没改。", 'warning')
            return

        self.settings.setValue(setting_key, new_key)
        button.setText(new_key)
        self.set_status(f"快捷键已改为 {new_key}，已经生效。", 'success')

    def browse_path(self, setting_key: str, dialog_title: str):
        """浏览路径"""
        path = QFileDialog.getExistingDirectory(
            self, dialog_title,
            self.settings.value(setting_key, ""),
            QFileDialog.Option.ShowDirsOnly
        )

        if path:
            if setting_key == "pretrained_path":
                self._pretrained_path_value = path
                self._refresh_pretrained_path_display()
                self.mark_dirty()

    def save_settings(self):
        """保存设置"""
        theme_key = self.theme_combo.currentData() or THEME_LIGHT
        self.settings.setValue(THEME_SETTING_KEY, theme_key)
        self.theme_changed.emit(theme_key)

        # 保存路径
        self.settings.setValue("pretrained_path", self._pretrained_path_value)

        # 保存自动保存设置
        self.settings.setValue("auto_save_interval", self.auto_save_interval.value())
        self.settings.setValue("auto_save_enabled", self.auto_save_enabled.isChecked())

        self.set_status("设置已保存。", 'success')

    def reset_settings(self):
        """恢复默认设置"""
        self.auto_save_enabled.setChecked(True)
        self.auto_save_interval.setValue(5)
        self._pretrained_path_value = str(DEFAULT_PRETRAINED_PATH)
        self._refresh_pretrained_path_display()

        # 快捷键和别处一样：改完立即写入
        for setting_key, _name, default in SHORTCUTS:
            self.settings.setValue(setting_key, default)
            self.shortcut_buttons[setting_key].setText(default)

        light_index = self.theme_combo.findData(THEME_LIGHT)
        if light_index >= 0:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(light_index)
            self.theme_combo.blockSignals(False)
        self.settings.setValue(THEME_SETTING_KEY, THEME_LIGHT)
        self.theme_changed.emit(THEME_LIGHT)

        self.set_status("已恢复默认值，路径和自动保存点「保存设置」后写入。", 'warning')
