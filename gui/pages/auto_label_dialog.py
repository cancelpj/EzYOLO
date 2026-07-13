# -*- coding: utf-8 -*-
"""
自动打标签弹窗
用于设置自动打标签的参数和选项

每个标签页都按同一条线索排：模型/服务 → 参数 → （可折叠的高级项），
底部固定一条「将要发生什么」的预览 + 风险/执行状态，主操作只有「保存设置」一个。
"""

from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QRadioButton, QDoubleSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QSplitter, QMessageBox,
    QFileDialog, QScrollArea, QWidget, QTabWidget, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import os
import json
from typing import Dict, List, Optional

from gui.styles import COLORS, CONTROL_HEIGHT_LG, RADIUS_SM
from gui.widgets.app_dialog import confirm, show_warning
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.context_help import ContextHelp

# LLM配置文件路径
LLM_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'llm_config.json')
SAM_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'sam_config.json')
SAM3_DOWNLOAD_URL = "https://huggingface.co/1038lab/sam3/discussions/1"

# 默认LLM配置
DEFAULT_LLM_CONFIG = {
    'api_key': '',
    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'model_name': 'qwen-vl-max',
    'system_prompt': '''你是面向计算机视觉数据集的目标检测标注专家，仅输出指定目标的边界框坐标。
核心要求：
1. 目标类别：仅处理用户指定的类别，需找出图片中所有该类别实例；
2. 坐标格式：每个边界框以 [xmin, ymin, xmax, ymax] 格式输出；
3. 输出格式：每行一个目标，格式为 "标签,[xmin,ymin,xmax,ymax]"；
4. 无目标时输出空内容；
5. 仅返回坐标数据，无任何说明文字。''',
    'user_prompt': '''请检测图片中的所有 {target}，按以下格式返回每行一个：
{target},[xmin,ymin,xmax,ymax]
{target},[xmin,ymin,xmax,ymax]
...'''
}

DEFAULT_SAM_CONFIG = {
    "sam_type": "SAM",
    "model_file": "sam_b.pt",
    "device": "cpu",
    "imgsz": 1024,
    "conf": 0.4,
    "iou": 0.9,
    "retina_masks": True,
    "usage_mode": "normal",
}

# 真实的Ultralytics模型配置（从train_page.py获取）
ULTRALYTICS_MODELS = {
    "YOLOv3": {
        "sizes": ["n", "u"],
        "tasks": ["detect"],
        "prefix": "yolov3",
    },
    "YOLOv5": {
        "sizes": ["nu", "su", "mu", "lu", "xu"],
        "tasks": ["detect"],
        "prefix": "yolov5",
    },
    "YOLOv8": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "obb", "pose", "segment", "world"],
        "prefix": "yolov8",
    },
    "YOLOv9": {
        "sizes": ["t", "s", "m", "c", "e"],
        "tasks": ["detect"],
        "prefix": "yolov9",
    },
    "YOLOv10": {
        "sizes": ["n", "s", "m", "b", "l", "x"],
        "tasks": ["detect"],
        "prefix": "yolov10",
    },
    "YOLOv11": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "obb", "pose", "segment"],
        "prefix": "yolo11",
    },
    "YOLOv12": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect"],
        "prefix": "yolo12",
    },
    "YOLOv26": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "obb", "pose", "segment"],
        "prefix": "yolo26",
    },
}

# SAM模型配置
SAM_MODELS = {
    "SAM": {
        "models": {
            "SAM base": "sam_b.pt",
            "SAM large": "sam_l.pt"
        }
    },
    "SAM2": {
        "models": {
            "SAM 2 tiny": "sam2_t.pt",
            "SAM 2 small": "sam2_s.pt",
            "SAM 2 base": "sam2_b.pt",
            "SAM 2 large": "sam2_l.pt",
            "SAM 2.1 tiny": "sam2.1_t.pt",
            "SAM 2.1 small": "sam2.1_s.pt",
            "SAM 2.1 base": "sam2.1_b.pt",
            "SAM 2.1 large": "sam2.1_l.pt"
        }
    },
    "MobileSAM": {
        "models": {
            "MobileSAM": "mobile_sam.pt"
        }
    },
    "FastSAM": {
        "models": {
            "FastSAM-s": "FastSAM-s.pt",
            "FastSAM-x": "FastSAM-x.pt"
        }
    },
    "SAM3": {
        "models": {
            "SAM 3": "sam3.pt"
        }
    }
}

# 型号显示名称
SIZE_NAMES = {
    "n": "nano (超轻量)",
    "s": "small (轻量)",
    "m": "medium (中等)",
    "l": "large (大)",
    "x": "xlarge (超大)",
    "nu": "nano-u (超轻量新版)",
    "su": "small-u (轻量新版)",
    "mu": "medium-u (中等新版)",
    "lu": "large-u (大新版)",
    "xu": "xlarge-u (超大新版)",
    "tiny": "tiny (超轻量)",
    "t": "tiny (超轻量)",
    "c": "compact (紧凑)",
    "e": "extended (扩展)",
    "b": "balanced (平衡)",
    "u": "ultra (超大)",
}


def _sam_model_exists(model_file: str) -> bool:
    """SAM 模型文件是否已经在本地（下载逻辑找的就是这几个位置）。"""
    if not model_file:
        return False
    candidates = [
        model_file,
        os.path.join('models', model_file),
        os.path.join(os.path.expanduser('~'), '.cache', 'ultralytics', model_file),
    ]
    return any(os.path.exists(path) for path in candidates)


class AutoLabelDialog(QDialog):
    """自动打标签弹窗"""

    # 信号定义
    single_inference_requested = pyqtSignal(str, float, float, dict, str, str)  # 单张推理请求
    batch_inference_requested = pyqtSignal(str, float, float, dict, list, bool, str)  # 批量推理请求

    def __init__(self, parent=None, project_classes=None):
        super().__init__(parent)
        self.setWindowTitle("自动打标签设置")
        self.setMinimumSize(700, 560)

        # 当前项目类别
        self.project_classes = project_classes or []

        # 模型信息
        self.selected_model_version = "YOLOv8"
        self.selected_model_size = "n"
        self.model_source = "official"  # official or custom
        self.custom_model_path = ""

        # 推理参数
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.infer_only_unlabeled = True
        self.overwrite_labels = False

        # 类别映射
        self.class_mappings = {}

        # 初始化UI
        self.init_ui()

        # 加载SAM配置
        self.load_sam_config()

        # 加载LLM配置
        self.load_llm_config()

        # 底部预览随当前标签页刷新
        self.update_preview()

    def init_ui(self):
        """初始化界面：一句话说明 → 使用说明 → 标签页（滚动）→ 固定的配置摘要 → 按钮。

        摘要和按钮不进滚动区，也不浮在内容上面：它们是主布局里的独立行，
        滚动区永远停在它们上边，不会被盖住。
        """
        self.setObjectName("autoLabelDialog")
        self.setStyleSheet(self._dialog_stylesheet())

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 16)
        main_layout.setSpacing(12)

        # 顶部只留一句结果说明，易错点收进轻量帮助
        subtitle = QLabel("选择标注方式并保存，标注页会立即使用这里的配置。")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        main_layout.addWidget(subtitle)

        self.context_help = ContextHelp(
            [
                "YOLO 生成检测框，SAM 分割轮廓，LLM 根据文字描述识别目标。",
                "置信度越高，结果通常越少；批量运行前请先确认模型和范围。",
                "覆盖原标签会替换已有标注，且无法撤销。",
            ],
            risk_steps=[3],
            title="配置提示",
        )
        main_layout.addWidget(self.context_help)

        # 三种标注方式各一个标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("autoLabelTabs")
        self.tab_widget.addTab(self._wrap_scroll(self.create_yolo_tab()), "YOLO 检测")
        self.tab_widget.addTab(self._wrap_scroll(self.create_sam_tab()), "SAM 分割")
        self.tab_widget.addTab(self._wrap_scroll(self.create_llm_tab()), "LLM 视觉")
        self.tab_widget.currentChanged.connect(lambda _: self.update_preview())

        main_layout.addWidget(self.tab_widget, 1)

        # 摘要区：一条分隔线划清「上面是可滚动的设置，下面是固定的结论」
        divider = QFrame()
        divider.setObjectName("divider")
        main_layout.addWidget(divider)

        # 预览确认 + 风险/执行状态
        self.lbl_preview = QLabel()
        self.lbl_preview.setWordWrap(True)
        main_layout.addWidget(self.lbl_preview)

        self.lbl_notice = QLabel()
        self.lbl_notice.setWordWrap(True)
        main_layout.addWidget(self.lbl_notice)

        # 按钮组：主操作只有「保存设置」。两个按钮同高同内边距，基线才对得齐
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("secondary")
        self.btn_cancel.setToolTip("放弃修改，保持原有设置")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("保存设置")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self.on_save_clicked)
        button_layout.addWidget(self.btn_save)

        main_layout.addLayout(button_layout)

        # 焦点落在标签页上，而不是「使用说明」：说明是可以 Tab 过去、Enter 展开的，
        # 但一打开就把焦点（和 Enter）交给它，等于把这一页的主线让给了帮助。
        self.tab_widget.setFocus()

    def _dialog_stylesheet(self) -> str:
        """只作用于本弹窗的样式：去掉白底套白底，把选中态说清楚。

        标签页内容底色改成画布色，白色的分组卡片才有「卡片」的样子；
        模型来源的两个选项各自是一块可选区域——选中 = 圆点实心 + 标题加粗 + 轻背景，
        三个信号一起说同一件事，不再只靠一个蓝点。

        底部两个按钮的高度写在样式表里而不是 setMinimumHeight()：
        控件一旦被样式表接管，QSS 里的 min-height 会盖掉代码设的最小高度。
        22 + 上下 padding 5 + 上下 border 1 = CONTROL_HEIGHT_LG。
        """
        return f"""
            QDialog#autoLabelDialog QPushButton#primary,
            QDialog#autoLabelDialog QPushButton#secondary {{
                min-height: {CONTROL_HEIGHT_LG - 12}px;
                min-width: 96px;
                padding: 5px 14px;
            }}
            QTabWidget#autoLabelTabs::pane {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: {RADIUS_SM}px;
                top: -1px;
            }}
            QTabWidget#autoLabelTabs QTabBar::tab:selected {{
                background-color: {COLORS['background']};
                border-bottom-color: {COLORS['background']};
            }}
            QFrame#sourceOption {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: {RADIUS_SM}px;
            }}
            QFrame#sourceOption[selected="true"] {{
                background-color: {COLORS['selected']};
                border-color: {COLORS['border']};
            }}
            QWidget#optionDetail {{
                background-color: transparent;
            }}
        """

    def _wrap_scroll(self, content: QWidget) -> QScrollArea:
        """标签页内容放进滚动区：窗口再小也只是出滚动条，不截断。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def create_yolo_tab(self) -> QWidget:
        """创建YOLO自动标注页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # 来源 → 参数 → 高级
        layout.addWidget(self.create_model_selection_group())
        layout.addWidget(self.create_inference_params_group())
        layout.addWidget(self.create_class_mapping_section())

        layout.addStretch()
        return tab

    def create_sam_tab(self) -> QWidget:
        """创建SAM自动标注页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        layout.addWidget(self.create_sam_model_group())
        layout.addWidget(self.create_sam_params_group())
        layout.addWidget(self.create_sam_usage_group())

        layout.addStretch()
        return tab

    def create_model_selection_group(self) -> QGroupBox:
        """创建模型选择组：先定来源，再定具体模型"""
        group = QGroupBox("模型来源")

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # 官方预训练模型
        self.rbtn_official = QRadioButton("官方预训练模型（首次使用会自动下载）")
        self.rbtn_official.setChecked(True)
        self.rbtn_official.toggled.connect(self.on_model_source_changed)

        # 官方模型的版本 / 型号 / 任务
        self.official_widget = QWidget()
        self.official_widget.setObjectName("optionDetail")
        official_layout = QFormLayout(self.official_widget)
        official_layout.setContentsMargins(24, 0, 0, 0)

        self.cb_model_version = QComboBox()
        self.cb_model_version.addItems(sorted(ULTRALYTICS_MODELS.keys()))
        self.cb_model_version.currentTextChanged.connect(self.on_model_version_changed)
        official_layout.addRow("模型版本:", self.cb_model_version)

        self.cb_model_size = QComboBox()
        self.cb_model_size.currentTextChanged.connect(lambda _: self.update_preview())
        official_layout.addRow("型号:", self.cb_model_size)

        self.cb_model_task = QComboBox()
        self.cb_model_task.currentTextChanged.connect(lambda _: self.update_preview())
        official_layout.addRow("任务类型:", self.cb_model_task)

        self.official_option = self._source_option(self.rbtn_official, self.official_widget)
        layout.addWidget(self.official_option)

        # 自定义模型
        self.rbtn_custom = QRadioButton("自定义模型（本地 .pt / .pth 文件）")
        self.rbtn_custom.toggled.connect(self.on_model_source_changed)

        self.custom_widget = QWidget()
        self.custom_widget.setObjectName("optionDetail")
        custom_layout = QHBoxLayout(self.custom_widget)
        custom_layout.setContentsMargins(24, 0, 0, 0)

        self.lbl_custom_model = QLabel("未选择模型文件")
        self.lbl_custom_model.setObjectName("caption")
        custom_layout.addWidget(self.lbl_custom_model, 1)

        self.btn_browse_model = QPushButton("浏览...")
        self.btn_browse_model.clicked.connect(self.browse_custom_model)
        custom_layout.addWidget(self.btn_browse_model)

        self.custom_option = self._source_option(self.rbtn_custom, self.custom_widget)
        layout.addWidget(self.custom_option)

        # 两个单选钮现在各自待在自己的选项块里。QRadioButton 的自动互斥是按「同一个父控件」
        # 算的，父控件不同就不再互斥——点自定义不会把官方那个取消掉。所以显式编组。
        self.source_button_group = QButtonGroup(self)
        self.source_button_group.addButton(self.rbtn_official)
        self.source_button_group.addButton(self.rbtn_custom)

        # 初始化模型型号列表 + 来源可用状态
        self.on_model_version_changed(self.cb_model_version.currentText())
        self.on_model_source_changed()

        return group

    def _source_option(self, radio: QRadioButton, detail: QWidget) -> QFrame:
        """一个模型来源 = 单选钮 + 它自己的详细设置，整块是一个可选区域。"""
        option = QFrame()
        option.setObjectName("sourceOption")
        option.setProperty("selected", False)

        layout = QVBoxLayout(option)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(radio)
        layout.addWidget(detail)

        return option

    def create_inference_params_group(self) -> QGroupBox:
        """创建推理参数组"""
        group = QGroupBox("推理参数")

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # 阈值设置
        form = QFormLayout()

        self.sb_conf_threshold = QDoubleSpinBox()
        self.sb_conf_threshold.setRange(0.0, 1.0)
        self.sb_conf_threshold.setSingleStep(0.05)
        self.sb_conf_threshold.setValue(0.5)
        self.sb_conf_threshold.valueChanged.connect(lambda _: self.update_preview())
        form.addRow("置信度:", self.sb_conf_threshold)

        self.sb_iou_threshold = QDoubleSpinBox()
        self.sb_iou_threshold.setRange(0.0, 1.0)
        self.sb_iou_threshold.setSingleStep(0.05)
        self.sb_iou_threshold.setValue(0.45)
        self.sb_iou_threshold.valueChanged.connect(lambda _: self.update_preview())
        form.addRow("IOU阈值:", self.sb_iou_threshold)

        layout.addLayout(form)

        # 推理选项
        self.chk_only_unlabeled = QCheckBox("仅推理无标签数据")
        self.chk_only_unlabeled.setChecked(True)
        self.chk_only_unlabeled.toggled.connect(lambda _: self.update_preview())
        layout.addWidget(self.chk_only_unlabeled)

        self.chk_overwrite = QCheckBox("覆盖原标签")
        self.chk_overwrite.setChecked(False)
        self.chk_overwrite.toggled.connect(lambda _: self.update_preview())
        layout.addWidget(self.chk_overwrite)

        overwrite_hint = QLabel("勾选后，推理结果会替换图片上已有的标注，且无法撤销。")
        overwrite_hint.setObjectName("caption")
        overwrite_hint.setWordWrap(True)
        layout.addWidget(overwrite_hint)

        return group

    def create_class_mapping_section(self) -> CollapsibleSection:
        """类别映射：进阶用法，默认折叠"""
        section = CollapsibleSection("类别映射（可选：模型类别与项目类别不一致时使用）")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        section.add_layout(layout)

        # 启用映射选项
        self.chk_enable_mapping = QCheckBox("启用类别映射")
        self.chk_enable_mapping.setChecked(False)
        self.chk_enable_mapping.stateChanged.connect(self.on_enable_mapping_changed)
        layout.addWidget(self.chk_enable_mapping)

        # 模型类别文件加载
        self.btn_load_classes = QPushButton("加载模型 classes.txt")
        self.btn_load_classes.clicked.connect(self.load_model_classes)
        self.btn_load_classes.setEnabled(False)
        layout.addWidget(self.btn_load_classes, alignment=Qt.AlignmentFlag.AlignLeft)
        self.model_classes_path = ""
        self.model_classes = []

        # 两侧类别对照
        splitter = QSplitter(Qt.Orientation.Horizontal)

        model_class_widget = QWidget()
        model_class_layout = QVBoxLayout(model_class_widget)
        model_class_layout.setContentsMargins(0, 0, 0, 0)
        model_class_layout.addWidget(QLabel("模型类别"))
        self.model_class_list = QListWidget()
        model_class_layout.addWidget(self.model_class_list)
        splitter.addWidget(model_class_widget)

        project_class_widget = QWidget()
        project_class_layout = QVBoxLayout(project_class_widget)
        project_class_layout.setContentsMargins(0, 0, 0, 0)
        project_class_layout.addWidget(QLabel("项目类别"))
        self.project_class_list = QListWidget()
        project_class_layout.addWidget(self.project_class_list)
        splitter.addWidget(project_class_widget)

        layout.addWidget(splitter)

        # 映射按钮
        mapping_buttons_layout = QHBoxLayout()

        self.btn_edit_mapping = QPushButton("编辑映射")
        self.btn_edit_mapping.clicked.connect(self.edit_mapping)
        self.btn_edit_mapping.setEnabled(False)
        mapping_buttons_layout.addWidget(self.btn_edit_mapping)

        self.btn_apply_all = QPushButton("一键应用模型类别")
        self.btn_apply_all.setToolTip("用模型的类别列表覆盖项目类别，保存后写入项目")
        self.btn_apply_all.clicked.connect(self.apply_all_model_classes)
        self.btn_apply_all.setEnabled(False)
        mapping_buttons_layout.addWidget(self.btn_apply_all)
        mapping_buttons_layout.addStretch()

        layout.addLayout(mapping_buttons_layout)

        # 初始化类别列表
        self.update_class_lists()

        return section

    def on_model_version_changed(self, version: str):
        """模型版本改变时更新型号和任务类型列表"""
        self.cb_model_size.clear()
        self.cb_model_task.clear()

        if version in ULTRALYTICS_MODELS:
            # 更新型号列表
            sizes = ULTRALYTICS_MODELS[version]['sizes']
            for size in sizes:
                display_name = SIZE_NAMES.get(size, size)
                self.cb_model_size.addItem(display_name, size)

            # 更新任务类型列表
            tasks = ULTRALYTICS_MODELS[version]['tasks']
            for task in tasks:
                self.cb_model_task.addItem(task)

            # 默认选择第一个任务类型
            if tasks:
                self.cb_model_task.setCurrentIndex(0)

        self.update_preview()

    def on_model_source_changed(self):
        """模型来源改变时更新界面：选中的那一块要一眼能认出来。"""
        if self.rbtn_custom.isChecked():
            self.model_source = "custom"
        else:
            self.model_source = "official"

        is_custom = self.model_source == "custom"
        self.btn_browse_model.setEnabled(is_custom)
        self.lbl_custom_model.setEnabled(is_custom)
        self.official_widget.setEnabled(not is_custom)

        # 圆点（单选钮自己画）+ 标题字重 + 轻背景，三个信号一起表达选中
        for option, radio, chosen in (
            (self.official_option, self.rbtn_official, not is_custom),
            (self.custom_option, self.rbtn_custom, is_custom),
        ):
            option.setProperty("selected", chosen)
            option.style().unpolish(option)
            option.style().polish(option)

            font = radio.font()
            font.setBold(chosen)
            radio.setFont(font)

        self.update_preview()

    def browse_custom_model(self):
        """浏览自定义模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件",
            "", "PyTorch models (*.pt *.pth)"
        )
        if file_path:
            self.custom_model_path = file_path
            self.lbl_custom_model.setText(os.path.basename(file_path))
            self.lbl_custom_model.setToolTip(file_path)
            self.update_preview()

    def edit_mapping(self):
        """编辑类别映射"""
        if not self.model_classes:
            QMessageBox.warning(self, "警告", "请先加载模型classes.txt文件")
            return

        # 这里可以实现一个更复杂的映射编辑界面
        # 暂时使用简单的消息框
        QMessageBox.information(self, "编辑映射", "类别映射编辑功能开发中...")

    def add_class(self):
        """添加新类别"""
        # 这里可以实现添加新类别的功能
        QMessageBox.information(self, "添加类别", "添加类别功能开发中...")

    def on_single_inference(self):
        """单张推理"""
        # 获取模型路径
        model_path = self.get_model_path()
        if not model_path:
            QMessageBox.warning(self, "错误", "请选择有效的模型")
            return

        # 获取任务类型
        model_task = self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'

        # 发送信号
        self.single_inference_requested.emit(
            model_path,
            self.sb_conf_threshold.value(),
            self.sb_iou_threshold.value(),
            self.class_mappings,
            getattr(self, 'current_image_path', ''),
            model_task
        )
        # 关闭弹窗
        self.accept()

    def on_batch_inference(self):
        """一键推理"""
        # 获取模型路径
        model_path = self.get_model_path()
        if not model_path:
            QMessageBox.warning(self, "错误", "请选择有效的模型")
            return

        # 获取任务类型
        model_task = self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'

        # 发送信号
        self.batch_inference_requested.emit(
            model_path,
            self.sb_conf_threshold.value(),
            self.sb_iou_threshold.value(),
            self.class_mappings,
            getattr(self, 'current_images', []),
            self.chk_only_unlabeled.isChecked(),
            model_task
        )
        # 关闭弹窗
        self.accept()

    def run_single_inference(self, image_path: str):
        """运行单张推理"""
        self.current_image_path = image_path
        self.exec()

    def run_batch_inference(self, images: list):
        """运行批量推理"""
        self.current_images = images
        self.exec()

    def get_model_path(self) -> str:
        """获取模型路径"""
        if self.model_source == "custom":
            return self.custom_model_path
        else:
            # 构建官方模型名称
            version = self.cb_model_version.currentText()
            size = self.cb_model_size.currentData() or self.cb_model_size.currentText()
            if version in ULTRALYTICS_MODELS:
                prefix = ULTRALYTICS_MODELS[version]['prefix']
                return f"{prefix}{size}"
        return ""

    def on_enable_mapping_changed(self, state):
        """启用映射选项改变时的处理"""
        enabled = state == Qt.CheckState.Checked.value
        self.btn_load_classes.setEnabled(enabled)
        self.btn_edit_mapping.setEnabled(enabled and len(self.model_classes) > 0)
        self.btn_apply_all.setEnabled(enabled and len(self.model_classes) > 0)

    def load_model_classes(self):
        """加载模型classes.txt文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型classes.txt文件",
            "", "Text files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                if classes:
                    self.model_classes_path = file_path
                    self.model_classes = classes
                    self.update_model_class_list()
                    self.btn_edit_mapping.setEnabled(True)
                    self.btn_apply_all.setEnabled(True)
                    QMessageBox.information(self, "成功", f"成功加载 {len(classes)} 个模型类别")
                else:
                    QMessageBox.warning(self, "警告", "classes.txt文件为空")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载classes.txt文件失败: {str(e)}")

    def update_model_class_list(self):
        """更新模型类别列表"""
        self.model_class_list.clear()
        if self.model_classes:
            for i, cls in enumerate(self.model_classes):
                item = QListWidgetItem(f"{i}: {cls}")
                self.model_class_list.addItem(item)
        else:
            # 还没加载：给出空状态，而不是一串看着像真数据的示例类别
            placeholder = QListWidgetItem("尚未加载模型类别，请点击「加载模型 classes.txt」")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QColor(COLORS['text_disabled']))
            self.model_class_list.addItem(placeholder)

    def apply_all_model_classes(self):
        """一键应用模型类别到项目"""
        if not self.model_classes:
            QMessageBox.warning(self, "警告", "请先加载模型classes.txt文件")
            return

        # 创建新的项目类别列表
        new_classes = []
        for i, cls_name in enumerate(self.model_classes):
            # 生成随机颜色
            import random
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            new_classes.append({
                'id': i,
                'name': cls_name,
                'color': color
            })

        # 更新项目类别
        self.project_classes = new_classes
        self.update_project_class_list()

        # 发送信号通知主窗口更新类别
        # 这里可以添加一个信号来通知主窗口
        QMessageBox.information(self, "成功", f"成功应用 {len(new_classes)} 个模型类别到项目")

    def update_project_class_list(self):
        """更新项目类别列表"""
        self.project_class_list.clear()
        for cls in self.project_classes:
            item = QListWidgetItem(f"{cls['id']}: {cls['name']}")
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            self.project_class_list.addItem(item)

    def update_class_lists(self):
        """更新类别列表"""
        self.update_model_class_list()
        self.update_project_class_list()

    def set_classes(self, classes: list):
        """设置项目类别"""
        self.project_classes = classes
        self.update_class_lists()

    def get_model_task(self) -> str:
        """获取当前选择的任务类型"""
        return self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'

    def get_class_mappings(self):
        """获取类别映射"""
        if not self.chk_enable_mapping.isChecked():
            return {}

        # 这里可以返回更复杂的映射
        # 暂时返回空映射
        return self.class_mappings

    # ==================== 预览与状态 ====================

    def update_preview(self):
        """刷新底部「将要发生什么」的预览，以及风险 / 状态提示。"""
        if not hasattr(self, 'lbl_preview') or not hasattr(self, 'tab_widget'):
            return

        index = self.tab_widget.currentIndex()
        if index == 0:
            preview, notice, color = self._yolo_preview()
        elif index == 1:
            preview, notice, color = self._sam_preview()
        else:
            preview, notice, color = self._llm_preview()

        self.lbl_preview.setText(preview)
        self.lbl_preview.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self._set_notice(notice, color)

    def _set_notice(self, text: str, color: str):
        """没有提示就把这一行收掉：空标签照样占一行高度，底部白白多出一条空带。"""
        self.lbl_notice.setText(text)
        self.lbl_notice.setStyleSheet(f"color: {color};")
        self.lbl_notice.setVisible(bool(text))

    def _yolo_preview(self):
        if self.model_source == "custom":
            model_name = os.path.basename(self.custom_model_path) if self.custom_model_path else "未选择模型文件"
        else:
            model_name = f"{self.cb_model_version.currentText()} {self.cb_model_size.currentText()}"

        scope = "仅未标注图片" if self.chk_only_unlabeled.isChecked() else "全部图片"
        preview = (
            f"保存后自动标注将使用：{model_name} · {self.get_model_task()} · "
            f"置信度 {self.sb_conf_threshold.value():.2f} / IoU {self.sb_iou_threshold.value():.2f} · {scope}"
        )

        if self.model_source == "custom" and not self.custom_model_path:
            return preview, "还没有选择自定义模型文件，保存前请先浏览选择。", COLORS['error']
        if self.chk_overwrite.isChecked():
            return preview, "已勾选「覆盖原标签」：推理结果会替换图片已有标注，无法撤销。", COLORS['error']
        return preview, "", COLORS['text_secondary']

    def _sam_preview(self):
        sam_type = self.cb_sam_type.currentText()
        model_file = self.cb_sam_model.currentData()
        model_name = self.cb_sam_model.currentText()
        mode = self.cb_sam_usage_mode.currentText()
        device = self.cb_sam_device.currentText()

        preview = f"SAM 标注将使用：{model_name} · 设备 {device} · {mode}"

        if not _sam_model_exists(model_file):
            if sam_type == "SAM3":
                return preview, f"{model_file} 不在本地，且不支持自动下载：请从 {SAM3_DOWNLOAD_URL} 下载后放到项目根目录。", COLORS['warning']
            return preview, f"{model_file} 不在本地，保存时会询问是否下载。", COLORS['warning']
        return preview, f"{model_file} 已在本地，可直接使用。", COLORS['success']

    def _llm_preview(self):
        model_name = self.le_llm_model_name.text().strip() or "未填写模型名称"
        base_url = self.le_llm_base_url.text().strip() or "未填写 Base URL"
        preview = f"LLM 标注将调用：{model_name} @ {base_url}"

        if not self.le_llm_api_key.text().strip():
            return preview, "还没有填写 API Key，LLM 自动标注不可用。", COLORS['warning']
        return preview, "API Key 以明文保存在 config/llm_config.json，请勿共享该文件。", COLORS['warning']

    def _set_status(self, text: str, color: str):
        """执行状态（当前只有 SAM 模型下载会用到）。"""
        self._set_notice(text, color)

    def on_save_clicked(self):
        """保存设置。

        原来这里不管写没写成功都直接 accept()——SAM / LLM 配置写失败（目录只读、
        磁盘满、文件被占用）时窗口照样关掉，用户看到的是「点了保存，什么都没发生」，
        下次打开发现设置根本没存上。现在：写成功才关窗，写失败就留在原地把原因说出来。
        """
        # 自定义来源却没选文件：存下去只会得到一个用不了的配置
        if self.model_source == "custom" and not self.custom_model_path:
            self.tab_widget.setCurrentIndex(0)
            self._set_status("已选择自定义模型，请先浏览选择模型文件。", COLORS['error'])
            return

        # 检查SAM模型是否存在
        if hasattr(self, 'cb_sam_type') and hasattr(self, 'cb_sam_model'):
            sam_type = self.cb_sam_type.currentText()
            model_file = self.cb_sam_model.currentData()

            if model_file and not _sam_model_exists(model_file):
                if sam_type == "SAM3":
                    self.tab_widget.setCurrentIndex(1)
                    show_warning(
                        self, "找不到 SAM3 模型",
                        f"{model_file} 不在本地，而 SAM3 不支持自动下载。",
                        detail=f"请到 {SAM3_DOWNLOAD_URL} 下载 sam3.pt，放到项目根目录后重试。",
                    )
                    return

                # 模型不存在，问一下要不要现在下
                if confirm(
                    self, "模型还没下载",
                    f"{sam_type} 要用的 {model_file} 不在本地，下载之后才能用 SAM 标注。",
                    detail="下载要花一些时间，取决于网络。也可以先保存其他设置，之后再下。",
                    confirm_text="现在下载",
                ):
                    self.download_sam_model(sam_type, model_file)

        # 两份配置都写盘成功，才算保存成功
        sam_saved = self.save_sam_config()
        llm_saved = self.save_llm_config()

        if not (sam_saved and llm_saved):
            # 只有一份写失败时，另一份是真的已经写进去了——不能笼统说「设置没有写入」，
            # 那会让用户以为可以放心重试或放弃，实际上磁盘上已经是半新半旧的状态。
            failed = []
            saved_ok = []
            if sam_saved:
                saved_ok.append("SAM 配置")
            else:
                failed.append(f"SAM 配置（{SAM_CONFIG_FILE}）")
            if llm_saved:
                saved_ok.append("LLM 配置")
            else:
                failed.append(f"LLM 配置（{LLM_CONFIG_FILE}）")

            message = "保存失败：" + "；".join(failed) + " 没有写入"
            if saved_ok:
                message += "；" + "、".join(saved_ok) + " 已经写入"
            message += "。请检查文件是否可写后重试。"

            self._set_status(message, COLORS['error'])
            return

        # 走到这里 = 两份配置都落盘了。accept() 只在这一条路上发生，
        # 所以调用方拿到 Accepted 就等于「真的保存成功了」。
        self.accept()

    def download_sam_model(self, sam_type: str, model_file: str):
        """下载SAM模型"""
        try:
            from PyQt6.QtWidgets import QProgressDialog
            from PyQt6.QtCore import Qt, QThread, pyqtSignal

            class ModelDownloadWorker(QThread):
                """模型下载工作线程"""
                download_finished = pyqtSignal(bool, str)

                def __init__(self, sam_type, model_file):
                    super().__init__()
                    self.sam_type = sam_type
                    self.model_file = model_file

                def run(self):
                    try:
                        # 根据类型导入正确的类来触发下载
                        if self.sam_type == "FastSAM":
                            from ultralytics import FastSAM
                            model = FastSAM(self.model_file)
                        else:
                            # SAM, SAM2, MobileSAM 都使用SAM类
                            from ultralytics import SAM
                            model = SAM(self.model_file)

                        self.download_finished.emit(True, f"模型 {self.model_file} 下载成功")
                    except Exception as e:
                        self.download_finished.emit(False, f"下载失败: {str(e)}")

            # 显示进度对话框
            self._set_status(f"正在下载模型 {model_file}...", COLORS['text_secondary'])
            progress = QProgressDialog(f"正在下载模型 {model_file}...", "取消", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setCancelButton(None)
            progress.show()

            # 创建下载线程
            self.download_worker = ModelDownloadWorker(sam_type, model_file)
            self.download_worker.download_finished.connect(
                lambda success, msg: self.on_model_download_finished(success, msg, progress)
            )
            self.download_worker.start()

        except Exception as e:
            self._set_status(f"启动下载失败: {str(e)}", COLORS['error'])
            QMessageBox.critical(self, "错误", f"启动下载失败: {str(e)}")

    def on_model_download_finished(self, success: bool, message: str, progress_dialog):
        """模型下载完成回调"""
        progress_dialog.close()

        if success:
            self._set_status(message, COLORS['success'])
            QMessageBox.information(self, "成功", message)
        else:
            self._set_status(message, COLORS['error'])
            QMessageBox.critical(self, "下载失败", message)

    # ==================== SAM相关方法 ====================

    def create_sam_model_group(self) -> QGroupBox:
        """创建SAM模型选择组"""
        group = QGroupBox("SAM 模型")

        layout = QFormLayout(group)

        # SAM类型选择
        self.cb_sam_type = QComboBox()
        self.cb_sam_type.addItems(list(SAM_MODELS.keys()))
        self.cb_sam_type.currentTextChanged.connect(self.on_sam_type_changed)
        layout.addRow("模型类型:", self.cb_sam_type)

        # SAM型号选择
        self.cb_sam_model = QComboBox()
        self.cb_sam_model.currentTextChanged.connect(lambda _: self.update_preview())
        layout.addRow("模型型号:", self.cb_sam_model)

        # 初始化型号列表
        self.on_sam_type_changed(self.cb_sam_type.currentText())

        # 设备选择
        self.cb_sam_device = QComboBox()
        self.cb_sam_device.addItems(["自动选择", "CPU", "CUDA:0", "CUDA:1", "CUDA:2", "CUDA:3"])
        self.cb_sam_device.currentTextChanged.connect(lambda _: self.update_preview())
        layout.addRow("设备:", self.cb_sam_device)

        return group

    def create_sam_params_group(self) -> QGroupBox:
        """创建SAM推理参数组：常用的留在外面，FastSAM 专用的收起来"""
        group = QGroupBox("推理参数")

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        form = QFormLayout()

        # 图像尺寸
        self.sb_sam_imgsz = QDoubleSpinBox()
        self.sb_sam_imgsz.setRange(256, 2048)
        self.sb_sam_imgsz.setValue(1024)
        self.sb_sam_imgsz.setSingleStep(64)
        form.addRow("图像尺寸:", self.sb_sam_imgsz)

        layout.addLayout(form)

        advanced = CollapsibleSection("高级参数（仅 FastSAM 生效）")
        advanced_layout = QFormLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced.add_layout(advanced_layout)

        # 置信度阈值（FastSAM用）
        self.sb_sam_conf = QDoubleSpinBox()
        self.sb_sam_conf.setRange(0.01, 1.0)
        self.sb_sam_conf.setValue(0.4)
        self.sb_sam_conf.setDecimals(2)
        self.sb_sam_conf.setSingleStep(0.05)
        advanced_layout.addRow("置信度阈值:", self.sb_sam_conf)

        # IoU阈值（FastSAM用）
        self.sb_sam_iou = QDoubleSpinBox()
        self.sb_sam_iou.setRange(0.1, 1.0)
        self.sb_sam_iou.setValue(0.9)
        self.sb_sam_iou.setDecimals(2)
        self.sb_sam_iou.setSingleStep(0.05)
        advanced_layout.addRow("IoU阈值:", self.sb_sam_iou)

        # Retina masks选项（FastSAM用）
        self.chk_sam_retina = QCheckBox("使用Retina Masks")
        self.chk_sam_retina.setChecked(True)
        advanced_layout.addRow(self.chk_sam_retina)

        layout.addWidget(advanced)

        return group

    def create_sam_usage_group(self) -> QGroupBox:
        """创建SAM使用方式组"""
        group = QGroupBox("使用方式")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        form = QFormLayout()
        self.cb_sam_usage_mode = QComboBox()
        self.cb_sam_usage_mode.currentTextChanged.connect(lambda _: self.update_preview())
        form.addRow("模式:", self.cb_sam_usage_mode)
        layout.addLayout(form)

        hint = QLabel("记忆标注只有 SAM2 / SAM3 提供，其他模型只有普通标注。")
        hint.setObjectName("caption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._update_sam_usage_mode_options(self.cb_sam_type.currentText())
        return group

    def on_sam_type_changed(self, sam_type: str):
        """SAM类型改变时更新型号列表"""
        self.cb_sam_model.clear()
        if sam_type in SAM_MODELS:
            models = SAM_MODELS[sam_type]["models"]
            for name, file in models.items():
                self.cb_sam_model.addItem(f"{name} ({file})", file)
        self._update_sam_usage_mode_options(sam_type)
        self.update_preview()

    def _update_sam_usage_mode_options(self, sam_type: str, target_mode: str = None):
        """根据SAM类型更新可选使用方式。"""
        if not hasattr(self, "cb_sam_usage_mode"):
            return
        self.cb_sam_usage_mode.blockSignals(True)
        self.cb_sam_usage_mode.clear()
        self.cb_sam_usage_mode.addItem("普通标注", "normal")
        if sam_type in ("SAM2", "SAM3"):
            self.cb_sam_usage_mode.addItem("记忆标注", "memory")
        if target_mode:
            idx = self.cb_sam_usage_mode.findData(target_mode)
            if idx >= 0:
                self.cb_sam_usage_mode.setCurrentIndex(idx)
        self.cb_sam_usage_mode.blockSignals(False)

    def get_sam_config(self) -> dict:
        """获取SAM配置"""
        sam_type = self.cb_sam_type.currentText()
        model_file = self.cb_sam_model.currentData()

        # 获取设备
        device = self.cb_sam_device.currentText()
        if device == "自动选择":
            try:
                import torch
                device = '0' if torch.cuda.is_available() else 'cpu'
            except:
                device = 'cpu'
        elif device == "CPU":
            device = 'cpu'
        elif device.startswith("CUDA:"):
            device = device.split(":")[1]

        return {
            'sam_type': sam_type,
            'model_file': model_file,
            'device': device,
            'imgsz': int(self.sb_sam_imgsz.value()),
            'conf': self.sb_sam_conf.value(),
            'iou': self.sb_sam_iou.value(),
            'retina_masks': self.chk_sam_retina.isChecked(),
            'usage_mode': self.cb_sam_usage_mode.currentData() or "normal",
        }

    def load_sam_config(self):
        """从配置文件加载SAM配置并应用到UI。"""
        config = DEFAULT_SAM_CONFIG.copy()
        if os.path.exists(SAM_CONFIG_FILE):
            try:
                with open(SAM_CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        config.update(saved)
            except Exception:
                pass

        sam_type = config.get("sam_type", "SAM")
        if sam_type in SAM_MODELS:
            self.cb_sam_type.setCurrentText(sam_type)
        else:
            sam_type = self.cb_sam_type.currentText()

        model_file = config.get("model_file", "sam_b.pt")
        model_index = self.cb_sam_model.findData(model_file)
        if model_index >= 0:
            self.cb_sam_model.setCurrentIndex(model_index)

        device = str(config.get("device", "cpu"))
        if device == "cpu":
            display_device = "CPU"
        elif device.isdigit():
            display_device = f"CUDA:{device}"
        else:
            display_device = "自动选择"
        device_index = self.cb_sam_device.findText(display_device)
        if device_index >= 0:
            self.cb_sam_device.setCurrentIndex(device_index)

        self.sb_sam_imgsz.setValue(float(config.get("imgsz", 1024)))
        self.sb_sam_conf.setValue(float(config.get("conf", 0.4)))
        self.sb_sam_iou.setValue(float(config.get("iou", 0.9)))
        self.chk_sam_retina.setChecked(bool(config.get("retina_masks", True)))
        self._update_sam_usage_mode_options(sam_type, config.get("usage_mode", "normal"))

    def save_sam_config(self) -> bool:
        """保存SAM配置到配置文件。写不进去返回 False（调用方要据此留住窗口）。

        建目录也要算在「写失败」里：原来 makedirs 在 try 外面，配置目录建不出来时
        直接抛异常，而不是返回 False——调用方以为只会拿到 True/False，结果是崩一下。
        """
        config = self.get_sam_config()
        try:
            config_dir = os.path.dirname(SAM_CONFIG_FILE)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            with open(SAM_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存SAM配置失败: {e}")
            return False

    @classmethod
    def get_saved_sam_config(cls) -> dict:
        """读取已保存的SAM配置（不依赖弹窗实例）。"""
        config = DEFAULT_SAM_CONFIG.copy()
        if os.path.exists(SAM_CONFIG_FILE):
            try:
                with open(SAM_CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        config.update(saved)
            except Exception:
                pass
        return config

    # ==================== LLM相关方法 ====================

    def create_llm_tab(self) -> QWidget:
        """创建LLM自动标注页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        layout.addWidget(self.create_llm_api_group())
        layout.addWidget(self.create_llm_prompt_section())

        layout.addStretch()
        return tab

    def create_llm_api_group(self) -> QGroupBox:
        """创建LLM API配置组"""
        group = QGroupBox("视觉模型服务")

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        form = QFormLayout()

        # API Key
        self.le_llm_api_key = QLineEdit()
        self.le_llm_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.le_llm_api_key.setPlaceholderText("请输入API Key")
        self.le_llm_api_key.textChanged.connect(lambda _: self.update_preview())
        form.addRow("API Key:", self.le_llm_api_key)

        # Base URL
        self.le_llm_base_url = QLineEdit()
        self.le_llm_base_url.setPlaceholderText("请输入Base URL")
        self.le_llm_base_url.textChanged.connect(lambda _: self.update_preview())
        form.addRow("Base URL:", self.le_llm_base_url)

        # Model Name
        self.le_llm_model_name = QLineEdit()
        self.le_llm_model_name.setPlaceholderText("请输入模型名称")
        self.le_llm_model_name.textChanged.connect(lambda _: self.update_preview())
        form.addRow("模型名称:", self.le_llm_model_name)

        layout.addLayout(form)

        hint = QLabel("兼容 OpenAI 接口的视觉模型即可；标注时按当前选中的类别逐张调用。")
        hint.setObjectName("caption")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return group

    def create_llm_prompt_section(self) -> CollapsibleSection:
        """提示词模板：默认折叠，不挡住上面的基础配置"""
        section = CollapsibleSection("提示词模板（高级）")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        section.add_layout(layout)

        # 系统提示词
        layout.addWidget(QLabel("系统提示词:"))
        self.te_llm_system_prompt = QTextEdit()
        self.te_llm_system_prompt.setMaximumHeight(120)
        layout.addWidget(self.te_llm_system_prompt)

        # 用户提示词
        layout.addWidget(QLabel("用户提示词:"))
        self.te_llm_user_prompt = QTextEdit()
        self.te_llm_user_prompt.setMaximumHeight(120)
        layout.addWidget(self.te_llm_user_prompt)

        hint = QLabel("{target} 会被替换成当前选中的类别名称。")
        hint.setObjectName("caption")
        layout.addWidget(hint)

        return section

    def load_llm_config(self):
        """加载LLM配置"""
        config = DEFAULT_LLM_CONFIG.copy()

        # 从文件加载配置
        if os.path.exists(LLM_CONFIG_FILE):
            try:
                with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    config.update(saved_config)
            except Exception as e:
                print(f"加载LLM配置失败: {e}")

        # 设置到UI
        self.le_llm_api_key.setText(config.get('api_key', ''))
        self.le_llm_base_url.setText(config.get('base_url', ''))
        self.le_llm_model_name.setText(config.get('model_name', ''))
        self.te_llm_system_prompt.setPlainText(config.get('system_prompt', ''))
        self.te_llm_user_prompt.setPlainText(config.get('user_prompt', ''))

    def save_llm_config(self):
        """保存LLM配置"""
        config = {
            'api_key': self.le_llm_api_key.text(),
            'base_url': self.le_llm_base_url.text(),
            'model_name': self.le_llm_model_name.text(),
            'system_prompt': self.te_llm_system_prompt.toPlainText(),
            'user_prompt': self.te_llm_user_prompt.toPlainText()
        }

        # 建目录和写文件都可能失败，一起算在「没保存成功」里
        try:
            config_dir = os.path.dirname(LLM_CONFIG_FILE)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            with open(LLM_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存LLM配置失败: {e}")
            return False

    def get_llm_config(self) -> dict:
        """获取LLM配置"""
        return {
            'api_key': self.le_llm_api_key.text(),
            'base_url': self.le_llm_base_url.text(),
            'model_name': self.le_llm_model_name.text(),
            'system_prompt': self.te_llm_system_prompt.toPlainText(),
            'user_prompt': self.te_llm_user_prompt.toPlainText()
        }
