# -*- coding: utf-8 -*-
"""
结果分析页面（主流程第 4 步）

页面按一条阅读顺序自上而下排：
    ① 这次训练能不能用  —— 一句话结论，新手先看这个
    ② 看哪一次训练      —— 一个项目可能训练过很多次
    ③ 关键指标          —— 四个数，每个配一行大白话
    ④ 训练曲线 / 预测示例 —— 图放在标签页里，一次只铺开一张
    ⑤ 模型文件与导出    —— 主操作（导出模型）在这里

训练目录的发现统一走 gui/workflow.find_project_runs，
和侧边栏「已训练 N 次」、前置条件页用的是同一份事实，不各自去猜。
指标仍然取 results.csv 最后一行，算法没有变。
"""

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QFrame, QScrollArea, QStackedWidget, QTabWidget, QSizePolicy,
    QMessageBox, QFileDialog, QDialog, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QMenu, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from gui.styles import COLORS, RADIUS_SM, set_menu_indicator
from gui.workflow import APP_ROOT, find_project_runs
from gui.widgets.context_help import ContextHelp
from gui.widgets.workflow_widgets import EmptyState


class ONNXExportDialog(QDialog):
    """ONNX导出配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ONNX导出配置")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本参数
        basic_group = QGroupBox("基本参数")
        basic_layout = QFormLayout(basic_group)
        
        # imgsz
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        basic_layout.addRow("输入尺寸 (imgsz):", self.spin_imgsz)
        
        # batch
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        basic_layout.addRow("批量大小 (batch):", self.spin_batch)
        
        # opset
        self.spin_opset = QSpinBox()
        self.spin_opset.setRange(7, 17)
        self.spin_opset.setValue(12)
        basic_layout.addRow("ONNX Opset:", self.spin_opset)
        
        # device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "0", "1", "2", "3"])
        basic_layout.addRow("设备 (device):", self.combo_device)
        
        layout.addWidget(basic_group)
        
        # 选项参数
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_half = QCheckBox("半精度 (half) - FP16")
        self.chk_half.setChecked(False)
        options_layout.addWidget(self.chk_half)
        
        self.chk_dynamic = QCheckBox("动态轴 (dynamic)")
        self.chk_dynamic.setChecked(False)
        options_layout.addWidget(self.chk_dynamic)
        
        self.chk_simplify = QCheckBox("简化模型 (simplify)")
        self.chk_simplify.setChecked(True)
        options_layout.addWidget(self.chk_simplify)
        
        self.chk_nms = QCheckBox("包含NMS (nms)")
        self.chk_nms.setChecked(False)
        options_layout.addWidget(self.chk_nms)
        
        layout.addWidget(options_group)
        
        # NMS参数组（仅在NMS选中时启用）
        self.nms_group = QGroupBox("NMS参数")
        self.nms_group.setEnabled(False)
        nms_layout = QFormLayout(self.nms_group)
        
        # conf
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        nms_layout.addRow("置信度阈值 (conf):", self.spin_conf)
        
        # iou
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        nms_layout.addRow("IoU阈值 (iou):", self.spin_iou)
        
        # agnostic_nms
        self.chk_agnostic_nms = QCheckBox("类别无关NMS (agnostic_nms)")
        self.chk_agnostic_nms.setChecked(False)
        nms_layout.addRow(self.chk_agnostic_nms)
        
        layout.addWidget(self.nms_group)
        
        # NMS选中时启用NMS参数组
        self.chk_nms.stateChanged.connect(
            lambda state: self.nms_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'imgsz': self.spin_imgsz.value(),
            'half': self.chk_half.isChecked(),
            'dynamic': self.chk_dynamic.isChecked(),
            'simplify': self.chk_simplify.isChecked(),
            'opset': self.spin_opset.value(),
            'nms': self.chk_nms.isChecked(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device.currentText()
        }
        
        # 如果启用NMS，添加NMS参数
        if self.chk_nms.isChecked():
            config['conf'] = self.spin_conf.value()
            config['iou'] = self.spin_iou.value()
            config['agnostic_nms'] = self.chk_agnostic_nms.isChecked()
        
        return config


class TensorRTExportDialog(QDialog):
    """TensorRT导出配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TensorRT导出配置")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本参数
        basic_group = QGroupBox("基本参数")
        basic_layout = QFormLayout(basic_group)
        
        # imgsz
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        basic_layout.addRow("输入尺寸 (imgsz):", self.spin_imgsz)
        
        # batch
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        basic_layout.addRow("批量大小 (batch):", self.spin_batch)
        
        # workspace
        self.spin_workspace = QSpinBox()
        self.spin_workspace.setRange(1, 16)
        self.spin_workspace.setValue(4)
        basic_layout.addRow("工作空间 (workspace GB):", self.spin_workspace)
        
        # device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["0", "1", "2", "3"])
        basic_layout.addRow("设备 (device):", self.combo_device)
        
        layout.addWidget(basic_group)
        
        # 选项参数
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_half = QCheckBox("半精度 (half) - FP16")
        self.chk_half.setChecked(True)
        options_layout.addWidget(self.chk_half)
        
        self.chk_int8 = QCheckBox("INT8量化 (int8)")
        self.chk_int8.setChecked(False)
        options_layout.addWidget(self.chk_int8)
        
        self.chk_dynamic = QCheckBox("动态轴 (dynamic)")
        self.chk_dynamic.setChecked(False)
        options_layout.addWidget(self.chk_dynamic)
        
        self.chk_simplify = QCheckBox("简化模型 (simplify)")
        self.chk_simplify.setChecked(True)
        options_layout.addWidget(self.chk_simplify)
        
        self.chk_nms = QCheckBox("包含NMS (nms)")
        self.chk_nms.setChecked(False)
        options_layout.addWidget(self.chk_nms)
        
        layout.addWidget(options_group)
        
        # NMS参数组（仅在NMS选中时启用）
        self.nms_group = QGroupBox("NMS参数")
        self.nms_group.setEnabled(False)
        nms_layout = QFormLayout(self.nms_group)
        
        # conf
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        nms_layout.addRow("置信度阈值 (conf):", self.spin_conf)
        
        # iou
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        nms_layout.addRow("IoU阈值 (iou):", self.spin_iou)
        
        # agnostic_nms
        self.chk_agnostic_nms = QCheckBox("类别无关NMS (agnostic_nms)")
        self.chk_agnostic_nms.setChecked(False)
        nms_layout.addRow(self.chk_agnostic_nms)
        
        layout.addWidget(self.nms_group)
        
        # NMS选中时启用NMS参数组
        self.chk_nms.stateChanged.connect(
            lambda state: self.nms_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # INT8校准参数（仅在INT8选中时启用）
        self.int8_group = QGroupBox("INT8校准参数")
        self.int8_group.setEnabled(False)
        int8_layout = QFormLayout(self.int8_group)
        
        self.edit_data = QLineEdit()
        self.edit_data.setPlaceholderText("校准数据集路径（如：coco128.yaml）")
        int8_layout.addRow("校准数据 (data):", self.edit_data)
        
        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(0.1, 1.0)
        self.spin_fraction.setValue(1.0)
        self.spin_fraction.setSingleStep(0.1)
        int8_layout.addRow("数据比例 (fraction):", self.spin_fraction)
        
        layout.addWidget(self.int8_group)
        
        # INT8选中时启用校准参数
        self.chk_int8.stateChanged.connect(
            lambda state: self.int8_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'imgsz': self.spin_imgsz.value(),
            'half': self.chk_half.isChecked(),
            'dynamic': self.chk_dynamic.isChecked(),
            'simplify': self.chk_simplify.isChecked(),
            'workspace': self.spin_workspace.value(),
            'int8': self.chk_int8.isChecked(),
            'nms': self.chk_nms.isChecked(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device.currentText()
        }
        
        # 如果启用NMS，添加NMS参数
        if self.chk_nms.isChecked():
            config['conf'] = self.spin_conf.value()
            config['iou'] = self.spin_iou.value()
            config['agnostic_nms'] = self.chk_agnostic_nms.isChecked()
        
        # 如果启用INT8，添加校准参数
        if self.chk_int8.isChecked():
            config['data'] = self.edit_data.text() or None
            config['fraction'] = self.spin_fraction.value()
        
        return config


# ==================== 文案表 ====================

# 四个关键指标：名字 + 大数字，说明只在悬停时给。
METRIC_SPECS = [
    {
        'key': 'mAP50',
        'title': 'mAP50',
        'tip': '把所有类别的准确度平均起来的总分。判定较宽松：预测框和真实框重叠一半以上就算认对。',
    },
    {
        'key': 'mAP50_95',
        'title': 'mAP50-95',
        'tip': '同样是总分，但要求预测框从「大致套住」到「几乎重合」都表现好。它比 mAP50 低很多是常态。',
    },
    {
        'key': '精确率',
        'title': '精确率',
        'tip': '英文 Precision。它高说明模型很少把背景或别的东西认成目标。',
    },
    {
        'key': '召回率',
        'title': '召回率',
        'tip': '英文 Recall。它高说明模型很少漏掉目标。和精确率通常此消彼长。',
    },
]

# 每张结果图是什么，一个短语说清楚
IMAGE_CAPTIONS = [
    ('results.', '训练全过程曲线'),
    ('confusion_matrix', '混淆矩阵'),
    ('PR_curve', '精确率-召回率曲线'),
    ('P_curve', '精确率-置信度曲线'),
    ('R_curve', '召回率-置信度曲线'),
    ('F1_curve', 'F1 曲线'),
    # 预测示例的两条要排在 labels 前面：val_batch0_labels.jpg 里也有 "labels"
    ('_pred', '模型预测框'),
    ('_labels', '标注真值（对照用）'),
    ('train_batch', '训练批次样本（含数据增强）'),
    ('labels_correlogram', '标注分布相关图'),
    ('labels', '各类别标注数量'),
]

# 曲线/图表 tab 里的排序：先看总览，再看细节
CHART_ORDER = [
    'results.', 'confusion_matrix_normalized', 'confusion_matrix',
    'PR_curve', 'F1_curve', 'P_curve', 'R_curve',
    'labels_correlogram', 'labels',
]

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# 导出格式：一个主按钮 + 一个菜单，不再把四个按钮并排铺开
EXPORT_FORMATS = [
    ('pt', 'PyTorch（.pt）', '训练直接产出的权重，本软件的「模型测试」和 Python 代码都能用'),
    ('onnx', 'ONNX（.onnx）', '通用部署格式，大多数推理框架都认'),
    ('torchscript', 'TorchScript', 'C++ / 移动端部署常用'),
    ('tensorrt', 'TensorRT（.engine）', 'NVIDIA 显卡上的加速格式，需要本机装好 TensorRT'),
]


def caption_for(filename: str) -> str:
    """这张图是干什么的。"""
    for keyword, text in IMAGE_CAPTIONS:
        if keyword in filename:
            return text
    return ""


def chart_rank(filename: str) -> int:
    for i, keyword in enumerate(CHART_ORDER):
        if keyword in filename:
            return i
    return len(CHART_ORDER)


class PreviewPane(QWidget):
    """一个「选一张图 + 看这张图」的面板：下拉选图、一句说明、随窗口缩放的大图。

    结果图有十几张，一次全铺开只会让人不知道看哪张，
    所以三类图（曲线 / 预测示例 / 其他）各用一个面板，放进标签页里。
    """

    def __init__(self, empty_text: str, parent=None):
        super().__init__(parent)
        self.empty_text = empty_text
        self._pixmap: Optional[QPixmap] = None
        self._images: List[Tuple[str, str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)

        pick_label = QLabel("查看")
        pick_label.setObjectName("caption")
        top.addWidget(pick_label)

        self.combo = QComboBox()
        self.combo.setMinimumWidth(160)
        self.combo.currentIndexChanged.connect(self._on_pick)
        top.addWidget(self.combo)
        top.addStretch()
        layout.addLayout(top)

        self.caption = QLabel()
        self.caption.setObjectName("caption")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(280)
        # Ignored：图片大小不再反过来撑大窗口，窗口才能自由缩小
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setStyleSheet(
            f"background-color: {COLORS['inset']};"
            f"border: 1px solid {COLORS['border']};"
            f"border-radius: {RADIUS_SM}px;"
            f"color: {COLORS['text_secondary']};"
        )
        layout.addWidget(self.image_label, 1)

    def refresh_theme(self):
        self.image_label.setStyleSheet(
            f"background-color: {COLORS['inset']};"
            f"border: 1px solid {COLORS['border']};"
            f"border-radius: {RADIUS_SM}px;"
            f"color: {COLORS['text_secondary']};"
        )

    def set_images(self, images: List[Tuple[str, str]]):
        """images: [(显示名, 绝对路径)]，按调用方给的顺序显示。"""
        self._images = images
        self._pixmap = None

        self.combo.blockSignals(True)
        self.combo.clear()
        for name, path in images:
            self.combo.addItem(name, path)
        self.combo.blockSignals(False)

        if not images:
            self.combo.setEnabled(False)
            self.caption.setText("")
            self.image_label.setText(self.empty_text)
            return

        self.combo.setEnabled(True)
        self.combo.setCurrentIndex(0)
        self._on_pick(0)

    def _on_pick(self, index: int):
        path = self.combo.itemData(index)
        if not path:
            return

        self.caption.setText(caption_for(os.path.basename(path)))

        if not os.path.exists(path):
            self._pixmap = None
            self.image_label.setText("这张图找不到了（文件可能被移动或删除）。点「刷新」重新扫描。")
            return

        self.image_label.setText("正在加载图像…")
        QApplication.processEvents()

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._pixmap = None
            self.image_label.setText("这张图打不开，文件可能已损坏。")
            return

        self._pixmap = pixmap
        self._render()

    def _render(self):
        """按当前可用空间重画。图比空间小就保持原样，不放大糊掉。"""
        if self._pixmap is None:
            return

        area = self.image_label.size()
        if area.width() < 40 or area.height() < 40:
            return

        if self._pixmap.width() > area.width() or self._pixmap.height() > area.height():
            shown = self._pixmap.scaled(
                area,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            shown = self._pixmap

        self.image_label.setPixmap(shown)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()


class ResultPage(QWidget):
    """结果分析页面"""

    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_run: Optional[Path] = None
        self.current_metrics: Dict[str, float] = {}

        self.init_ui()
        # 初始时不自动扫描，等待设置项目

    def set_project(self, project_id: int):
        """由主窗口调用：切换当前项目。"""
        self.current_project_id = project_id
        if project_id:
            print(f"[ResultPage] 已切换到项目: {project_id}")
            self.scan_runs_directory()
        else:
            print("[ResultPage] 项目已取消选择")
            self.current_run = None
            self.current_metrics = {}
            self._show_empty(
                "还没有选择项目",
                "训练结果是跟着项目走的。先在左上角「当前项目」里选一个项目。",
                show_action=False,
            )

    # ==================== 界面搭建 ====================

    def init_ui(self):
        """初始化界面"""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # 内容 / 空状态 / 加载中，三选一
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.create_content_view())   # 0
        self.view_stack.addWidget(self.create_empty_view())     # 1
        self.view_stack.addWidget(self.create_loading_view())   # 2
        root.addWidget(self.view_stack)

        self._show_empty(
            "还没有选择项目",
            "训练结果是跟着项目走的。先在左上角「当前项目」里选一个项目。",
            show_action=False,
        )

    def create_content_view(self) -> QWidget:
        """整页放进滚动区：窗口再矮也能一路看下去。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        self.context_help = ContextHelp(
            [
                "优先看 mAP50 和预测示例，曲线用于辅助判断。",
                "损失下降并趋稳，通常表示训练正在收敛。",
                "导出使用最佳权重，不会修改原来的训练结果。",
            ],
            title="指标怎么看",
        )
        layout.addWidget(self.context_help)

        layout.addWidget(self.create_verdict_card())
        layout.addWidget(self.create_run_card())
        layout.addWidget(self.create_metrics_card())
        layout.addWidget(self.create_gallery_card(), 1)
        layout.addWidget(self.create_export_card())

        scroll.setWidget(inner)
        return scroll

    def create_verdict_card(self) -> QFrame:
        """① 一句话结论：这次训练能不能用。"""
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)

        self.verdict_badge = QLabel()
        self.verdict_badge.setToolTip(
            "按 mAP50 的粗略判断，实际效果以「模型测试」里跑新图片为准。"
        )
        head.addWidget(self.verdict_badge)

        self.verdict_title = QLabel()
        self.verdict_title.setObjectName("h2")
        self.verdict_title.setWordWrap(True)
        # 会换行的标签在 HBox 里必须拿到伸展权重，否则它只分到 sizeHint 那点宽度，
        # 明明整行都空着，标题却在「可以/用」中间断开
        head.addWidget(self.verdict_title, 1)
        layout.addLayout(head)

        self.verdict_detail = QLabel()
        self.verdict_detail.setObjectName("subtitle")
        self.verdict_detail.setWordWrap(True)
        layout.addWidget(self.verdict_detail)

        return card

    def create_run_card(self) -> QFrame:
        """② 看哪一次训练。"""
        card = QFrame()
        card.setObjectName("card")

        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(10)

        label = QLabel("查看训练")
        label.setObjectName("caption")
        layout.addWidget(label)

        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(180)
        self.run_combo.setToolTip("同一个项目每训练一次，就多一条记录。默认看最新的一次")
        self.run_combo.currentIndexChanged.connect(self.on_run_changed)
        layout.addWidget(self.run_combo)

        self.run_hint = QLabel()
        self.run_hint.setObjectName("caption")
        layout.addWidget(self.run_hint)

        layout.addStretch()

        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setObjectName("ghost")
        self.btn_refresh.setToolTip("重新扫描训练结果目录")
        self.btn_refresh.clicked.connect(self.scan_runs_directory)
        layout.addWidget(self.btn_refresh)

        return card

    def create_metrics_card(self) -> QFrame:
        """③ 四个关键指标，每个配一行人话。"""
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel("关键指标")
        title.setObjectName("title")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(10)

        self.metric_labels = {}
        for i, spec in enumerate(METRIC_SPECS):
            tile = QFrame()
            # 选择器必须写死在 QFrame#metric_tile 上：不带选择器的属性会顺着
            # 继承链落到里面的两个 QLabel，指标名和数值各自套一个框，白底上一眼就看出来
            tile.setObjectName("metric_tile")
            tile.setStyleSheet(
                f"QFrame#metric_tile {{"
                f"  background-color: {COLORS['background']};"
                f"  border: 1px solid {COLORS['border']};"
                f"  border-radius: {RADIUS_SM}px;"
                f"}}"
            )
            tile.setToolTip(spec['tip'])

            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(14, 10, 14, 10)
            tile_layout.setSpacing(2)

            name = QLabel(spec['title'])
            name.setObjectName("caption")
            tile_layout.addWidget(name)

            value = QLabel("--")
            value.setObjectName("value")
            self.metric_labels[spec['key']] = value
            tile_layout.addWidget(value)

            grid.addWidget(tile, 0, i)
            grid.setColumnStretch(i, 1)

        # 一行四个，不是 2×2：四个指标是并列的，横着排一眼扫完。
        # 排成 2×2 之后，宽屏下每块要被拉到 570px 宽，里面只有一个小标题和一个数字，
        # 剩下的全是空白——看起来不是「留白」，是「没做完」。
        layout.addLayout(grid)

        self.metrics_source = QLabel()
        self.metrics_source.setObjectName("caption")
        self.metrics_source.setWordWrap(True)
        layout.addWidget(self.metrics_source)

        return card

    def create_gallery_card(self) -> QFrame:
        """④ 曲线 / 预测示例 / 其他图，一次只铺开一张。"""
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("训练过程与预测示例")
        title.setObjectName("title")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        self.chart_pane = PreviewPane("这次训练没有生成曲线图。训练可能中途停了。")
        self.tabs.addTab(self.chart_pane, "训练曲线")

        self.pred_pane = PreviewPane("这次训练没有留下预测示例图。")
        self.tabs.addTab(self.pred_pane, "预测示例")

        self.other_pane = PreviewPane("没有其他图像。")
        self.tabs.addTab(self.other_pane, "其他图像")

        layout.addWidget(self.tabs, 1)
        return card

    def create_export_card(self) -> QFrame:
        """⑤ 模型文件在哪、能不能导出。主操作就这一个。"""
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("模型文件与导出")
        title.setObjectName("title")
        layout.addWidget(title)

        self.model_info = QLabel()
        self.model_info.setObjectName("subtitle")
        self.model_info.setWordWrap(True)
        layout.addWidget(self.model_info)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.btn_export_model = QPushButton("导出模型")
        self.btn_export_model.setObjectName("primary")
        self.btn_export_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_model.setMinimumHeight(34)

        export_menu = QMenu(self)
        for fmt, text, tip in EXPORT_FORMATS:
            action = export_menu.addAction(text)
            action.setToolTip(tip)
            action.triggered.connect(lambda _checked, f=fmt: self.on_export_format(f))
        self.btn_export_model.setMenu(export_menu)
        set_menu_indicator(self.btn_export_model)
        row.addWidget(self.btn_export_model)

        self.btn_export_folder = QPushButton("导出结果文件夹")
        self.btn_export_folder.setToolTip("把这次训练的整个目录（曲线图、指标、权重）复制到别处")
        self.btn_export_folder.clicked.connect(self.export_result_folder)
        row.addWidget(self.btn_export_folder)

        row.addStretch()
        layout.addLayout(row)

        self.export_status = QLabel()
        self.export_status.setObjectName("caption")
        self.export_status.setWordWrap(True)
        layout.addWidget(self.export_status)

        return card

    def create_empty_view(self) -> EmptyState:
        state = EmptyState("", "")
        self.btn_recheck = state.add_action("重新检查", self.scan_runs_directory, primary=True)
        self.empty_state = state
        return state

    def create_loading_view(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("正在读取训练结果…")
        label.setObjectName("subtitle")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        return holder

    # ==================== 页面状态 ====================

    def _show_empty(self, title: str, message: str, show_action: bool = True):
        self.empty_state.set_text(title, message)
        self.btn_recheck.setVisible(show_action)
        self.view_stack.setCurrentIndex(1)

    def _show_loading(self):
        self.view_stack.setCurrentIndex(2)
        QApplication.processEvents()

    # ==================== 训练记录 ====================

    def scan_runs_directory(self):
        """扫描当前项目的训练记录。

        目录发现走 gui/workflow.find_project_runs，与侧边栏、前置条件页同源。
        """
        if not self.current_project_id:
            self.current_run = None
            self._show_empty(
                "还没有选择项目",
                "训练结果是跟着项目走的。先在左上角「当前项目」里选一个项目。",
                show_action=False,
            )
            return

        self._show_loading()
        runs = find_project_runs(self.current_project_id)

        if not runs:
            self.current_run = None
            self.current_metrics = {}
            self._show_empty(
                "这个项目还没有训练记录",
                "训练完成后，点「重新检查」就能看到。",
            )
            return

        # 最新的一次放在最上面，默认就看它
        runs = list(reversed(runs))

        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for i, run in enumerate(runs):
            text = f"{run.name}（最新）" if i == 0 else run.name
            self.run_combo.addItem(text, str(run))
        self.run_combo.setCurrentIndex(0)
        self.run_combo.blockSignals(False)

        self.run_hint.setText(f"共 {len(runs)} 次训练")
        self.view_stack.setCurrentIndex(0)
        self.load_run(runs[0])

    def on_run_changed(self, index: int):
        path = self.run_combo.itemData(index)
        if path:
            self.load_run(Path(path))

    def load_run(self, run_dir: Path):
        """把一次训练的指标、图、模型文件都摆出来。"""
        self.current_run = run_dir
        self._set_export_status("")

        self.read_training_metrics(str(run_dir))
        self._load_images(run_dir)
        self._update_model_info(run_dir)

    # ==================== 指标 ====================

    def read_training_metrics(self, project_dir) -> Optional[Dict[str, float]]:
        """从 results.csv 读取训练指标（取最后一轮，算法与原来一致）。"""
        results_csv = os.path.join(str(project_dir), "results.csv")
        self.current_metrics = {}
        self.metrics_source.setToolTip("")

        if not os.path.exists(results_csv):
            self._clear_metrics()
            self.metrics_source.setText("这次训练目录里没有 results.csv，读不到指标。")
            self.metrics_source.setToolTip(
                "训练很可能中途停了。下面的图和模型文件如果存在，仍然可以看、可以导出。"
            )
            self._apply_verdict(None)
            return None

        try:
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                self._clear_metrics()
                self.metrics_source.setText("results.csv 是空的，读不到指标。")
                self._apply_verdict(None)
                return None

            # 有些 ultralytics 版本会在表头列名前后留空格
            last_row = {
                (key or '').strip(): value
                for key, value in rows[-1].items()
            }

            metrics = {
                'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                '精确率': float(last_row.get('metrics/precision(B)', 0)),
                '召回率': float(last_row.get('metrics/recall(B)', 0)),
            }

            self.current_metrics = metrics
            self.update_metrics(metrics)
            self.metrics_source.setText(
                f"最后一轮在验证集上的成绩 · 共 {len(rows)} 轮"
            )
            return metrics

        except Exception as e:
            print(f"读取指标失败: {e}")
            self._clear_metrics()
            self.metrics_source.setText(f"读取 results.csv 失败：{e}")
            self._apply_verdict(None)
            return None

    def update_metrics(self, metrics: Dict):
        """更新指标显示。"""
        for metric, value in metrics.items():
            if metric in self.metric_labels:
                self.metric_labels[metric].setText(f"{value:.4f}")
        self._apply_verdict(metrics.get('mAP50'))

    def _clear_metrics(self):
        for label in self.metric_labels.values():
            label.setText("--")

    def _apply_verdict(self, map50: Optional[float]):
        """把 mAP50 翻译成一句新手能直接用的结论。"""
        if map50 is None:
            state, mark, title = 'unknown', '?', "读不到这次训练的成绩"
            detail = "没有找到指标文件，无法判断这次训练好不好。"
            tip = "训练可能中途停了，可以回到「模型训练」重新跑一次。"
        elif map50 >= 0.7:
            state, mark, title = 'success', '✓', "这次训练看起来可以用"
            detail = f"mAP50 {map50:.2f}，在验证集上表现不错。"
            tip = "导出前可以先去「模型测试」用几张新图片确认一下。"
        elif map50 >= 0.4:
            state, mark, title = 'warning', '!', "勉强能用，但还有明显提升空间"
            detail = f"mAP50 {map50:.2f}，模型能认出不少目标，但漏检和误报还比较多。"
            tip = "想更好：多标一些图片，或把训练轮数调高再训一次。"
        else:
            state, mark, title = 'error', '×', "效果偏弱，先别急着用"
            detail = f"mAP50 {map50:.2f}，模型还没学明白。"
            tip = "最常见原因：标注太少、类别标得不一致、训练轮数太少。"

        self.verdict_detail.setToolTip(tip)

        color = {
            'success': COLORS['success'],
            'warning': COLORS['warning'],
            'error': COLORS['error'],
            'unknown': COLORS['text_secondary'],
        }[state]

        # 不只靠颜色区分：符号 + 文字本身也说明了结论
        self.verdict_badge.setText(mark)
        self.verdict_badge.setStyleSheet(
            f"color: {color};"
            f"border: 1px solid {color};"
            f"border-radius: 11px;"
            f"min-width: 22px; max-width: 22px;"
            f"min-height: 22px; max-height: 22px;"
            f"font-weight: 600;"
        )
        self.verdict_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_title.setText(title)
        self.verdict_detail.setText(detail)

    # ==================== 图像 ====================

    def _load_images(self, run_dir: Path):
        """把结果图按「曲线 / 预测示例 / 其他」分到三个标签页。"""
        charts: List[Tuple[str, str]] = []
        preds: List[Tuple[str, str]] = []
        others: List[Tuple[str, str]] = []

        for root, _, files in os.walk(run_dir):
            for filename in sorted(files):
                if not filename.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                path = os.path.join(root, filename)
                if filename.startswith(('val_batch', 'train_batch')):
                    preds.append((filename, path))
                elif chart_rank(filename) < len(CHART_ORDER):
                    charts.append((filename, path))
                else:
                    others.append((filename, path))

        charts.sort(key=lambda item: chart_rank(item[0]))

        self.chart_pane.set_images(charts)
        self.pred_pane.set_images(preds)
        self.other_pane.set_images(others)

        self.tabs.setTabEnabled(2, bool(others))

    # ==================== 模型文件与导出 ====================

    def _best_model(self) -> Optional[Path]:
        if not self.current_run:
            return None
        best = self.current_run / "weights" / "best.pt"
        return best if best.exists() else None

    def _update_model_info(self, run_dir: Path):
        """模型文件在不在、多大、在哪。不在就把导出按钮禁掉并说明原因。"""
        best = self._best_model()

        if best:
            size_mb = best.stat().st_size / (1024 * 1024)
            self.model_info.setText(
                f"best.pt · {size_mb:.1f} MB · {self._display_path(best.parent)}"
            )
            self.model_info.setToolTip("best.pt 是这次训练中表现最好的一轮权重，导出、测试用的都是它。")
            self.model_info.setStyleSheet("")
            self.btn_export_model.setEnabled(True)
            self.btn_export_model.setToolTip(
                "选择一种格式导出这次训练的 best.pt；ONNX / TensorRT 转换可能要几十秒，期间界面会没有响应。"
            )
        else:
            self.model_info.setText(f"没有在 {self._display_path(run_dir / 'weights')} 找到 best.pt。")
            self.model_info.setToolTip("训练可能中途被停掉了，没来得及保存权重。回到「模型训练」重新跑一次即可。")
            self.model_info.setStyleSheet(f"color: {COLORS['warning']};")
            self.btn_export_model.setEnabled(False)
            self.btn_export_model.setToolTip("没有 best.pt，没有可导出的模型")

        self.btn_export_folder.setEnabled(run_dir.exists())

    def _display_path(self, path: Path) -> str:
        """尽量显示成 runs/train/exp_1 这样的短路径。

        基准是应用根目录，不是当前工作目录：打包成 EzYOLO.app 双击启动时 cwd 是 /，
        以前这里会退回去显示一长串绝对路径。
        """
        try:
            return str(path.relative_to(APP_ROOT))
        except ValueError:
            return str(path)

    def _set_export_status(self, text: str, state: str = 'idle'):
        self._export_status_state = state
        color = {
            'idle': COLORS['text_secondary'],
            'busy': COLORS['text_secondary'],
            'success': COLORS['success'],
            'error': COLORS['error'],
        }[state]
        self.export_status.setText(text)
        self.export_status.setToolTip("")
        self.export_status.setStyleSheet(f"color: {color};")

    def _begin_export(self, text: str):
        self.btn_export_model.setEnabled(False)
        self.btn_export_folder.setEnabled(False)
        self._set_export_status(text, 'busy')
        QApplication.processEvents()

    def _end_export(self, text: str, state: str):
        self.btn_export_model.setEnabled(self._best_model() is not None)
        self.btn_export_folder.setEnabled(bool(self.current_run))
        self._set_export_status(text, state)

    def _end_export_success(self, path: str):
        """导出成功，路径可能很长：卡片里只显示省略后的样子，完整路径放 tooltip。"""
        prefix = "导出成功："
        metrics = self.export_status.fontMetrics()
        available = max(self.export_status.width() - metrics.horizontalAdvance(prefix) - 8, 160)
        elided = metrics.elidedText(path, Qt.TextElideMode.ElideMiddle, available)
        self._end_export(f"{prefix}{elided}", 'success')
        self.export_status.setToolTip(path)

    def on_export_format(self, format_type: str):
        if format_type == 'pt':
            self.export_model_pt()
        else:
            self.export_model(format_type)

    def export_model(self, format_type):
        """导出模型（onnx / torchscript / tensorrt）。"""
        best_model = self._best_model()
        if not best_model:
            QMessageBox.warning(self, "提示", "这次训练没有 best.pt，无法导出")
            return

        project_dir = str(self.current_run)

        # 显示导出配置对话框
        export_config = {}
        if format_type == 'onnx':
            dialog = ONNXExportDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_config = dialog.get_config()
        elif format_type == 'tensorrt':
            dialog = TensorRTExportDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_config = dialog.get_config()

        # 选择导出路径
        file_filter = ""
        if format_type == 'onnx':
            file_filter = "ONNX files (*.onnx)"
        elif format_type == 'tensorrt':
            file_filter = "TensorRT files (*.engine)"
        elif format_type == 'torchscript':
            file_filter = "TorchScript files (*.pt)"

        default_name = f"best.{format_type}"
        if format_type == 'tensorrt':
            default_name = "best.engine"

        save_path, _ = QFileDialog.getSaveFileName(
            self, f"导出为{format_type.upper()}",
            os.path.join(project_dir, default_name),
            file_filter
        )

        if not save_path:
            return

        self._begin_export(f"正在导出 {format_type.upper()}…")

        try:
            from ultralytics import YOLO

            # 加载模型
            model = YOLO(str(best_model))

            # 构建导出参数（Ultralytics 的 TensorRT 导出格式名是 engine）
            export_format = 'engine' if format_type == 'tensorrt' else format_type
            export_kwargs = {'format': export_format}

            # 添加配置参数（针对ONNX和TensorRT）
            if format_type in ['onnx', 'tensorrt']:
                export_kwargs.update(export_config)
                # 过滤掉None值
                export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}

            print(f"[导出] 参数: {export_kwargs}")

            # 导出
            model.export(**export_kwargs)

            # 移动文件
            if format_type == 'onnx':
                export_path = os.path.join(project_dir, "weights", "best.onnx")
            elif format_type == 'tensorrt':
                export_path = os.path.join(project_dir, "weights", "best.engine")
            elif format_type == 'torchscript':
                export_path = os.path.join(project_dir, "weights", "best.torchscript.pt")

            if os.path.exists(export_path):
                shutil.move(export_path, save_path)
                self._end_export_success(save_path)
                QMessageBox.information(self, "成功", f"模型已导出为 {save_path}")
            else:
                self._end_export("导出失败：没有生成模型文件，详情见控制台日志。", 'error')
                QMessageBox.warning(self, "失败", "导出失败，请检查日志")

        except Exception as e:
            self._end_export(f"导出失败：{e}", 'error')
            QMessageBox.warning(self, "错误", f"导出出错: {str(e)}")

    def export_model_pt(self):
        """导出为pt文件（直接复制 best.pt）。"""
        best_model = self._best_model()
        if not best_model:
            QMessageBox.warning(self, "提示", "这次训练没有 best.pt，无法导出")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出为pt",
            os.path.join(str(self.current_run), "best.pt"),
            "PyTorch files (*.pt)"
        )

        if not save_path:
            return

        self._begin_export("正在导出 PyTorch 权重…")

        try:
            shutil.copy2(str(best_model), save_path)
            self._end_export_success(save_path)
            QMessageBox.information(self, "成功", f"模型已导出为 {save_path}")

        except Exception as e:
            self._end_export(f"导出失败：{e}", 'error')
            QMessageBox.warning(self, "错误", f"导出出错: {str(e)}")

    def export_result_folder(self):
        """导出整个结果文件夹。"""
        if not self.current_run or not self.current_run.exists():
            QMessageBox.warning(self, "提示", "没有可导出的训练结果")
            return

        run_dir = self.current_run

        # 选择目标文件夹
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目标文件夹",
            str(run_dir.parent)
        )

        if not save_dir:
            return

        target_dir = os.path.join(save_dir, run_dir.name)

        # 会覆盖同名文件夹，先说清楚再动手
        if os.path.exists(target_dir):
            confirm = QMessageBox.question(
                self, "目标文件夹已存在",
                f"{target_dir}\n\n这个文件夹已经存在，继续会先删除它再写入。要继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        self._begin_export("正在复制结果文件夹…")

        try:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)

            shutil.copytree(str(run_dir), target_dir)
            self._end_export_success(target_dir)
            QMessageBox.information(self, "成功", f"结果文件夹已导出到 {target_dir}")

        except Exception as e:
            self._end_export(f"导出失败：{e}", 'error')
            QMessageBox.warning(self, "错误", f"导出出错: {str(e)}")

    def refresh_theme(self):
        """主题切换后刷新预览区与导出状态色。"""
        for pane in (
            getattr(self, 'chart_pane', None),
            getattr(self, 'pred_pane', None),
            getattr(self, 'other_pane', None),
        ):
            refresh = getattr(pane, 'refresh_theme', None)
            if callable(refresh):
                refresh()
        if hasattr(self, 'export_status'):
            state = getattr(self, '_export_status_state', 'idle')
            self._set_export_status(self.export_status.text(), state)
        if hasattr(self, 'model_info') and self.model_info.text().startswith("没有在"):
            self.model_info.setStyleSheet(f"color: {COLORS['warning']};")
        elif hasattr(self, 'model_info'):
            self.model_info.setStyleSheet("")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = ResultPage()
    window.show()
    sys.exit(app.exec())
