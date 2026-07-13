# -*- coding: utf-8 -*-
"""批量标注的确认框，以及它确认的那份「计划快照」。

为什么要有快照：确认框上写的和真正交给线程跑的，必须是同一份东西。
以前的写法是确认框读一遍界面控件、执行时再读一遍——中间用户切了个项目、
改了个类别、动了个滑杆，就会出现「看到的是 A，跑的是 B」。BatchPlan 是
frozen dataclass，构造时把图片列表和类别映射都拷贝一份；确认框把它原样
还给调用方，调用方直接拿它去跑，中途不再回头看任何可变的 UI 状态。

YOLO 和 LLM 共用这一个确认框：只修一处，两个入口都得到同样的保护。
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

SCOPE_UNLABELED = 'unlabeled'
SCOPE_ALL = 'all'
SCOPE_RANGE = 'range'

SCOPE_LABELS = {
    SCOPE_UNLABELED: "仅未标注的图片",
    SCOPE_ALL: "全部图片",
    SCOPE_RANGE: "指定范围",
}

# 这两个参数只有 YOLO 有；LLM 走的是自然语言提示词，没有阈值可调。
# 与其在确认框上留两个空格子让人猜，不如明说「不适用」。
NOT_APPLICABLE = "不适用"


@dataclass(frozen=True)
class BatchPlan:
    """一次批量标注的完整快照。确认之后不再变，执行只认它。"""

    project_id: int
    engine: str                       # 'yolo' | 'llm'
    images: Tuple[Dict[str, Any], ...]
    scope: str
    model_label: str                  # 确认框上显示的模型/版本
    model_path: str = ''
    model_task: str = 'detect'
    conf: Optional[float] = None      # None 表示这个引擎没有这个参数
    iou: Optional[float] = None
    overwrite: bool = False
    class_mapping: Dict = field(default_factory=dict)
    class_id: Optional[int] = None    # LLM：整批都打这一个类别
    class_name: str = ''

    def __post_init__(self):
        # 拷贝而不是引用：调用方后面再怎么动它自己那份列表，都影响不到这份快照
        object.__setattr__(self, 'images', tuple(dict(img) for img in self.images))
        object.__setattr__(self, 'class_mapping', dict(self.class_mapping or {}))

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def image_ids(self) -> List[int]:
        return [img.get('id') for img in self.images]

    @property
    def image_paths(self) -> List[str]:
        return [img.get('storage_path', '') for img in self.images]

    def with_images(self, images) -> 'BatchPlan':
        """按选定范围裁出一份新快照（仍然是冻结的）。"""
        return replace(self, images=tuple(images))

    # ---- 确认框上的措辞 ----

    def scope_text(self) -> str:
        return SCOPE_LABELS.get(self.scope, self.scope)

    def conf_text(self) -> str:
        return NOT_APPLICABLE if self.conf is None else f"{self.conf:.2f}"

    def iou_text(self) -> str:
        return NOT_APPLICABLE if self.iou is None else f"{self.iou:.2f}"

    def overwrite_text(self) -> str:
        if self.overwrite:
            return "是（覆盖已有标注，不可撤销）"
        return "否（保留已有标注）"


class BatchConfirmDialog(QDialog):
    """批量标注前的确认：把要做什么、对多少张图做、按什么参数做，全摆出来。

    默认焦点在「取消」——一整批覆盖标注不该是一个回车就发生的事。
    测试里不用真开模态，直接点 btn_confirm / btn_cancel 就行。
    """

    def __init__(self, parent, plan: BatchPlan, allow_range: bool = False):
        super().__init__(parent)

        self._plan = plan
        self._allow_range = allow_range

        self.setWindowTitle("确认批量标注")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(10)

        title = QLabel("批量标注")
        title.setObjectName("h2")
        layout.addWidget(title)

        subtitle = QLabel("确认下面的范围和参数，再开始。")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # 起止范围（LLM 用）：以前这是一个单独的弹窗，确认完范围还要再确认一次参数。
        # 合并进来——一次看全，一次决定。
        self.spin_start = None
        self.spin_end = None
        if allow_range and plan.count:
            layout.addWidget(self._build_range_row(plan.count))

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(8)

        self.lbl_count = QLabel()
        self.lbl_scope = QLabel(plan.scope_text())
        self.lbl_model = QLabel(plan.model_label or "未指定")
        self.lbl_conf = QLabel(plan.conf_text())
        self.lbl_iou = QLabel(plan.iou_text())
        self.lbl_overwrite = QLabel(plan.overwrite_text())

        form.addRow("处理张数", self.lbl_count)
        form.addRow("范围", self.lbl_scope)
        form.addRow("模型 / 版本", self.lbl_model)
        form.addRow("置信度", self.lbl_conf)
        form.addRow("IoU", self.lbl_iou)
        form.addRow("覆盖已有标注", self.lbl_overwrite)

        if plan.class_name:
            form.addRow("目标类别", QLabel(plan.class_name))

        layout.addLayout(form)

        if plan.overwrite:
            warning = QLabel("已有标注会被删除后重写，无法撤销。")
            warning.setObjectName("caption")
            warning.setWordWrap(True)
            layout.addWidget(warning)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 10, 0, 0)
        button_row.setSpacing(8)
        button_row.addStretch()

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(self.btn_cancel)

        self.btn_confirm = QPushButton("开始批量标注")
        self.btn_confirm.setObjectName("primary")
        self.btn_confirm.clicked.connect(self.accept)
        button_row.addWidget(self.btn_confirm)

        layout.addLayout(button_row)

        # 默认焦点给取消：一整批推理是要花时间、还可能覆盖标注的事
        self.btn_cancel.setDefault(True)
        self.btn_cancel.setFocus()

        self._refresh_count()

    def _build_range_row(self, total: int) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        row_layout.addWidget(QLabel("起始图片"))
        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, total)
        self.spin_start.setValue(1)
        self.spin_start.valueChanged.connect(self._on_range_changed)
        row_layout.addWidget(self.spin_start)

        row_layout.addWidget(QLabel("结束图片"))
        self.spin_end = QSpinBox()
        self.spin_end.setRange(1, total)
        self.spin_end.setValue(total)
        self.spin_end.valueChanged.connect(self._on_range_changed)
        row_layout.addWidget(self.spin_end)

        row_layout.addStretch()
        return row

    def _on_range_changed(self):
        # 起点不能越过终点，否则「处理张数」会算出负数
        if self.spin_start.value() > self.spin_end.value():
            sender = self.sender()
            if sender is self.spin_start:
                self.spin_end.setValue(self.spin_start.value())
            else:
                self.spin_start.setValue(self.spin_end.value())
        self._refresh_count()

    def _selected_images(self):
        if self.spin_start is None or self.spin_end is None:
            return list(self._plan.images)
        start = self.spin_start.value() - 1
        end = self.spin_end.value()
        return list(self._plan.images[start:end])

    def _refresh_count(self):
        count = len(self._selected_images())
        self.lbl_count.setText(f"{count} 张")
        self.btn_confirm.setEnabled(count > 0)

    def selected_plan(self) -> BatchPlan:
        """用户确认的那一份快照——执行时直接用它，不要再回头读界面。"""
        if self.spin_start is None:
            return self._plan
        return self._plan.with_images(self._selected_images())


def confirm_batch_plan(parent, plan: BatchPlan,
                       allow_range: bool = False) -> Optional[BatchPlan]:
    """弹确认框。用户确认返回那份快照，取消返回 None。

    取消这条路上什么都不做：不建线程、不写数据库、不改图片状态。
    """
    dialog = BatchConfirmDialog(parent, plan, allow_range=allow_range)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.selected_plan()
