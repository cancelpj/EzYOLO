# -*- coding: utf-8 -*-
"""
标注页面
支持矩形框、多边形标注，快捷键操作，自动保存
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QFileDialog,
    QMenu, QMessageBox, QComboBox, QLineEdit, QSplitter,
    QListWidget, QListWidgetItem, QButtonGroup,
    QRadioButton, QSpinBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QCheckBox, QSlider, QTextEdit, QInputDialog,
    QSizePolicy, QToolButton,
    QColorDialog, QDialog, QApplication, QAbstractSpinBox
)
import math
import copy

# 导入自动标注相关模块
from gui.pages.auto_label_dialog import AutoLabelDialog
from gui.pages.batch_process_dialog import BatchProcessDialog
from gui.pages.settings_page import event_matches_shortcut
from gui.widgets.batch_confirm_dialog import (
    BatchPlan, SCOPE_ALL, SCOPE_RANGE, SCOPE_UNLABELED, confirm_batch_plan,
)
from core.auto_labeler import BatchLabelingManager
from core.model_manager import ModelManager
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QPoint, QRect, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QFontMetrics, QKeyEvent, QMouseEvent, QWheelEvent, QAction, QIcon, QPen, QBrush, QShortcut, QKeySequence
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import gc
import json
import threading
import random
import sys

from gui.styles import COLORS, RADIUS_SM, get_primary_font_family, set_menu_indicator
from gui.display_names import (
    build_project_display_names, display_name, display_names, parse_display_name_rule,
)
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.context_help import ContextHelp
from models.database import db
from gui.widgets.loading_dialog import LoadingOverlay
from gui.view_zoom import (
    ZOOM_MAX, ZOOM_MIN, native_zoom_factor, pinch_zoom_factor,
    wheel_zoom_factor, zoom_at,
)

SAM3_DOWNLOAD_URL = "https://huggingface.co/1038lab/sam3/discussions/1"
NEGATIVE_SAMPLE_CLASS_ID = -1

_ASSETS_DIR = Path(__file__).parent.parent / "assets"


def _asset_icon(name: str) -> QIcon:
    """gui/assets 下的 SVG 图标。"""
    return QIcon(str(_ASSETS_DIR / name))

# 工具栏按钮基础高度：实际高度还会按当前字体和 sizeHint 动态向上增长。
# QToolButton（带下拉箭头）和 QPushButton 的默认高度不一致，所以需要统一下限。
TOOLBAR_BUTTON_HEIGHT = 32


def _readable_on_light(color: str) -> str:
    """把类别色压到浅底上读得出来的程度。

    类别色是用户挑给「画在图片上的框」用的，亮黄、亮青这类颜色画在照片上很显眼，
    但直接拿去当白底上的文字色就糊了。这里只在太亮时压暗，色相保持不变——
    当前类别的颜色线索还在，字也还看得清。
    """
    if not color:
        return COLORS['text_primary']

    qcolor = QColor(color)
    if not qcolor.isValid():
        return COLORS['text_primary']

    # 感知亮度（ITU-R BT.601）：白底上超过这个值的颜色基本没法当正文看
    luminance = (
        0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()
    ) / 255
    while luminance > 0.55:
        qcolor = qcolor.darker(130)
        luminance = (
            0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()
        ) / 255
    return qcolor.name()


class SAMModelManager:
    """SAM模型缓存管理器：同配置复用，切换配置自动释放重载。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._model_key = None

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_model(self, sam_type: str, model_path: str, device: str):
        """获取可复用模型；配置变化时释放旧模型并重载。"""
        model_key = (sam_type, model_path, str(device))
        with self._lock:
            if self._model is not None and self._model_key == model_key:
                return self._model

            self._release_model_locked()

            if sam_type == "FastSAM":
                from ultralytics import FastSAM
                self._model = FastSAM(model_path)
            else:
                # SAM/SAM2/MobileSAM/SAM3 均使用 SAM 类
                from ultralytics import SAM
                self._model = SAM(model_path)

            self._model_key = model_key
            return self._model

    def release_model(self):
        """主动释放当前缓存模型。"""
        with self._lock:
            self._release_model_locked()

    def _release_model_locked(self):
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None
            self._model_key = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class SAMMemoryPredictorManager:
    """SAM2/SAM3 记忆预测器管理：按配置复用，切换配置释放。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._predictor = None
        self._predictor_key = None

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_predictor(self, sam_config: dict):
        sam_type = sam_config.get("sam_type", "SAM")
        model_file = sam_config.get("model_file", "sam2_t.pt")
        device = sam_config.get("device", "cpu")
        imgsz = int(sam_config.get("imgsz", 1024))
        # 动态记忆分割对阈值更敏感，使用较低conf避免把有效mask过滤空
        conf = 0.01
        model_key = (sam_type, model_file, str(device), imgsz)

        with self._lock:
            if self._predictor is not None and self._predictor_key == model_key:
                return self._predictor

            self._release_locked()

            from ultralytics.models.sam import SAM2DynamicInteractivePredictor
            overrides = dict(
                conf=conf,
                task="segment",
                mode="predict",
                imgsz=imgsz,
                model=model_file,
                save=False,
                device=device,
                verbose=False,
            )
            self._predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=20)
            self._predictor_key = model_key
            return self._predictor

    def clear(self):
        with self._lock:
            self._release_locked()

    def _release_locked(self):
        if self._predictor is not None:
            try:
                del self._predictor
            except Exception:
                pass
            self._predictor = None
            self._predictor_key = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class SAMMemoryObjectsDialog(QDialog):
    """记忆标注对象管理弹窗（可在画布上交互后添加对象）。"""

    add_requested = pyqtSignal()
    delete_requested = pyqtSignal(int)
    save_requested = pyqtSignal()
    closed = pyqtSignal()  # 窗口关闭信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM记忆对象管理")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.info_label = QLabel("请在图片上用SAM提示（点/框）后点击“添加对象”。")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.count_label = QLabel("当前对象数: 0")
        layout.addWidget(self.count_label)

        self.obj_list = QListWidget()
        layout.addWidget(self.obj_list)

        row = QHBoxLayout()
        self.btn_add = QPushButton("添加对象")
        self.btn_del = QPushButton("删除对象")
        self.btn_save = QPushButton("保存并更新记忆")
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_save)
        layout.addLayout(row)

        self.btn_add.clicked.connect(self.add_requested.emit)
        self.btn_del.clicked.connect(self._on_delete_clicked)
        self.btn_save.clicked.connect(self.save_requested.emit)

    def closeEvent(self, event):
        """窗口关闭时发送信号"""
        self.closed.emit()
        super().closeEvent(event)

    def _on_delete_clicked(self):
        row = self.obj_list.currentRow()
        if row >= 0:
            self.delete_requested.emit(row)

    def update_objects(self, objects: list):
        self.obj_list.clear()
        for obj in objects:
            desc = f"ID={obj.get('obj_id')}  点={len(obj.get('points', []))}  框={len(obj.get('bboxes', []))}"
            self.obj_list.addItem(desc)
        self.count_label.setText(f"当前对象数: {len(objects)}")


class SAMInferenceWorker(QThread):
    """SAM推理工作线程"""
    
    inference_finished = pyqtSignal(bool, str, object)  # 成功, 消息, mask
    
    def __init__(self, sam_config: dict, image_path: str, points: list = None, bboxes: list = None):
        super().__init__()
        self.sam_config = sam_config
        self.image_path = image_path
        self.points = points or []
        self.bboxes = bboxes or []

    def _build_predict_kwargs(self) -> dict:
        """构建统一推理参数，确保UI配置生效。"""
        kwargs = {"verbose": False}
        if self.sam_config.get('imgsz'):
            kwargs["imgsz"] = int(self.sam_config['imgsz'])
        if self.sam_config.get('device') is not None:
            kwargs["device"] = self.sam_config['device']
        if self.sam_config.get('retina_masks') is not None:
            kwargs["retina_masks"] = bool(self.sam_config['retina_masks'])
        return kwargs
        
    def run(self):
        """运行SAM推理"""
        try:
            sam_type = self.sam_config.get('sam_type', 'SAM')
            model_file = self.sam_config.get('model_file', 'sam_b.pt')
            device = self.sam_config.get('device', 'cpu')
            
            # 检查模型文件是否存在
            import os
            model_paths = [
                model_file,
                os.path.join('models', model_file),
                os.path.join(os.path.expanduser('~'), '.cache', 'ultralytics', model_file),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                if sam_type == "SAM3":
                    self.inference_finished.emit(
                        False,
                        "SAM3模型未找到: sam3.pt\n"
                        "SAM3不支持自动下载。\n"
                        "请到以下页面下载 sam3.pt 后放到项目根目录：\n"
                        f"{SAM3_DOWNLOAD_URL}",
                        None
                    )
                else:
                    self.inference_finished.emit(False, f"模型文件不存在: {model_file}\n请下载模型后重试", None)
                return
            
            # 模型缓存复用：同配置仅加载一次，切换配置时自动释放重载
            model = SAMModelManager.instance().get_model(sam_type, model_path, device)

            predict_kwargs = self._build_predict_kwargs()
            if sam_type == "FastSAM":
                predict_kwargs["conf"] = float(self.sam_config.get("conf", 0.4))
                predict_kwargs["iou"] = float(self.sam_config.get("iou", 0.9))
            
            # 准备提示
            if self.points:
                # 点提示
                points_array = [[p[0], p[1]] for p in self.points]
                labels = [1] * len(points_array)  # 1表示前景
                results = model(self.image_path, points=points_array, labels=labels, **predict_kwargs)
            elif self.bboxes:
                # 框提示
                bbox = self.bboxes[-1]  # 使用最后一个框
                results = model(self.image_path, bboxes=[bbox], **predict_kwargs)
            else:
                self.inference_finished.emit(False, "没有提供提示", None)
                return
            
            # 提取mask
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy() if hasattr(result.masks.data, 'cpu') else result.masks.data
                    scores = None
                    if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result.boxes, 'conf'):
                        confs = result.boxes.conf
                        if confs is not None:
                            scores = confs.cpu().numpy() if hasattr(confs, 'cpu') else confs
                    self.inference_finished.emit(True, "推理成功", {"masks": masks, "scores": scores})
                else:
                    self.inference_finished.emit(False, "未检测到分割结果", None)
            else:
                self.inference_finished.emit(False, "推理无结果", None)
                
        except Exception as e:
            sam_type = self.sam_config.get('sam_type', 'SAM')
            if sam_type == "SAM3":
                self.inference_finished.emit(
                    False,
                    "SAM3加载/推理失败。\n"
                    "请确认 sam3.pt 已放在项目根目录且文件可用。\n"
                    f"下载说明: {SAM3_DOWNLOAD_URL}\n"
                    f"错误详情: {str(e)}",
                    None
                )
            else:
                self.inference_finished.emit(False, f"推理出错: {str(e)}", None)


def redact_secret(text: str, secret: str) -> str:
    """把 API Key 从要显示/打印的文字里抹掉。

    异常信息里偶尔会带上请求参数；密钥不该进日志、不该进弹窗、更不该进测试输出。
    """
    if not secret or not text:
        return text
    return text.replace(secret, "***")


def llm_detect(config: dict, image_path: str, target: str) -> List[Dict]:
    """调一次视觉大模型，返回 [{'label', 'bbox'}]。

    只做网络请求和解析，不碰界面也不碰数据库——单张和批量共用同一条路径，
    调用方负责把它放进后台线程。
    """
    import base64
    import re
    from openai import OpenAI

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])

    completion = client.chat.completions.create(
        model=config['model_name'],
        messages=[
            {"role": "system", "content": config['system_prompt']},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config['user_prompt'].format(target=target)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            }
        ]
    )

    response_text = completion.choices[0].message.content or ""

    # 约定的返回格式: target,[xmin,ymin,xmax,ymax]
    detections = []
    for label, xmin, ymin, xmax, ymax in re.findall(
        r'([^,\n]+),\[(\d+),(\d+),(\d+),(\d+)\]', response_text
    ):
        detections.append({
            "label": label.strip(),
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
        })
    return detections


class LLMBatchWorker(QThread):
    """LLM 批量推理：一整批图片的网络请求都在这个线程里跑。

    原来这段是在界面线程里 for 循环逐张发请求，靠 QProgressDialog 撑场面——
    每张图要等几秒，这几秒里窗口是死的（拖不动、点不了、取消键也要等当前这张
    回来才轮得到）。搬到后台之后：进度实时、取消立刻生效、失败的图片单独记账。

    线程只负责「发请求 + 解析」；标注写库仍然回到界面线程做（signal 带回结果），
    数据库的写入路径不多一个并发来源。
    """

    image_started = pyqtSignal(int, str)     # 第几张（从 1 数）, 文件名
    image_done = pyqtSignal(int, list, str)  # image_id, 检测结果, 出错原因（''=成功）
    batch_finished = pyqtSignal(int, int, bool)  # 成功数, 失败数, 是否被取消

    def __init__(self, config: dict, images: List[Dict], target: str):
        super().__init__()
        self.config = config
        self.images = images
        self.target = target
        self._cancelled = False

    def cancel(self):
        """请求取消：只置标志，调用方不阻塞等待。"""
        self._cancelled = True

    def run(self):
        succeeded = 0
        failed = 0

        for i, image_data in enumerate(self.images):
            if self._cancelled:
                break

            image_id = image_data['id']
            self.image_started.emit(i + 1, image_data.get('filename', ''))

            image_path = image_data.get('storage_path', '')
            if not image_path or not os.path.exists(image_path):
                failed += 1
                self.image_done.emit(image_id, [], "图片文件不存在")
                continue

            try:
                detections = llm_detect(self.config, image_path, self.target)
            except Exception as e:  # noqa: BLE001  单张失败不该中断整批
                failed += 1
                reason = redact_secret(str(e), self.config.get('api_key', ''))
                self.image_done.emit(image_id, [], reason or "推理失败")
                continue

            succeeded += 1
            self.image_done.emit(image_id, detections, "")

        self.batch_finished.emit(succeeded, failed, self._cancelled)


class AnnotateImageLoadWorker(QThread):
    """标注页面图片加载工作线程"""
    
    image_loaded = pyqtSignal(int, object)  # 索引, 缩略图
    finished_loading = pyqtSignal()
    
    def __init__(self, images: List[Dict]):
        super().__init__()
        self.images = images
        self._is_running = True
    
    def run(self):
        """在后台线程中加载图片"""
        for i, image_data in enumerate(self.images):
            if not self._is_running:
                break
            
            storage_path = image_data.get('storage_path', '')
            pixmap = None
            
            if storage_path and os.path.exists(storage_path):
                try:
                    img = cv2.imread(storage_path)
                    if img is not None:
                        img = cv2.resize(img, (80, 80))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                except Exception:
                    pass
            
            if pixmap is None or pixmap.isNull():
                pixmap = QPixmap(80, 80)
                pixmap.fill(QColor(COLORS['sidebar']))
            
            self.image_loaded.emit(i, pixmap)
            
            if i % 10 == 0:
                self.msleep(1)
        
        self.finished_loading.emit()
    
    def stop(self):
        self._is_running = False


class RandomSampleDeleteWorker(QThread):
    """随机删除样本工作线程"""

    progress_updated = pyqtSignal(int, int, str)  # 当前进度, 总数, 文件名
    delete_finished = pyqtSignal(object)  # 删除结果
    delete_failed = pyqtSignal(str)

    def __init__(self, project_id: int, target_class_id: int, delete_count: int, current_image_id: Optional[int]):
        super().__init__()
        self.project_id = project_id
        self.target_class_id = target_class_id
        self.delete_count = delete_count
        self.current_image_id = current_image_id
        self._is_running = True

    def run(self):
        try:
            if self.target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
                candidate_images = db.get_negative_sample_images(self.project_id, annotated_only=True)
            else:
                candidate_images = db.get_project_images_by_class(self.project_id, self.target_class_id)

            available_count = len(candidate_images)
            total_count = min(self.delete_count, available_count)
            if total_count <= 0:
                self.delete_finished.emit({
                    'deleted_count': 0,
                    'failed_files': [],
                    'current_image_deleted': False,
                    'canceled': False,
                    'available_count': available_count,
                    'total_count': 0,
                })
                return

            target_images = random.sample(candidate_images, total_count)
            deleted_count = 0
            failed_files = []
            current_image_deleted = False

            for index, image in enumerate(target_images, start=1):
                if not self._is_running:
                    self.delete_finished.emit({
                        'deleted_count': deleted_count,
                        'failed_files': failed_files,
                        'current_image_deleted': current_image_deleted,
                        'canceled': True,
                        'available_count': available_count,
                        'total_count': total_count,
                    })
                    return

                filename = image.get('filename', str(image.get('id')))
                self.progress_updated.emit(index - 1, total_count, filename)

                if image['id'] == self.current_image_id:
                    current_image_deleted = True

                if db.delete_image(image['id']):
                    deleted_count += 1
                else:
                    failed_files.append(filename)

                self.progress_updated.emit(index, total_count, filename)

            self.delete_finished.emit({
                'deleted_count': deleted_count,
                'failed_files': failed_files,
                'current_image_deleted': current_image_deleted,
                'canceled': False,
                'available_count': available_count,
                'total_count': total_count,
            })
        except Exception as e:
            self.delete_failed.emit(str(e))

    def stop(self):
        self._is_running = False


class AnnotationCanvas(QFrame):
    """标注画布组件"""
    
    annotation_created = pyqtSignal(dict)  # 标注创建信号
    annotation_selected = pyqtSignal(int)  # 标注选中信号
    annotation_modified = pyqtSignal(int, dict, dict)  # 标注修改信号：id, 修改前 data, 修改后 data
    annotation_deleted = pyqtSignal(int)  # 标注删除信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        
        # 图像数据
        self.current_image = None
        self.current_image_path = None
        self.image_scale = 1.0
        self.image_offset = QPoint(0, 0)

        # 视图锁：锁上之后切换图片不再重新适配窗口，保持当前缩放和位置
        self.view_locked = False
        self.lock_button = None

        # 标注数据
        self.annotations = []  # 当前图像的所有标注
        self.selected_annotation_id = None
        self.current_tool = 'rectangle'  # rectangle, polygon, move
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        # 多边形绘制
        self.polygon_points = []
        
        # 关键点绘制
        self.keypoints = []
        self.drawing_keypoint = False
        
        # OBB绘制状态
        self.obb_state = 0  # 0: 未开始, 1: 已确定第一个点, 2: 已确定第一条边
        self.obb_points = []
        
        # 编辑状态
        self.editing = False
        self.dragging = False
        self.resizing = False
        # 多边形顶点拖动（move工具下）
        self.dragging_vertex = False
        self.drag_vertex_index = None
        self.drag_start = None
        self.drag_start_annotation = None
        self.resize_handle = None
        self.resize_start_rect = None
        
        # 图片平移
        self.panning = False
        self.pan_start = None
        self.pan_start_offset = None
        
        # 手柄大小
        self.handle_size = 8
        # 多边形顶点命中阈值（控件坐标，像素）
        self.vertex_hit_radius = 8
        # 多边形顶点“磁吸/辅助命中”半径（比命中半径略大一点）
        self.vertex_snap_radius = 14
        # 当前悬停的多边形顶点 (annotation_id, vertex_index) / None
        self.hover_vertex = None
        
        # 类别颜色（动态获取）
        self.class_colors = {}
        
        # 当前选中的类别ID
        self.current_class_id = 0
        
        # 批量处理模式
        self.batch_process_mode = False
        self.batch_process_points = []
        self.batch_process_dialog = None
        
        # SAM标注模式
        self.sam_mode_active = False
        self.sam_config = None
        self.sam_image_path = None
        self.sam_points = []  # [(x, y), ...]
        self.sam_bboxes = []  # [(x1, y1, x2, y2), ...]
        self.sam_mode = 'point'  # 'point' 或 'bbox'
        self.sam_drawing_bbox = False
        self.sam_start_point = None
        self.sam_current_point = None
        # SAM运行模式：normal=普通交互(允许自动推理), memory_collect=记忆采集(禁止自动推理)
        self.sam_operation_mode = "normal"
        # SAM记忆模式显示列表
        self.memory_display_points = []  # [{"x": x, "y": y, "obj_id": id}, ...]
        self.memory_display_bboxes = []  # [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "obj_id": id}, ...]

        # 没有图片时画布中央显示的话，由页面按当前状态填写
        self.empty_hint = "请选择一张图片开始标注"

        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        # 初始样式会在主题变化时被更新
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['sidebar']};
                border: 2px solid {COLORS['border']};
            }}
        """)
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # macOS 的 Cocoa ZoomNativeGesture 已经是原生捏合输入；再让 Qt 合成
        # PinchGesture 会把同一次手势缩放两遍。其他平台保留 Qt Pinch 作为入口。
        self.pinch_gesture_registered = sys.platform != 'darwin'
        if self.pinch_gesture_registered:
            self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            self.grabGesture(Qt.GestureType.PinchGesture)

        # 视图锁按钮：浮在画布右上角
        self.lock_button = QToolButton(self)
        self.lock_button.setCheckable(True)
        self.lock_button.setFixedSize(28, 28)
        self.lock_button.setIconSize(QSize(16, 16))
        self.lock_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lock_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lock_button.toggled.connect(self._on_view_lock_toggled)
        self.lock_button.hide()
        self._refresh_lock_button()

    def _refresh_lock_button(self):
        """锁按钮的图标、提示、配色跟着锁的状态走。"""
        if self.lock_button is None:
            return

        locked = self.lock_button.isChecked()
        self.lock_button.setIcon(
            _asset_icon('lock-closed.svg' if locked else 'lock-open.svg')
        )
        self.lock_button.setToolTip(
            "已锁定：切换图片保持缩放和位置" if locked else "锁定视图"
        )
        background = COLORS['selected'] if locked else COLORS['panel']
        border = COLORS['primary'] if locked else COLORS['border']
        # padding 必须显式清零：全局 QToolButton 给了 5px 10px，28px 的方按钮会把
        # 图标压成一个小点
        self.lock_button.setStyleSheet(f"""
            QToolButton {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: {RADIUS_SM}px;
                padding: 0px;
            }}
            QToolButton:hover {{
                border-color: {COLORS['primary']};
            }}
        """)

    def _on_view_lock_toggled(self, checked: bool):
        """只改锁的状态，不动当前视图——锁上是为了保住现在看到的画面。"""
        self.view_locked = checked
        self._refresh_lock_button()

    def _position_lock_button(self):
        """右上角，距离右边和上边各 8px。"""
        if self.lock_button is None:
            return
        self.lock_button.move(self.width() - self.lock_button.width() - 8, 8)
        self.lock_button.raise_()

    def _sync_lock_button(self):
        """没有图片就没有视图可锁，按钮跟着藏起来。"""
        if self.lock_button is None:
            return
        self.lock_button.setVisible(self.current_image is not None)
        self._position_lock_button()

    def _capture_locked_view(self) -> Optional[Tuple[float, float, float]]:
        """锁定时记下缩放，以及视口中心落在当前图上的归一化位置（0~1）。"""
        if not self.view_locked or self.current_image is None or self.image_scale <= 0:
            return None

        center = self.rect().center()
        img_x, img_y = self.widget_to_image(center.x(), center.y())
        return (
            self.image_scale,
            img_x / max(self.current_image.width(), 1),
            img_y / max(self.current_image.height(), 1),
        )

    def _apply_locked_view(self, view: Tuple[float, float, float]):
        """把上一张图的缩放和归一化中心搬到新图上：同尺寸时等于原样保留。"""
        scale, norm_x, norm_y = view
        self.image_scale = scale

        width = self.current_image.width()
        height = self.current_image.height()
        center = self.rect().center()
        self.image_offset = QPoint(
            self._clamp_offset(center.x() - norm_x * width * scale, width * scale, self.width()),
            self._clamp_offset(center.y() - norm_y * height * scale, height * scale, self.height()),
        )

    @staticmethod
    def _clamp_offset(offset: float, scaled_length: float, viewport_length: int) -> int:
        """新旧图尺寸差得离谱时，别让图整个滑出画布——至少留一条边在视口里。"""
        overlap = min(40.0, scaled_length)
        low = overlap - scaled_length
        high = viewport_length - overlap
        return int(round(min(max(offset, low), high)))

    def load_image(self, image_path: str):
        """加载图像"""
        if not image_path or not os.path.exists(image_path):
            self.current_image = None
            self.current_image_path = None
            self._sync_lock_button()
            self.update()
            return

        # 使用OpenCV加载图像
        img = cv2.imread(image_path)
        if img is not None:
            # 换图之前先问上一张：锁着就把它的缩放和视口中心留下来
            locked_view = self._capture_locked_view()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.current_image = QPixmap.fromImage(qt_image)
            self.current_image_path = image_path

            if locked_view is None:
                # 没锁（或上一张压根没图）：照旧适配窗口
                self.reset_view()
            else:
                self._apply_locked_view(locked_view)

            self._sync_lock_button()
            self.update()

    def reset_view(self):
        """重置视图"""
        if self.current_image is None:
            return
        
        # 计算适应窗口的缩放比例
        widget_rect = self.rect()
        img_width = self.current_image.width()
        img_height = self.current_image.height()
        
        scale_x = (widget_rect.width() - 40) / img_width
        scale_y = (widget_rect.height() - 40) / img_height
        self.image_scale = min(scale_x, scale_y, 1.0)
        
        # 居中显示
        scaled_width = img_width * self.image_scale
        scaled_height = img_height * self.image_scale
        self.image_offset = QPoint(
            int((widget_rect.width() - scaled_width) / 2),
            int((widget_rect.height() - scaled_height) / 2)
        )
    
    def set_annotations(self, annotations: List[Dict]):
        """设置标注数据"""
        self.annotations = annotations
        self.selected_annotation_id = None
        self.update()
    
    def set_tool(self, tool: str):
        """设置当前工具"""
        self.current_tool = tool
        self.drawing = False
        self.polygon_points = []
        # 清除关键点绘制状态
        if hasattr(self, 'keypoints'):
            self.keypoints = []
        
        # 处理SAM工具
        if tool == 'sam':
            self.sam_mode_active = True
            self.sam_points = []
            self.sam_bboxes = []
            self.sam_mode = 'point'
        else:
            self.sam_mode_active = False
            self.sam_operation_mode = "normal"
        
        self.update()

    def set_sam_operation_mode(self, mode: str):
        """设置SAM交互模式。"""
        self.sam_operation_mode = mode if mode in ("normal", "memory_collect") else "normal"

    def _sam_auto_infer_enabled(self) -> bool:
        """当前状态下是否允许SAM自动推理。"""
        return self.sam_mode_active and self.sam_operation_mode == "normal"
    
    def image_to_widget(self, x: float, y: float) -> QPoint:
        """图像坐标转换为控件坐标"""
        widget_x = int(x * self.image_scale + self.image_offset.x())
        widget_y = int(y * self.image_scale + self.image_offset.y())
        return QPoint(widget_x, widget_y)
    
    def widget_to_image(self, x: int, y: int) -> Tuple[float, float]:
        """控件坐标转换为图像坐标"""
        img_x = (x - self.image_offset.x()) / self.image_scale
        img_y = (y - self.image_offset.y()) / self.image_scale
        return (img_x, img_y)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(COLORS['sidebar']))
        
        if self.current_image is None:
            # 显示提示文字
            painter.setPen(QColor(COLORS['text_secondary']))
            painter.setFont(QFont(get_primary_font_family(), 14))
            painter.drawText(
                self.rect().adjusted(40, 0, -40, 0),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.empty_hint
            )
            return
        
        # 绘制图像
        scaled_pixmap = self.current_image.scaled(
            int(self.current_image.width() * self.image_scale),
            int(self.current_image.height() * self.image_scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(self.image_offset, scaled_pixmap)
        
        # 绘制标注
        self.draw_annotations(painter)
        
        # 绘制正在绘制的矩形
        if self.drawing and self.current_tool == 'rectangle' and self.start_point and self.current_point:
            self.draw_drawing_rectangle(painter)
        
        # 绘制正在绘制的多边形
        if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
            self.draw_drawing_polygon(painter)
        
        # 绘制正在绘制的旋转矩形
        if self.current_tool == 'obb' and len(self.obb_points) > 0:
            self.draw_drawing_obb(painter)
        
        # 绘制鼠标辅助线
        if self.current_image and self.current_point:
            self.draw_guide_lines(painter)
        
        # 批量处理模式：绘制已选择的像素点
        if self.batch_process_mode and self.batch_process_points:
            self.draw_batch_process_points(painter)
        
        # SAM模式：绘制点和框
        if self.sam_mode_active:
            self.draw_sam_elements(painter)
    
    def draw_sam_elements(self, painter: QPainter):
        """绘制SAM模式的点和框"""
        # 绘制记忆对象的所有标记（不同ID用不同颜色）
        memory_points = getattr(self, 'memory_display_points', [])
        memory_bboxes = getattr(self, 'memory_display_bboxes', [])
        
        if memory_points:
            # 为不同ID分配不同颜色
            id_colors = {}
            color_list = [
                QColor(255, 0, 0),    # 红
                QColor(0, 255, 0),    # 绿
                QColor(0, 0, 255),    # 蓝
                QColor(255, 255, 0),  # 黄
                QColor(255, 0, 255),  # 紫
                QColor(0, 255, 255),  # 青
                QColor(255, 128, 0),  # 橙
                QColor(128, 0, 255),  # 紫红
            ]
            
            for point_data in memory_points:
                obj_id = point_data.get("obj_id", 0)
                if obj_id not in id_colors:
                    id_colors[obj_id] = color_list[obj_id % len(color_list)]
                color = id_colors[obj_id]
                
                img_x = point_data["x"]
                img_y = point_data["y"]
                widget_pos = self.image_to_widget(img_x, img_y)
                
                # 绘制点
                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(color))
                radius = 8
                painter.drawEllipse(widget_pos, radius, radius)
                
                # 绘制ID
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont(get_primary_font_family(), 10, QFont.Weight.Bold))
                painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, f"ID:{obj_id}")
        
        # 绘制记忆对象的所有框
        if memory_bboxes:
            id_colors = {}
            color_list = [
                QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
                QColor(255, 128, 0), QColor(128, 0, 255),
            ]
            
            for bbox_data in memory_bboxes:
                obj_id = bbox_data.get("obj_id", 0)
                if obj_id not in id_colors:
                    id_colors[obj_id] = color_list[obj_id % len(color_list)]
                color = id_colors[obj_id]
                
                x1 = bbox_data["x1"]
                y1 = bbox_data["y1"]
                x2 = bbox_data["x2"]
                y2 = bbox_data["y2"]
                
                p1 = self.image_to_widget(x1, y1)
                p2 = self.image_to_widget(x2, y2)
                
                # 绘制框
                pen = QPen(color, 2)
                pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                rect = QRect(p1, p2)
                painter.drawRect(rect)
                
                # 绘制ID标签
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color, 1))
                label_rect = QRect(p1.x(), p1.y() - 20, 40, 20)
                painter.drawRect(label_rect)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont(get_primary_font_family(), 9))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, f"ID:{obj_id}")
        
        # 绘制当前正在添加的点（临时标记）
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        for i, (img_x, img_y) in enumerate(self.sam_points):
            widget_pos = self.image_to_widget(img_x, img_y)
            radius = 8
            painter.drawEllipse(widget_pos, radius, radius)
            # 绘制点编号
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont(get_primary_font_family(), 10, QFont.Weight.Bold))
            painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, str(i + 1))
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
        
        # 绘制正在绘制的框（虚线）
        if self.sam_drawing_bbox and self.sam_start_point and self.sam_current_point:
            pen = QPen(QColor(255, 0, 0), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRect(self.sam_start_point, self.sam_current_point)
            painter.drawRect(rect)
    
    def draw_batch_process_points(self, painter: QPainter):
        """绘制批量处理模式下选择的像素点"""
        # 设置绘制样式
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        
        # 绘制每个像素点
        for i, (img_x, img_y) in enumerate(self.batch_process_points):
            # 将图像坐标转换为控件坐标
            widget_pos = self.image_to_widget(img_x, img_y)
            
            # 绘制圆点
            radius = 6
            painter.drawEllipse(widget_pos, radius, radius)
            
            # 绘制点编号
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont(get_primary_font_family(), 10, QFont.Weight.Bold))
            painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, str(i + 1))
            
            # 恢复绘制样式
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0)))
    
    def draw_annotations(self, painter: QPainter):
        """绘制所有标注"""
        for annotation in self.annotations:
            ann_id = annotation['id']
            ann_type = annotation.get('type', 'bbox')
            data = annotation.get('data', {})
            class_id = annotation.get('class_id', 0)
            
            # 获取颜色（优先从class_colors字典，否则使用默认灰色）
            color = self.class_colors.get(class_id, QColor(128, 128, 128))
            if isinstance(color, str):
                color = QColor(color)
            
            # 如果是选中的标注，使用高亮颜色
            is_selected = (ann_id == self.selected_annotation_id)
            pen_width = 3 if is_selected else 2
            
            pen = QPen(color)
            pen.setWidth(pen_width)
            painter.setPen(pen)
            
            brush = QBrush(color)
            brush.setStyle(Qt.BrushStyle.NoBrush)
            painter.setBrush(brush)
            
            if ann_type == 'bbox':
                self.draw_bbox(painter, data, is_selected)
            elif ann_type == 'polygon':
                self.draw_polygon(painter, data, is_selected)
            elif ann_type == 'keypoint':
                self.draw_keypoints(painter, data, is_selected)
            elif ann_type == 'obb':
                self.draw_obb(painter, data, is_selected)
    
    def draw_bbox(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制矩形框"""
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 0)
        height = data.get('height', 0)
        
        top_left = self.image_to_widget(x, y)
        bottom_right = self.image_to_widget(x + width, y + height)
        
        rect = QRect(top_left, bottom_right)
        painter.drawRect(rect)
        
        # 如果是选中状态，绘制调整手柄
        if is_selected:
            self.draw_resize_handles(painter, rect)
    
    def draw_polygon(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制多边形"""
        points = data.get('points', [])
        if len(points) < 3:
            return
        
        widget_points = []
        for point in points:
            widget_point = self.image_to_widget(point['x'], point['y'])
            widget_points.append(widget_point)
        
        # 绘制多边形
        for i in range(len(widget_points)):
            p1 = widget_points[i]
            p2 = widget_points[(i + 1) % len(widget_points)]
            painter.drawLine(p1, p2)
        
        # 绘制顶点
        for point in widget_points:
            painter.drawEllipse(point, 4, 4)
    
    def draw_keypoints(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制关键点"""
        keypoints = data.get('keypoints', [])
        if not keypoints:
            return
        
        # 绘制关键点之间的连接线
        if len(keypoints) > 1:
            for i in range(len(keypoints) - 1):
                kp1 = keypoints[i]
                kp2 = keypoints[i + 1]
                if kp1.get('visible', 1) and kp2.get('visible', 1):
                    p1 = self.image_to_widget(kp1['x'], kp1['y'])
                    p2 = self.image_to_widget(kp2['x'], kp2['y'])
                    painter.drawLine(p1, p2)
        
        # 绘制关键点
        for kp in keypoints:
            if kp.get('visible', 1):
                point = self.image_to_widget(kp['x'], kp['y'])
                # 绘制关键点圆圈
                radius = 6 if is_selected else 4
                painter.drawEllipse(point, radius, radius)
                # 绘制关键点索引（如果有）
                if 'id' in kp:
                    painter.drawText(point.x() + radius + 2, point.y() - radius, str(kp['id']))
    
    def draw_obb(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制旋转矩形"""
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 0)
        height = data.get('height', 0)
        angle = data.get('angle', 0)  # 旋转角度（弧度）
        
        # 计算矩形的四个顶点
        center = self.image_to_widget(x, y)
        half_width = width * self.image_scale / 2
        half_height = height * self.image_scale / 2
        
        # 计算四个顶点
        import math
        points = []
        # 定义四个顶点相对于中心点的基础偏移（未旋转时）
        # 顺序：p1, p2, p3, p4 对应 create_obb_annotation_with_points 中的四个点
        base_offsets = [
            (half_width, -half_height),  # p1: 右上
            (-half_width, -half_height),  # p2: 左上
            (-half_width, half_height),   # p3: 左下
            (half_width, half_height)     # p4: 右下
        ]
        
        for dx, dy in base_offsets:
            # 应用旋转
            rotated_dx = dx * math.cos(angle) - dy * math.sin(angle)
            rotated_dy = dx * math.sin(angle) + dy * math.cos(angle)
            # 计算最终顶点坐标
            vertex_x = center.x() + rotated_dx
            vertex_y = center.y() + rotated_dy
            points.append(QPoint(int(vertex_x), int(vertex_y)))
        
        # 绘制旋转矩形
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            painter.drawLine(p1, p2)
        
        # 如果是选中状态，绘制调整手柄
        if is_selected:
            for point in points:
                handle_rect = QRect(
                    point.x() - self.handle_size // 2,
                    point.y() - self.handle_size // 2,
                    self.handle_size,
                    self.handle_size
                )
                painter.drawRect(handle_rect)
    
    def draw_resize_handles(self, painter: QPainter, rect: QRect):
        """绘制调整大小的手柄"""
        handle_size = 8
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        
        # 四个角
        corners = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]
        
        for corner in corners:
            handle_rect = QRect(
                corner.x() - handle_size // 2,
                corner.y() - handle_size // 2,
                handle_size,
                handle_size
            )
            painter.drawRect(handle_rect)
    
    def draw_drawing_rectangle(self, painter: QPainter):
        """绘制正在绘制的矩形"""
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        # 使用主色调的半透明版本
        primary_color = QColor(COLORS['primary'])
        primary_color.setAlpha(50)
        painter.setBrush(QBrush(primary_color))
        
        rect = QRect(self.start_point, self.current_point)
        painter.drawRect(rect)
    
    def draw_drawing_polygon(self, painter: QPainter):
        """绘制正在绘制的多边形"""
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # 绘制已有点
        for point in self.polygon_points:
            painter.drawEllipse(point, 4, 4)
        
        # 绘制连线
        if len(self.polygon_points) > 1:
            for i in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
        
        # 绘制从最后一点到当前鼠标的线
        if len(self.polygon_points) > 0 and self.current_point:
            painter.drawLine(self.polygon_points[-1], self.current_point)
    
    def draw_drawing_obb(self, painter: QPainter):
        """绘制正在绘制的旋转矩形"""
        if len(self.obb_points) == 0:
            return
        
        # 设置绘制样式
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        # 使用主色调的半透明版本
        primary_color = QColor(COLORS['primary'])
        primary_color.setAlpha(50)
        painter.setBrush(QBrush(primary_color))
        
        if len(self.obb_points) == 1:
            # 只绘制第一个点
            point = self.obb_points[0]
            painter.drawEllipse(point, 4, 4)
        elif len(self.obb_points) == 2:
            # 绘制第一条边
            p1 = self.obb_points[0]
            p2 = self.obb_points[1]
            painter.drawLine(p1, p2)
            painter.drawEllipse(p1, 4, 4)
            painter.drawEllipse(p2, 4, 4)
            # 如果有鼠标位置，绘制辅助线、垂线和半透明OBB
            if self.current_point:
                # 计算垂直线
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                # 计算垂线方向向量
                perp_dx = -dy
                perp_dy = dx
                # 归一化
                length = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
                if length > 0:
                    perp_dx /= length
                    perp_dy /= length
                # 绘制垂线
                perp_p1 = QPoint(int(p2.x() + perp_dx * 100), int(p2.y() + perp_dy * 100))
                perp_p2 = QPoint(int(p2.x() - perp_dx * 100), int(p2.y() - perp_dy * 100))
                painter.setPen(QPen(QColor(COLORS['primary']), 1, Qt.PenStyle.DotLine))
                painter.drawLine(perp_p1, perp_p2)
                # 计算垂线上的点（鼠标位置在垂线上的投影）
                mouse_dx = self.current_point.x() - p2.x()
                mouse_dy = self.current_point.y() - p2.y()
                # 计算投影长度
                proj_length = mouse_dx * perp_dx + mouse_dy * perp_dy
                # 计算垂线上的点
                perp_point = QPoint(int(p2.x() + perp_dx * proj_length), int(p2.y() + perp_dy * proj_length))
                # 计算第四个点
                p4 = QPoint(int(p1.x() + (perp_point.x() - p2.x())), int(p1.y() + (perp_point.y() - p2.y())))
                # 绘制OBB
                painter.setPen(QPen(QColor(COLORS['primary']), 2, Qt.PenStyle.DashLine))
                painter.drawLine(p1, p2)
                painter.drawLine(p2, perp_point)
                painter.drawLine(perp_point, p4)
                painter.drawLine(p4, p1)
                # 绘制所有点
                painter.drawEllipse(p1, 4, 4)
                painter.drawEllipse(p2, 4, 4)
                painter.drawEllipse(perp_point, 4, 4)
                painter.drawEllipse(p4, 4, 4)
                # 绘制鼠标到垂点的辅助线
                painter.setPen(QPen(QColor(COLORS['primary']), 1, Qt.PenStyle.DotLine))
                painter.drawLine(self.current_point, perp_point)
                painter.setPen(QPen(QColor(COLORS['primary']), 2, Qt.PenStyle.DashLine))
        elif len(self.obb_points) == 3:
            # 绘制完整的OBB
            p1 = self.obb_points[0]
            p2 = self.obb_points[1]
            p3 = self.obb_points[2]
            
            # 计算第四个点
            dx = p3.x() - p2.x()
            dy = p3.y() - p2.y()
            p4 = QPoint(p1.x() + dx, p1.y() + dy)
            
            # 绘制OBB
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
            painter.drawLine(p3, p4)
            painter.drawLine(p4, p1)
            
            # 绘制所有点
            painter.drawEllipse(p1, 4, 4)
            painter.drawEllipse(p2, 4, 4)
            painter.drawEllipse(p3, 4, 4)
            painter.drawEllipse(p4, 4, 4)
    
    def draw_guide_lines(self, painter: QPainter):
        """绘制鼠标辅助线"""
        if not self.current_image or not self.current_point:
            return
        
        # 获取图像区域
        img_rect = self.rect()
        scaled_width = int(self.current_image.width() * self.image_scale)
        scaled_height = int(self.current_image.height() * self.image_scale)
        
        # 计算图像显示区域的边界
        img_left = self.image_offset.x()
        img_top = self.image_offset.y()
        img_right = img_left + scaled_width
        img_bottom = img_top + scaled_height
        
        # 获取鼠标位置
        mouse_x = self.current_point.x()
        mouse_y = self.current_point.y()
        
        # 检查鼠标是否在图像区域内
        if not (img_left <= mouse_x <= img_right and img_top <= mouse_y <= img_bottom):
            return
        
        # 设置辅助线样式
        pen = QPen(QColor(255, 255, 255, 150))  # 半透明白色
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)  # 加粗
        painter.setPen(pen)
        
        # 绘制水平线（穿过鼠标）
        painter.drawLine(img_left, mouse_y, img_right, mouse_y)
        
        # 绘制垂直线（穿过鼠标）
        painter.drawLine(mouse_x, img_top, mouse_x, img_bottom)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if self.current_image is None:
            return
        
        # 批量处理模式：点击选择像素点
        if self.batch_process_mode and event.button() == Qt.MouseButton.LeftButton:
            # 将鼠标位置转换为图像坐标
            img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
            
            # 检查是否在图像范围内
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            if 0 <= img_x <= img_width and 0 <= img_y <= img_height:
                # 添加像素点
                self.batch_process_points.append((int(img_x), int(img_y)))
                
                # 更新对话框中的显示
                if self.batch_process_dialog:
                    self.batch_process_dialog.add_point(int(img_x), int(img_y))
                
                self.update()
            return
        
        # SAM模式处理
        if self.sam_mode_active:
            if event.button() == Qt.MouseButton.LeftButton:
                # 左键添加点提示
                img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
                if self.current_image:
                    img_w = self.current_image.width()
                    img_h = self.current_image.height()
                    img_x = max(0, min(img_x, img_w - 1))
                    img_y = max(0, min(img_y, img_h - 1))
                self.sam_points.append((img_x, img_y))
                self.update()
                
                # 检查是否按住Ctrl
                is_ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                
                # 如果没有按住Ctrl，立即进行推理
                if not is_ctrl_pressed:
                    if self._sam_auto_infer_enabled():
                        project_type = getattr(self, 'sam_project_type', 'segment')
                        self.run_sam_inference(project_type)
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # 右键开始画框
                self.sam_mode = 'bbox'
                self.sam_drawing_bbox = True
                self.sam_start_point = event.pos()
                self.sam_current_point = event.pos()
                return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == 'rectangle':
                self.drawing = True
                self.start_point = event.pos()
                self.current_point = event.pos()
            elif self.current_tool == 'polygon':
                # 检查是否是吸附到初始点的情况
                if len(self.polygon_points) > 0:
                    initial_point = self.polygon_points[0]
                    distance = (event.pos() - initial_point).manhattanLength()
                    # 如果距离小于吸附阈值，完成多边形标注
                    if distance < 10:
                        self.create_polygon_annotation()
                        return
                # 否则添加新点
                self.polygon_points.append(event.pos())
                self.update()
            elif self.current_tool == 'keypoint':
                # 转换为图像坐标
                x, y = self.widget_to_image(event.pos().x(), event.pos().y())
                
                # 限制坐标在图像范围内
                if self.current_image:
                    img_width = self.current_image.width()
                    img_height = self.current_image.height()
                    x = max(0, min(x, img_width))
                    y = max(0, min(y, img_height))
                
                # 创建关键点标注
                keypoint = {
                    'x': x,
                    'y': y,
                    'visible': 1
                }
                
                # 添加到关键点列表
                self.keypoints.append(keypoint)
                
                # 如果已经有多个关键点，创建标注
                if len(self.keypoints) >= 1:
                    annotation = {
                        'type': 'keypoint',
                        'class_id': self.current_class_id,
                        'data': {
                            'keypoints': self.keypoints.copy()
                        }
                    }
                    self.annotation_created.emit(annotation)
                    # 重置关键点列表，准备下一个标注
                    self.keypoints = []
            elif self.current_tool == 'obb':
                if self.obb_state == 0:
                    # 第一步：确定第一个点（固定角度的起始点）
                    self.obb_state = 1
                    self.obb_points = [event.pos()]
                    self.update()
                elif self.obb_state == 1:
                    # 第二步：确定第一条边的另一个端点
                    self.obb_state = 2
                    self.obb_points.append(event.pos())
                    self.update()
                elif self.obb_state == 2:
                    # 第三步：确定邻边的另一个端点，完成OBB创建
                    # 计算垂线上的点（与draw_drawing_obb方法相同的逻辑）
                    p1 = self.obb_points[0]
                    p2 = self.obb_points[1]
                    # 计算垂直线
                    dx = p2.x() - p1.x()
                    dy = p2.y() - p1.y()
                    # 计算垂线方向向量
                    perp_dx = -dy
                    perp_dy = dx
                    # 归一化
                    length = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
                    if length > 0:
                        perp_dx /= length
                        perp_dy /= length
                    # 计算从p2到当前鼠标的向量
                    mouse_dx = event.pos().x() - p2.x()
                    mouse_dy = event.pos().y() - p2.y()
                    # 计算投影长度
                    proj_length = mouse_dx * perp_dx + mouse_dy * perp_dy
                    # 计算垂线上的点
                    perp_point = QPoint(int(p2.x() + perp_dx * proj_length), int(p2.y() + perp_dy * proj_length))
                    # 将垂线上的点添加到obb_points列表中
                    self.obb_points.append(perp_point)
                    # 创建OBB标注
                    self.create_obb_annotation_with_points()
                    # 重置状态
                    self.obb_state = 0
                    self.obb_points = []
                    self.update()
            elif self.current_tool == 'move':
                # 检查是否点击了调整手柄
                handle_info = self.get_resize_handle_at(event.pos())
                if handle_info:
                    self.resizing = True
                    self.resize_handle = handle_info['handle']
                    self.selected_annotation_id = handle_info['annotation_id']
                    # 获取选中的标注数据
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self.drag_start = event.pos()
                        self.resize_start_rect = annotation['data'].copy()
                        self.annotation_selected.emit(self.selected_annotation_id)
                else:
                    # 检查是否点击了某个标注
                    clicked_annotation = self.get_annotation_at(event.pos())
                    if clicked_annotation:
                        self.selected_annotation_id = clicked_annotation['id']
                        ann_type = clicked_annotation.get('type', 'bbox')
                        ann_data = clicked_annotation.get('data', {})

                        # 点到多边形顶点：拖动该点，而不是拖动整个多边形
                        if ann_type == 'polygon':
                            # 先用更宽松的“最近点磁吸”选择顶点，提升易用性
                            vertex_idx = self.get_nearest_polygon_vertex_at(event.pos(), clicked_annotation, self.vertex_snap_radius)
                            if vertex_idx is None:
                                vertex_idx = self.get_polygon_vertex_at(event.pos(), clicked_annotation)
                            if vertex_idx is not None:
                                self.dragging_vertex = True
                                self.drag_vertex_index = vertex_idx
                                self.drag_start = event.pos()
                                self.drag_start_annotation = (
                                    copy.deepcopy(ann_data) if isinstance(ann_data, dict) else {}
                                )
                                self.annotation_selected.emit(self.selected_annotation_id)
                                self.update()
                                return

                        # 否则：拖动整个标注
                        self.dragging = True
                        self.drag_start = event.pos()
                        # 深度复制标注数据，特别是多边形的点
                        if ann_type == 'polygon' and 'points' in ann_data:
                            # 对多边形点进行深度复制
                            self.drag_start_annotation = {'points': [p.copy() for p in ann_data.get('points', [])]}
                        elif ann_type == 'keypoint' and 'keypoints' in ann_data:
                            # 对关键点进行深度复制
                            self.drag_start_annotation = {'keypoints': [kp.copy() for kp in ann_data.get('keypoints', [])]}
                        else:
                            self.drag_start_annotation = ann_data.copy() if isinstance(ann_data, dict) else {}
                        self.annotation_selected.emit(self.selected_annotation_id)
                    else:
                        # 没有点击标注，开始平移图片
                        self.panning = True
                        self.pan_start = event.pos()
                        self.pan_start_offset = QPoint(self.image_offset)
                        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        # SAM模式：更新框绘制
        if self.sam_mode_active and self.sam_drawing_bbox:
            self.sam_current_point = event.pos()
            self.update()
            return
        
        # 多边形标注：添加初始点吸附效果
        if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
            initial_point = self.polygon_points[0]
            distance = (event.pos() - initial_point).manhattanLength()
            # 吸附阈值：10像素
            if distance < 10:
                self.current_point = initial_point
            else:
                self.current_point = event.pos()
        else:
            self.current_point = event.pos()
        
        if self.drawing and self.current_tool == 'rectangle':
            self.update()
        elif self.current_tool == 'polygon':
            self.update()
        elif self.current_tool == 'move':
            if self.resizing and self.resize_handle and self.resize_start_rect:
                # 调整大小
                self.resize_annotation(event.pos())
                self.update()
            elif self.dragging_vertex and self.selected_annotation_id is not None:
                # 拖动多边形的某个顶点
                annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                if annotation and annotation.get('type') == 'polygon':
                    points = annotation.get('data', {}).get('points', [])
                    idx = self.drag_vertex_index
                    if idx is not None and 0 <= idx < len(points):
                        img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
                        if self.current_image:
                            img_w = self.current_image.width()
                            img_h = self.current_image.height()
                            img_x = max(0, min(img_x, img_w))
                            img_y = max(0, min(img_y, img_h))
                        points[idx]['x'] = img_x
                        points[idx]['y'] = img_y
                self.update()
            elif self.dragging and self.drag_start and self.drag_start_annotation:
                # 拖动标注
                self.drag_annotation(event.pos())
                self.update()
            elif self.panning and self.pan_start and self.pan_start_offset:
                # 平移图片
                delta = event.pos() - self.pan_start
                self.image_offset = QPoint(
                    self.pan_start_offset.x() + delta.x(),
                    self.pan_start_offset.y() + delta.y()
                )
                self.update()
            else:
                # 检查鼠标是否在手柄上，改变光标
                handle_info = self.get_resize_handle_at(event.pos())
                if handle_info:
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    annotation = self.get_annotation_at(event.pos())
                    if annotation:
                        # 轻微磁吸：靠近多边形顶点时，更容易“抓住点”
                        if annotation.get('type') == 'polygon':
                            vidx = self.get_nearest_polygon_vertex_at(event.pos(), annotation, self.vertex_snap_radius)
                            if vidx is not None:
                                self.hover_vertex = (annotation.get('id'), vidx)
                                self.setCursor(Qt.CursorShape.PointingHandCursor)
                            else:
                                self.hover_vertex = None
                                self.setCursor(Qt.CursorShape.OpenHandCursor)
                        else:
                            self.hover_vertex = None
                        self.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        self.hover_vertex = None
                        # 图片放大时可以平移
                        if self.image_scale > 1.0:
                            self.setCursor(Qt.CursorShape.OpenHandCursor)
                        else:
                            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # 非移动工具时，设置为箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # 更新鼠标位置信息
        if self.current_image:
            img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
        
        # 触发重绘以显示辅助线
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        # SAM模式：右键释放完成框绘制
        if self.sam_mode_active and event.button() == Qt.MouseButton.RightButton:
            if self.sam_drawing_bbox:
                self.sam_drawing_bbox = False
                # 转换框坐标到图像坐标
                x1, y1 = self.widget_to_image(self.sam_start_point.x(), self.sam_start_point.y())
                x2, y2 = self.widget_to_image(self.sam_current_point.x(), self.sam_current_point.y())
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.sam_bboxes.append((x1, y1, x2, y2))
                self.update()
                # 运行推理
                if self._sam_auto_infer_enabled():
                    project_type = getattr(self, 'sam_project_type', 'segment')
                    self.run_sam_inference(project_type)
                return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing and self.current_tool == 'rectangle':
                self.drawing = False
                self.create_rectangle_annotation()
            elif self.drawing and self.current_tool == 'obb':
                self.drawing = False
                self.create_obb_annotation()
            elif self.resizing:
                old_data = copy.deepcopy(self.resize_start_rect) if self.resize_start_rect else None
                self.resizing = False
                self.resize_handle = None
                self.resize_start_rect = None
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self._emit_annotation_modified(annotation, old_data)
            elif self.dragging_vertex:
                old_data = copy.deepcopy(self.drag_start_annotation) if self.drag_start_annotation else None
                self.dragging_vertex = False
                self.drag_vertex_index = None
                self.drag_start = None
                self.drag_start_annotation = None
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self._emit_annotation_modified(annotation, old_data)
            elif self.dragging:
                old_data = copy.deepcopy(self.drag_start_annotation) if self.drag_start_annotation else None
                self.dragging = False
                self.drag_start = None
                self.drag_start_annotation = None
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self._emit_annotation_modified(annotation, old_data)
            elif self.panning:
                self.panning = False
                self.pan_start = None
                self.pan_start_offset = None
        
        # 根据当前状态设置光标
        if self.current_tool == 'move':
            if self.image_scale > 1.0:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # 非移动工具时，始终设置为箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        self.update()
    
    def get_annotation_at(self, pos: QPoint) -> Optional[Dict]:
        """获取指定位置的标注"""
        for annotation in reversed(self.annotations):
            if self.is_point_in_annotation(pos, annotation):
                return annotation
        return None

    def get_polygon_vertex_at(self, pos: QPoint, annotation: Dict) -> Optional[int]:
        """如果pos命中多边形顶点，返回顶点索引，否则None（控件坐标判定）"""
        if not annotation or annotation.get('type') != 'polygon':
            return None
        data = annotation.get('data', {})
        points = data.get('points', [])
        if not points:
            return None

        for idx, pt in enumerate(points):
            if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
                continue
            wp = self.image_to_widget(pt['x'], pt['y'])
            if (pos - wp).manhattanLength() <= self.vertex_hit_radius:
                return idx
        return None

    def get_nearest_polygon_vertex_at(self, pos: QPoint, annotation: Dict, radius: int) -> Optional[int]:
        """返回radius范围内最近的多边形顶点索引（控件坐标），否则None。

        用于提供轻微“磁吸/辅助命中”，让用户更容易选中顶点。
        """
        if not annotation or annotation.get('type') != 'polygon':
            return None
        data = annotation.get('data', {})
        points = data.get('points', [])
        if not points:
            return None

        best_idx = None
        best_dist = None
        for idx, pt in enumerate(points):
            if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
                continue
            wp = self.image_to_widget(pt['x'], pt['y'])
            d = (pos - wp).manhattanLength()
            if d <= radius and (best_dist is None or d < best_dist):
                best_dist = d
                best_idx = idx
        return best_idx
    
    def get_resize_handle_at(self, pos: QPoint) -> Optional[Dict]:
        """获取指定位置的调整手柄信息"""
        if self.selected_annotation_id is None:
            return None
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation or annotation.get('type') != 'bbox':
            return None
        
        data = annotation['data']
        x, y, width, height = data['x'], data['y'], data['width'], data['height']
        
        # 转换为控件坐标
        top_left = self.image_to_widget(x, y)
        bottom_right = self.image_to_widget(x + width, y + height)
        
        # 四个角的手柄
        handles = {
            'top_left': QRect(top_left.x() - self.handle_size, top_left.y() - self.handle_size, 
                             self.handle_size * 2, self.handle_size * 2),
            'top_right': QRect(bottom_right.x() - self.handle_size, top_left.y() - self.handle_size,
                              self.handle_size * 2, self.handle_size * 2),
            'bottom_left': QRect(top_left.x() - self.handle_size, bottom_right.y() - self.handle_size,
                                self.handle_size * 2, self.handle_size * 2),
            'bottom_right': QRect(bottom_right.x() - self.handle_size, bottom_right.y() - self.handle_size,
                                 self.handle_size * 2, self.handle_size * 2),
        }
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(pos):
                return {'handle': handle_name, 'annotation_id': annotation['id']}
        
        return None
    
    def _emit_annotation_modified(self, annotation: Dict, old_data: Optional[dict]):
        """拖动/缩放结束后上报修改，并带上修改前的 data 供撤销使用。"""
        if not old_data or self.selected_annotation_id is None:
            return
        new_data = copy.deepcopy(annotation.get('data', {}))
        self.annotation_modified.emit(
            self.selected_annotation_id,
            copy.deepcopy(old_data),
            new_data,
        )

    def drag_annotation(self, pos: QPoint):
        """拖动标注"""
        if self.drag_start is None or self.drag_start_annotation is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation:
            return
        
        # 计算拖动偏移（控件坐标）
        delta_x = pos.x() - self.drag_start.x()
        delta_y = pos.y() - self.drag_start.y()
        
        # 转换为图像坐标偏移
        img_delta_x = delta_x / self.image_scale
        img_delta_y = delta_y / self.image_scale
        
        ann_type = annotation.get('type', 'bbox')
        data = annotation['data']
        
        # 获取图像尺寸
        img_width = self.current_image.width() if self.current_image else 0
        img_height = self.current_image.height() if self.current_image else 0
        
        if ann_type == 'bbox':
            new_x = self.drag_start_annotation['x'] + img_delta_x
            new_y = self.drag_start_annotation['y'] + img_delta_y
            width = data.get('width', 0)
            height = data.get('height', 0)
            
            # 限制在图像范围内
            data['x'] = max(0, min(new_x, img_width - width))
            data['y'] = max(0, min(new_y, img_height - height))
        elif ann_type == 'polygon':
            # 确保drag_start_annotation包含正确的点数据
            if 'points' in self.drag_start_annotation and 'points' in data:
                start_points = self.drag_start_annotation['points']
                # 确保点的数量匹配
                if len(start_points) == len(data['points']):
                    for i, point in enumerate(data['points']):
                        if i < len(start_points):
                            # 直接使用原始点加上偏移量，避免累积误差
                            new_x = start_points[i]['x'] + img_delta_x
                            new_y = start_points[i]['y'] + img_delta_y
                            # 限制在图像范围内
                            point['x'] = max(0, min(new_x, img_width))
                            point['y'] = max(0, min(new_y, img_height))
        elif ann_type == 'obb':
            # 确保drag_start_annotation包含正确的OBB数据
            if 'x' in self.drag_start_annotation and 'y' in self.drag_start_annotation:
                new_x = self.drag_start_annotation['x'] + img_delta_x
                new_y = self.drag_start_annotation['y'] + img_delta_y
                # 限制在图像范围内
                data['x'] = max(0, min(new_x, img_width))
                data['y'] = max(0, min(new_y, img_height))
        elif ann_type == 'keypoint':
            # 确保drag_start_annotation包含正确的关键点数据
            if 'keypoints' in self.drag_start_annotation and 'keypoints' in data:
                start_keypoints = self.drag_start_annotation['keypoints']
                # 确保关键点数量匹配
                if len(start_keypoints) == len(data['keypoints']):
                    for i, kp in enumerate(data['keypoints']):
                        if i < len(start_keypoints):
                            new_x = start_keypoints[i]['x'] + img_delta_x
                            new_y = start_keypoints[i]['y'] + img_delta_y
                            # 限制在图像范围内
                            kp['x'] = max(0, min(new_x, img_width))
                            kp['y'] = max(0, min(new_y, img_height))
    
    def resize_annotation(self, pos: QPoint):
        """调整标注大小"""
        if self.resize_handle is None or self.resize_start_rect is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation:
            return
        
        data = annotation['data']
        start = self.resize_start_rect
        
        # 将鼠标位置转换为图像坐标
        img_x, img_y = self.widget_to_image(pos.x(), pos.y())
        
        # 限制鼠标位置在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            img_x = max(0, min(img_x, img_width))
            img_y = max(0, min(img_y, img_height))
        
        if self.resize_handle == 'top_left':
            new_x = min(img_x, start['x'] + start['width'])
            new_y = min(img_y, start['y'] + start['height'])
            data['x'] = max(0, new_x)
            data['y'] = max(0, new_y)
            data['width'] = start['x'] + start['width'] - data['x']
            data['height'] = start['y'] + start['height'] - data['y']
        elif self.resize_handle == 'top_right':
            new_y = min(img_y, start['y'] + start['height'])
            data['x'] = start['x']
            data['y'] = max(0, new_y)
            data['width'] = min(img_x, img_width) - start['x'] if self.current_image else img_x - start['x']
            data['height'] = start['y'] + start['height'] - data['y']
        elif self.resize_handle == 'bottom_left':
            new_x = min(img_x, start['x'] + start['width'])
            data['x'] = max(0, new_x)
            data['y'] = start['y']
            data['width'] = start['x'] + start['width'] - data['x']
            data['height'] = min(img_y, img_height) - start['y'] if self.current_image else img_y - start['y']
        elif self.resize_handle == 'bottom_right':
            data['x'] = start['x']
            data['y'] = start['y']
            data['width'] = min(img_x, img_width) - start['x'] if self.current_image else img_x - start['x']
            data['height'] = min(img_y, img_height) - start['y'] if self.current_image else img_y - start['y']
        
        # 确保宽度和高度为正且不超过图像范围
        if data['width'] < 0:
            data['x'] += data['width']
            data['width'] = abs(data['width'])
        if data['height'] < 0:
            data['y'] += data['height']
            data['height'] = abs(data['height'])
        
        # 最终限制在图像范围内
        if self.current_image:
            data['x'] = max(0, min(data['x'], img_width))
            data['y'] = max(0, min(data['y'], img_height))
            data['width'] = min(data['width'], img_width - data['x'])
            data['height'] = min(data['height'], img_height - data['y'])
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """鼠标双击事件 - 完成多边形绘制"""
        if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
            self.create_polygon_annotation()
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放"""
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        self._zoom_by_factor(wheel_zoom_factor(delta), event.position())
        event.accept()

    def event(self, event):
        """统一接住原生捏合和 Qt PinchGesture，避免平台分支各自算缩放。"""
        event_type = event.type()
        if event_type == QEvent.Type.NativeGesture:
            gesture_type = event.gestureType()
            if gesture_type in (
                Qt.NativeGestureType.BeginNativeGesture,
                Qt.NativeGestureType.EndNativeGesture,
            ):
                event.accept()
                return True
            if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
                self._zoom_by_factor(native_zoom_factor(event.value()), event.position())
                event.accept()
                return True
            return super().event(event)

        if event_type == QEvent.Type.Gesture:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if pinch is None:
                return super().event(event)

            anchor = self.mapFromGlobal(pinch.centerPoint().toPoint())
            if not self.rect().contains(anchor):
                anchor = self.rect().center()
            self._zoom_by_factor(pinch_zoom_factor(pinch.scaleFactor()), anchor)
            event.accept(pinch)
            return True

        return super().event(event)

    def _zoom_by_factor(self, factor: float, anchor) -> bool:
        """所有缩放入口共用这里；只在最终落到像素坐标时取整，并至多刷新一次。"""
        if self.current_image is None:
            return False

        new_scale, offset_x, offset_y = zoom_at(
            self.image_scale,
            self.image_offset.x(),
            self.image_offset.y(),
            anchor.x(),
            anchor.y(),
            factor,
            (ZOOM_MIN, ZOOM_MAX),
        )
        new_offset = QPoint(int(round(offset_x)), int(round(offset_y)))
        if new_scale == self.image_scale and new_offset == self.image_offset:
            return False

        self.image_scale = new_scale
        self.image_offset = new_offset
        self.update()
        return True
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        from PyQt6.QtCore import QSettings
        
        # 获取快捷键设置
        settings = QSettings("EzYOLO", "Settings")
        reset_view_key = str(settings.value("reset_view_shortcut", "R"))
        
        # 重置视图快捷键
        if event_matches_shortcut(event, reset_view_key):
            self.reset_view()
            self.update()
            return
        
        if event.key() == Qt.Key.Key_Escape:
            # SAM模式：退出SAM模式
            if self.sam_mode_active:
                self.sam_mode_active = False
                self.sam_points = []
                self.sam_bboxes = []
                self.sam_drawing_bbox = False
                self.set_tool('rectangle')  # 切换回默认工具
                self.update()
                return
            # 取消当前操作
            if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
                self.polygon_points = []
                self.update()
            elif self.drawing:
                self.drawing = False
                self.update()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # 完成多边形绘制
            if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
                self.create_polygon_annotation()
        elif event.key() == Qt.Key.Key_Delete:
            # 删除选中的标注
            if self.selected_annotation_id is not None:
                self.annotation_deleted.emit(self.selected_annotation_id)
        else:
            # 将未处理的事件传递给父组件
            self.parent().keyPressEvent(event)
    
    def check_annotation_selection(self, pos: QPoint):
        """检查是否选中了某个标注"""
        for annotation in reversed(self.annotations):  # 从后往前检查，先检查上面的
            if self.is_point_in_annotation(pos, annotation):
                self.selected_annotation_id = annotation['id']
                self.annotation_selected.emit(annotation['id'])
                self.update()
                return
        
        # 没有选中任何标注
        self.selected_annotation_id = None
        self.update()
    
    def is_point_in_annotation(self, pos: QPoint, annotation: Dict) -> bool:
        """检查点是否在标注内"""
        ann_type = annotation.get('type', 'bbox')
        data = annotation.get('data', {})
        
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            
            top_left = self.image_to_widget(x, y)
            bottom_right = self.image_to_widget(x + width, y + height)
            
            return (top_left.x() <= pos.x() <= bottom_right.x() and
                    top_left.y() <= pos.y() <= bottom_right.y())
        
        elif ann_type == 'polygon':
            # 简化的多边形检测
            points = data.get('points', [])
            if len(points) < 3:
                return False
            
            # 转换为控件坐标
            widget_points = []
            for point in points:
                widget_point = self.image_to_widget(point['x'], point['y'])
                widget_points.append((widget_point.x(), widget_point.y()))
            
            # 使用射线法检测点是否在多边形内
            return self.point_in_polygon(pos.x(), pos.y(), widget_points)
        elif ann_type == 'obb':
            # 计算OBB的四个顶点
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            angle = data.get('angle', 0)
            
            # 转换为控件坐标的四个顶点
            widget_points = []
            for i in range(4):
                vertex_angle = angle + i * math.pi / 2
                vertex_x = x + width * math.cos(vertex_angle) - height * math.sin(vertex_angle)
                vertex_y = y + width * math.sin(vertex_angle) + height * math.cos(vertex_angle)
                widget_point = self.image_to_widget(vertex_x, vertex_y)
                widget_points.append((widget_point.x(), widget_point.y()))
            
            # 使用射线法检测点是否在OBB内
            return self.point_in_polygon(pos.x(), pos.y(), widget_points)
        elif ann_type == 'keypoint':
            # 检查是否点击了任何一个关键点
            keypoints = data.get('keypoints', [])
            for kp in keypoints:
                kp_x = kp.get('x', 0)
                kp_y = kp.get('y', 0)
                # 转换为控件坐标
                widget_kp = self.image_to_widget(kp_x, kp_y)
                # 计算鼠标位置与关键点的距离
                distance = math.sqrt((pos.x() - widget_kp.x()) ** 2 + (pos.y() - widget_kp.y()) ** 2)
                # 设置阈值，10像素范围内视为点击了关键点
                if distance <= 10:
                    return True
        
        return False
    
    def point_in_polygon(self, x: int, y: int, polygon: List[Tuple[int, int]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def create_rectangle_annotation(self):
        """创建矩形标注"""
        if self.start_point is None or self.current_point is None:
            return
        
        # 转换为图像坐标
        x1, y1 = self.widget_to_image(self.start_point.x(), self.start_point.y())
        x2, y2 = self.widget_to_image(self.current_point.x(), self.current_point.y())
        
        # 确保 x1 < x2, y1 < y2
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 限制坐标在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            # 限制x和y在图像范围内
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            
            # 限制width和height不超出图像范围
            width = min(width, img_width - x)
            height = min(height, img_height - y)
        
        # 过滤太小的标注
        if width < 5 or height < 5:
            return
        
        annotation = {
            'type': 'bbox',
            'class_id': self.current_class_id,
            'data': {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }
        }
        
        self.annotation_created.emit(annotation)
        self.start_point = None
        self.current_point = None
    
    def create_polygon_annotation(self):
        """创建多边形标注"""
        if len(self.polygon_points) < 3:
            return
        
        # 转换为图像坐标
        points = []
        for point in self.polygon_points:
            x, y = self.widget_to_image(point.x(), point.y())
            
            # 限制坐标在图像范围内
            if self.current_image:
                img_width = self.current_image.width()
                img_height = self.current_image.height()
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
            
            points.append({'x': x, 'y': y})
        
        annotation = {
            'type': 'polygon',
            'class_id': self.current_class_id,
            'data': {
                'points': points
            }
        }
        
        self.annotation_created.emit(annotation)
        self.polygon_points = []
    
    def run_sam_inference(self, project_type: str = 'segment'):
        """运行SAM推理"""
        if not self._sam_auto_infer_enabled():
            # 非normal模式下禁止自动推理
            return
        if not self.sam_config or not self.sam_image_path:
            return
        
        if not self.sam_points and not self.sam_bboxes:
            return
        
        # 保存项目类型供回调使用
        self.sam_project_type = project_type
        
        # 创建推理线程
        self.sam_worker = SAMInferenceWorker(
            self.sam_config,
            self.sam_image_path,
            points=self.sam_points,
            bboxes=self.sam_bboxes
        )
        self.sam_worker.inference_finished.connect(
            lambda success, msg, masks: self.on_sam_inference_finished(success, msg, masks, self.sam_project_type)
        )
        self.sam_worker.start()
    
    def on_sam_inference_finished(self, success: bool, message: str, masks, project_type: str = 'segment'):
        """SAM推理完成回调"""
        if self.sam_operation_mode != "normal":
            # 记忆采集阶段丢弃任何异步返回，避免误落标注
            self.sam_points = []
            self.sam_bboxes = []
            self.update()
            return

        mask_array = masks
        mask_scores = None
        if isinstance(masks, dict):
            mask_array = masks.get("masks")
            mask_scores = masks.get("scores")

        if mask_array is not None and not isinstance(mask_array, np.ndarray):
            mask_array = np.asarray(mask_array)

        if success and mask_array is not None and len(mask_array) > 0:
            # SAM常返回多个候选mask，优先取最高分mask，避免并集导致过分膨胀
            if len(mask_array.shape) == 2:
                selected_mask = mask_array
            else:
                selected_index = 0
                if mask_scores is not None and len(mask_scores) == len(mask_array):
                    selected_index = int(np.argmax(mask_scores))
                selected_mask = mask_array[selected_index]

            if len(selected_mask.shape) > 2:
                selected_mask = selected_mask.squeeze()
            mask_uint8 = (selected_mask > 0).astype(np.uint8) * 255
            
            # 查找所有轮廓
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.sam_points = []
                self.sam_bboxes = []
                self.update()
                return
            
            # 获取所有轮廓的边界框（合并所有轮廓）
            all_x, all_y, all_x2, all_y2 = [], [], [], []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                all_x.append(x)
                all_y.append(y)
                all_x2.append(x + w)
                all_y2.append(y + h)
            
            # 计算合并后的边界框
            min_x = min(all_x)
            min_y = min(all_y)
            max_x2 = max(all_x2)
            max_y2 = max(all_y2)
            total_w = max_x2 - min_x
            total_h = max_y2 - min_y
            
            if project_type == 'detect':
                # detect任务：创建合并后的边界框
                annotation = {
                    'type': 'bbox',
                    'class_id': self.current_class_id,
                    'data': {
                        'x': float(min_x),
                        'y': float(min_y),
                        'width': float(total_w),
                        'height': float(total_h)
                    }
                }
                
                self.annotation_created.emit(annotation)
            else:
                # segment或其他任务：创建多边形标注（使用最大轮廓）
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 简化轮廓
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 转换为点列表
                points = []
                for point in approx_contour:
                    x, y = point[0]
                    points.append({'x': float(x), 'y': float(y)})
                
                # 创建多边形标注
                annotation = {
                    'type': 'polygon',
                    'class_id': self.current_class_id,
                    'data': {
                        'points': points
                    }
                }
                
                self.annotation_created.emit(annotation)
            
            # 清除SAM状态
            self.sam_points = []
            self.sam_bboxes = []
            self.update()
        else:
            # 推理失败，清除状态
            self.sam_points = []
            self.sam_bboxes = []
            self.update()
    
    def create_obb_annotation_with_points(self):
        """根据三个点创建旋转矩形标注"""
        if len(self.obb_points) != 3:
            return
        
        # 转换为图像坐标
        p1 = self.obb_points[0]
        p2 = self.obb_points[1]
        p3 = self.obb_points[2]
        
        # 转换控件坐标为图像坐标
        x1, y1 = self.widget_to_image(p1.x(), p1.y())
        x2, y2 = self.widget_to_image(p2.x(), p2.y())
        x3, y3 = self.widget_to_image(p3.x(), p3.y())
        
        # 计算向量
        vec1 = (x2 - x1, y2 - y1)
        
        # 计算垂线方向向量
        perp_dx = -vec1[1]
        perp_dy = vec1[0]
        
        # 归一化
        length = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
        if length > 0:
            perp_dx /= length
            perp_dy /= length
        
        # 计算从p2到p3的向量
        vec3 = (x3 - x2, y3 - y2)
        
        # 计算投影长度
        proj_length = vec3[0] * perp_dx + vec3[1] * perp_dy
        
        # 计算垂线上的点
        perp_x = x2 + perp_dx * proj_length
        perp_y = y2 + perp_dy * proj_length
        
        # 计算第四个点
        x4 = x1 + (perp_x - x2)
        y4 = y1 + (perp_y - y2)
        
        # 计算宽度和高度
        width = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        height = math.sqrt((perp_x - x2) ** 2 + (perp_y - y2) ** 2)
        
        # 计算旋转角度（弧度）
        angle = math.atan2(vec1[1], vec1[0])
        
        # 计算中心点
        center_x = (x1 + x2 + perp_x + x4) / 4
        center_y = (y1 + y2 + perp_y + y4) / 4
        
        # 限制坐标在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            center_x = max(0, min(center_x, img_width))
            center_y = max(0, min(center_y, img_height))
            width = min(width, img_width)
            height = min(height, img_height)
        
        # 过滤太小的标注
        if width < 5 or height < 5:
            return
        
        annotation = {
            'type': 'obb',
            'class_id': self.current_class_id,
            'data': {
                'x': center_x,
                'y': center_y,
                'width': width,
                'height': height,
                'angle': angle
            }
        }
        
        self.annotation_created.emit(annotation)
    
    def resizeEvent(self, event):
        """窗口大小改变"""
        super().resizeEvent(event)
        self._position_lock_button()
        # 锁定时不重新适配：用户锁的就是当前这个缩放和位置
        if self.current_image and not self.view_locked:
            self.reset_view()


class AnnotatePage(QWidget):
    """标注页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_image_id = None
        self.current_image_data = None
        self.images = []
        self.annotations = []
        self.classes = []
        self.current_class_id = 0
        self.history = []  # 撤销历史
        self.history_index = -1
        self.load_worker = None  # 加载线程
        self.random_delete_worker = None
        self.random_delete_progress = None
        self._sample_stats_project_id = None
        self._sample_class_counts_cache = {}
        self._negative_sample_count_cache = 0
        self._sample_stats_dirty = True
        self.default_draw_tool = 'rectangle'
        # 图片栏收起前的三栏宽度，展开时照着还回去；初值和 init_ui 里的初始比例一致
        self._splitter_sizes = [205, 412, 247]

        # 自动标注相关属性
        self.auto_label_dialog = None
        self.model_manager = None
        self.batch_labeling_manager = None
        # 批处理对话框（非模态）。同名属性在 AnnotationCanvas 上也有一份，
        # 但页面自己这份必须先存在：show_batch_process_dialog 一进来就要读它。
        self.batch_process_dialog = None
        self.sam_memory_objects = []
        self.sam_memory_dialog = None
        self.prev_image_shortcut = None
        self.next_image_shortcut = None

        # LLM 推理：单张一个线程，批量一个线程；引用留在页面上，别让它跑着就被回收。
        # 已经收工、但线程还没真正退出的那些，先挪进 _retired_llm_workers 继续持有，
        # 等 QThread.finished 再销毁——见 _on_llm_worker_finished。
        self.llm_worker = None
        self.llm_batch_worker = None
        self._retired_llm_workers = []
        self.llm_batch_total = 0
        self.llm_batch_added = 0

        # 批量标注（YOLO / LLM 共用这一组状态）。
        # 正在跑的那份快照留在这儿：写库时读它，不读 self.current_project_id ——
        # 用户完全可能在跑批的时候切到别的项目去，那时页面上的 current_project_id
        # 已经是另一个项目了，照它写就会把标注写进错的项目。
        self._active_batch_plan = None
        self._batch_running = False
        self._batch_status_text = ""

        self.init_ui()

    def shutdown(self):
        """关窗前收工：先把两边的取消旗都插上，再挨个等线程退出。

        QThread 对象在 run() 还没结束时被销毁会直接崩。这里以前只管 LLM：
        YOLO 批量推理跑着的时候关窗，BatchLabelingManager 跟着页面一起没了，
        它手上那个还在跑的线程正好撞在这个崩点上。

        先取消、后等待，是因为「取消」都是放个标志就返回，「等待」才真堵着：
        两边的取消一起发出去，谁先退出都行；反过来先站在 LLM 那儿等满 5 秒，
        YOLO 线程这 5 秒里还在一张张往下推图，白跑。

        两种等待的性质不一样，各按各的来：LLM 是网络请求，可能吊死，所以等待
        给 5 秒上限，不让它把退出流程拖住；YOLO 是本地推理，不会吊死，但停不到
        半张图上——**当前这张必须先跑完，manager.cleanup() 会一直等到它返回**，
        所以正在跑大图时关窗，可能要多等一张图的推理时间。

        可以重复调用：没有 manager、线程已经退了，都走空路径。
        """
        manager = getattr(self, 'batch_labeling_manager', None)
        llm_workers = [self.llm_batch_worker, self.llm_worker, *self._retired_llm_workers]

        # 第一步：只发取消，一个都不等
        if manager is not None:
            manager.request_cancel()
        for worker in llm_workers:
            if worker is None or not worker.isRunning():
                continue
            if hasattr(worker, 'cancel'):
                worker.cancel()

        # 第二步：等 YOLO 线程真的退出（cleanup 里 stop + wait），顺带卸掉模型
        if manager is not None:
            manager.cleanup()

        # 第三步：等 LLM 线程退出
        for worker in llm_workers:
            if worker is None or not worker.isRunning():
                continue
            worker.wait(5000)

    def refresh_theme(self):
        """主题切换后刷新工具栏、状态栏与画布。"""
        base_tool_style = self._toolbar_chip_style('tool')
        for button in (
            getattr(self, 'btn_draw_tool', None),
            getattr(self, 'btn_keypoint', None),
            getattr(self, 'btn_move', None),
            getattr(self, 'btn_undo', None),
        ):
            if button is not None:
                button.setStyleSheet(base_tool_style)
        if hasattr(self, 'btn_delete'):
            self.btn_delete.setStyleSheet(self._toolbar_chip_style('danger'))
        if hasattr(self, 'status_bar'):
            self.status_bar.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['panel']};
                    border: none;
                    border-top: 1px solid {COLORS['border']};
                    border-radius: 0px;
                }}
                QLabel {{
                    color: {COLORS['text_secondary']};
                    font-size: 12px;
                }}
            """)
        if hasattr(self, 'canvas') and hasattr(self.canvas, '_refresh_lock_button'):
            self.canvas._refresh_lock_button()
        if hasattr(self, 'canvas'):
            self.canvas.update()
        if hasattr(self, 'current_class_chip'):
            self._update_current_class_chip()

    def _track_llm_worker(self, worker):
        """所有 LLM 线程都从这里登记生命周期。"""
        worker.finished.connect(self._on_llm_worker_finished)

    def _on_llm_worker_finished(self):
        """线程真的退出了才销毁它。

        业务信号（image_done / batch_finished）是 run() 里的最后几句，那会儿
        run() 还没返回；在那里 deleteLater() 或者把最后一个引用丢掉，QThread
        就会在自己还在跑的时候被销毁——直接崩。QThread.finished 是 run() 返回
        之后才发的，只有这里放手才安全。
        """
        worker = self.sender()
        if worker is None:
            return
        if worker is self.llm_worker:
            self.llm_worker = None
        if worker is self.llm_batch_worker:
            self.llm_batch_worker = None
        if worker in self._retired_llm_workers:
            self._retired_llm_workers.remove(worker)
        worker.deleteLater()
    
    def init_ui(self):
        """初始化界面

        自上而下四层：
            信息条 —— 现在在哪个项目、标哪张图、标了多少、当前是什么类别
            主区   —— 左：图片列表 / 中：工具栏 + 画布 + 翻页 / 右：类别、AI、样本、导出
            状态栏 —— 位置、本图标注数、当前工具（快捷键收进「快捷键」按钮的提示里）
        """
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 顶部信息条（任务类型选择器在这里创建，中间面板初始化时要读它）
        self.context_bar = self.create_context_bar()
        self.main_layout.addWidget(self.context_bar)

        # 轻量帮助：默认收起，跟顶栏左右对齐
        self.main_layout.addWidget(self.create_context_help_bar())

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter

        # 左侧：图片列表
        self.left_panel = self.create_left_panel()
        splitter.addWidget(self.left_panel)

        # 中间：标注画布
        self.center_panel = self.create_center_panel()
        splitter.addWidget(self.center_panel)

        # 右侧：属性面板
        self.right_panel = self.create_right_panel()
        splitter.addWidget(self.right_panel)

        # 初始比例要落在三个面板各自的 min/max 区间内（左 175~320 / 中 ≥412 / 右 232~280），
        # 否则第一次绘制时会被重新夹紧，看起来像「跳」了一下。这里按 1100px 窗口的可用宽度分
        # （内容区约 884）：中栏先拿够画布要的 412，剩下的给图片栏和属性栏。
        splitter.setSizes([205, 412, 247])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.main_layout.addWidget(splitter, 1)

        # 底部状态栏
        self.status_bar = self.create_status_bar()
        self.main_layout.addWidget(self.status_bar)

        self._init_navigation_shortcuts()
        self._restore_image_list_collapsed()
        self._update_context_bar()
        self._update_action_availability()

    def showEvent(self, event):
        """显示页面时刷新快捷键配置。"""
        self._refresh_navigation_shortcuts()
        super().showEvent(event)

    def create_context_help_bar(self) -> QWidget:
        """轻量帮助那一条：主布局是零边距的，靠这层容器跟顶栏对齐。"""
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 0)
        layout.setSpacing(0)

        self.context_help = ContextHelp(
            [
                "锁定视图后，切换图片会保留当前缩放和画面位置。",
                "自动标注适合先生成结果，再逐张检查和修正。",
                "批量覆盖会替换范围内的已有标注，开始前请确认处理范围。",
            ],
            risk_steps=[3],
            title="标注技巧",
        )
        layout.addWidget(self.context_help)

        return bar

    def create_context_bar(self) -> QWidget:
        """顶部信息条：当前图片 / 工具 / 标注方式。

        这里不再放「当前项目」——左侧流程栏一直显示着当前项目，同一屏写两遍
        不会让人更清楚自己在哪，只是把宽度从「当前图片」那一格里抠走。

        「标注方式」留着并放在最右：它决定工具栏给的是方框、多边形还是关键点，
        不是纯展示，撤掉的话用户就没有地方换标注形状了。
        """
        bar = QWidget()
        bar.setObjectName("page_header")

        layout = QGridLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(2)

        # 当前图片：第几张 + 是哪张 + 标没标。文件名可省略，但不会挤动工具。
        self.image_name_label = QLabel()
        self.image_name_label.setObjectName("title")
        self.image_name_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.image_name_label.setMinimumWidth(240)
        self.image_name_label.setMaximumWidth(320)
        self.image_name_label.setMinimumHeight(TOOLBAR_BUTTON_HEIGHT)
        self._register_elided_label(self.image_name_label)
        self._set_elided_text(self.image_name_label, "未选择图片")

        # 标注方式（任务类型）：决定用什么形状标注。
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "pose", "classify"])
        self.task_combo.setFixedWidth(128)
        self.task_combo.setMinimumHeight(TOOLBAR_BUTTON_HEIGHT)
        self.task_combo.setToolTip("标注方式：决定画框/多边形/关键点/整图分类")
        self.task_combo.currentTextChanged.connect(self.on_task_changed)

        # 工具栏直接落在顶栏的值行，不再占画布上方的一整行。
        # 高度不钉死；赋值完成后再刷新一次，这时 refresh 方法才能拿到
        # self.toolbar，并把按钮按字体算出的共同最小高度同步给容器。
        self.toolbar = self.create_toolbar()
        self.refresh_toolbar_button_layout()

        for column, caption in ((0, "当前图片"), (1, "工具"), (3, "标注方式")):
            label = QLabel(caption)
            label.setObjectName("caption")
            layout.addWidget(label, 0, column)
        layout.addWidget(self.image_name_label, 1, 0)
        layout.addWidget(self.toolbar, 1, 1)
        layout.setColumnStretch(2, 1)
        layout.addWidget(self.task_combo, 1, 3)

        return bar

    def _register_elided_label(self, label: QLabel):
        """登记一个会随宽度打省略号的标签：自己变宽变窄时重算。"""
        if not hasattr(self, '_elided_labels'):
            self._elided_labels = []
        if label not in self._elided_labels:
            self._elided_labels.append(label)
            label.installEventFilter(self)

    def _set_elided_text(self, label: QLabel, text: str, tooltip: Optional[str] = None):
        """给标签设文本：显示时按当前宽度截断，完整内容放进 tooltip。"""
        label.setProperty('_full_text', text)
        label.setToolTip(text if tooltip is None else tooltip)
        self._apply_elide(label)

    def _apply_elide(self, label: QLabel):
        """按标签当前宽度重新截断，永远不会切在半个字上。"""
        full = label.property('_full_text')
        if full is None:
            return
        metrics = QFontMetrics(label.font())
        available = max(label.width(), 32)
        label.setText(metrics.elidedText(str(full), Qt.TextElideMode.ElideRight, available))

    def eventFilter(self, obj, event):
        """信息条上的标签一变宽/变窄，就重新算省略号。"""
        if event.type() == QEvent.Type.Resize and obj in getattr(self, '_elided_labels', ()):
            self._apply_elide(obj)
        return super().eventFilter(obj, event)

    def _init_navigation_shortcuts(self):
        """初始化翻页快捷键，避免依赖控件焦点。"""
        self.prev_image_shortcut = QShortcut(QKeySequence(), self)
        self.prev_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.prev_image_shortcut.activated.connect(self._trigger_prev_image_shortcut)

        self.next_image_shortcut = QShortcut(QKeySequence(), self)
        self.next_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.next_image_shortcut.activated.connect(self._trigger_next_image_shortcut)

        self._refresh_navigation_shortcuts()

    def _shortcut_keys(self) -> Dict[str, str]:
        """当前生效的快捷键，与 keyPressEvent 读的是同一份设置。"""
        from PyQt6.QtCore import QSettings

        settings = QSettings("EzYOLO", "Settings")
        return {
            'prev': str(settings.value("prev_image_shortcut", "A")).upper(),
            'next': str(settings.value("next_image_shortcut", "D")).upper(),
            'rect': str(settings.value("rect_tool_shortcut", "W")).upper(),
            'poly': str(settings.value("poly_tool_shortcut", "P")).upper(),
            'move': str(settings.value("move_tool_shortcut", "V")).upper(),
            'delete': str(settings.value("delete_shortcut", "DELETE")).upper(),
        }

    def _refresh_navigation_shortcuts(self):
        """读取设置中的快捷键，并同步到翻页按钮和各处工具提示。"""
        keys = self._shortcut_keys()
        self.prev_image_shortcut.setKey(QKeySequence(keys['prev']))
        self.next_image_shortcut.setKey(QKeySequence(keys['next']))

        if hasattr(self, 'btn_prev'):
            self.btn_prev.setText(f"◀ 上一张 ({keys['prev']})")
            self.btn_next.setText(f"下一张 ({keys['next']}) ▶")
            self._sync_navigation_button_sizes()

        if hasattr(self, 'btn_move'):
            self.btn_move.setToolTip(f"拖动画布和已有标注\n快捷键: {keys['move']}")
        if hasattr(self, 'btn_delete'):
            self.btn_delete.setToolTip(
                f"删除当前选中的标注\n快捷键: {keys['delete']}\n下拉菜单里可以删除整张图片"
            )
        if hasattr(self, 'btn_draw_tool'):
            self.refresh_draw_tool_button()

        # 快捷键表不再常驻状态栏（1100px 下会被切断），改挂在「快捷键」按钮的提示里
        shortcut_text = "\n".join([
            f"{keys['prev']} / {keys['next']}　上一张 / 下一张",
            f"{keys['rect']}　矩形",
            f"{keys['poly']}　多边形",
            f"{keys['move']}　移动",
            "1-9　切换类别",
            f"{keys['delete']}　删除标注",
            "Ctrl+Z　撤销",
        ])
        if hasattr(self, 'btn_shortcut_help'):
            self.btn_shortcut_help.setToolTip(shortcut_text)
        if hasattr(self, 'status_bar'):
            self.status_bar.setToolTip(shortcut_text)

        self._update_canvas_placeholder()

    def _should_handle_navigation_shortcut(self) -> bool:
        """在当前焦点状态下是否允许触发翻页快捷键。"""
        if QApplication.activeWindow() is not self.window():
            return False
        if QApplication.activePopupWidget() is not None:
            return False

        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return True

        if isinstance(focus_widget, (QLineEdit, QTextEdit, QAbstractSpinBox)):
            return False
        if focus_widget.inherits("QPlainTextEdit"):
            return False

        return True

    def _trigger_prev_image_shortcut(self):
        """上一张快捷键入口。"""
        if self._should_handle_navigation_shortcut():
            self.prev_image()

    def _trigger_next_image_shortcut(self):
        """下一张快捷键入口。"""
        if self._should_handle_navigation_shortcut():
            self.next_image()

    def create_left_panel(self) -> QWidget:
        """创建左侧面板 - 图片列表"""
        panel = QWidget()
        panel.setMinimumWidth(175)
        panel.setMaximumWidth(320)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 6, 10)
        layout.setSpacing(8)

        # 标题行：「图片 · 张数」+ 收起按钮。收起后这一整栏的宽度让给画布。
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        self.image_list_title = QLabel("图片")
        self.image_list_title.setObjectName("h2")
        title_row.addWidget(self.image_list_title)
        title_row.addStretch(1)

        self.btn_collapse_image_list = self._ghost_icon_button(
            'chevron-left.svg', 24, "收起图片列表"
        )
        self.btn_collapse_image_list.clicked.connect(
            lambda: self._set_image_list_collapsed(True)
        )
        title_row.addWidget(self.btn_collapse_image_list)

        layout.addLayout(title_row)

        # 这个项目里有多少图、标了多少
        self.image_list_caption = QLabel("还没有图片")
        self.image_list_caption.setObjectName("caption")
        self.image_list_caption.setWordWrap(True)
        layout.addWidget(self.image_list_caption)

        # 筛选
        self.image_filter = QComboBox()
        self.image_filter.addItems(["全部", "未标注", "已标注"])
        self.image_filter.setToolTip("只看还没标的图片，可以避免漏标")
        self.image_filter.currentTextChanged.connect(self.filter_images)
        layout.addWidget(self.image_filter)

        # 图片列表（✓ = 已标注，○ = 还没标）
        self.image_list = QListWidget()
        # 缩略图和文件名共用一行的宽度：80px 的图会把文件名挤成「sho…」，
        # 64px 刚好让 shot0.jpg 这种长度完整显示
        self.image_list.setIconSize(QSize(64, 64))
        self.image_list.setSpacing(2)
        self.image_list.setToolTip("✓ 已标注　○ 还没标注")
        # 缩略图占掉大半行宽，文件名再长也只能就地省略：
        # 让列表横向滚动的话，用户得拖着滚动条才能看全一个文件名，反而更糟
        self.image_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setWordWrap(False)
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list, 1)

        return panel

    def _ghost_icon_button(self, icon_name: str, size: int, tooltip: str) -> QToolButton:
        """只有图标、没有底色的小按钮：收起 / 展开图片栏用。"""
        button = QToolButton()
        button.setIcon(_asset_icon(icon_name))
        button.setIconSize(QSize(12, 12))
        button.setFixedSize(size, size)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # padding 显式清零：全局 QToolButton 有 5px 10px，会把这么小的按钮里的图标压没
        button.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: {RADIUS_SM}px;
                padding: 0px;
            }}
            QToolButton:hover {{
                background-color: {COLORS['hover']};
            }}
        """)
        return button

    def _set_image_list_collapsed(self, collapsed: bool, persist: bool = True):
        """收起 / 展开左侧图片栏。

        收起前先记下三栏当前宽度：QSplitter 把左栏让出来的空间分给中栏，
        展开时再按记下的宽度还回去（受窗口约束可能有几个像素出入）。
        """
        if collapsed and self.left_panel.isVisible():
            self._splitter_sizes = self.splitter.sizes()

        self.left_panel.setVisible(not collapsed)
        self.btn_expand_image_list.setVisible(collapsed)

        if not collapsed and self._splitter_sizes:
            self.splitter.setSizes(self._splitter_sizes)

        if persist:
            from PyQt6.QtCore import QSettings
            settings = QSettings("EzYOLO", "Settings")
            settings.setValue("annotate_image_list_collapsed", collapsed)

    def _restore_image_list_collapsed(self):
        """按上次退出时的状态决定图片栏是收着还是开着。"""
        from PyQt6.QtCore import QSettings
        settings = QSettings("EzYOLO", "Settings")
        collapsed = str(settings.value("annotate_image_list_collapsed", False)).lower() in ('true', '1')
        self._set_image_list_collapsed(collapsed, persist=False)

    def create_center_panel(self) -> QWidget:
        """创建中间面板 - 标注画布 + 翻页"""
        panel = QWidget()
        # 画布自己的最小宽度就是 400，加上左右各 6px 边距，这一栏少于 412 就会把画布
        # 的右缘（以及贴在右上角的视图锁）裁掉。左 175 + 中 412 + 右 232 + 手柄，
        # 1100px 窗口下仍然放得下。
        panel.setMinimumWidth(412)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 10)
        layout.setSpacing(8)

        # 标注画布
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_created.connect(self.on_annotation_created)
        self.canvas.annotation_selected.connect(self.on_annotation_selected)
        self.canvas.annotation_modified.connect(self.on_annotation_modified)
        self.canvas.annotation_deleted.connect(self.on_annotation_deleted)
        layout.addWidget(self.canvas, stretch=1)

        # 翻页：这一页最主要的下一步动作就是「下一张」
        layout.addWidget(self.create_navigation_bar())

        # 初始化时根据当前任务类型调整工具按钮的可见性
        current_task = self.task_combo.currentText()
        self.adjust_tool_visibility(current_task)

        return panel

    def create_navigation_bar(self) -> QWidget:
        """画布下方的翻页条：上一张 / 下一张（主操作）。"""
        bar = QWidget()
        bar.setToolTip("标注自动保存，不需要手动保存")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setMinimumHeight(34)
        self.btn_prev.clicked.connect(self.prev_image)
        layout.addWidget(self.btn_prev)

        layout.addStretch(1)

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setObjectName("primary")
        self.btn_next.setMinimumHeight(34)
        self.btn_next.clicked.connect(self.next_image)
        layout.addWidget(self.btn_next)

        self._sync_navigation_button_sizes()

        return bar

    def _sync_navigation_button_sizes(self):
        """翻页按钮始终取两者所需的较大尺寸，避免左右一大一小。"""
        buttons = (self.btn_prev, self.btn_next)
        for button in buttons:
            button.setMinimumSize(0, 0)
            button.setMaximumSize(16777215, 16777215)

        width = max(button.sizeHint().width() for button in buttons)
        height = max(34, *(button.sizeHint().height() for button in buttons))
        for button in buttons:
            button.setFixedSize(width, height)
    
    def create_toolbar(self) -> QWidget:
        """创建工具栏：一行安静的控件条，只放画标注和改标注的动作。

        以前这里是一张带边框的卡片，还给每组按钮顶了一行小标题（「标注工具」「修改」）。
        三五个按钮不需要目录：标题多占一行高度，边框把它框成一个和画布平起平坐的区块，
        画布因此矮了一截——而画布才是这一页真正要看的东西。
        现在只留按钮本身，中间一根竖线把「画」和「改/删」分开。
        """
        toolbar = QWidget()
        toolbar.setObjectName("annotate_toolbar")
        # 卡片底色和边框都不要：它是浮在画布上方的一条控件，不是又一张卡片
        toolbar.setStyleSheet(
            "QWidget#annotate_toolbar { background-color: transparent; border: none; }"
        )
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)
        base_tool_button_style = self._toolbar_chip_style('tool')

        # 图片栏收起时，这里是把它叫回来的唯一入口；平时不占位
        self.btn_expand_image_list = self._ghost_icon_button(
            'chevron-right.svg', 28, "展开图片列表"
        )
        self.btn_expand_image_list.clicked.connect(
            lambda: self._set_image_list_collapsed(False)
        )
        self.btn_expand_image_list.hide()
        toolbar_layout.addWidget(self.btn_expand_image_list)

        # 工具按钮组
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        # 绘制工具（矩形/多边形合并）
        self.btn_draw_tool = QToolButton()
        self.btn_draw_tool.setCheckable(True)
        self.btn_draw_tool.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_draw_tool.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_draw_tool.setStyleSheet(base_tool_button_style)
        self.btn_draw_tool.clicked.connect(self.activate_default_draw_tool)
        self.draw_tool_menu = QMenu(self)
        self.action_draw_rectangle = self.draw_tool_menu.addAction("矩形")
        self.action_draw_polygon = self.draw_tool_menu.addAction("多边形")
        self.action_draw_rectangle.triggered.connect(lambda: self.select_draw_tool('rectangle'))
        self.action_draw_polygon.triggered.connect(lambda: self.select_draw_tool('polygon'))
        self.btn_draw_tool.setMenu(self.draw_tool_menu)
        self.btn_draw_tool.hide()  # 默认隐藏
        self.tool_group.addButton(self.btn_draw_tool)

        # 关键点工具
        self.btn_keypoint = QPushButton("关键点")
        self.btn_keypoint.setToolTip("在目标上依次点关键点（姿态任务）")
        self.btn_keypoint.setCheckable(True)
        self.btn_keypoint.clicked.connect(lambda: self.set_tool('keypoint'))
        self.btn_keypoint.setStyleSheet(base_tool_button_style)
        self.btn_keypoint.hide()  # 默认隐藏
        self.tool_group.addButton(self.btn_keypoint)

        # # OBB工具
        # self.btn_obb = QPushButton("🔲 旋转矩形 (O)")
        # self.btn_obb.setCheckable(True)
        # self.btn_obb.clicked.connect(lambda: self.set_tool('obb'))
        # toolbar.addWidget(self.btn_obb)
        # self.btn_obb.hide()  # 默认隐藏
        # self.tool_group.addButton(self.btn_obb)

        # 移动工具
        self.btn_move = QPushButton("移动")
        self.btn_move.setToolTip("拖动画布和已有标注")
        self.btn_move.setCheckable(True)
        self.btn_move.clicked.connect(lambda: self.set_tool('move'))
        self.btn_move.setStyleSheet(base_tool_button_style)
        self.tool_group.addButton(self.btn_move)

        # 撤销按钮（和左边几个工具用同一套尺寸，一行排开时高度、内边距才对得齐）
        self.btn_undo = QPushButton("撤销")
        self.btn_undo.setToolTip("撤销上一步\n快捷键: Ctrl+Z")
        self.btn_undo.setStyleSheet(base_tool_button_style)
        self.btn_undo.clicked.connect(self.undo)

        # 删除按钮（主动作删标注，下拉菜单删图片）
        self.btn_delete = QToolButton()
        self.btn_delete.setText("删除")
        self.btn_delete.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_delete.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_delete.setStyleSheet(self._toolbar_chip_style('danger'))
        self.btn_delete.clicked.connect(self.delete_selected_annotation)
        self.delete_menu = QMenu(self)
        self.action_delete_current_image = self.delete_menu.addAction("删除当前图片")
        self.action_delete_current_image.triggered.connect(self.delete_current_image)
        self.delete_menu.aboutToShow.connect(self.update_delete_menu_state)
        self.btn_delete.setMenu(self.delete_menu)

        # 画的工具 | 改和删。一行排开，中间一根细线分开——
        # 「删除」离「画方框」远一点，手滑的代价小一点。
        self.toolbar_draw_buttons = [self.btn_draw_tool, self.btn_keypoint, self.btn_move]
        self.toolbar_edit_buttons = [self.btn_undo, self.btn_delete]
        self.toolbar_buttons = self.toolbar_draw_buttons + self.toolbar_edit_buttons

        for button in self.toolbar_draw_buttons:
            toolbar_layout.addWidget(button)
        self.toolbar_separator = self._toolbar_separator()
        toolbar_layout.addWidget(self.toolbar_separator)
        for button in self.toolbar_edit_buttons:
            toolbar_layout.addWidget(button)
        toolbar_layout.addStretch(1)

        self.refresh_draw_tool_button()
        self.refresh_toolbar_button_layout()

        return toolbar

    def _toolbar_separator(self) -> QFrame:
        """工具栏里的竖直分隔线。"""
        line = QFrame()
        line.setFixedWidth(1)
        line.setMinimumHeight(TOOLBAR_BUTTON_HEIGHT - 8)
        line.setStyleSheet(
            f"background-color: {COLORS['border']}; border: none; border-radius: 0px;"
        )
        return line

    def _toolbar_chip_style(self, role: str) -> str:
        """工具栏按钮样式。

        选中的工具 = 实心蓝（就是它在生效），未选中 = 描边；
        删除是破坏性动作，单独用红色描边，不和普通工具混在一起。
        """
        palette_map = {
            'tool': {
                'text': COLORS['text_primary'],
                'border': COLORS['border_strong'],
                'hover_border': COLORS['primary_hover'],
                'checked': COLORS['primary'],
                'checked_hover': COLORS['primary_hover'],
            },
            'danger': {
                'text': COLORS['error'],
                'border': COLORS['border_strong'],
                'hover_border': COLORS['error'],
                'checked': COLORS['error'],
                'checked_hover': COLORS['error'],
            },
        }
        palette = palette_map[role]
        arrow_name = 'chevron_down.svg' if role == 'tool' else 'chevron_down_red.svg'
        arrow_url = (_ASSETS_DIR / arrow_name).as_posix()
        disabled_arrow_url = (_ASSETS_DIR / 'chevron_down_disabled.svg').as_posix()
        checked_menu_background = COLORS['panel'] if role == 'tool' else palette['checked']
        checked_menu_border = COLORS['border'] if role == 'tool' else COLORS['panel']
        return f"""
            QPushButton, QToolButton {{
                background-color: {COLORS['panel']};
                color: {palette['text']};
                border: 1px solid {palette['border']};
                border-radius: {RADIUS_SM}px;
                font-weight: 600;
            }}
            QPushButton {{
                padding: 4px 10px;
            }}
            /* QToolButton 在这里永远带菜单：右边留 46px 给 34px 宽的 menu-button
               子控件（+ 一点呼吸空间），文字才不会被压进箭头区。 */
            QToolButton {{
                padding: 4px 46px 4px 10px;
            }}
            QPushButton:hover, QToolButton:hover {{
                background-color: {COLORS['hover']};
                border-color: {palette['hover_border']};
            }}
            QPushButton:checked, QToolButton:checked {{
                background-color: {palette['checked']};
                color: #FFFFFF;
                border-color: {palette['checked']};
            }}
            QPushButton:disabled, QToolButton:disabled {{
                background-color: {COLORS['panel']};
                color: {COLORS['text_disabled']};
                border-color: {COLORS['border']};
            }}
            QToolButton::menu-button {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 34px;
                background-color: {COLORS['panel']};
                border-left: 1px solid {COLORS['border']};
                border-top-right-radius: {RADIUS_SM}px;
                border-bottom-right-radius: {RADIUS_SM}px;
            }}
            QToolButton::menu-button:hover {{
                background-color: {COLORS['hover']};
            }}
            /* 子控件要写在伪状态前面。写成 QToolButton:checked::menu-button，
               Qt 认不出子控件，会把这条的底色刷满整个按钮——没选中的「删除」
               也会变成一整块实心红。 */
            QToolButton::menu-button:checked {{
                background-color: {checked_menu_background};
                border-left: 1px solid {checked_menu_border};
            }}
            QToolButton::menu-button:checked:hover {{
                background-color: {checked_menu_background if role == 'tool' else palette['checked_hover']};
                border-left: 1px solid {checked_menu_border};
            }}
            QToolButton::menu-button:disabled {{
                background-color: {COLORS['panel']};
                border-left: 1px solid {COLORS['border']};
            }}
            QToolButton::menu-arrow,
            QToolButton::menu-indicator {{
                image: url({arrow_url});
                width: 16px;
                height: 16px;
            }}
            QToolButton::menu-arrow:disabled,
            QToolButton::menu-indicator:disabled {{
                image: url({disabled_arrow_url});
            }}
        """

    def _get_toolbar_button_width(self, button) -> int:
        """计算按钮建议宽度，菜单按钮额外预留箭头空间。

        菜单按钮不能信 QToolButton 原生的 minimumSizeHint——它不知道我们把
        menu-button 子控件挤宽到了多少，算出来的宽度会比实际需要的窄，文字
        就被顶进箭头区。这里按实际的内边距（左 10 + 右 46）和双边框（2px）
        自己算，再留 6px 安全边，最后跟 sizeHint 取较大值兜底。
        """
        text_width = button.fontMetrics().horizontalAdvance(button.text())
        if isinstance(button, QToolButton) and button.menu() is not None:
            width = text_width + 10 + 46 + 2 + 6
            return max(width, button.sizeHint().width())
        return text_width + 24

    def refresh_toolbar_button_layout(self):
        """统一工具栏按钮高度，宽度按各自文字走。

        以前是按分组统一宽度（都撑到组里最宽的那个），工具栏因此要 577px，
        窗口一小就把删除按钮挤出可视区。这里只保证高度一致和一个最小宽度。

        高度对所有按钮一视同仁——带下拉箭头的 QToolButton（画方框、删除）和
        普通 QPushButton（撤销）默认的 sizeHint 不一样，不钉住的话一行排开就是
        参差不齐的。高度不再写死成 TOOLBAR_BUTTON_HEIGHT：换一套更高的系统字体时，
        固定高度会把文字压扁，这里改成按实际按钮取需要的最大高度，再用
        minimumHeight 兜底，长得下的字体可以自己撑高。
        """
        min_button_width = 78
        visible_buttons = [
            button for button in getattr(self, 'toolbar_buttons', [])
            if button is not None and not button.isHidden()
        ]
        toolbar_height = max(
            [TOOLBAR_BUTTON_HEIGHT]
            + [button.sizeHint().height() for button in visible_buttons]
            + [button.fontMetrics().height() + 12 for button in visible_buttons]
        )
        for button in visible_buttons:
            button.setMinimumHeight(toolbar_height)
            button.setMinimumWidth(
                max(min_button_width, self._get_toolbar_button_width(button))
            )
        if hasattr(self, 'toolbar'):
            self.toolbar.setMinimumHeight(toolbar_height)
        if hasattr(self, 'toolbar_separator'):
            self.toolbar_separator.setFixedHeight(max(1, toolbar_height - 8))

    def refresh_draw_tool_button(self):
        """刷新绘制工具按钮文本与选项状态。"""
        if not hasattr(self, 'btn_draw_tool'):
            return

        keys = self._shortcut_keys()
        if self.default_draw_tool == 'polygon':
            self.btn_draw_tool.setText("画多边形")
            self.btn_draw_tool.setToolTip(
                f"沿目标轮廓依次点，回到起点闭合\n快捷键: {keys['poly']}\n点右侧箭头可换成矩形"
            )
        else:
            self.btn_draw_tool.setText("画方框")
            self.btn_draw_tool.setToolTip(
                f"按住左键拖出一个框\n快捷键: {keys['rect']}\n点右侧箭头可换成多边形"
            )

        if hasattr(self, 'action_draw_rectangle'):
            self.action_draw_rectangle.setCheckable(True)
            self.action_draw_rectangle.setChecked(self.default_draw_tool == 'rectangle')
        if hasattr(self, 'action_draw_polygon'):
            self.action_draw_polygon.setCheckable(True)
            self.action_draw_polygon.setChecked(self.default_draw_tool == 'polygon')

        self.refresh_toolbar_button_layout()

    def select_draw_tool(self, tool: str):
        """选择默认绘制工具，并立即切换到该工具。"""
        if tool not in ('rectangle', 'polygon'):
            return
        self.default_draw_tool = tool
        self.refresh_draw_tool_button()
        self.btn_draw_tool.setChecked(True)
        self.set_tool(tool)

    def activate_default_draw_tool(self):
        """激活当前默认绘制工具。"""
        self.btn_draw_tool.setChecked(True)
        self.set_tool(self.default_draw_tool)

    def update_delete_menu_state(self):
        """更新删除菜单状态。"""
        if hasattr(self, 'action_delete_current_image'):
            self.action_delete_current_image.setEnabled(bool(self.current_image_id))
    
    def apply_sam_button_mode(self):
        """根据SAM设置刷新按钮行为：普通模式/记忆模式。"""
        if not hasattr(self, "btn_sam"):
            return
        try:
            self.btn_sam.clicked.disconnect()
        except Exception:
            pass
        self.btn_sam.setMenu(None)
        set_menu_indicator(self.btn_sam, False)

        sam_config = AutoLabelDialog.get_saved_sam_config()
        sam_type = sam_config.get("sam_type", "SAM")
        usage_mode = sam_config.get("usage_mode", "normal")

        if usage_mode == "memory" and sam_type in ("SAM2", "SAM3"):
            self.btn_sam.setText("SAM 记忆标注")
            self.btn_sam.setToolTip("SAM 记忆标注：先教一次，之后自动标同类")
            self.btn_sam.setMenu(self.create_sam_memory_menu())
            set_menu_indicator(self.btn_sam, True)
        else:
            self.btn_sam.setText("SAM 交互分割")
            self.btn_sam.setToolTip("点目标自动分割轮廓")
            self.btn_sam.clicked.connect(self.start_sam_annotation)
        self.refresh_toolbar_button_layout()

    def create_sam_memory_menu(self) -> QMenu:
        menu = QMenu(self)
        act_update = QAction("更新记忆", self)
        act_clear = QAction("清空记忆", self)
        act_single = QAction("单张推理", self)
        act_batch = QAction("批量推理", self)
        act_update.triggered.connect(self.start_sam_memory_update)
        act_clear.triggered.connect(self.clear_sam_memory)
        act_single.triggered.connect(self.run_sam_memory_single)
        act_batch.triggered.connect(self.run_sam_memory_batch)
        menu.addAction(act_update)
        menu.addAction(act_clear)
        menu.addSeparator()
        menu.addAction(act_single)
        menu.addAction(act_batch)
        return menu
    
    def create_right_panel(self) -> QWidget:
        """创建右侧面板 - 类别、AI 自动标注、样本管理、导出

        整块可以滚动：窗口再矮也不会把下面的按钮压没。
        """
        panel = QWidget()
        panel.setMinimumWidth(232)
        panel.setMaximumWidth(280)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 横向滚动条平时不会出现；万一字体更宽把内容撑出去，也是能滚到而不是被切掉
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(10)

        layout.addWidget(self._create_class_group())
        layout.addWidget(self._create_ai_group())
        layout.addWidget(self._create_sample_group())
        layout.addWidget(self._create_export_group())
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

        return panel

    def _create_class_group(self) -> QGroupBox:
        """类别：标注的第一步——先说清楚要画的是什么。"""
        class_group = QGroupBox("类别")
        class_group.setObjectName("annotate_class_group")
        class_group.setStyleSheet(self._compact_group_style("annotate_class_group"))
        class_layout = QVBoxLayout(class_group)
        class_layout.setSpacing(8)

        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(96)
        self.class_list.setSpacing(2)
        self.class_list.setToolTip("数字键 1-9 切换类别；右键类别可改名或删除")
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        class_layout.addWidget(self.class_list)

        # 隐藏的类别下拉：仍然承担「选中标注 → 改类别」的数据流，不再占版面
        self.attr_class = QComboBox(class_group)
        self.attr_class.currentIndexChanged.connect(self.on_attr_class_changed)
        self.attr_class.hide()

        self.btn_add_class = QPushButton("+ 添加类别")
        self.btn_add_class.clicked.connect(self.add_class)

        # 只有「选中了一个标注、且选的类别和它现在的不一样」时才可点
        self.btn_apply_attr = QPushButton("改为选中类别")
        self.btn_apply_attr.setToolTip("把画布上选中的那个标注，改成当前选中的类别")
        self.btn_apply_attr.clicked.connect(self.apply_annotation_changes)
        self.btn_apply_attr.setEnabled(False)

        # 两个按钮真 1:1 等宽：QHBoxLayout 的 stretch 只分「多出来的」空间，
        # 文字长的那个起点就更宽，最后还是不等。栅格按列分宽度，才真的一样宽。
        class_button_row = QGridLayout()
        class_button_row.setContentsMargins(0, 0, 0, 0)
        class_button_row.setHorizontalSpacing(8)
        class_button_row.setColumnStretch(0, 1)
        class_button_row.setColumnStretch(1, 1)
        for column, button in enumerate((self.btn_add_class, self.btn_apply_attr)):
            # 最小宽度放到 10：窄栏里由栅格来分宽度，按钮自己不许把列撑开
            button.setMinimumWidth(10)
            button.setMinimumHeight(28)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            class_button_row.addWidget(button, 0, column)
        class_layout.addLayout(class_button_row)

        return class_group

    def _compact_group_style(self, object_name: str) -> str:
        """只给这一页的某个分组用的紧凑内边距（全局 QGroupBox / QPushButton 不动）。

        组里的按钮也一并收紧左右内边距：右栏窄，全局那份 padding 会让「改为选中类别」
        这种六个字的按钮在 1100px 窗口下切字。
        """
        return f"""
            QGroupBox#{object_name} {{
                margin-top: 20px;
                padding: 10px 12px 12px 12px;
            }}
            QGroupBox#{object_name} QPushButton {{
                padding: 5px 6px;
            }}
            QGroupBox#{object_name} QPushButton[menuIndicator="true"] {{
                padding-right: 30px;
            }}
        """

    def _create_ai_group(self) -> QGroupBox:
        """AI 自动标注：让模型先画一遍，人只做检查和微调。"""
        ai_group = QGroupBox("AI 自动标注")
        ai_group.setObjectName("annotate_ai_group")
        ai_group.setStyleSheet(self._compact_group_style("annotate_ai_group"))
        ai_group.setToolTip("模型先标一遍，你只做检查和微调")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(8)

        # 已有 YOLO 权重 → 自动画框
        self.btn_auto_label = QPushButton("用已有模型标注")
        self.btn_auto_label.setToolTip("用已训练模型自动标注")
        self._attach_action_menu(self.btn_auto_label, self.create_auto_label_menu())

        # SAM：点一下就分割（文本和菜单由 apply_sam_button_mode 按设置决定）
        self.btn_sam = QPushButton("SAM")

        # 多模态大模型
        self.btn_llm_label = QPushButton("大模型标注")
        self.btn_llm_label.setToolTip("用多模态大模型识别目标")
        self._attach_action_menu(self.btn_llm_label, self.create_llm_menu())

        # 批处理不在这里：它不是「让模型帮你标」，是按像素点批量改图，
        # 归到下面的样本管理（进阶）里，见 _create_sample_group
        self.ai_action_buttons = [self.btn_auto_label, self.btn_sam, self.btn_llm_label]
        for button in self.ai_action_buttons:
            button.setMinimumHeight(32)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            ai_layout.addWidget(button)

        self.apply_sam_button_mode()

        return ai_group

    def _create_sample_group(self) -> CollapsibleSection:
        """样本管理：批处理 + 负样本标记 + 按类别随机删图，用来平衡数据。

        默认收起：这几件事不是每天都干，但里面有一个会真删图片文件的按钮——
        既不该常驻占掉类别和 AI 入口的位置，也不该藏进菜单里让人找不到。
        """
        sample_group = CollapsibleSection("样本管理（进阶）")
        sample_group.setToolTip("按整张图片随机删除，用来平衡类别；负样本 = 已标注但没有任何框的图片")
        sample_layout = sample_group.content_layout()

        self.btn_batch_process = QPushButton("批处理")
        self.btn_batch_process.setToolTip("按选中的像素点批量处理图片")
        self.btn_batch_process.clicked.connect(self.show_batch_process_dialog)
        self.btn_batch_process.setMinimumHeight(30)
        self.btn_batch_process.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sample_layout.addWidget(self.btn_batch_process)

        self.btn_mark_negative_sample = QPushButton("把当前图片标为负样本")
        self.btn_mark_negative_sample.setToolTip("这张图里没有任何目标，也是有用的学习材料")
        self.btn_mark_negative_sample.clicked.connect(self.mark_current_image_as_negative_sample)
        self.btn_mark_negative_sample.setMinimumHeight(30)
        self.btn_mark_negative_sample.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sample_layout.addWidget(self.btn_mark_negative_sample)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # 面板窄的时候标签自动换到字段上一行，而不是把面板顶宽
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.sample_target_class = QComboBox()
        self.sample_target_class.setMinimumWidth(90)
        self.sample_target_class.currentIndexChanged.connect(self.on_sample_target_changed)
        form.addRow("删哪个类别:", self.sample_target_class)

        self.sample_count_label = QLabel("0")
        self.sample_count_label.setMinimumHeight(30)
        self.sample_count_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.sample_count_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_primary']};
                padding: 6px 10px;
                border: 1px solid {COLORS['border']};
                background-color: {COLORS['inset']};
                border-radius: {RADIUS_SM}px;
            }}
        """)
        form.addRow("现有图片数:", self.sample_count_label)

        self.sample_delete_count = QSpinBox()
        self.sample_delete_count.setRange(0, 0)
        self.sample_delete_count.setMinimumWidth(90)
        form.addRow("随机删去:", self.sample_delete_count)

        sample_layout.addLayout(form)

        self.btn_delete_random_samples = QPushButton("随机删除样本")
        self.btn_delete_random_samples.setObjectName("danger")
        self.btn_delete_random_samples.setToolTip("按整张图片随机删除，会连图片文件一起删掉，不可恢复")
        self.btn_delete_random_samples.clicked.connect(self.delete_random_samples_for_target)
        self.btn_delete_random_samples.setMinimumHeight(30)
        self.btn_delete_random_samples.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sample_layout.addWidget(self.btn_delete_random_samples)

        return sample_group

    def _create_export_group(self) -> CollapsibleSection:
        """导出：训练不需要手动导出，这里是给外部工具用的。默认收起。"""
        export_group = CollapsibleSection("数据导出（可选）")
        export_group.setToolTip("训练会直接读标注，不用先导出；只有要把数据给别的工具用时才需要")
        export_layout = export_group.content_layout()

        format_layout = QFormLayout()
        format_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        format_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.export_format = QComboBox()
        self.export_format.addItems(["YOLO格式", "COCO格式"])
        self.export_format.setMinimumWidth(90)
        format_layout.addRow("导出格式:", self.export_format)
        export_layout.addLayout(format_layout)

        self.btn_export_annotations = QPushButton("导出标注文件")
        self.btn_export_annotations.clicked.connect(self.export_annotations)
        self.btn_export_annotations.setMinimumHeight(30)
        export_layout.addWidget(self.btn_export_annotations)

        self.btn_export_dataset = QPushButton("导出完整数据集")
        self.btn_export_dataset.setToolTip("图片 + 标注 + 训练用的 data.yaml")
        self.btn_export_dataset.clicked.connect(self.export_dataset)
        self.btn_export_dataset.setMinimumHeight(30)
        export_layout.addWidget(self.btn_export_dataset)

        return export_group
    
    def _new_action_menu(self) -> QMenu:
        """「选一个操作」用的菜单外壳：40px 行高、16px 图标、浅色 hover。

        样式挂在 objectName 上（见 gui/styles.py 的 QMenu#actionMenu），
        不在这里写一次性样式表。
        """
        menu = QMenu(self)
        menu.setObjectName("actionMenu")
        return menu

    def _attach_action_menu(self, button: QPushButton, menu: QMenu):
        """把菜单挂到按钮上，并保证菜单不比按钮窄。

        菜单比入口按钮还窄的话，看着就像点歪了弹出来的系统菜单，而不是这个按钮
        本身展开的操作列表。宽度要等按钮真的布局完才知道，所以推迟到弹出前再量。
        """
        menu.aboutToShow.connect(
            lambda m=menu, b=button: m.setMinimumWidth(max(m.minimumWidth(), b.width()))
        )
        button.setMenu(menu)
        set_menu_indicator(button)

    def create_auto_label_menu(self) -> QMenu:
        """「用已有模型标注」的操作菜单。

        三项的份量完全不同，所以文字和图标都要把这件事说清楚：
        「设置」只是打开一个窗口；「标注当前图片」当场跑一张；
        「批量标注…」——省略号是承诺后面还有一步——会动到一整批图片，
        点它只会打开确认框，绝不会直接开跑。
        """
        menu = self._new_action_menu()

        action_settings = menu.addAction(_asset_icon("action_settings.svg"), "设置")
        action_settings.setToolTip("选择模型、置信度、IoU 和覆盖规则")
        action_settings.triggered.connect(self.show_auto_label_settings)

        action_single = menu.addAction(_asset_icon("action_single.svg"), "标注当前图片")
        action_single.setToolTip("只处理当前这一张，立刻执行")
        action_single.triggered.connect(self.run_single_inference)

        action_batch = menu.addAction(_asset_icon("action_batch.svg"), "批量标注…")
        action_batch.setToolTip("先显示范围和参数，确认后才开始")
        action_batch.triggered.connect(self.run_batch_inference)

        self.auto_label_actions = {
            'settings': action_settings,
            'single': action_single,
            'batch': action_batch,
        }
        return menu

    def create_llm_menu(self) -> QMenu:
        """大模型标注的操作菜单：和 YOLO 那个一模一样的三项，不让人重新学一遍。"""
        menu = self._new_action_menu()

        action_settings = menu.addAction(_asset_icon("action_settings.svg"), "设置")
        action_settings.setToolTip("填 API Key、模型名和提示词")
        action_settings.triggered.connect(self.show_llm_settings)

        action_single = menu.addAction(_asset_icon("action_single.svg"), "标注当前图片")
        action_single.setToolTip("只处理当前这一张，立刻执行")
        action_single.triggered.connect(self.run_llm_single_inference)

        action_batch = menu.addAction(_asset_icon("action_batch.svg"), "批量标注…")
        action_batch.setToolTip("先显示范围和参数，确认后才开始")
        action_batch.triggered.connect(self.run_llm_batch_inference)

        self.llm_actions = {
            'settings': action_settings,
            'single': action_single,
            'batch': action_batch,
        }
        return menu

    def show_llm_settings(self):
        """打开自动标注设置，直接落在 LLM 那一页。"""
        self.open_auto_label_config('llm')
    
    def create_status_bar(self) -> QFrame:
        """创建状态栏：图片、进度、类别、工具和临时批处理状态。"""
        status_bar = QFrame()
        # 固定高度会在字体放大时把文字切掉，这里只给下限
        status_bar.setMinimumHeight(34)
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border: none;
                border-top: 1px solid {COLORS['border']};
                border-radius: 0px;
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 12px;
            }}
        """)

        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self.status_image = QLabel("当前: 0/0")
        layout.addWidget(self.status_image)

        layout.addWidget(QLabel("|"))

        self.status_progress = QLabel("标注: 0/0")
        layout.addWidget(self.status_progress)

        layout.addWidget(QLabel("|"))

        # 不能用 Ignored：布局会把它按近零宽度排，随后控件又被 minimumWidth 撑开，
        # 造成与右侧分隔符重叠。Preferred 让布局按实际可见宽度为它留出位置。
        # 当前类别仍保留颜色方块；长名称省略，完整名称放在 tooltip。
        self.current_class_chip = QLabel()
        self.current_class_chip.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.current_class_chip.setMinimumWidth(120)
        self.current_class_chip.setMaximumWidth(150)
        self._register_elided_label(self.current_class_chip)
        self._set_elided_text(self.current_class_chip, "类别: 未选择")
        layout.addWidget(self.current_class_chip)

        layout.addWidget(QLabel("|"))

        self.status_annotation = QLabel("本图标注: 0")
        layout.addWidget(self.status_annotation)

        layout.addWidget(QLabel("|"))

        self.status_tool = QLabel("工具: 矩形")
        layout.addWidget(self.status_tool)

        # 临时进度用（批量自动标注时显示正在处理哪张图），平时为空。
        self.status_batch = QLabel("")
        self.status_batch.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._register_elided_label(self.status_batch)
        layout.addWidget(self.status_batch, 1)

        # 批量任务的取消入口就挂在进度文字旁边，只在跑着的时候露出来。
        # 不用模态的 QProgressDialog：那东西会把整个页面按住，用户连切去看看
        # 设置都做不到，而批量推理恰恰是最该让人边跑边干别的事的地方。
        self.btn_cancel_batch = QPushButton("取消")
        self.btn_cancel_batch.setObjectName("ghost")
        self.btn_cancel_batch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_cancel_batch.setStyleSheet(
            "QPushButton#ghost { padding: 1px 8px; min-height: 0px; font-size: 12px; }"
        )
        self.btn_cancel_batch.setFixedHeight(24)
        self.btn_cancel_batch.setVisible(False)
        self.btn_cancel_batch.clicked.connect(self.cancel_active_batch)
        layout.addWidget(self.btn_cancel_batch)

        # 快捷键表以前是一整条常驻文字，窗口一窄就被切断；现在收进这个按钮的提示里
        self.btn_shortcut_help = QPushButton("快捷键")
        self.btn_shortcut_help.setObjectName("ghost")
        self.btn_shortcut_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_shortcut_help.setStyleSheet(
            "QPushButton#ghost { padding: 1px 8px; min-height: 0px; font-size: 12px; }"
        )
        self.btn_shortcut_help.setFixedHeight(24)
        self.btn_shortcut_help.clicked.connect(self._show_shortcut_help)
        layout.addWidget(self.btn_shortcut_help)

        return status_bar

    def _show_shortcut_help(self):
        """点「快捷键」直接把提示弹出来，不用非得悬停等。"""
        from PyQt6.QtWidgets import QToolTip

        QToolTip.showText(
            self.btn_shortcut_help.mapToGlobal(QPoint(0, 0)),
            self.btn_shortcut_help.toolTip(),
            self.btn_shortcut_help
        )

    def _current_image_index(self) -> int:
        """当前图片在列表中的位置，没有则 -1。"""
        if not self.current_image_id:
            return -1
        return next(
            (i for i, img in enumerate(self.images) if img['id'] == self.current_image_id),
            -1
        )

    def _update_context_bar(self):
        """刷新顶部信息条和左侧图片计数。"""
        if not hasattr(self, 'image_name_label'):
            return

        total = len(self.images)
        annotated = sum(1 for img in self.images if img.get('status') == 'annotated')
        index = self._current_image_index()

        if not self.current_project_id:
            self._set_elided_text(self.image_name_label, "未选择项目")
        elif not self.images:
            self._set_elided_text(self.image_name_label, "这个项目还没有图片")
        elif index < 0:
            self._set_elided_text(self.image_name_label, "未选择图片")
        else:
            image = self.images[index]
            status = "已标注" if image.get('status') == 'annotated' else "还没标注"
            # 位置和状态在前：它们每张图都有、长度稳定，窗口再窄也看得见。
            # 名字放最后，压不下时省略掉的是它——不是「第几张」和「标没标」。
            self._set_elided_text(
                self.image_name_label,
                f"第 {index + 1}/{total} 张 · {self._image_display_name(image)} · {status}",
                tooltip=self._image_tooltip(image)
            )

        if hasattr(self, 'image_list_title'):
            self.image_list_title.setText(f"图片 · {total}" if total else "图片")

        if hasattr(self, 'image_list_caption'):
            if total:
                self.image_list_caption.setText(
                    f"共 {total} 张 · 已标注 {annotated} · 还剩 {total - annotated}"
                )
            else:
                self.image_list_caption.setText("还没有图片，请回到「数据导入」")

        self._update_current_class_chip()
        self._update_canvas_placeholder()

    def _update_current_class_chip(self):
        """状态栏上的「当前类别」：画上去的就是它。"""
        if not hasattr(self, 'current_class_chip'):
            return

        current = next((cls for cls in self.classes if cls['id'] == self.current_class_id), None)
        if current is None:
            self.current_class_chip.setStyleSheet(f"color: {COLORS['text_secondary']};")
            self._set_elided_text(
                self.current_class_chip, "类别: 未选择",
                tooltip="画上去的标注算哪个类别，在右侧「类别」里换"
            )
            return

        self.current_class_chip.setStyleSheet(
            f"color: {_readable_on_light(current.get('color'))}; font-weight: 600;"
        )
        self._set_elided_text(
            self.current_class_chip, f"类别: ■ {current['name']}",
            tooltip=f"当前类别：{current['name']}\n在右侧「类别」里切换"
        )

    def _update_canvas_placeholder(self):
        """画布空着的时候，告诉用户下一步该做什么。"""
        if not hasattr(self, 'canvas'):
            return

        if not self.current_project_id:
            hint = "还没有选择项目"
        elif not self.images:
            hint = "这个项目里还没有图片"
        else:
            hint = "从左边选一张图片开始"

        if self.canvas.empty_hint != hint:
            self.canvas.empty_hint = hint
            if self.canvas.current_image is None:
                self.canvas.update()

    def _update_action_availability(self):
        """没有项目/图片时就把对应的按钮关掉，别让人点了没反应。"""
        has_project = bool(self.current_project_id)
        has_image = bool(self.current_image_id)
        index = self._current_image_index()

        if hasattr(self, 'btn_prev'):
            self.btn_prev.setEnabled(index > 0)
            self.btn_next.setEnabled(0 <= index < len(self.images) - 1)

        for name in ('btn_draw_tool', 'btn_keypoint', 'btn_move', 'btn_delete'):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(has_image)

        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(has_image and self.history_index >= 0)

        # AI 入口只要求有项目：它们的菜单里还有「设置」和「批量标注…」，
        # 没选图片也该点得开；只对当前图片生效的那些，处理函数自己会提示先选图片。
        # 但正在跑一批的时候一律关掉：这是防重复启动的一环，而且 update_status_bar
        # 会反复调到这里，不带上 _batch_running 的话刚禁掉的按钮转头又被打开了。
        batch_running = getattr(self, '_batch_running', False)
        for button in getattr(self, 'ai_action_buttons', []):
            button.setEnabled(has_project and not batch_running)

        if hasattr(self, 'btn_add_class'):
            self.btn_add_class.setEnabled(has_project)
        if hasattr(self, 'btn_mark_negative_sample'):
            self.btn_mark_negative_sample.setEnabled(has_image)
        # 批处理跟着项目走（和它还在 AI 组里时一样）：它的对话框自己会提示先选图片
        for name in ('btn_batch_process', 'btn_delete_random_samples',
                     'btn_export_annotations', 'btn_export_dataset'):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(has_project)
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        # 即使项目ID相同，也重新加载数据（确保图片列表更新）
        self.current_project_id = project_id
        self._invalidate_sample_stats_cache()
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在加载项目数据...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来加载项目数据
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class ProjectLoadThread(QThread):
            """项目数据加载线程"""
            
            data_loaded = pyqtSignal(dict)
            finished = pyqtSignal()
            
            def __init__(self, project_id):
                super().__init__()
                self.project_id = project_id
            
            def run(self):
                """运行线程"""
                try:
                    # 加载项目信息
                    project = db.get_project(self.project_id)
                    classes = []
                    
                    if project:
                        # 加载类别
                        import json
                        try:
                            classes = json.loads(project.get('classes', '[]'))
                        except:
                            classes = []
                        
                        if not classes:
                            # 添加默认类别
                            classes = [
                                {'id': 0, 'name': 'person', 'color': '#FF0000'},
                                {'id': 1, 'name': 'car', 'color': '#00FF00'}
                            ]
                    
                    # 加载图片列表数据
                    images = db.get_project_images(self.project_id)
                    
                    # 发送加载完成信号
                    self.data_loaded.emit({'classes': classes, 'images': images})
                finally:
                    self.finished.emit()
        
        # 创建并启动线程
        self.load_thread = ProjectLoadThread(project_id)
        self.load_thread.data_loaded.connect(self.on_project_data_loaded)
        self.load_thread.finished.connect(self.on_project_load_finished)
        self.load_thread.start()
    
    def on_project_data_loaded(self, data):
        """项目数据加载完成回调"""
        # 更新类别
        self.classes = data.get('classes', [])
        self.update_class_list()
        
        # 保存图片数据
        self.images = data.get('images', [])
        
        # 同步任务类型选择器（项目名归左侧流程栏显示，这一页不再重复）
        if self.current_project_id:
            project = db.get_project(self.current_project_id)
            if project:
                task_type = project.get('type')
                if task_type in ['detect', 'segment', 'pose', 'classify', 'obb']:
                    index = self.task_combo.findText(task_type)
                    if index >= 0:
                        self.task_combo.setCurrentIndex(index)

        # 开始加载图片列表（使用多线程加载缩略图）
        self.load_image_list()
    
    def on_project_load_finished(self):
        """项目加载完成回调"""
        # 隐藏加载动画
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')
        
        # 清理线程
        if hasattr(self, 'load_thread'):
            self.load_thread.wait()
            delattr(self, 'load_thread')
    
    def load_image_list(self):
        """加载图片列表 - 使用多线程"""
        # 停止之前的加载
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.stop()
            self.load_worker.wait()
        
        self.image_list.clear()
        self.images = []
        
        if not self.current_project_id:
            return
        
        # 从数据库获取图片列表（很快）
        self.images = db.get_project_images(self.current_project_id)
        self._invalidate_sample_stats_cache()
        self._refresh_image_display_names()

        # 先创建所有列表项（显示占位符）
        for image in self.images:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, image['id'])

            # 设置显示文本
            status_text = "✓" if image.get('status') == 'annotated' else "○"
            item.setText(f"{status_text} {self._image_display_name(image)}")
            item.setToolTip(self._image_tooltip(image))

            self.image_list.addItem(item)

        self.update_status_bar()
        self.update_sample_control_panel()
        
        # 启动后台加载线程加载缩略图
        if self.images:
            self.load_worker = AnnotateImageLoadWorker(self.images)
            self.load_worker.image_loaded.connect(self.on_image_loaded)
            self.load_worker.finished_loading.connect(self.on_load_finished)
            self.load_worker.start()
    
    def on_image_loaded(self, index: int, pixmap: QPixmap):
        """单个图片加载完成回调"""
        if index < self.image_list.count():
            item = self.image_list.item(index)
            if item:
                item.setIcon(QIcon(pixmap))
    
    def on_load_finished(self):
        """加载完成回调"""
        pass
    
    def _refresh_image_display_names(self):
        """整个列表一起算显示名：按项目的显示名称规则来（默认等价于旧的「帧号化名」逻辑）。
        重名/编号是否独一份只有看全列表才知道，所以必须整批算，不能一张一张各算各的。"""
        project = db.get_project(self.current_project_id) if self.current_project_id else None
        rule = parse_display_name_rule((project or {}).get('display_name_rule'))
        project_name = (project or {}).get('name', '')
        try:
            aliases = build_project_display_names(rule, self.images, project_name)
        except ValueError:
            # 规则本该在保存前就校验过全量图片；万一还是生成失败，退回「保留原名」，
            # 不能让整页刷不出来。
            aliases = display_names([img.get('filename', '') for img in self.images])
        self._image_display_names = {
            img['id']: alias for img, alias in zip(self.images, aliases)
        }

    def _image_display_name(self, image: Dict) -> str:
        """列表里显示的名字：按项目规则来（默认抽帧名念成「帧 223」，普通文件名原样）。"""
        cached = getattr(self, '_image_display_names', {}).get(image['id'])
        return cached or display_name(image.get('filename', ''))

    def _image_tooltip(self, image: Dict) -> str:
        """完整文件名、分辨率、来源路径不丢——显示名可能被规则改写得完全认不出，
        这里必须永远留一个能找到真实文件的地方。某项元数据缺失时只把那一项换成
        「未知/未记录」，不能整行消失，否则会被当成软件漏读了数据。"""
        width = image.get('width')
        height = image.get('height')
        resolution = f"{width}x{height}" if width and height else "未知"
        original_path = image.get('original_path') or "未记录"
        return "\n".join([
            image.get('filename', ''),
            f"分辨率: {resolution}",
            f"来源: {original_path}",
        ])

    def update_image_list_display(self):
        """更新图片列表显示"""
        # 重新加载图片数据
        if self.current_project_id:
            self.images = db.get_project_images(self.current_project_id)
            self._refresh_image_display_names()

            # 更新图片列表项
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                image_id = item.data(Qt.ItemDataRole.UserRole)

                # 找到对应的图片数据
                image_data = next((img for img in self.images if img['id'] == image_id), None)
                if image_data:
                    # 更新显示文本
                    status_text = "✓" if image_data.get('status') == 'annotated' else "○"
                    item.setText(f"{status_text} {self._image_display_name(image_data)}")
                    item.setToolTip(self._image_tooltip(image_data))

            # 图片的已标注状态变了，进度也要跟着变
            self.update_status_bar()

    def on_display_name_rule_changed(self, project_id: int):
        """导入页改了这个项目的显示名称规则：只重刷列表文字/tooltip 和顶部信息条，
        不重新加载画布、不碰当前选中的图片和标注状态。"""
        if project_id != self.current_project_id:
            return
        self.update_image_list_display()
        self._update_context_bar()

    def update_class_list(self):
        """更新类别列表"""
        current_attr_class = self.attr_class.currentData()
        current_sample_target = self.sample_target_class.currentData()
        current_selected_class = self.current_class_id
        class_sample_counts, negative_sample_count = self._get_sample_stats()
        self.class_list.clear()
        self.attr_class.clear()
        self.sample_target_class.blockSignals(True)
        self.sample_target_class.clear()
        
        # 更新canvas的类别颜色
        class_colors = {}
        for cls in self.classes:
            class_colors[cls['id']] = cls.get('color', '#808080')
        self.canvas.class_colors = class_colors
        
        for cls in self.classes:
            sample_count = class_sample_counts.get(cls['id'], 0)
            # 创建带颜色的列表项
            item = QListWidgetItem(f"■ {cls['name']} ({sample_count})")
            item.setData(Qt.ItemDataRole.UserRole, cls['id'])
            
            # 设置颜色
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            item.setSizeHint(QSize(item.sizeHint().width(), 28))
            
            self.class_list.addItem(item)
            
            # 添加到属性面板的下拉框
            self.attr_class.addItem(cls['name'], cls['id'])
            self.sample_target_class.addItem(cls['name'], cls['id'])

        self.sample_target_class.addItem("负样本", NEGATIVE_SAMPLE_CLASS_ID)

        if current_attr_class is not None:
            attr_index = self.attr_class.findData(current_attr_class)
            if attr_index >= 0:
                self.attr_class.setCurrentIndex(attr_index)

        if current_sample_target is None:
            current_sample_target = self.current_class_id if self.classes else NEGATIVE_SAMPLE_CLASS_ID
        sample_index = self.sample_target_class.findData(current_sample_target)
        if sample_index >= 0:
            self.sample_target_class.setCurrentIndex(sample_index)
        elif self.sample_target_class.count() > 0:
            self.sample_target_class.setCurrentIndex(0)
        self.sample_target_class.blockSignals(False)
        self.update_sample_control_panel(class_sample_counts, negative_sample_count)
        self._update_current_class_chip()

        # 默认选中第一个类别
        if self.class_list.count() > 0:
            selected_row = next(
                (row for row in range(self.class_list.count())
                 if self.class_list.item(row).data(Qt.ItemDataRole.UserRole) == current_selected_class),
                0
            )
            self.class_list.setCurrentRow(selected_row)
            self.on_class_selected()

    def get_sample_target_images(self, target_class_id):
        """获取指定标签对应的样本图像列表"""
        if not self.current_project_id or target_class_id is None:
            return []

        if target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
            return db.get_negative_sample_images(self.current_project_id, annotated_only=True)

        return db.get_project_images_by_class(self.current_project_id, target_class_id)

    def _invalidate_sample_stats_cache(self):
        """标记样本统计缓存失效。"""
        self._sample_stats_dirty = True

    def _get_sample_stats(self, force_refresh=False):
        """获取当前项目样本统计缓存。"""
        if not self.current_project_id:
            self._sample_stats_project_id = None
            self._sample_class_counts_cache = {}
            self._negative_sample_count_cache = 0
            self._sample_stats_dirty = False
            return {}, 0

        should_refresh = (
            force_refresh
            or self._sample_stats_dirty
            or self._sample_stats_project_id != self.current_project_id
        )
        if should_refresh:
            self._sample_class_counts_cache = db.get_project_image_counts_by_class(self.current_project_id)
            self._negative_sample_count_cache = db.get_negative_sample_image_count(
                self.current_project_id,
                annotated_only=True
            )
            self._sample_stats_project_id = self.current_project_id
            self._sample_stats_dirty = False

        return self._sample_class_counts_cache, self._negative_sample_count_cache

    def get_project_sample_counts(self, force_refresh=False):
        """获取当前项目各类别对应的样本图数量。"""
        return self._get_sample_stats(force_refresh=force_refresh)[0]

    def get_negative_sample_count(self, force_refresh=False):
        """获取当前项目负样本图数量。"""
        return self._get_sample_stats(force_refresh=force_refresh)[1]

    def get_class_sample_count(self, class_id):
        """获取某个类别对应的样本图数量"""
        return self.get_project_sample_counts().get(class_id, 0)

    def _get_selected_class_list_id(self) -> Optional[int]:
        """获取类别列表当前选中的类别ID。"""
        current_item = self.class_list.currentItem()
        if current_item is None:
            return None
        return current_item.data(Qt.ItemDataRole.UserRole)

    def _set_class_list_selection(self, class_id: Optional[int]):
        """按类别ID同步类别列表选中状态。"""
        if class_id is None:
            self.class_list.clearSelection()
            self._refresh_annotation_class_controls()
            return

        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == class_id:
                self.class_list.setCurrentRow(row)
                self.current_class_id = class_id
                self.canvas.current_class_id = class_id
                break

        self._sync_attr_class_combo_from_list()
        self._refresh_annotation_class_controls()
        self._update_current_class_chip()

    def _sync_attr_class_combo_from_list(self):
        """将类别列表当前选择同步到属性下拉框。"""
        class_id = self._get_selected_class_list_id()
        self.attr_class.blockSignals(True)
        if class_id is None:
            self.attr_class.setCurrentIndex(-1)
        else:
            index = self.attr_class.findData(class_id)
            if index >= 0:
                self.attr_class.setCurrentIndex(index)
        self.attr_class.blockSignals(False)

    def _refresh_annotation_class_controls(self):
        """根据当前标注和类别选择刷新“应用修改”按钮状态。"""
        selected_annotation_id = self.canvas.selected_annotation_id
        if selected_annotation_id is None:
            self.btn_apply_attr.setEnabled(False)
            return

        annotation = next((ann for ann in self.annotations if ann['id'] == selected_annotation_id), None)
        if annotation is None:
            self.btn_apply_attr.setEnabled(False)
            return

        selected_class_id = self._get_selected_class_list_id()
        self.btn_apply_attr.setEnabled(
            selected_class_id is not None and selected_class_id != annotation.get('class_id')
        )

    def refresh_class_list_counts(self, class_sample_counts=None):
        """刷新类别列表中的样本数量显示"""
        if class_sample_counts is None:
            class_sample_counts = self.get_project_sample_counts()

        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if not item:
                continue
            class_id = item.data(Qt.ItemDataRole.UserRole)
            class_info = next((cls for cls in self.classes if cls['id'] == class_id), None)
            if not class_info:
                continue
            sample_count = class_sample_counts.get(class_id, 0)
            item.setText(f"■ {class_info['name']} ({sample_count})")

    def update_sample_control_panel(self, class_sample_counts=None, negative_sample_count=None):
        """刷新样本调节面板"""
        if class_sample_counts is None or negative_sample_count is None:
            cached_class_counts, cached_negative_sample_count = self._get_sample_stats()
            if class_sample_counts is None:
                class_sample_counts = cached_class_counts
            if negative_sample_count is None:
                negative_sample_count = cached_negative_sample_count

        self.refresh_class_list_counts(class_sample_counts)
        target_class_id = self.sample_target_class.currentData()
        if target_class_id is None:
            self.sample_count_label.setText("0")
            self.sample_delete_count.blockSignals(True)
            self.sample_delete_count.setRange(0, 0)
            self.sample_delete_count.setValue(0)
            self.sample_delete_count.blockSignals(False)
            return

        if target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
            sample_count = negative_sample_count
        else:
            sample_count = class_sample_counts.get(target_class_id, 0)
        self.sample_count_label.setText(str(sample_count))

        self.sample_delete_count.blockSignals(True)
        self.sample_delete_count.setRange(0, sample_count)
        if self.sample_delete_count.value() > sample_count:
            self.sample_delete_count.setValue(sample_count)
        self.sample_delete_count.blockSignals(False)

    def on_sample_target_changed(self, index):
        """删样目标切换事件"""
        self.update_sample_control_panel()
    
    def init_auto_label_components(self):
        """初始化自动标注组件"""
        if not self.model_manager:
            self.model_manager = ModelManager()
        
        if not self.auto_label_dialog:
            self.auto_label_dialog = AutoLabelDialog(self)
            self.auto_label_dialog.single_inference_requested.connect(self.on_single_inference_requested)
            self.auto_label_dialog.batch_inference_requested.connect(self.on_batch_inference_requested)
        
        if not self.batch_labeling_manager:
            self.batch_labeling_manager = BatchLabelingManager()
            self.batch_labeling_manager.progress_updated.connect(self.on_batch_inference_progress)
            self.batch_labeling_manager.batch_completed.connect(self.on_batch_inference_completed)
        
        # 初始化加载动画
        if not hasattr(self, 'loading_label'):
            self.loading_label = QLabel("加载中...")
            self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.loading_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.7);
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 20px;
                    border-radius: 8px;
                }
            """)
            self.loading_label.hide()
            # 将加载动画添加到主布局
            self.main_layout.addWidget(self.loading_label)
            self.loading_label.setGeometry(
                self.width() // 2 - 100,
                self.height() // 2 - 50,
                200,
                100
            )
            self.loading_label.raise_()
    
    def show_loading_animation(self, message):
        """显示加载动画"""
        if not hasattr(self, 'loading_label'):
            self.init_auto_label_components()
        
        self.loading_label.setText(message)
        self.loading_label.setGeometry(
            self.width() // 2 - 150,
            self.height() // 2 - 50,
            300,
            100
        )
        self.loading_label.show()
        self.loading_label.raise_()
        # 强制刷新界面
        self.repaint()
    
    def hide_loading_animation(self):
        """隐藏加载动画"""
        if hasattr(self, 'loading_label'):
            self.loading_label.hide()
    
    # 配置窗口里的三页：设置页和「去配置」入口按名字点进去，不让用户自己找
    AUTO_LABEL_TABS = {'yolo': 0, 'sam': 1, 'llm': 2}

    def open_auto_label_config(self, section: str = "") -> bool:
        """打开自动标注配置窗口，可指定直接落在哪一页。

        设置页和各处「去配置」入口都走这里——配置文件（SAM / LLM）是全局的，
        没有项目也能配，只是类别映射没东西可写，这时不碰数据库。

        返回用户有没有点保存。
        """
        self.init_auto_label_components()
        self.auto_label_dialog.set_classes(self.classes)

        tab_index = self.AUTO_LABEL_TABS.get(section)
        if tab_index is not None:
            self.auto_label_dialog.tab_widget.setCurrentIndex(tab_index)

        if self.auto_label_dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        # 保存 YOLO 预标注的参数（SAM / LLM 的配置由对话框自己写进 config/*.json）
        self.auto_label_settings = {
            'model_path': self.auto_label_dialog.get_model_path(),
            'model_task': self.auto_label_dialog.get_model_task(),
            'conf_threshold': self.auto_label_dialog.sb_conf_threshold.value(),
            'iou_threshold': self.auto_label_dialog.sb_iou_threshold.value(),
            'class_mapping': self.auto_label_dialog.get_class_mappings(),
            'only_unlabeled': self.auto_label_dialog.chk_only_unlabeled.isChecked(),
            'overwrite_labels': self.auto_label_dialog.chk_overwrite.isChecked(),
        }

        # 类别是项目的东西：没有项目就没有类别可写，这时一个字都不往数据库里落
        new_classes = getattr(self.auto_label_dialog, 'project_classes', None)
        if self.current_project_id and new_classes is not None and new_classes != self.classes:
            self.classes = new_classes
            db.update_project(self.current_project_id, classes=self.classes)
            self.update_class_list()
            QMessageBox.information(self, "成功", "类别列表已更新")

        self.apply_sam_button_mode()
        return True

    def _offer_auto_label_config(self, title: str, message: str, section: str) -> bool:
        """缺配置时不只是「告诉用户去哪配」，而是给一个真能点开配置的按钮。

        用户点「去配置」→ 直接开配置窗口的对应页；点了保存就返回 True，
        调用方可以就地重读配置继续干活，不用再走一遍菜单。
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(message)
        config_btn = box.addButton("去配置", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        if box.clickedButton() is not config_btn:
            return False
        return self.open_auto_label_config(section)

    # ==================== 批量标注：状态、取消、前置条件 ====================

    def _refresh_batch_status(self):
        """把批量任务的状态写回状态栏。

        单独一个方法，是因为 update_status_bar / set_project / 切页回来都会经过
        状态栏刷新——它们必须重画这一格，而不是把它清空。
        """
        self._set_elided_text(self.status_batch, self._batch_status_text)
        self.btn_cancel_batch.setVisible(self._batch_running)

    def _set_batch_status(self, text: str, running: bool):
        self._batch_status_text = text
        self._batch_running = running
        self._refresh_batch_status()
        self._update_action_availability()

    def _batch_in_progress(self) -> bool:
        """现在有没有一批在跑（YOLO 或 LLM 都算）。

        防双击的第一层：页面这边先拦一道。第二层在 BatchLabelingManager 里，
        因为「确认之后、线程真正起来之前」还有一个窗口期，光看 isRunning() 拦不住。
        """
        if self._batch_running:
            return True
        manager = self.batch_labeling_manager
        if manager is not None and manager.is_running():
            return True
        if self.llm_batch_worker is not None and self.llm_batch_worker.isRunning():
            return True
        return False

    def cancel_active_batch(self):
        """取消正在跑的这一批：只发请求，界面线程不等它退出。"""
        self.btn_cancel_batch.setEnabled(False)
        self._set_batch_status("正在取消…", True)

        manager = self.batch_labeling_manager
        if manager is not None and manager.is_running():
            manager.request_cancel()

        if self.llm_batch_worker is not None and self.llm_batch_worker.isRunning():
            self.llm_batch_worker.cancel()

    def _finish_batch_ui(self):
        """一批跑完（或被取消）之后，把界面收回常态。"""
        self._active_batch_plan = None
        self.btn_cancel_batch.setEnabled(True)
        self._set_batch_status("", False)

    def _require_yolo_settings(self) -> Optional[dict]:
        """拿到用户保存过的 YOLO 设置；没有就把设置页打开，让他先选个模型。

        以前没有设置时会悄悄退回 "yolov8n"。那不是一个无害的默认值：
        ultralytics 拿到这个名字会去联网下载一个 COCO 通用模型，然后把它认识的
        80 个类别（人、车、猫……）写进用户的标注里——而用户的项目类别可能是
        「安全帽」。宁可拦下来让他选，也不能替他决定要跑哪个模型。
        """
        settings = getattr(self, 'auto_label_settings', None)
        if settings and settings.get('model_path'):
            return settings

        if not self._offer_auto_label_config(
            "还没有配置模型",
            "自动标注要先在「YOLO 检测」里选好模型和推理参数。",
            'yolo',
        ):
            return None

        settings = getattr(self, 'auto_label_settings', None)
        if not settings or not settings.get('model_path'):
            return None
        return settings

    @staticmethod
    def _load_llm_config() -> dict:
        """读 LLM 配置：默认值 + 用户存过的那份。"""
        from gui.pages.auto_label_dialog import LLM_CONFIG_FILE, DEFAULT_LLM_CONFIG

        llm_config = DEFAULT_LLM_CONFIG.copy()
        if os.path.exists(LLM_CONFIG_FILE):
            try:
                with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    llm_config.update(json.load(f))
            except Exception as e:
                print(f"加载LLM配置失败: {e}")
        return llm_config

    def _require_llm_config(self) -> Optional[dict]:
        """拿到能用的 LLM 配置；没填 API Key 就当场给一个能点开的配置入口。"""
        llm_config = self._load_llm_config()
        if llm_config.get('api_key'):
            return llm_config

        if not self._offer_auto_label_config(
            "还没填 API Key",
            "用大模型标注要先在 LLM 视觉里填好 API Key 和模型名。",
            'llm',
        ):
            return None

        llm_config = self._load_llm_config()
        if not llm_config.get('api_key'):
            QMessageBox.warning(self, "提示", "还是没有 API Key，填好之后再试。")
            return None
        return llm_config

    def show_auto_label_settings(self):
        """显示自动标注设置对话框（工具栏「自动标注 → 设置」）"""
        self.open_auto_label_config()


    def start_sam_annotation(self):
        """开始SAM交互式标注"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        
        # 获取当前图片路径
        current_image = None
        for img in self.images:
            if img['id'] == self.current_image_id:
                current_image = img
                break
        
        if not current_image:
            QMessageBox.warning(self, "提示", "无法获取当前图片信息")
            return
        
        image_path = current_image.get('storage_path', '')
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "图片文件不存在")
            return
        
        # 获取已保存的SAM配置（避免临时弹窗回落到默认值）
        sam_config = AutoLabelDialog.get_saved_sam_config()

        if not sam_config or not sam_config.get('model_file'):
            # 配完就地继续，不用再点一遍「自动标注 → 设置」
            if not self._offer_auto_label_config(
                "还没配置 SAM 模型",
                "SAM 交互分割要先选好分割模型和权重文件。",
                'sam',
            ):
                return
            sam_config = AutoLabelDialog.get_saved_sam_config()
            if not sam_config or not sam_config.get('model_file'):
                QMessageBox.warning(self, "提示", "还是没有可用的 SAM 权重，先配好再来。")
                return


        # 获取项目类型
        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        
        # 进入SAM标注模式
        self.canvas.set_tool('sam')
        self.canvas.set_sam_operation_mode("normal")
        self.canvas.sam_config = sam_config
        self.canvas.sam_image_path = image_path
        self.canvas.sam_project_type = project_type
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.sam_mode = 'point'  # 'point' 或 'bbox'
        
        # 显示提示
        task_name = "边界框" if project_type == 'detect' else "分割"
        QMessageBox.information(self, "SAM标注模式", 
            f"进入SAM交互式标注模式 ({task_name}任务):\n"
            "• 左键点击: 添加点提示并推理\n"
            "• Ctrl+左键: 添加多个点（不立即推理）\n"
            "• 右键拖拽: 绘制框提示\n"
            "• 松开鼠标: 自动推理\n"
            "• 按ESC键: 退出SAM模式")

    def _get_current_image_path(self) -> str:
        if not self.current_image_id:
            return ""
        for img in self.images:
            if img['id'] == self.current_image_id:
                return img.get('storage_path', '')
        return ""

    def _prepare_memory_mode_context(self):
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return None, None
        image_path = self._get_current_image_path()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "请先选择有效图片")
            return None, None
        sam_config = AutoLabelDialog.get_saved_sam_config()
        if sam_config.get("usage_mode") != "memory" or sam_config.get("sam_type") not in ("SAM2", "SAM3"):
            if not self._offer_auto_label_config(
                "当前不是记忆标注模式",
                "记忆标注需要把 SAM 设成 SAM2 或 SAM3，并把用法选成「记忆」。",
                'sam',
            ):
                return None, None
            sam_config = AutoLabelDialog.get_saved_sam_config()
            if sam_config.get("usage_mode") != "memory" or sam_config.get("sam_type") not in ("SAM2", "SAM3"):
                QMessageBox.warning(self, "提示", "SAM 设置仍然不是记忆标注模式（仅 SAM2/SAM3 支持）。")
                return None, None
        return sam_config, image_path

    def start_sam_memory_update(self):
        """打开记忆对象管理并允许在画布上输入对象提示。"""
        sam_config, image_path = self._prepare_memory_mode_context()
        if not sam_config:
            return

        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'

        self.canvas.set_tool('sam')
        self.canvas.set_sam_operation_mode("memory_collect")
        # 清理进入记忆模式前残留的提示与异步回调影响
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.sam_config = sam_config
        self.canvas.sam_image_path = image_path
        self.canvas.sam_project_type = project_type

        if self.sam_memory_dialog is None:
            self.sam_memory_dialog = SAMMemoryObjectsDialog(self)
            self.sam_memory_dialog.add_requested.connect(self.add_sam_memory_object_from_canvas)
            self.sam_memory_dialog.delete_requested.connect(self.delete_sam_memory_object)
            self.sam_memory_dialog.save_requested.connect(self.save_sam_memory_and_infer_current)
            self.sam_memory_dialog.closed.connect(self.on_sam_memory_dialog_closed)
        self.sam_memory_dialog.update_objects(self.sam_memory_objects)
        self.sam_memory_dialog.show()
        self.sam_memory_dialog.raise_()
        # 清空画布上的临时标记，并显示所有已有对象的标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self._draw_all_memory_objects_on_canvas()

        QMessageBox.information(self, "记忆标注", "请在图片上添加点/框提示后点击“添加对象“，\n可以指定ID来更新已有对象。")

    def add_sam_memory_object_from_canvas(self):
        print(f"[SAM Memory] 添加对象被调用")
        points = list(self.canvas.sam_points)
        bboxes = list(self.canvas.sam_bboxes)
        print(f"[SAM Memory] 当前画布标记: points={len(points)}, bboxes={len(bboxes)}")
        
        if not points and not bboxes:
            QMessageBox.warning(self, "提示", "请先在图片上添加点或框提示")
            return
        
        # 弹窗输入对象ID
        from PyQt6.QtWidgets import QInputDialog
        
        # 获取建议的ID（已有ID的最大值+1）
        suggested_id = 0
        if self.sam_memory_objects:
            suggested_id = max(x["obj_id"] for x in self.sam_memory_objects) + 1
        
        print(f"[SAM Memory] 建议ID: {suggested_id}")
        
        try:
            obj_id, ok = QInputDialog.getInt(
                self,
                "添加记忆对象",
                "请输入对象ID:",
                value=suggested_id,
                min=0,
                max=999
            )
        except Exception as e:
            print(f"[SAM Memory] 输入对话框出错: {e}")
            return
        
        if not ok:
            print(f"[SAM Memory] 用户取消输入")
            return
        
        print(f"[SAM Memory] 用户输入ID: {obj_id}")
        
        # 检查是否已存在该ID，如果存在则更新
        existing_idx = None
        for i, obj in enumerate(self.sam_memory_objects):
            if obj["obj_id"] == obj_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # 更新已有对象
            reply = QMessageBox.question(
                self,
                "ID已存在",
                f"ID {obj_id} 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.sam_memory_objects[existing_idx] = {
                    "obj_id": obj_id,
                    "points": points,
                    "bboxes": bboxes,
                }
                print(f"[SAM Memory] 更新已有对象 ID={obj_id}")
        else:
            # 添加新对象
            self.sam_memory_objects.append({
                "obj_id": obj_id,
                "points": points,
                "bboxes": bboxes,
            })
            print(f"[SAM Memory] 添加新对象 ID={obj_id}")
        
        # 清空画布上的临时标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.update()
        
        # 更新对话框列表
        if self.sam_memory_dialog:
            self.sam_memory_dialog.update_objects(self.sam_memory_objects)
        
        # 在图上显示所有对象的标记
        print(f"[SAM Memory] 开始绘制所有对象标记")
        self._draw_all_memory_objects_on_canvas()
        print(f"[SAM Memory] 绘制完成")

    def delete_sam_memory_object(self, index: int):
        if 0 <= index < len(self.sam_memory_objects):
            del self.sam_memory_objects[index]
            if self.sam_memory_dialog:
                self.sam_memory_dialog.update_objects(self.sam_memory_objects)
            # 重新绘制所有对象标记
            self._draw_all_memory_objects_on_canvas()
    
    def _draw_all_memory_objects_on_canvas(self):
        """在画布上绘制所有记忆对象的标记"""
        print(f"[SAM Memory] _draw_all_memory_objects_on_canvas 被调用")
        
        # 清空当前临时标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        
        # 收集所有对象的标记用于显示
        all_points = []
        all_bboxes = []
        
        print(f"[SAM Memory] 当前记忆对象数: {len(self.sam_memory_objects)}")
        
        for obj in self.sam_memory_objects:
            obj_id = obj.get("obj_id", 0)
            points = obj.get("points", [])
            bboxes = obj.get("bboxes", [])
            
            print(f"[SAM Memory] 对象 ID={obj_id}: points={len(points)}, bboxes={len(bboxes)}")
            
            # 为每个点添加ID信息
            for p in points:
                all_points.append({
                    "x": p[0],
                    "y": p[1],
                    "obj_id": obj_id
                })
            
            # 为每个框添加ID信息
            for bbox in bboxes:
                all_bboxes.append({
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "obj_id": obj_id
                })
        
        print(f"[SAM Memory] 总显示点数: {len(all_points)}, 总显示框数: {len(all_bboxes)}")
        
        # 保存到画布的显示列表
        self.canvas.memory_display_points = all_points
        self.canvas.memory_display_bboxes = all_bboxes
        self.canvas.update()
        print(f"[SAM Memory] 画布更新完成")

    def on_sam_memory_dialog_closed(self):
        """记忆对象管理对话框关闭时的处理"""
        print(f"[SAM Memory] 对话框关闭，清除画布上的记忆标记")
        # 清除画布上的记忆标记显示
        self.canvas.memory_display_points = []
        self.canvas.memory_display_bboxes = []
        # 清空临时标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.update()
        # 退出SAM记忆采集模式
        self.canvas.set_sam_operation_mode("normal")
        self.canvas.sam_mode_active = False
        self.canvas.set_tool('rectangle')
        print(f"[SAM Memory] 已清除标记并退出SAM模式")

    def clear_sam_memory(self):
        SAMMemoryPredictorManager.instance().clear()
        self.sam_memory_objects = []
        if self.sam_memory_dialog:
            self.sam_memory_dialog.update_objects(self.sam_memory_objects)
        QMessageBox.information(self, "提示", "SAM记忆已清空")

    def _run_memory_predictor(self, sam_config: dict, image_path: str, update_objects: list = None):
        predictor = SAMMemoryPredictorManager.instance().get_predictor(sam_config)
        last_update_results = None
        if update_objects:
            for obj in update_objects:
                kwargs = {
                    "source": image_path,
                    "obj_ids": [obj["obj_id"]],
                    "update_memory": True,
                }
                if obj.get("bboxes"):
                    kwargs["bboxes"] = [obj["bboxes"][-1]]
                if obj.get("points"):
                    kwargs["points"] = [[p[0], p[1]] for p in obj["points"]]
                    kwargs["labels"] = [1] * len(obj["points"])
                update_results = predictor(**kwargs)
                if update_results:
                    last_update_results = update_results
        infer_results = predictor(source=image_path)
        # 某些场景下“同图更新记忆后立刻推理同图”会返回空，回退到最近一次更新结果
        if self._memory_results_empty(infer_results) and not self._memory_results_empty(last_update_results):
            return last_update_results
        return infer_results

    @staticmethod
    def _memory_results_empty(results) -> bool:
        if not results or len(results) == 0:
            return True
        result = results[0]
        if not hasattr(result, "masks") or result.masks is None or result.masks.data is None:
            return True
        try:
            return len(result.masks.data) == 0
        except Exception:
            return False

    def save_sam_memory_and_infer_current(self):
        sam_config, image_path = self._prepare_memory_mode_context()
        if not sam_config:
            return
        if not self.sam_memory_objects:
            QMessageBox.warning(self, "提示", "请先添加至少一个记忆对象")
            return
        try:
            self.canvas.set_sam_operation_mode("normal")
            results = self._run_memory_predictor(sam_config, image_path, update_objects=self.sam_memory_objects)
            ok, msg, ann_count = self._apply_memory_results_to_current_image(results)
            if ok:
                QMessageBox.information(
                    self,
                    "记忆推理完成",
                    f"记忆对象数: {len(self.sam_memory_objects)}\n当前图新增标注: {ann_count}"
                )
            else:
                QMessageBox.warning(self, "记忆推理无结果", msg)
        except Exception as e:
            QMessageBox.critical(self, "记忆推理失败", str(e))

    def _apply_memory_results_to_current_image(self, results):
        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        anns = self._build_annotations_from_memory_results(results, project_type)
        if not anns:
            summary = self._summarize_memory_results(results)
            return False, f"模型返回了空结果（没有可用mask/轮廓）。\n结果摘要: {summary}\n请尝试补充提示点或框后重试。", 0
        for ann in anns:
            self.on_annotation_created(ann)
        self.load_annotations()
        return True, "ok", len(anns)

    @staticmethod
    def _summarize_memory_results(results) -> str:
        if not results or len(results) == 0:
            return "results=empty"
        result = results[0]
        if not hasattr(result, "masks") or result.masks is None or result.masks.data is None:
            return "masks=None"
        data = result.masks.data
        try:
            shape = tuple(data.shape)
        except Exception:
            shape = "unknown"
        try:
            count = len(data)
        except Exception:
            count = "unknown"
        return f"masks.shape={shape}, count={count}"

    def run_sam_memory_single(self):
        sam_config, image_path = self._prepare_memory_mode_context()
        if not sam_config:
            return
        project = db.get_project(self.current_project_id)
        _ = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        try:
            self.canvas.set_sam_operation_mode("normal")
            results = self._run_memory_predictor(sam_config, image_path)
            ok, msg, ann_count = self._apply_memory_results_to_current_image(results)
            if ok:
                QMessageBox.information(self, "单张记忆推理完成", f"当前图新增标注: {ann_count}")
            else:
                QMessageBox.warning(self, "单张记忆推理无结果", msg)
        except Exception as e:
            QMessageBox.critical(self, "推理失败", str(e))

    def run_sam_memory_batch(self):
        sam_config, _ = self._prepare_memory_mode_context()
        if not sam_config:
            return
        if not self.images:
            QMessageBox.warning(self, "提示", "当前项目没有图片")
            return
        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择当前图片，批量推理将从该图片开始")
            return
        if not self.sam_memory_objects:
            QMessageBox.warning(self, "提示", "请先执行“更新记忆”并保存至少一个对象")
            return

        start_idx = next((i for i, img in enumerate(self.images) if img.get("id") == self.current_image_id), -1)
        if start_idx < 0:
            QMessageBox.warning(self, "提示", "未找到当前图片在列表中的位置")
            return
        images_to_process = self.images[start_idx:]
        if not images_to_process:
            QMessageBox.warning(self, "提示", "当前图片之后没有可处理图片")
            return

        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        processed = 0
        failed = 0
        for img in images_to_process:
            image_path = img.get("storage_path", "")
            if not image_path or not os.path.exists(image_path):
                failed += 1
                continue
            try:
                results = self._run_memory_predictor(sam_config, image_path)
                anns = self._build_annotations_from_memory_results(results, project_type)
                for ann in anns:
                    class_id = ann.get('class_id', self.current_class_id)
                    class_name = self.classes[class_id]['name'] if class_id < len(self.classes) else 'unknown'
                    db.add_annotation(
                        image_id=img['id'],
                        project_id=self.current_project_id,
                        class_id=class_id,
                        class_name=class_name,
                        annotation_type=ann['type'],
                        data=ann['data']
                    )
                if anns:
                    db.update_image_status(img['id'], 'annotated')
                processed += 1
            except Exception:
                failed += 1
        self.load_annotations()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
        QMessageBox.information(
            self,
            "批量记忆推理",
            f"处理范围: 从当前图片开始，共 {len(images_to_process)} 张\n完成: {processed} 张，失败: {failed} 张"
        )

    def _build_annotations_from_memory_results(self, results, project_type: str) -> list:
        if not results or len(results) == 0:
            return []
        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy() if hasattr(result.masks.data, 'cpu') else result.masks.data
        masks = np.asarray(masks)
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [m for m in masks]
        elif masks.ndim >= 4:
            # 兼容 [N, K, H, W]，统一拉平为多个2D mask
            h, w = masks.shape[-2], masks.shape[-1]
            mask_list = [m for m in masks.reshape(-1, h, w)]
        else:
            return []
        annotations = []
        for mask in mask_list:
            if mask.ndim > 2:
                mask = mask.squeeze()
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            if project_type == 'detect':
                all_x, all_y, all_x2, all_y2 = [], [], [], []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    all_x.append(x)
                    all_y.append(y)
                    all_x2.append(x + w)
                    all_y2.append(y + h)
                annotations.append({
                    'type': 'bbox',
                    'class_id': self.current_class_id,
                    'data': {
                        'x': float(min(all_x)),
                        'y': float(min(all_y)),
                        'width': float(max(all_x2) - min(all_x)),
                        'height': float(max(all_y2) - min(all_y)),
                    }
                })
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                points = [{'x': float(p[0][0]), 'y': float(p[0][1])} for p in approx_contour]
                if points:
                    annotations.append({
                        'type': 'polygon',
                        'class_id': self.current_class_id,
                        'data': {'points': points}
                    })
        return annotations
    
    def _exit_batch_process_mode(self):
        """退出批处理点选模式。

        幂等：确认执行、点取消、直接关窗三条路径都走这里，重复调用没有副作用。
        以前只有「确认执行」会清理，用户一取消就卡在点选模式里——画布上每点一下
        还在继续加点，而且是加给一个已经关掉的对话框。
        """
        canvas = getattr(self, 'canvas', None)
        if canvas is not None:
            canvas.batch_process_mode = False
            canvas.batch_process_points = []
            canvas.batch_process_dialog = None
            canvas.update()
        self.batch_process_dialog = None

    def show_batch_process_dialog(self):
        """显示批量处理标注对话框"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return

        if not self.images:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return

        # 上一次的对话框可能还开着，先收干净再开新的
        previous = self.batch_process_dialog
        self._exit_batch_process_mode()
        if previous is not None:
            previous.close()
            previous.deleteLater()

        # 创建对话框
        dialog = BatchProcessDialog(self, self.classes, len(self.images))
        dialog.process_requested.connect(self.on_batch_process_requested)
        # 接受 / 取消 / 直接关窗都会触发 finished，统一在这里退出点选模式
        dialog.finished.connect(lambda _result: self._exit_batch_process_mode())

        # 进入像素点选择模式
        self.batch_process_dialog = dialog
        self.canvas.batch_process_mode = True
        self.canvas.batch_process_points = []
        self.canvas.batch_process_dialog = dialog
        self.canvas.update()

        # 显示对话框（非模态，允许在图片上点击）
        dialog.show()

    def on_batch_process_requested(self, config):
        """处理批量处理请求"""
        # 退出像素点选择模式
        self._exit_batch_process_mode()

        # 执行批量处理
        self.execute_batch_process(config)
    
    def execute_batch_process(self, config):
        """执行批量处理"""
        points = config['points']
        start_idx = config['start_idx']
        end_idx = config['end_idx']
        operation = config['operation']
        
        # 获取处理范围内的图片
        images_to_process = self.images[start_idx:end_idx+1]
        
        if not images_to_process:
            QMessageBox.warning(self, "提示", "没有需要处理的图片")
            return
        
        # 显示进度对话框
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("正在批量处理标注...", "取消", 0, len(images_to_process), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        processed_count = 0
        modified_count = 0
        
        try:
            for i, image_data in enumerate(images_to_process):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"正在处理第 {i+1}/{len(images_to_process)} 张图片...")
                
                image_id = image_data['id']
                annotations = db.get_image_annotations(image_id)
                
                if not annotations:
                    continue
                
                # 检查每个标注是否覆盖选择的像素点
                for annotation in annotations:
                    annotation_data = annotation.get('data', {})
                    annotation_type = annotation.get('type', 'bbox')
                    
                    # 检查标注是否覆盖任何选择的像素点
                    covers_point = False
                    for point in points:
                        px, py = point
                        if self.is_point_in_annotation_data(px, py, annotation_data, annotation_type):
                            covers_point = True
                            break
                    
                    if covers_point:
                        if operation == 'delete':
                            # 批量删除：检查类别是否在目标类别列表中
                            target_classes = config.get('target_classes', [])
                            if annotation.get('class_id') in target_classes:
                                db.delete_annotation(annotation['id'])
                                modified_count += 1
                        else:
                            # 批量修改：检查类别是否在源类别列表中
                            source_classes = config.get('source_classes', [])
                            target_class = config.get('target_class')
                            if annotation.get('class_id') in source_classes:
                                target_class_info = next(
                                    (cls for cls in self.classes if cls['id'] == target_class),
                                    None
                                )
                                db.update_annotation(
                                    annotation['id'],
                                    class_id=target_class,
                                    class_name=target_class_info['name'] if target_class_info else None
                                )
                                modified_count += 1
                
                processed_count += 1
            
            progress.setValue(len(images_to_process))
            
            # 显示结果
            QMessageBox.information(
                self, 
                "批量处理完成", 
                f"处理完成！\n处理了 {processed_count} 张图片\n修改了 {modified_count} 个标注"
            )
            
            # 刷新当前图片的标注显示
            self._invalidate_sample_stats_cache()
            if self.current_image_id:
                self.load_annotations()
            self.update_sample_control_panel()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量处理出错: {str(e)}")
    
    def is_point_in_annotation_data(self, px: int, py: int, data: dict, ann_type: str) -> bool:
        """检查点是否在标注数据内"""
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            return x <= px <= x + width and y <= py <= y + height
        elif ann_type == 'polygon':
            points = data.get('points', [])
            if len(points) < 3:
                return False
            # 使用射线法判断点是否在多边形内
            return self.point_in_polygon(px, py, points)
        return False
    
    def point_in_polygon(self, x: int, y: int, polygon: list) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]['x'], polygon[0]['y']
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]['x'], polygon[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def run_single_inference(self):
        """标注当前图片：单张不弹二次确认，直接跑——但模型必须是用户自己选过的。"""
        if not self.current_image_data:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return

        # 没有保存过设置就不能开跑：宁可把设置页推到用户面前，也不能拿一个
        # 他没选过的模型去写他的标注。这一步必须在加载动画之前，否则设置窗口
        # 会开在一块「正在推理…」的遮罩后面。
        settings = self._require_yolo_settings()
        if not settings:
            return

        model_path = settings['model_path']
        model_task = settings.get('model_task', 'detect')
        conf_threshold = settings.get('conf_threshold', 0.5)
        iou_threshold = settings.get('iou_threshold', 0.45)
        class_mapping = settings.get('class_mapping', {})
        overwrite_labels = settings.get('overwrite_labels', False)

        # 显示加载动画
        self.show_loading_animation("正在标注当前图片...")

        try:
            from core.auto_labeler import AutoLabeler

            # 使用process_single_image方法，支持model_task
            labeler = AutoLabeler(model_path, self.model_manager)
            
            # 加载模型配置
            import re
            match = re.match(r'(yolov)(\d+)([a-z])', model_path)
            if match:
                version_num = match.group(2)
                size = match.group(3)
                version = f"YOLOv{version_num}"
                load_config = {
                    'model_version': version,
                    'model_size': size,
                    'model_source': 'official',
                    'model_task': model_task,  # 使用保存的任务类型
                    'class_mappings': class_mapping
                }
                labeler.load_model(load_config)
            else:
                # 自定义模型
                if os.path.exists(model_path):
                    # 对于自定义模型，也需要设置model_task
                    load_config = {
                        'model_version': '',  # 自定义模型不需要版本
                        'model_size': '',  # 自定义模型不需要大小
                        'model_source': 'custom',
                        'model_task': model_task,  # 使用保存的任务类型
                        'custom_model_path': model_path,
                        'class_mappings': class_mapping
                    }
                    labeler.load_model(load_config)
            
            # 处理图像
            config = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'class_mappings': class_mapping
            }
            annotations = labeler.process_single_image(
                self.current_image_data['storage_path'], 
                self.current_image_id,
                config
            )
            
            # 更新当前图像的标注
            if annotations and self.current_image_id:
                # 保存标注到数据库
                # 如果需要覆盖原标签，先删除所有原标注
                if overwrite_labels:
                    db.delete_image_annotations(self.current_image_id)
                
                for annotation in annotations:
                    # 获取类别名称
                    class_id = annotation['class_id']
                    
                    # 检查类别ID是否在项目类别范围内
                    existing_class = next((cls for cls in self.classes if cls['id'] == class_id), None)
                    if existing_class:
                        class_name = existing_class['name']
                    else:
                        # 创建新类别
                        class_name = f"class_{class_id}"
                        # 生成随机颜色
                        import random
                        color = f"#{random.randint(0, 0xFFFFFF):06x}"
                        new_class = {
                            'id': class_id,
                            'name': class_name,
                            'color': color
                        }
                        self.classes.append(new_class)
                        # 更新项目类别
                        db.update_project(self.current_project_id, classes=self.classes)
                    
                    # 保存标注
                    db.add_annotation(
                        self.current_image_id,
                        self.current_project_id,
                        class_id,
                        class_name,
                        annotation['type'],
                        annotation['data']
                    )
                
                # 重新加载当前图像的标注
                self.load_current_image_annotations()
                
                QMessageBox.information(self, "成功", "自动标注完成！")
            else:
                # 如果没有检测到目标，但需要覆盖原标签，也删除原标注
                if overwrite_labels and self.current_image_id:
                    db.delete_image_annotations(self.current_image_id)
                    self.load_current_image_annotations()
                QMessageBox.information(self, "提示", "未检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {str(e)}")
        finally:
            # 隐藏加载动画
            self.hide_loading_animation()
    
    def _build_yolo_batch_plan(self, settings: dict) -> Optional[BatchPlan]:
        """把「保存过的 YOLO 设置 + 当前图片列表」冻成一份计划快照。"""
        only_unlabeled = settings.get('only_unlabeled', True)
        if only_unlabeled:
            # status 可能是 None 或空字符串（老数据），一并当成未标注
            targets = [
                img for img in self.images
                if img.get('status') not in ('annotated', 'completed')
            ]
            scope = SCOPE_UNLABELED
        else:
            targets = list(self.images)
            scope = SCOPE_ALL

        if not targets:
            QMessageBox.information(
                self, "没有要处理的图片",
                f"当前范围是「{'仅未标注的图片' if only_unlabeled else '全部图片'}」，"
                f"里面一张图片都没有。\n项目共 {len(self.images)} 张图片。",
            )
            return None

        return BatchPlan(
            project_id=self.current_project_id,
            engine='yolo',
            images=targets,
            scope=scope,
            model_label=settings['model_path'],
            model_path=settings['model_path'],
            model_task=settings.get('model_task', 'detect'),
            conf=settings.get('conf_threshold', 0.5),
            iou=settings.get('iou_threshold', 0.45),
            overwrite=settings.get('overwrite_labels', False),
            class_mapping=settings.get('class_mapping', {}),
        )

    def run_batch_inference(self):
        """批量标注：只负责问清楚、拿到确认，然后把那份快照交出去。

        这里绝不会「顺手就开跑」——菜单项后面的省略号承诺的就是这一步。
        """
        if not self.current_project_id or len(self.images) == 0:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return

        if self._batch_in_progress():
            QMessageBox.information(self, "提示", "上一批还在跑，等它结束或者先取消。")
            return

        settings = self._require_yolo_settings()
        if not settings:
            return

        plan = self._build_yolo_batch_plan(settings)
        if plan is None:
            return

        confirmed = confirm_batch_plan(self, plan)
        if confirmed is None:
            # 取消：没建线程、没写库、没动任何图片状态
            return

        self._start_yolo_batch(confirmed)

    def _start_yolo_batch(self, plan: BatchPlan) -> bool:
        """按快照开跑。执行用的就是确认框还回来的那一份，不再回头读界面。"""
        if self._batch_in_progress():
            return False

        self.init_auto_label_components()

        self._active_batch_plan = plan
        self._set_batch_status(f"准备中：{plan.count} 张图片", True)

        started = self.batch_labeling_manager.start_batch_processing(
            plan, self.model_manager,
        )
        if not started:
            self._finish_batch_ui()
        return started


    def on_single_inference_requested(self, model_path, conf_threshold, iou_threshold, class_mapping, image_path, model_task='detect'):
        """单张推理请求回调"""
        from core.auto_labeler import AutoLabeler
        
        try:
            labeler = AutoLabeler(model_path, self.model_manager)
            
            # 加载模型，指定任务类型
            import re
            match = re.match(r'(yolov)(\d+)([a-z])', model_path)
            if match:
                version_num = match.group(2)
                size = match.group(3)
                version = f"YOLOv{version_num}"
                load_config = {
                    'model_version': version,
                    'model_size': size,
                    'model_source': 'official',
                    'model_task': model_task,
                    'class_mappings': class_mapping
                }
                labeler.load_model(load_config)
            else:
                # 自定义模型
                if os.path.exists(model_path):
                    labeler.current_model = self.model_manager.load_custom_model(model_path)
            
            # 处理图像
            config = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'class_mappings': class_mapping
            }
            annotations = labeler.process_single_image(image_path, self.current_image_id, config)
            
            # 获取覆盖标签设置
            overwrite_labels = False
            if hasattr(self, 'auto_label_settings'):
                overwrite_labels = self.auto_label_settings.get('overwrite_labels', False)
            
            # 更新当前图像的标注
            if annotations and self.current_image_id:
                # 保存标注到数据库
                # 如果需要覆盖原标签，先删除所有原标注
                if overwrite_labels:
                    db.delete_image_annotations(self.current_image_id)
                
                for annotation in annotations:
                    # 获取类别名称
                    class_id = annotation['class_id']
                    
                    # 检查类别ID是否在项目类别范围内
                    existing_class = next((cls for cls in self.classes if cls['id'] == class_id), None)
                    if existing_class:
                        class_name = existing_class['name']
                    else:
                        # 创建新类别
                        class_name = f"class_{class_id}"
                        # 生成随机颜色
                        import random
                        color = f"#{random.randint(0, 0xFFFFFF):06x}"
                        new_class = {
                            'id': class_id,
                            'name': class_name,
                            'color': color
                        }
                        self.classes.append(new_class)
                        # 更新项目类别
                        db.update_project(self.current_project_id, classes=self.classes)
                    
                    # 保存标注
                    db.add_annotation(
                        self.current_image_id,
                        self.current_project_id,
                        class_id,
                        class_name,
                        annotation['type'],
                        annotation['data']
                    )
                
                # 重新加载当前图像的标注
                self.load_current_image_annotations()
                
                QMessageBox.information(self, "成功", "自动标注完成！")
            else:
                # 如果没有检测到目标，但需要覆盖原标签，也删除原标注
                if overwrite_labels and self.current_image_id:
                    db.delete_image_annotations(self.current_image_id)
                    self.load_current_image_annotations()
                QMessageBox.information(self, "提示", "未检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {str(e)}")
    
    def on_batch_inference_requested(self, model_path, conf_threshold, iou_threshold, class_mapping, images, only_unlabeled, model_task='detect'):
        """设置窗口发来的批量推理请求：也要先过确认框，不能绕过去。"""
        if only_unlabeled:
            filtered_images = [img for img in images if img.get('status') != 'annotated']
        else:
            filtered_images = images

        if not filtered_images:
            QMessageBox.warning(self, "提示", "没有符合条件的图片")
            return

        if self._batch_in_progress():
            QMessageBox.information(self, "提示", "上一批还在跑，等它结束或者先取消。")
            return

        plan = BatchPlan(
            project_id=self.current_project_id,
            engine='yolo',
            images=filtered_images,
            scope=SCOPE_UNLABELED if only_unlabeled else SCOPE_ALL,
            model_label=model_path,
            model_path=model_path,
            model_task=model_task,
            conf=conf_threshold,
            iou=iou_threshold,
            overwrite=(getattr(self, 'auto_label_settings', None) or {}).get('overwrite_labels', False),
            class_mapping=class_mapping,
        )

        confirmed = confirm_batch_plan(self, plan)
        if confirmed is None:
            return

        self._start_yolo_batch(confirmed)

    def on_batch_inference_progress(self, progress, current, total, image_name):
        """批量推理进度回调"""
        self._set_batch_status(f"批量标注: {current}/{total} · {image_name}", True)

    def on_batch_inference_completed(self, success, message, processed_count, cancelled=False):
        """批量推理完成回调。取消不是失败，别拿红色错误框砸用户一脸。"""
        self._finish_batch_ui()

        # 重新加载图片列表以更新状态
        self.load_image_list()
        self.update_status_bar()

        if cancelled:
            QMessageBox.information(self, "已取消", message)
        elif success:
            QMessageBox.information(self, "成功", f"批量自动标注完成！\n处理了 {processed_count} 张图片")
        else:
            QMessageBox.critical(self, "错误", f"批量自动标注失败: {message}")
    
    def load_current_image_annotations(self):
        """加载当前图像的标注"""
        if self.current_image_id:
            self.annotations = db.get_image_annotations(self.current_image_id)
            self.canvas.set_annotations(self.annotations)
            self.update_status_bar()
            self._invalidate_sample_stats_cache()
            self.update_sample_control_panel()
    
    def update_status_bar(self):
        """更新状态栏"""
        if self.images and self.current_image_id:
            # 找到当前图像的索引
            current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), -1)
            if current_index >= 0:
                self.status_image.setText(f"当前: {current_index + 1}/{len(self.images)}")
        else:
            self.status_image.setText("当前: 0/0")
        
        # 更新标注数量
        self.status_annotation.setText(f"标注: {len(self.annotations)}")
        
        # 更新工具状态
        tool_names = {'rectangle': '矩形', 'polygon': '多边形', 'move': '移动'}
        tool_name = tool_names.get(self.canvas.current_tool, '矩形')
        self.status_tool.setText(f"工具: {tool_name}")
    
    def on_class_selected(self):
        """类别选中事件"""
        current_item = self.class_list.currentItem()
        if current_item:
            self.current_class_id = current_item.data(Qt.ItemDataRole.UserRole)
            # 更新画布当前类别
            self.canvas.current_class_id = self.current_class_id
            self._sync_attr_class_combo_from_list()
        self._refresh_annotation_class_controls()
        self._update_current_class_chip()
    
    def filter_images(self, filter_text: str):
        """筛选图片"""
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_id = item.data(Qt.ItemDataRole.UserRole)
            
            # 找到对应的图片数据
            image_data = next((img for img in self.images if img['id'] == image_id), None)
            if not image_data:
                continue
            
            status = image_data.get('status', 'pending')
            
            if filter_text == "全部":
                item.setHidden(False)
            elif filter_text == "未标注":
                item.setHidden(status != 'pending')
            elif filter_text == "已标注":
                item.setHidden(status == 'pending')
    
    def on_image_selected(self, item: QListWidgetItem):
        """图片选中事件"""
        image_id = item.data(Qt.ItemDataRole.UserRole)
        self.load_image(image_id)
    
    def on_task_changed(self, task_type):
        """任务类型切换事件"""
        if self.current_project_id:
            # 更新项目的任务类型
            db.update_project(self.current_project_id, type=task_type)
            # 重新加载当前图片的标注
            if self.current_image_id:
                self.load_annotations()
        
        # 根据任务类型调整标注工具显示
        self.adjust_tool_visibility(task_type)
    
    def adjust_tool_visibility(self, task_type):
        """根据任务类型调整标注工具的显示"""
        # 隐藏所有标注工具
        self.btn_draw_tool.hide()
        self.btn_keypoint.hide()
        
        # 根据任务类型显示对应的工具
        if task_type == 'detect':
            self.btn_draw_tool.show()
            self.select_draw_tool('rectangle')
        elif task_type == 'segment':
            self.btn_draw_tool.show()
            self.select_draw_tool('polygon')
        elif task_type == 'pose':
            self.btn_keypoint.show()
            # 默认选中关键点工具
            self.btn_keypoint.setChecked(True)
            self.set_tool('keypoint')
        elif task_type == 'classify':
            # 分类任务不需要标注工具
            pass
        
        # 移动工具始终显示
        self.btn_move.show()
        self.refresh_toolbar_button_layout()
    
    def load_image(self, image_id: int):
        """加载图片"""
        self.current_image_id = image_id
        self.update_delete_menu_state()
        
        # 找到图片数据
        image_data = next((img for img in self.images if img['id'] == image_id), None)
        if not image_data:
            return
        
        self.current_image_data = image_data
        
        # 加载到画布
        if image_data.get('storage_path'):
            self.canvas.load_image(image_data['storage_path'])
            # SAM模式下切图时，同步推理上下文到当前图片，避免继续使用旧图推理
            if self.canvas.sam_mode_active:
                self.canvas.sam_image_path = image_data['storage_path']
                self.canvas.sam_config = AutoLabelDialog.get_saved_sam_config()
                self.canvas.sam_points = []
                self.canvas.sam_bboxes = []
                self.canvas.sam_drawing_bbox = False
        
        # 加载标注
        self.load_annotations()
        
        # 更新状态栏
        self.update_status_bar()
        
        # 高亮当前项并滚动到该项
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_id:
                item.setSelected(True)
                # 滚动到当前项
                self.image_list.scrollToItem(item)
            else:
                item.setSelected(False)
    
    def load_annotations(self):
        """加载标注"""
        if not self.current_image_id:
            return
        
        self.annotations = db.get_image_annotations(self.current_image_id)
        self.canvas.set_annotations(self.annotations)
        self.update_status_bar()
        self._refresh_annotation_class_controls()
    
    def set_tool(self, tool: str):
        """设置工具"""
        self.canvas.set_tool(tool)

        if tool in ('rectangle', 'polygon'):
            self.default_draw_tool = tool
            if hasattr(self, 'btn_draw_tool'):
                self.btn_draw_tool.setChecked(True)
                self.refresh_draw_tool_button()
        elif tool == 'keypoint' and hasattr(self, 'btn_keypoint'):
            self.btn_keypoint.setChecked(True)
        elif tool == 'move' and hasattr(self, 'btn_move'):
            self.btn_move.setChecked(True)
        
        # 确保status_tool存在时才更新
        if hasattr(self, 'status_tool'):
            tool_names = {
                'rectangle': '矩形',
                'polygon': '多边形',
                'move': '移动',
                'keypoint': '关键点',
                'obb': '旋转矩形'
            }
            self.status_tool.setText(f"工具: {tool_names.get(tool, tool)}")

    def clear_current_image_view(self):
        """清空当前图片显示与相关状态。"""
        self.current_image_id = None
        self.current_image_data = None
        self.annotations = []
        self.history = []
        self.history_index = -1
        self.canvas.selected_annotation_id = None
        self.canvas.set_annotations([])
        self.canvas.load_image("")
        self.clear_attribute_panel(refresh_sample_panel=False)
        self.update_status_bar()
        self.update_delete_menu_state()

    def delete_current_image(self):
        """删除当前图片及其全部标注和实际文件。"""
        if not self.current_project_id or not self.current_image_id:
            QMessageBox.warning(self, "提示", "当前没有可删除的图片")
            return

        current_index = next(
            (i for i, image in enumerate(self.images) if image['id'] == self.current_image_id),
            -1
        )
        current_image = next(
            (image for image in self.images if image['id'] == self.current_image_id),
            None
        )
        if current_index < 0 or current_image is None:
            QMessageBox.warning(self, "提示", "当前图片不存在或已失效，请先重新加载")
            return

        filename = current_image.get('filename', str(self.current_image_id))
        reply = QMessageBox.question(
            self,
            "确认删除当前图片",
            (
                f"将删除当前图片“{filename}”及其全部标注，并删除实际文件。\n"
                "此操作不可恢复，是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        fallback_index = current_index if current_index < len(self.images) - 1 else current_index - 1
        deleted = db.delete_image(self.current_image_id)
        if not deleted:
            QMessageBox.warning(self, "提示", "删除当前图片失败")
            return

        self.load_image_list()
        self.update_class_list()

        if self.images:
            fallback_index = max(0, min(fallback_index, len(self.images) - 1))
            next_image_id = self.images[fallback_index]['id']
            self.load_image(next_image_id)
        else:
            self.clear_current_image_view()
    
    def on_annotation_created(self, annotation: dict):
        """标注创建事件"""
        if not self.current_image_id or not self.current_project_id:
            return
        
        # 从标注数据中获取类别ID（由画布传递）
        class_id = annotation.get('class_id', self.current_class_id)
        class_name = self.classes[class_id]['name'] if class_id < len(self.classes) else 'unknown'
        
        # 保存到数据库
        ann_id = db.add_annotation(
            image_id=self.current_image_id,
            project_id=self.current_project_id,
            class_id=class_id,
            class_name=class_name,
            annotation_type=annotation['type'],
            data=annotation['data']
        )
        
        # 添加到历史记录
        self.add_history('create', {'annotation_id': ann_id})
        
        # 更新图片状态为已标注
        db.update_image_status(self.current_image_id, 'annotated')
        
        # 重新加载标注
        self.load_annotations()
        
        # 更新图片列表显示
        self.update_image_list_display()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
    
    def on_annotation_selected(self, annotation_id: int):
        """标注选中事件"""
        # 更新属性面板
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            self.update_attribute_panel(annotation)
    
    def on_annotation_modified(self, annotation_id: int, old_data: dict, new_data: dict):
        """标注修改事件（拖动或调整大小后）"""
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            db.update_annotation(annotation_id, data=new_data)
            annotation['data'] = new_data

            self.update_attribute_panel(annotation)
            self.add_history('modify', {
                'annotation_id': annotation_id,
                'old_data': copy.deepcopy(old_data),
            })
    
    def on_annotation_deleted(self, annotation_id: int):
        """标注删除事件"""
        # 保存到历史记录
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            self.add_history('delete', annotation)
        
        # 从数据库删除
        db.delete_annotation(annotation_id)
        
        # 重新加载
        self.load_annotations()
        self.canvas.selected_annotation_id = None
        self.clear_attribute_panel(refresh_sample_panel=False)
        
        # 检查图片是否还有标注
        remaining_annotations = self.annotations
        if not remaining_annotations:
            # 如果没有标注了，更新状态为未标注
            db.update_image_status(self.current_image_id, 'pending')
            # 更新图片列表显示
            self.update_image_list_display()

        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
    
    def delete_selected_annotation(self):
        """删除选中的标注"""
        if self.canvas.selected_annotation_id is not None:
            self.on_annotation_deleted(self.canvas.selected_annotation_id)
    
    def update_attribute_panel(self, annotation: dict):
        """更新属性面板"""
        # 设置类别
        class_id = annotation.get('class_id', 0)
        self._set_class_list_selection(class_id)
    
    def clear_attribute_panel(self, refresh_sample_panel=True):
        """清空属性面板"""
        self.attr_class.blockSignals(True)
        self.attr_class.setCurrentIndex(-1)
        self.attr_class.blockSignals(False)
        self.sample_delete_count.blockSignals(True)
        self.sample_delete_count.setValue(0)
        self.sample_delete_count.blockSignals(False)
        if refresh_sample_panel:
            self.update_sample_control_panel()
        self._refresh_annotation_class_controls()
    
    def on_attr_class_changed(self, index):
        """属性类别改变事件"""
        if self.canvas.selected_annotation_id is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        class_id = self.attr_class.currentData()
        if class_id is not None:
            annotation['class_id'] = class_id
            self._set_class_list_selection(class_id)
            self.canvas.update()
    
    def apply_annotation_changes(self):
        """应用标注修改到数据库"""
        if self.canvas.selected_annotation_id is None:
            QMessageBox.warning(self, "提示", "请先选中一个标注再修改类别")
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        # 获取新的类别信息
        class_id = self._get_selected_class_list_id()
        if class_id is None:
            QMessageBox.warning(self, "提示", "当前没有可用类别")
            return

        class_info = next((cls for cls in self.classes if cls['id'] == class_id), None)
        class_name = class_info['name'] if class_info else 'unknown'

        if class_id == annotation.get('class_id'):
            self._refresh_annotation_class_controls()
            return

        # 更新标注数据
        annotation['class_id'] = class_id
        annotation['class_name'] = class_name

        # 更新到数据库
        updated = db.update_annotation(annotation['id'], class_id=class_id, class_name=class_name)
        if not updated:
            QMessageBox.warning(self, "提示", "标注类别修改失败")
            return
        
        # 刷新显示
        self.load_annotations()
        self.canvas.selected_annotation_id = annotation['id']
        updated_annotation = next((ann for ann in self.annotations if ann['id'] == annotation['id']), None)
        if updated_annotation:
            self.update_attribute_panel(updated_annotation)
        self.canvas.update()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
        
        QMessageBox.information(self, "成功", "标注修改已保存")

    def delete_random_samples_for_target(self):
        """按目标标签随机删除样本图像"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return

        if self.random_delete_worker and self.random_delete_worker.isRunning():
            QMessageBox.warning(self, "提示", "随机删样任务仍在执行，请等待完成后再试")
            return

        target_class_id = self.sample_target_class.currentData()
        target_name = self.sample_target_class.currentText()
        delete_count = self.sample_delete_count.value()

        if target_class_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个删样标签")
            return

        if delete_count <= 0:
            QMessageBox.warning(self, "提示", "删样数量必须大于 0")
            return

        class_sample_counts, negative_sample_count = self._get_sample_stats()
        if target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
            available_count = negative_sample_count
        else:
            available_count = class_sample_counts.get(target_class_id, 0)
        if available_count == 0:
            QMessageBox.warning(self, "提示", f"当前标签“{target_name}”没有可删除的样本")
            return

        delete_count = min(delete_count, available_count)
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"将随机删除 {delete_count} 张“{target_name}”样本图。\n"
            f"这是删除整张图片及其标注，不是只删框，且不可撤销。\n"
            f"是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from PyQt6.QtWidgets import QProgressDialog

        self.random_delete_progress = QProgressDialog(
            f"正在准备删除“{target_name}”样本图...",
            "取消",
            0,
            delete_count,
            self
        )
        self.random_delete_progress.setWindowTitle("随机删除样本")
        self.random_delete_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.random_delete_progress.setAutoClose(False)
        self.random_delete_progress.setAutoReset(False)
        self.random_delete_progress.setMinimumDuration(0)
        self.random_delete_progress.setValue(0)

        self.btn_delete_random_samples.setEnabled(False)
        self.random_delete_worker = RandomSampleDeleteWorker(
            self.current_project_id,
            target_class_id,
            delete_count,
            self.current_image_id
        )
        self.random_delete_worker.progress_updated.connect(self.on_random_delete_progress)
        self.random_delete_worker.delete_finished.connect(
            lambda result, current_target_name=target_name: self.on_random_delete_finished(current_target_name, result)
        )
        self.random_delete_worker.delete_failed.connect(self.on_random_delete_failed)
        self.random_delete_progress.canceled.connect(self.random_delete_worker.stop)
        self.random_delete_worker.start()
        self.random_delete_progress.show()

    def on_random_delete_progress(self, current: int, total: int, filename: str):
        """随机删样本进度回调"""
        if not self.random_delete_progress:
            return
        self.random_delete_progress.setMaximum(max(total, 1))
        self.random_delete_progress.setValue(current)
        self.random_delete_progress.setLabelText(
            f"正在删除样本图 ({current}/{total})：{filename}"
        )

    def _cleanup_random_delete_task(self):
        """清理随机删样本任务状态"""
        if self.random_delete_progress:
            self.random_delete_progress.close()
            self.random_delete_progress.deleteLater()
            self.random_delete_progress = None
        if self.random_delete_worker:
            self.random_delete_worker.deleteLater()
            self.random_delete_worker = None
        self.btn_delete_random_samples.setEnabled(True)

    def on_random_delete_finished(self, target_name: str, result: dict):
        """随机删样本完成回调"""
        self._cleanup_random_delete_task()
        self._invalidate_sample_stats_cache()

        deleted_count = result.get('deleted_count', 0)
        failed_files = result.get('failed_files', [])
        current_image_deleted = result.get('current_image_deleted', False)
        canceled = result.get('canceled', False)
        total_count = result.get('total_count', 0)

        self.load_image_list()

        remaining_images = self.images
        if remaining_images:
            remaining_ids = {image['id'] for image in remaining_images}
            if current_image_deleted or self.current_image_id not in remaining_ids:
                self.load_image(remaining_images[0]['id'])
            elif self.current_image_id:
                self.load_image(self.current_image_id)
        else:
            self.clear_current_image_view()

        if canceled:
            QMessageBox.information(
                self,
                "删除已取消",
                f"随机删样已取消。\n已删除 {deleted_count}/{total_count} 张“{target_name}”样本图"
            )
            return

        if failed_files:
            QMessageBox.warning(
                self,
                "部分删除失败",
                f"成功删除 {deleted_count} 张图片，失败 {len(failed_files)} 张。\n"
                f"失败文件：{', '.join(failed_files[:5])}"
            )
        else:
            QMessageBox.information(
                self,
                "删除完成",
                f"已随机删除 {deleted_count} 张“{target_name}”样本图"
            )

    def on_random_delete_failed(self, error_message: str):
        """随机删样本失败回调"""
        self._cleanup_random_delete_task()
        QMessageBox.critical(self, "删除失败", f"随机删除样本出错: {error_message}")

    def _get_next_image_shortcut_key(self) -> str:
        """获取当前“下一张”快捷键。"""
        from PyQt6.QtCore import QSettings

        settings = QSettings("EzYOLO", "Settings")
        return str(settings.value("next_image_shortcut", "D")).upper()

    def _exec_message_box_with_shortcut(
        self,
        *,
        icon,
        title: str,
        text: str,
        buttons,
        default_button=None,
        shortcut_button=None,
        shortcut_key: Optional[str] = None,
    ) -> Tuple[QMessageBox.StandardButton, bool]:
        """执行消息框，并允许指定快捷键触发某个按钮。"""
        message_box = QMessageBox(self)
        message_box.setIcon(icon)
        message_box.setWindowTitle(title)
        message_box.setText(text)
        message_box.setStandardButtons(buttons)
        if default_button is not None:
            message_box.setDefaultButton(default_button)

        shortcut_triggered = False
        if shortcut_button is not None and shortcut_key:
            target_button = message_box.button(shortcut_button)
            if target_button is not None:
                shortcut = QShortcut(QKeySequence(shortcut_key), message_box)
                message_box._shortcut = shortcut
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

                def on_shortcut_activated():
                    nonlocal shortcut_triggered
                    shortcut_triggered = True
                    target_button.click()

                shortcut.activated.connect(on_shortcut_activated)

        result = QMessageBox.StandardButton(message_box.exec())
        return result, shortcut_triggered

    def mark_current_image_as_negative_sample(self):
        """将当前图片标注为负样本"""
        if not self.current_project_id or not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return

        existing_annotations = db.get_image_annotations(self.current_image_id)
        annotation_count = len(existing_annotations)
        next_image_key = self._get_next_image_shortcut_key()

        if annotation_count > 0:
            reply, _ = self._exec_message_box_with_shortcut(
                icon=QMessageBox.Icon.Question,
                title="确认标注为负样本",
                text=(
                    f"当前图片已有 {annotation_count} 个标注。\n"
                    f"继续后会清空这些标注，并将当前图片标记为负样本。\n"
                    f"是否继续？\n\n"
                    f"按 {next_image_key} 可直接确认"
                ),
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                shortcut_button=QMessageBox.StandardButton.Yes,
                shortcut_key=next_image_key,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        db.delete_image_annotations(self.current_image_id)
        db.update_image_status(self.current_image_id, 'annotated')
        self.current_image_data = db.get_image(self.current_image_id)

        self.load_annotations()
        self.canvas.selected_annotation_id = None
        self.clear_attribute_panel(refresh_sample_panel=False)
        self.update_image_list_display()
        self.update_sample_control_panel()

        _, shortcut_triggered = self._exec_message_box_with_shortcut(
            icon=QMessageBox.Icon.Information,
            title="完成",
            text=f"当前图片已标注为负样本。\n按 {next_image_key} 可确认并进入下一张。",
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
            shortcut_button=QMessageBox.StandardButton.Ok,
            shortcut_key=next_image_key,
        )
        if shortcut_triggered:
            self.next_image()
    
    def show_class_context_menu(self, position):
        """显示类别右键菜单"""
        item = self.class_list.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        
        edit_action = menu.addAction("编辑")
        delete_action = menu.addAction("删除")
        
        action = menu.exec(self.class_list.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_class(item)
        elif action == delete_action:
            self.delete_class(item)
    
    def edit_class(self, item: QListWidgetItem):
        """编辑类别"""
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_info = next((c for c in self.classes if c['id'] == class_id), None)
        if not class_info:
            return
        
        # 编辑名称
        name, ok = QInputDialog.getText(
            self, "编辑类别", 
            "请输入类别名称:",
            text=class_info['name']
        )
        if not ok or not name:
            return
        
        # 编辑颜色
        color = QColorDialog.getColor(
            QColor(class_info.get('color', '#FF0000')), 
            self, "选择类别颜色"
        )
        if not color.isValid():
            color = QColor(class_info.get('color', '#FF0000'))
        
        # 更新类别信息
        class_info['name'] = name
        class_info['color'] = color.name()
        
        # 保存到数据库
        db.update_project(self.current_project_id, classes=self.classes)
        
        # 更新显示
        self.update_class_list()
    
    def delete_class(self, item: QListWidgetItem):
        """删除类别"""
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_info = next((c for c in self.classes if c['id'] == class_id), None)
        if not class_info:
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除类别 '{class_info['name']}' 吗？\n该类别下的所有标注将被删除！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 删除该类别下的所有标注
        for annotation in self.annotations:
            if annotation.get('class_id') == class_id:
                db.delete_annotation(annotation['id'])
        
        # 从列表中删除类别
        self.classes = [c for c in self.classes if c['id'] != class_id]
        
        # 重新编号
        for i, cls in enumerate(self.classes):
            cls['id'] = i
        
        # 保存到数据库
        db.update_project(self.current_project_id, classes=self.classes)
        
        # 更新显示
        self._invalidate_sample_stats_cache()
        self.update_class_list()
        self.load_annotations()
    
    def add_class(self):
        """添加类别"""
        name, ok = QInputDialog.getText(self, "添加类别", "请输入类别名称:")
        if ok and name:
            # 选择颜色
            color = QColorDialog.getColor(QColor(255, 0, 0), self, "选择类别颜色")
            if not color.isValid():
                color = QColor(255, 0, 0)
            
            class_id = len(self.classes)
            self.classes.append({
                'id': class_id,
                'name': name,
                'color': color.name()
            })
            
            # 更新项目类别
            db.update_project(self.current_project_id, classes=self.classes)
            
            self.update_class_list()
            # 选中新添加的类别
            self.class_list.setCurrentRow(self.class_list.count() - 1)
            self.on_class_selected()
    
    def prev_image(self):
        """上一张图片"""
        if not self.images or not self.current_image_id:
            return
        
        current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), 0)
        if current_index > 0:
            new_image_id = self.images[current_index - 1]['id']
            self.load_image(new_image_id)
    
    def next_image(self):
        """下一张图片"""
        if not self.images or not self.current_image_id:
            return
        
        current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), -1)
        if current_index < len(self.images) - 1:
            new_image_id = self.images[current_index + 1]['id']
            self.load_image(new_image_id)
    
    def add_history(self, action: str, data: dict):
        """添加历史记录"""
        # 删除当前位置之后的历史
        self.history = self.history[:self.history_index + 1]
        
        # 添加新记录
        self.history.append({
            'action': action,
            'data': data
        })
        self.history_index = len(self.history) - 1
        
        # 限制历史记录数量
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        """撤销"""
        if self.history_index < 0:
            return
        
        record = self.history[self.history_index]
        action = record['action']
        data = record['data']
        
        if action == 'create':
            # 撤销创建 = 删除
            ann_id = data.get('annotation_id')
            if ann_id:
                db.delete_annotation(ann_id)
        elif action == 'delete':
            # 撤销删除 = 创建
            db.add_annotation(
                image_id=data['image_id'],
                project_id=data['project_id'],
                class_id=data['class_id'],
                class_name=data['class_name'],
                annotation_type=data['type'],
                data=data['data']
            )
        elif action == 'modify':
            ann_id = data.get('annotation_id')
            old_data = data.get('old_data')
            if ann_id is not None and old_data is not None:
                db.update_annotation(ann_id, data=old_data)
        
        self.history_index -= 1
        self.load_annotations()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
    
    def update_status_bar(self):
        """更新状态栏，并顺带刷新顶部信息条和按钮可用性。"""
        total = len(self.images)
        annotated = sum(1 for img in self.images if img.get('status') == 'annotated')
        index = self._current_image_index()
        current = index + 1 if index >= 0 else 0

        self.status_image.setText(f"当前: {current}/{total}")
        self.status_progress.setText(f"标注: {annotated}/{total}")
        self.status_annotation.setText(f"本图标注: {len(self.annotations)}")
        tool_names = {
            'rectangle': '矩形',
            'polygon': '多边形',
            'move': '移动',
            'keypoint': '关键点',
            'obb': '旋转矩形'
        }
        self.status_tool.setText(f"工具: {tool_names.get(self.canvas.current_tool, self.canvas.current_tool)}")
        # 批量任务的进度和取消入口不能被这里顺手抹掉。
        # 以前这句是无条件 `self._set_elided_text(self.status_batch, "")` ——
        # 而 update_status_bar 是每标一个框、每切一张图都会调的，于是正在跑的
        # 批量任务刚写上一行进度就被清空，用户看着像是任务没了。
        self._refresh_batch_status()

        self._update_context_bar()
        self._update_action_availability()

    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        from PyQt6.QtCore import QSettings
        
        # 获取快捷键设置
        settings = QSettings("EzYOLO", "Settings")
        rect_tool_key = str(settings.value("rect_tool_shortcut", "W"))
        poly_tool_key = str(settings.value("poly_tool_shortcut", "P"))
        move_tool_key = str(settings.value("move_tool_shortcut", "V"))
        delete_key = str(settings.value("delete_shortcut", "DELETE"))
        
        # 处理工具快捷键（按键码比对，DELETE/SPACE/方向键这些没有字符的键才认得出来）
        key_text = event.text().upper()
        if event_matches_shortcut(event, rect_tool_key):
            self.select_draw_tool('rectangle')
            return
        elif event_matches_shortcut(event, poly_tool_key):
            self.select_draw_tool('polygon')
            return
        elif event_matches_shortcut(event, move_tool_key):
            self.btn_move.setChecked(True)
            self.set_tool('move')
            return
        elif event_matches_shortcut(event, delete_key):
            self.delete_selected_annotation()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            self.undo()
        else:
            # 处理数字键1-9切换标签
            if key_text in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                class_index = int(key_text) - 1
                if class_index < self.class_list.count():
                    self.class_list.setCurrentRow(class_index)
                    self.on_class_selected()
                    return
            super().keyPressEvent(event)
    
    def export_annotations(self):
        """导出标注文件"""
        import shutil
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        
        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return
        
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", 
            os.path.expanduser("~")
        )
        
        if not export_dir:
            return
        
        try:
            # 获取导出格式
            export_format = self.export_format.currentText()
            
            # 获取项目信息
            project = db.get_project(self.current_project_id)
            if not project:
                QMessageBox.warning(self, "导出失败", "无法获取项目信息")
                return
            
            project_name = project.get('name', 'untitled')
            
            # 创建导出目录结构
            export_path = os.path.join(export_dir, f"{project_name}_annotations")
            os.makedirs(export_path, exist_ok=True)
            
            # 获取项目图片
            images = db.get_project_images(self.current_project_id)
            if not images:
                QMessageBox.warning(self, "导出失败", "项目中没有图片")
                return
            
            # 获取类别映射
            class_mapping = {cls['id']: cls['name'] for cls in self.classes}
            
            if export_format == "YOLO格式":
                # 创建labels目录
                labels_dir = os.path.join(export_path, 'labels')
                os.makedirs(labels_dir, exist_ok=True)
                
                # 导出每个图片的标注
                exported_count = 0
                negative_count = 0
                for image in images:
                    image_id = image['id']
                    annotations = db.get_image_annotations(image_id)
                    image_status = image.get('status', 'pending')
                    
                    if annotations or image_status == 'annotated':
                        # 已标注图片都应生成标签文件；负样本生成空txt
                        filename = os.path.splitext(image['filename'])[0] + '.txt'
                        label_file = os.path.join(labels_dir, filename)
                        
                        with open(label_file, 'w', encoding='utf-8') as f:
                            for ann in annotations:
                                class_id = ann.get('class_id', 0)
                                ann_type = ann.get('type', 'bbox')
                                data = ann.get('data', {})
                                
                                if ann_type == 'bbox':
                                    # YOLO格式：class_id x_center y_center width height
                                    x = data.get('x', 0)
                                    y = data.get('y', 0)
                                    width = data.get('width', 0)
                                    height = data.get('height', 0)

                                    # 计算中心点和归一化
                                    img_width = image.get('width', 1920)  # 默认宽度
                                    img_height = image.get('height', 1080)  # 默认高度

                                    x_center = (x + width/2) / img_width
                                    y_center = (y + height/2) / img_height
                                    norm_width = width / img_width
                                    norm_height = height / img_height

                                    # 写入文件
                                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                                elif ann_type == 'mask':
                                    # YOLO分割格式：class_id x1 y1 x2 y2 ... xn yn
                                    mask = data.get('mask', [])
                                    if mask:
                                        # 计算归一化
                                        img_width = image.get('width', 1920)  # 默认宽度
                                        img_height = image.get('height', 1080)  # 默认高度

                                        # 构建归一化的多边形坐标
                                        normalized_points = []
                                        for point in mask:
                                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                                x, y = point[0], point[1]
                                                norm_x = x / img_width
                                                norm_y = y / img_height
                                                normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                                        if normalized_points:
                                            # 写入文件
                                            f.write(f"{class_id} {' '.join(normalized_points)}\n")
                        
                        exported_count += 1
                        if not annotations:
                            negative_count += 1
                
                # 创建classes.txt文件
                classes_file = os.path.join(export_path, 'classes.txt')
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for cls in sorted(self.classes, key=lambda x: x['id']):
                        f.write(f"{cls['name']}\n")
                
                QMessageBox.information(
                    self,
                    "导出成功",
                    f"已导出 {exported_count} 个标注文件到\n{export_path}\n\n"
                    f"- 负样本空标签: {negative_count} 个"
                )
                
            elif export_format == "COCO格式":
                # 导出COCO格式
                import json
                
                # 创建COCO格式的标注数据
                coco_data = {
                    "info": {
                        "description": f"Annotations for {project_name}",
                        "version": "1.0",
                        "year": 2024
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }
                
                # 添加类别
                for cls in self.classes:
                    coco_data["categories"].append({
                        "id": cls['id'],
                        "name": cls['name'],
                        "supercategory": "object"
                    })
                
                # 添加图片和标注
                annotation_id = 1
                for image in images:
                    # 添加图片信息
                    image_info = {
                        "id": image['id'],
                        "file_name": image['filename'],
                        "width": image.get('width', 1920),
                        "height": image.get('height', 1080),
                        "date_captured": "",
                        "license": 0,
                        "coco_url": "",
                        "flickr_url": ""
                    }
                    coco_data["images"].append(image_info)
                    
                    # 添加标注
                    annotations = db.get_image_annotations(image['id'])
                    for ann in annotations:
                        ann_type = ann.get('type', 'bbox')
                        data = ann.get('data', {})
                        
                        if ann_type == 'bbox':
                            # COCO格式：x y width height
                            x = int(data.get('x', 0))
                            y = int(data.get('y', 0))
                            width = int(data.get('width', 0))
                            height = int(data.get('height', 0))
                            
                            coco_annotation = {
                                "id": annotation_id,
                                "image_id": image['id'],
                                "category_id": ann.get('class_id', 0),
                                "segmentation": [],
                                "area": width * height,
                                "bbox": [x, y, width, height],
                                "iscrowd": 0
                            }
                            coco_data["annotations"].append(coco_annotation)
                            annotation_id += 1
                
                # 保存COCO格式文件
                coco_file = os.path.join(export_path, 'annotations.json')
                with open(coco_file, 'w', encoding='utf-8') as f:
                    json.dump(coco_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "导出成功", f"已导出 COCO 格式标注到\n{coco_file}")
                
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中出错:\n{str(e)}")
    
    def run_llm_single_inference(self):
        """运行LLM单张推理"""
        # 检查项目类型
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        # 获取项目类型
        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        
        # 检查是否为detect任务
        if project_type != 'detect':
            QMessageBox.information(self, "提示", "功能还在完善，敬请期待")
            return
        
        # 检查是否有选中的图片
        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        
        # 获取当前图片路径
        current_image = None
        for img in self.images:
            if img['id'] == self.current_image_id:
                current_image = img
                break
        
        if not current_image:
            QMessageBox.warning(self, "提示", "无法获取当前图片信息")
            return
        
        image_path = current_image.get('storage_path', '')
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "图片文件不存在")
            return
        
        # 获取当前选中的类别
        if not self.classes or self.current_class_id >= len(self.classes):
            QMessageBox.warning(self, "提示", "请先选择一个类别")
            return
        
        target_class = self.classes[self.current_class_id]['name']

        # 缺 API Key 时这里会给出「去配置」入口，配完直接往下走
        llm_config = self._require_llm_config()
        if not llm_config:
            return

        # 显示进度对话框
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("正在使用LLM进行目标检测...", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()

        # 单张也走后台线程：请求要等几秒，界面不能在这几秒里僵着。
        # 上一张还在跑就不再起新的：直接改写 self.llm_worker 会把最后一个引用丢掉，
        # 那个还在 run() 里的 QThread 当场被回收——直接崩。
        if self.llm_worker is not None and self.llm_worker.isRunning():
            progress.close()
            QMessageBox.information(self, "提示", "上一张还在跑，等它出结果再试。")
            return

        self.llm_worker = LLMBatchWorker(llm_config, [current_image], target_class)
        self._track_llm_worker(self.llm_worker)
        self.llm_worker.image_done.connect(
            lambda _image_id, detections, error:
                self.on_llm_inference_finished(
                    not error,
                    f"推理出错: {error}" if error else f"检测到 {len(detections)} 个目标",
                    detections, progress,
                )
        )
        self.llm_worker.start()


    def on_llm_inference_finished(self, success, message, detections, progress_dialog):
        """LLM推理完成回调"""
        progress_dialog.close()
        
        if not success:
            QMessageBox.critical(self, "错误", message)
            return
        
        if not detections:
            QMessageBox.information(self, "完成", "未检测到目标")
            return
        
        # 添加检测结果为标注
        class_id = self.current_class_id
        added_count = 0
        
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = bbox
            
            # 创建标注数据
            annotation = {
                'type': 'bbox',
                'class_id': class_id,
                'data': {
                    'x': float(xmin),
                    'y': float(ymin),
                    'width': float(xmax - xmin),
                    'height': float(ymax - ymin)
                }
            }
            
            # 保存到数据库
            class_name = self.classes[class_id]['name'] if class_id < len(self.classes) else 'unknown'
            ann_id = db.add_annotation(
                image_id=self.current_image_id,
                project_id=self.current_project_id,
                class_id=class_id,
                class_name=class_name,
                annotation_type='bbox',
                data=annotation['data']
            )
            
            if ann_id:
                added_count += 1
        
        # 更新图片状态
        if added_count > 0:
            db.update_image_status(self.current_image_id, 'annotated')
        
        # 重新加载标注
        self.load_annotations()
        self.update_image_list_display()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
        
        QMessageBox.information(self, "完成", f"{message}\n已添加 {added_count} 个标注")
    
    def run_llm_batch_inference(self):
        """运行LLM批量推理"""
        # 检查项目类型
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        # 获取项目类型
        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        
        # 检查是否为detect任务
        if project_type != 'detect':
            QMessageBox.information(self, "提示", "功能还在完善，敬请期待")
            return
        
        # 检查是否有图片
        if not self.images:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return
        
        # 获取当前选中的类别
        if not self.classes or self.current_class_id >= len(self.classes):
            QMessageBox.warning(self, "提示", "请先选择一个类别")
            return
        
        # 已经在跑就别再起一批：两批同时写标注，谁都说不清结果
        if self._batch_in_progress():
            QMessageBox.information(self, "提示", "上一批还在跑，等它结束或者先取消。")
            return

        target_class = self.classes[self.current_class_id]['name']

        # 缺 API Key 时给「去配置」入口，配完继续
        llm_config = self._require_llm_config()
        if not llm_config:
            return

        # 范围选择以前是一个单独的弹窗，选完范围直接就开跑了——用户从来没见过
        # 「要用哪个模型、会不会覆盖」。现在它并进共享的确认框里：一个框看全，
        # 一次决定。conf / IoU 对 LLM 不适用，确认框会照实写「不适用」。
        plan = BatchPlan(
            project_id=self.current_project_id,
            engine='llm',
            images=list(self.images),
            scope=SCOPE_RANGE,
            model_label=llm_config.get('model_name', '') or "未指定",
            conf=None,
            iou=None,
            overwrite=False,   # LLM 这条路只追加标注，从不删已有的
            class_id=self.current_class_id,
            class_name=target_class,
        )

        confirmed = confirm_batch_plan(self, plan, allow_range=True)
        if confirmed is None:
            # 取消：没建线程、没写库、没动任何图片状态
            return

        self._start_llm_batch(confirmed, llm_config)

    def _start_llm_batch(self, plan: BatchPlan, llm_config: dict) -> bool:
        """按快照跑一批 LLM。进度和取消都在状态栏，不再拿模态进度框挡住页面。"""
        if self._batch_in_progress():
            return False

        self._active_batch_plan = plan
        self.llm_batch_total = plan.count
        self.llm_batch_added = 0

        self._set_batch_status(f"准备中：{plan.count} 张图片", True)

        self.llm_batch_worker = LLMBatchWorker(
            llm_config, list(plan.images), plan.class_name,
        )
        self.llm_batch_worker.image_started.connect(self.on_llm_batch_image_started)
        self.llm_batch_worker.image_done.connect(self.on_llm_batch_image_done)
        self.llm_batch_worker.batch_finished.connect(self.on_llm_batch_finished)
        self._track_llm_worker(self.llm_batch_worker)

        self.llm_batch_worker.start()
        return True

    def on_llm_batch_image_started(self, position: int, filename: str):
        """第几张、哪一张，写在状态栏上。"""
        self._set_batch_status(
            f"大模型标注: {position}/{self.llm_batch_total} · {filename}", True,
        )

    def on_llm_batch_image_done(self, image_id: int, detections: list, error: str):
        """一张图片跑完：标注写库在界面线程做。

        写库用的是快照里的 project_id / 类别，不是 self.current_project_id ——
        这一批可能跑好几分钟，用户完全来得及切到别的项目去。照页面上的当前项目写，
        标注就落到别人家里去了。
        """
        if error:
            # 失败的图片只记账，不打断整批；错误里的密钥已经在线程里抹掉了
            print(f"[LLM批量] 图片 {image_id} 处理失败: {error}")
            return

        plan = self._active_batch_plan
        if plan is None:
            return

        class_id = plan.class_id
        class_name = plan.class_name
        added = 0

        for det in detections:
            xmin, ymin, xmax, ymax = det.get("bbox", [0, 0, 0, 0])
            ann_id = db.add_annotation(
                image_id=image_id,
                project_id=plan.project_id,
                class_id=class_id,
                class_name=class_name,
                annotation_type='bbox',
                data={
                    'x': float(xmin),
                    'y': float(ymin),
                    'width': float(xmax - xmin),
                    'height': float(ymax - ymin),
                },
            )
            if ann_id:
                added += 1

        if added:
            db.update_image_status(image_id, 'annotated')
            self.llm_batch_added += added

    def on_llm_batch_finished(self, succeeded: int, failed: int, cancelled: bool):
        """整批结束：收尾、刷新界面、把成绩单一次说清楚。"""
        # 这里不销毁线程：batch_finished 是 run() 的最后一句，run() 还没返回，
        # 一销毁就是「QThread: Destroyed while thread is still running」。
        # 先挪到「等它退出」的列表里继续持有，_on_llm_worker_finished 再放手。
        worker = self.llm_batch_worker
        self.llm_batch_worker = None
        if worker is not None and worker not in self._retired_llm_workers:
            self._retired_llm_workers.append(worker)

        self._finish_batch_ui()

        self.load_annotations()
        self.update_image_list_display()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()

        headline = "批量标注已取消。" if cancelled else "批量标注完成！"
        QMessageBox.information(
            self, "完成",
            f"{headline}\n"
            f"成功 {succeeded} 张，失败 {failed} 张，共 {self.llm_batch_total} 张\n"
            f"新增 {self.llm_batch_added} 个标注"
        )


    def export_dataset(self):
        """导出完整数据集"""
        import shutil
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        
        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return
        
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", 
            os.path.expanduser("~")
        )
        
        if not export_dir:
            return
        
        try:
            # 获取导出格式
            export_format = self.export_format.currentText()
            
            # 获取项目信息
            project = db.get_project(self.current_project_id)
            if not project:
                QMessageBox.warning(self, "导出失败", "无法获取项目信息")
                return
            
            project_name = project.get('name', 'untitled')
            
            # 创建导出目录结构
            dataset_dir = os.path.join(export_dir, project_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 创建images和labels目录
            images_dir = os.path.join(dataset_dir, 'images')
            labels_dir = os.path.join(dataset_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # 获取项目图片
            images = db.get_project_images(self.current_project_id)
            if not images:
                QMessageBox.warning(self, "导出失败", "项目中没有图片")
                return
            
            # 获取类别映射
            class_mapping = {cls['id']: cls['name'] for cls in self.classes}
            
            # 复制图片并导出标注
            copied_count = 0
            exported_count = 0
            negative_count = 0
            
            for image in images:
                # 复制图片
                src_image = image.get('storage_path', '')
                if src_image and os.path.exists(src_image):
                    dst_image = os.path.join(images_dir, image['filename'])
                    shutil.copy2(src_image, dst_image)
                    copied_count += 1
                
                # 导出标注
                image_id = image['id']
                annotations = db.get_image_annotations(image_id)
                image_status = image.get('status', 'pending')
                
                if annotations or image_status == 'annotated':
                    # 已标注图片都应生成标签文件；负样本生成空txt
                    filename = os.path.splitext(image['filename'])[0] + '.txt'
                    label_file = os.path.join(labels_dir, filename)
                    
                    with open(label_file, 'w', encoding='utf-8') as f:
                        for ann in annotations:
                            class_id = ann.get('class_id', 0)
                            ann_type = ann.get('type', 'bbox')
                            data = ann.get('data', {})
                            
                            # 获取图片尺寸
                            img_width = image.get('width', 1920)  # 默认宽度
                            img_height = image.get('height', 1080)  # 默认高度
                            
                            if ann_type == 'bbox':
                                # YOLO格式：class_id x_center y_center width height
                                x = data.get('x', 0)
                                y = data.get('y', 0)
                                width = data.get('width', 0)
                                height = data.get('height', 0)
                                
                                # 计算中心点和归一化
                                x_center = (x + width/2) / img_width
                                y_center = (y + height/2) / img_height
                                norm_width = width / img_width
                                norm_height = height / img_height
                                
                                # 写入文件
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                            
                            elif ann_type == 'polygon':
                                # YOLO分割格式：class_id x1 y1 x2 y2 ... xn yn
                                points = data.get('points', [])
                                if points:
                                    # 构建归一化的多边形坐标
                                    normalized_points = []
                                    for point in points:
                                        if isinstance(point, dict):
                                            x = point.get('x', 0)
                                            y = point.get('y', 0)
                                        elif isinstance(point, (list, tuple)) and len(point) >= 2:
                                            x, y = point[0], point[1]
                                        else:
                                            continue
                                        norm_x = x / img_width
                                        norm_y = y / img_height
                                        normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                                    
                                    if normalized_points:
                                        # 写入文件
                                        f.write(f"{class_id} {' '.join(normalized_points)}\n")
                    
                    exported_count += 1
                    if not annotations:
                        negative_count += 1
            
            # 创建classes.txt文件
            classes_file = os.path.join(dataset_dir, 'classes.txt')
            with open(classes_file, 'w', encoding='utf-8') as f:
                for cls in sorted(self.classes, key=lambda x: x['id']):
                    f.write(f"{cls['name']}\n")
            
            # 创建data.yaml文件（YOLO格式）
            yaml_file = os.path.join(dataset_dir, 'data.yaml')
            yaml_content = f"""
train: images
test: images
val: images

nc: {len(self.classes)}
names: {[cls['name'] for cls in sorted(self.classes, key=lambda x: x['id'])]}
"""
            
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            QMessageBox.information(
                self,
                "导出成功",
                f"已导出完整数据集到\n{dataset_dir}\n\n"
                f"- 复制图片: {copied_count} 张\n"
                f"- 导出标注: {exported_count} 个\n"
                f"- 负样本空标签: {negative_count} 个"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中出错:\n{str(e)}")
