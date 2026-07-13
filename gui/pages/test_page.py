# -*- coding: utf-8 -*-
"""
测试页面 - 模型推理测试（流程第 5 步）

页面按新手能读懂的顺序排：
    ① 选择训练好的模型 → ② 选择图片/视频 → ③ 设置（高级项折叠）
    → 开始测试 → 右侧看结果、知道结果存到哪
支持图片、图片文件夹、视频的推理测试。
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QFormLayout,
    QProgressBar, QTextEdit, QSplitter, QFileDialog, QMessageBox,
    QScrollArea, QFrame, QSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QStackedWidget, QTabWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QUrl, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QDesktopServices, QFontMetrics
from pathlib import Path
import os
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from gui.styles import COLORS, mono_font_family_css
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.context_help import ContextHelp
from gui.widgets.workflow_widgets import EmptyState
from gui.workflow import find_project_weights
from models.database import db

UNGROUPED_GROUP_ID = 0

# 推理输出的命名规则：图片 → test_images/<stem>_annotated.jpg，视频 → test_videos/<stem>_annotated.mp4。
# 写入和清理都按这一份规则算目标文件，两边不会漂移。
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')


def is_video_path(path: str) -> bool:
    """按扩展名判断是不是视频（与推理线程的分支同源）。"""
    return str(path).lower().endswith(VIDEO_EXTENSIONS)


def annotated_output_path(source_path: str, image_dir: Path, video_dir: Path) -> Path:
    """这次推理会为某个输入写到哪个文件。"""
    stem = Path(source_path).stem
    if is_video_path(source_path):
        return video_dir / f"{stem}_annotated.mp4"
    return image_dir / f"{stem}_annotated.jpg"

# 从train_page导入模型配置
from gui.pages.train_page import ULTRALYTICS_MODELS, TASK_NAMES, SIZE_NAMES


class InferenceThread(QThread):
    """推理后台线程"""
    
    progress_updated = pyqtSignal(int, int)  # 当前进度, 总数
    inference_finished = pyqtSignal(bool, str)  # 是否成功, 消息
    log_message = pyqtSignal(str)
    result_ready = pyqtSignal(dict)  # 单张图片推理结果
    frame_ready = pyqtSignal(np.ndarray, dict)  # 视频帧和检测结果
    
    def __init__(self, model_path: str, config: dict, data_paths: List[str], 
                 project_classes: List[dict] = None, class_mapping: dict = None):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.data_paths = data_paths
        self.project_classes = project_classes or []
        self.class_mapping = class_mapping or {}  # 类别映射
        self._is_running = False
        self.model = None
        self.output_root = Path(__file__).parent.parent.parent / "outputs"
        self.image_output_dir = self.output_root / "test_images"
        self.video_output_dir = self.output_root / "test_videos"
        
    def run(self):
        """运行推理"""
        self._is_running = True
        
        try:
            from ultralytics import YOLO
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            self.video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载模型
            self.log_message.emit(f"加载模型: {self.model_path}")
            
            # 检测模型类型
            model_type = "PyTorch"
            if self.model_path.endswith('.onnx'):
                model_type = "ONNX"
            elif self.model_path.endswith('.engine') or self.model_path.endswith('.trt'):
                model_type = "TensorRT"
            self.log_message.emit(f"模型类型: {model_type}")
            
            try:
                # 获取任务类型，用于ONNX/TensorRT模型
                task = self.config.get('task', 'detect')
                
                # 对于ONNX和TensorRT模型，需要传入task参数
                if model_type in ["ONNX", "TensorRT"]:
                    self.log_message.emit(f"使用任务类型: {task}")
                    self.model = YOLO(self.model_path, task=task)
                else:
                    self.model = YOLO(self.model_path)
                self.log_message.emit("✓ 模型加载成功")
            except Exception as e:
                self.log_message.emit(f"✗ 模型加载失败: {e}")
                # 针对不同模型类型给出更详细的错误提示
                if model_type == "ONNX":
                    self.log_message.emit("提示: ONNX模型需要安装onnxruntime (pip install onnxruntime)")
                elif model_type == "TensorRT":
                    self.log_message.emit("提示: TensorRT模型需要安装tensorrt并配置CUDA环境")
                self.inference_finished.emit(False, f"模型加载失败: {e}")
                return
            
            # 获取模型类别信息
            model_classes = self.model.names if hasattr(self.model, 'names') else {}
            self.log_message.emit(f"模型类别数: {len(model_classes)}")
            
            # 输出类别映射信息
            if self.class_mapping:
                self.log_message.emit(f"类别映射: {self.class_mapping}")
            
            # 推理参数
            conf = self.config.get('conf', 0.25)
            iou = self.config.get('iou', 0.45)
            imgsz = self.config.get('imgsz', 640)
            device = self.config.get('device', 'cpu')
            
            # 标准化设备值
            device = str(device).lower().strip()
            
            if device in ['自动选择', 'auto', '']:
                import torch
                # 更健壮的CUDA检测
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        # 测试CUDA是否真正可用
                        torch.cuda.current_device()
                        device = '0'
                        self.log_message.emit("✓ 检测到可用GPU，使用CUDA:0")
                    else:
                        device = 'cpu'
                        self.log_message.emit("✗ 未检测到可用GPU，使用CPU")
                except Exception as cuda_e:
                    device = 'cpu'
                    self.log_message.emit(f"⚠ CUDA检测失败，使用CPU: {cuda_e}")
            elif device in ['cpu', 'CPU']:
                device = 'cpu'
                self.log_message.emit("✓ 使用CPU进行推理")
            elif device.startswith('cuda:') or device.startswith('CUDA:'):
                device = device.split(':')[1]
            elif device.isdigit():
                # 纯数字，认为是GPU ID
                device = device
                self.log_message.emit(f"✓ 使用CUDA:{device}")
            else:
                # 默认使用CPU
                device = 'cpu'
                self.log_message.emit(f"⚠ 未知设备设置 '{device}'，默认使用CPU")
            
            self.log_message.emit(f"推理参数: conf={conf}, iou={iou}, imgsz={imgsz}, device={device}")
            
            total = len(self.data_paths)
            success_count = 0
            
            for i, data_path in enumerate(self.data_paths):
                if not self._is_running:
                    break
                
                self.progress_updated.emit(i + 1, total)
                self.log_message.emit(f"[{i+1}/{total}] 处理: {os.path.basename(data_path)}")
                
                try:
                    # 检查是否为视频文件
                    if is_video_path(data_path):
                        # 视频推理
                        self.process_video(data_path, conf, iou, imgsz, device)
                    else:
                        # 图片推理
                        results = self.model(
                            data_path,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            device=device,
                            verbose=False
                        )
                        
                        # 处理结果
                        result_data = self.process_result(results[0], data_path)
                        self.result_ready.emit(result_data)
                    
                    success_count += 1
                    
                except Exception as e:
                    self.log_message.emit(f"✗ 处理失败 {data_path}: {e}")
            
            if self._is_running:
                self.inference_finished.emit(True, f"推理完成！成功: {success_count}/{total}")
            else:
                self.inference_finished.emit(False, "推理已停止")
                
        except ImportError as e:
            self.log_message.emit(f"✗ 未检测到Ultralytics库: {e}")
            self.inference_finished.emit(False, "未安装Ultralytics库")
        except Exception as e:
            self.log_message.emit(f"✗ 推理出错: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            self.inference_finished.emit(False, f"推理出错: {e}")
    
    def process_video(self, video_path: str, conf: float, iou: float, imgsz: int, device: str):
        """处理视频文件"""
        self.log_message.emit(f"开始处理视频: {os.path.basename(video_path)}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_message.emit(f"✗ 无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 25.0
        
        self.log_message.emit(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        # 创建输出视频（边推理边写盘，避免帧缓存导致内存暴涨）
        app_root = Path(__file__).parent.parent.parent
        output_video_path = annotated_output_path(
            video_path, self.image_output_dir, self.video_output_dir
        )
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            self.log_message.emit(f"✗ 无法创建输出视频: {output_video_path}")
            return
        
        frame_count = 0
        processed_count = 0
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每几帧处理一次（可以根据需要调整）
            if frame_count % 1 == 0:  # 处理每一帧
                try:
                    # 对单帧进行推理
                    results = self.model(
                        frame,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                        verbose=False
                    )
                    
                    # 获取标注后的帧
                    annotated_frame = results[0].plot()
                    # 统一输出帧尺寸，避免播放器首段出现“逐步放大/缩小”的观感
                    if annotated_frame is not None:
                        ah, aw = annotated_frame.shape[:2]
                        if aw != width or ah != height:
                            annotated_frame = cv2.resize(
                                annotated_frame,
                                (width, height),
                                interpolation=cv2.INTER_LINEAR
                            )
                    if annotated_frame is not None:
                        writer.write(annotated_frame)
                    
                    # 处理检测结果
                    detections = []
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        for box in results[0].boxes:
                            class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                            # 应用类别映射
                            mapped_class_id = self.class_mapping.get(class_id, class_id)
                            
                            detection = {
                                'class_id': mapped_class_id,
                                'class_name': results[0].names.get(class_id, 'unknown'),
                                'confidence': float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf),
                                'bbox': box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                            }
                            detections.append(detection)
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.log_message.emit(f"✗ 处理帧 {frame_count} 失败: {e}")
            
            frame_count += 1
            
            # 每30帧更新一次进度
            if frame_count % 30 == 0:
                self.log_message.emit(f"  已处理 {frame_count}/{total_frames} 帧")
        
        cap.release()
        writer.release()
        
        # 发送视频处理完成信号
        result_data = {
            'source_path': video_path,
            'filename': os.path.basename(video_path),
            'is_video': True,
            'output_video': str(output_video_path),
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'fps': fps,
            'detections': []  # 视频不存储所有检测结果
        }
        self.result_ready.emit(result_data)
        
        self.log_message.emit(f"✓ 视频处理完成: {processed_count} 帧")
    
    def process_result(self, result, source_path: str) -> dict:
        """处理单张推理结果"""
        result_data = {
            'source_path': source_path,
            'filename': os.path.basename(source_path),
            'detections': [],
            'masks': [],  # 添加masks字段用于分割模型
            'speed': {},
            'annotated_image_path': None
        }
        
        # 获取检测框
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                # 应用类别映射
                mapped_class_id = self.class_mapping.get(class_id, class_id)
                
                detection = {
                    'class_id': mapped_class_id,
                    'class_name': result.names.get(class_id, 'unknown'),
                    'confidence': float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf),
                    'bbox': box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                }
                result_data['detections'].append(detection)
        
        # 获取分割masks（分割模型）
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            if hasattr(masks, 'xy') and masks.xy is not None:
                for i, mask_xy in enumerate(masks.xy):
                    # mask_xy 是多边形点列表 [(x1,y1), (x2,y2), ...]
                    if len(mask_xy) > 0:
                        # 获取对应的类别信息
                        class_id = 0
                        class_name = 'unknown'
                        confidence = 0.0
                        if hasattr(result, 'boxes') and result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                            class_name = result.names.get(class_id, 'unknown')
                            confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        
                        mask_data = {
                            'class_id': self.class_mapping.get(class_id, class_id),
                            'class_name': class_name,
                            'confidence': confidence,
                            'points': mask_xy.tolist() if hasattr(mask_xy, 'tolist') else list(mask_xy)
                        }
                        result_data['masks'].append(mask_data)
        
        # 获取推理速度
        if hasattr(result, 'speed'):
            result_data['speed'] = result.speed
        
        # 获取标注后的图像（Ultralytics会自动处理分割可视化）
        if hasattr(result, 'plot'):
            annotated_img = result.plot()
            if annotated_img is not None:
                output_path = annotated_output_path(
                    source_path, self.image_output_dir, self.video_output_dir
                )
                cv2.imwrite(str(output_path), annotated_img)
                result_data['annotated_image_path'] = str(output_path)
        
        return result_data
    
    def stop(self):
        """停止推理"""
        self._is_running = False


class ImageViewer(QWidget):
    """图像查看器组件"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_result = None
        self.scale_factor = 1.0
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        self.image_label.setMinimumSize(240, 180)
        self.layout.addWidget(self.image_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # 信息标签
        self.info_label = QLabel("未加载图像")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.layout.addWidget(self.info_label)
        self._apply_aspect_ratio()
    
    def show_image(self, image_path: str):
        """显示图像"""
        if not os.path.exists(image_path):
            self.info_label.setText(f"图像不存在: {image_path}")
            return
        
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.info_label.setText(f"无法加载图像: {image_path}")
            return
        
        self.current_image = pixmap
        self._update_display()
        self.info_label.setText(f"{os.path.basename(image_path)} ({pixmap.width()}x{pixmap.height()})")
    
    def show_array(self, image_array: np.ndarray):
        """显示numpy数组图像"""
        if image_array is None:
            return
        
        # 转换BGR到RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array
        
        # 转换为QPixmap
        height, width = rgb_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.current_image = pixmap
        self._update_display()
        self.info_label.setText(f"推理结果 ({width}x{height})")
    
    def _update_display(self):
        """更新显示"""
        if self.current_image is None:
            return
        
        # 缩放以适应窗口
        scaled_pixmap = self.current_image.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        self._apply_aspect_ratio()
        self._update_display()

    def _apply_aspect_ratio(self):
        """将图像显示区域固定为4:3比例。"""
        available_w = max(1, self.width())
        available_h = max(1, self.height() - self.info_label.sizeHint().height() - 8)
        target_w = min(available_w, int(available_h * 4 / 3))
        target_h = int(target_w * 3 / 4)
        if target_h > available_h:
            target_h = available_h
            target_w = int(target_h * 4 / 3)
        self.image_label.setFixedSize(max(1, target_w), max(1, target_h))


class VideoPlayer(QWidget):
    """视频播放器组件"""
    
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_frames = []  # 兼容旧逻辑，默认不再存完整视频帧
        self.current_frame_index = 0
        self.video_path = None
        self.video_cap = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        self.video_label.setMinimumSize(240, 180)
        self.layout.addWidget(self.video_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # 控制按钮
        self.controls_widget = QWidget()
        controls_layout = QHBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_play_pause = QPushButton("▶ 播放")
        self.btn_play_pause.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.btn_play_pause)
        
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop)
        controls_layout.addWidget(self.btn_stop)
        
        controls_layout.addStretch()
        
        # 帧计数
        self.frame_label = QLabel("0 / 0")
        controls_layout.addWidget(self.frame_label)
        
        self.layout.addWidget(self.controls_widget)
        self._apply_aspect_ratio()
    
    def add_frame(self, frame: np.ndarray):
        """添加视频帧"""
        # 保留兼容能力（调试场景），正式流程不再依赖帧缓存
        self.video_frames.append(frame.copy())
        self.frame_label.setText(f"{self.current_frame_index + 1} / {len(self.video_frames)}")
        if len(self.video_frames) == 1 and not self.is_playing:
            self.display_frame(self.video_frames[0])

    def load_video(self, video_path: str):
        """加载视频文件进行播放（无帧缓存模式）。"""
        self.clear()
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            self.video_cap = None
            self.frame_label.setText("0 / 0")
            return

        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        self._seek_and_show_first_frame()
    
    def toggle_play(self):
        """切换播放/暂停"""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """开始播放"""
        if self.video_cap is not None:
            self.is_playing = True
            self.btn_play_pause.setText("⏸ 暂停")
            interval = max(1, int(1000 / self.video_fps))
            self.timer.start(interval)
            return

        if not self.video_frames:
            return
        self.is_playing = True
        self.btn_play_pause.setText("⏸ 暂停")
        self.timer.start(33)  # 约30fps
    
    def pause(self):
        """暂停播放"""
        self.is_playing = False
        self.btn_play_pause.setText("▶ 播放")
        self.timer.stop()
    
    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.btn_play_pause.setText("▶ 播放")
        self.timer.stop()
        if self.video_cap is not None:
            self.current_frame_index = 0
            self._seek_and_show_first_frame()
            return
        self.current_frame_index = 0
        if self.video_frames:
            self.display_frame(self.video_frames[0])
    
    def update_frame(self):
        """更新帧"""
        if self.video_cap is not None:
            ret, frame = self.video_cap.read()
            if not ret:
                self.stop()
                return
            self.display_frame(frame)
            self.current_frame_index += 1
            self.frame_label.setText(f"{self.current_frame_index} / {self.total_frames}")
            return

        if not self.video_frames:
            return
        
        if self.current_frame_index < len(self.video_frames):
            self.display_frame(self.video_frames[self.current_frame_index])
            self.current_frame_index += 1
            self.frame_label.setText(f"{self.current_frame_index} / {len(self.video_frames)}")
        else:
            # 播放结束
            self.stop()
    
    def display_frame(self, frame: np.ndarray):
        """显示单帧"""
        if frame is None:
            return
        self.current_frame = frame
        
        # 转换BGR到RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # 转换为QPixmap
        height, width = rgb_frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 缩放以适应窗口
        target_size = self.video_label.contentsRect().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def clear(self):
        """清空视频帧"""
        self.timer.stop()
        self.is_playing = False
        self.btn_play_pause.setText("▶ 播放")
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.video_path = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.video_frames.clear()
        self.current_frame_index = 0
        self.current_frame = None
        self.video_label.clear()
        self.frame_label.setText("0 / 0")

    def _seek_and_show_first_frame(self):
        """跳转到第一帧并显示。"""
        if self.video_cap is None:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video_cap.read()
        if ret:
            self.display_frame(frame)
            self.current_frame_index = 1
            self.frame_label.setText(f"{self.current_frame_index} / {self.total_frames}")
            # 回到起点，保证播放从第一帧开始
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_index = 0
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放"""
        super().resizeEvent(event)
        self._apply_aspect_ratio()
        if self.current_frame is not None:
            self.display_frame(self.current_frame)

    def _apply_aspect_ratio(self):
        """将视频显示区域固定为4:3比例。"""
        controls_h = self.controls_widget.sizeHint().height() if hasattr(self, "controls_widget") else 0
        available_w = max(1, self.width())
        available_h = max(1, self.height() - controls_h - 8)
        target_w = min(available_w, int(available_h * 4 / 3))
        target_h = int(target_w * 3 / 4)
        if target_h > available_h:
            target_h = available_h
            target_w = int(target_h * 4 / 3)
        self.video_label.setFixedSize(max(1, target_w), max(1, target_h))


def make_card(title: str, hint: str = "") -> Tuple[QFrame, QVBoxLayout]:
    """一张卡片：标题 + 可选的一句说明。返回卡片本身和它的内容区布局。"""
    card = QFrame()
    card.setObjectName("card")

    outer = QVBoxLayout(card)
    outer.setContentsMargins(16, 14, 16, 16)
    outer.setSpacing(10)

    title_row = QHBoxLayout()
    title_row.setContentsMargins(0, 0, 0, 0)
    title_row.setSpacing(8)

    title_label = QLabel(title)
    title_label.setObjectName("h2")
    title_row.addWidget(title_label)
    title_row.addStretch()
    outer.addLayout(title_row)

    if hint:
        hint_label = QLabel(hint)
        hint_label.setObjectName("caption")
        hint_label.setWordWrap(True)
        outer.addWidget(hint_label)

    body = QVBoxLayout()
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(10)
    outer.addLayout(body)

    return card, body


def elide_label_text(label: QLabel, text: str, width_hint: int = 0) -> str:
    """按标签宽度做尾部省略，避免长文件名/路径把卡片撑宽或挤成多行。

    width_hint 用于标签还没参与过真实布局的场合（比如对话框 exec() 之前，
    label.width() 拿到的不是最终宽度）——这时用一个够用的估计宽度兜底。
    """
    metrics = QFontMetrics(label.font())
    width = width_hint or label.width()
    return metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(width, 1))


class TestPage(QWidget):
    """测试页面

    页面按新手能读懂的顺序排：
        ① 选择训练好的模型 → ② 选择图片/视频 → ③ 常用设置（高级项折叠）
        → 开始测试（左下角常驻，唯一主操作）→ 右侧看结果、知道存到哪
    推理逻辑、线程、参数语义都沿用原来的，这里只重排界面和状态提示。
    """

    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_project = None
        self.inference_thread = None
        self.model_path = None
        self.data_paths = []  # 待推理的数据路径列表
        self.current_data_index = 0
        self.inference_results = []  # 推理结果列表
        self.video_frames = []  # 视频帧缓存
        self.class_mapping = {}  # 类别映射
        self.model_classes = []  # 模型类别列表
        self.is_running = False
        self._stopping = False
        self._model_note = ""  # 模型来源那一行的补充说明
        self._model_user_chosen = False  # 用户自己挑过模型文件，别再被默认值盖掉
        self.output_root = Path(__file__).parent.parent.parent / "outputs"
        self.image_output_dir = self.output_root / "test_images"
        self.video_output_dir = self.output_root / "test_videos"

        self.init_ui()

    def _clear_outputs_for_run(self, data_paths):
        """只删掉这次测试将要覆盖的那几个输出文件。

        以前这里是构造页面时 rmtree 整个 outputs/test_images 和 outputs/test_videos——
        应用一启动（MainWindow 会构造所有页面）用户上次的测试结果就没了，而且用户根本
        没点过「开始测试」。现在只在用户真的点了「开始测试」之后，按本次输入算出目标
        文件名删掉，别人的结果一个都不碰。
        """
        for source_path in data_paths:
            target = annotated_output_path(
                source_path, self.image_output_dir, self.video_output_dir
            )
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.log_message(f"清理旧结果失败（忽略）: {target} - {exc}")

    def eventFilter(self, obj, event):
        """模型路径那一行自己变宽变窄时，重算省略号。

        省略要按标签的真实宽度算。只在设值的时候算一次是不行的：那会儿页面还没布局，
        拿到的是默认宽度，1100 宽下那行路径就会又断行又带省略号。
        也不能挂在页面的 resizeEvent 上——页面收到 resize 时，里面的标签还没被重新
        布局，读到的仍是旧宽度，省出来的文字比实际能画的宽，尾巴上的 .pt 被切掉。
        只有标签自己的 Resize 事件到达时，它的 width() 才是最终值。
        """
        if (
            event.type() == QEvent.Type.Resize
            and obj is getattr(self, 'model_source_label', None)
        ):
            self._update_model_summary()
        return super().eventFilter(obj, event)

    def set_project(self, project_id: int):
        """设置当前项目。

        每次切回测试页，主窗口都会把当前项目再设一遍。所以「同一个项目重设」
        必须是无损的：正在跑的测试不能被打断，用户自己挑好的模型也不能被
        悄悄换回项目默认的那个（换项目才重新推荐）。
        """
        project_changed = project_id != self.current_project_id
        if project_changed:
            self._model_user_chosen = False

        self.current_project_id = project_id
        if project_id:
            self.current_project = db.get_project(project_id)
            if self.current_project:
                project_name = self.current_project.get('name', 'Unknown') if isinstance(self.current_project, dict) else self.current_project.name
                print(f"[TestPage] 已切换到项目: {project_name} (ID: {project_id})")
                # 自动选用这个项目训练出来的模型
                if not self.is_running and not self._model_user_chosen:
                    self.set_default_model_path()
        else:
            self.current_project = None
            self.model_path = None
            self._model_note = "还没有选择项目"
            self._update_model_summary()
            print("[TestPage] 项目已取消选择")

        self.refresh_data_group_combo()
        self._update_ready_state()

    def set_default_model_path(self):
        """选用本项目训练产出的模型（runs/**/exp_<id>/weights/best.pt）。"""
        if not self.current_project_id:
            self.model_path = None
            self._model_note = "还没有选择项目"
            self._update_model_summary()
            self._update_ready_state()
            return

        weights = find_project_weights(self.current_project_id)
        if weights:
            self.model_path = str(weights)
            self._model_note = "来自本项目最近一次训练"
            self.log_message(f"已选用本项目训练好的模型: {self.model_path}")
        else:
            self.model_path = None
            self._model_note = "这个项目还没有训练结果"

        self._update_model_summary()
        self._update_ready_state()

    # ==================== 界面 ====================

    def init_ui(self):
        """初始化界面：左边按顺序设置，右边看结果。"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        setup_panel = self.create_setup_panel()
        setup_panel.setMinimumWidth(300)
        splitter.addWidget(setup_panel)

        result_panel = self.create_result_panel()
        result_panel.setMinimumWidth(320)
        splitter.addWidget(result_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 520])

        main_layout.addWidget(splitter)

        self.refresh_data_group_combo()
        self._update_model_summary()
        self._update_ready_state()

    def create_setup_panel(self) -> QWidget:
        """左栏：三步设置（可滚动）+ 常驻的运行区。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 0)
        content_layout.setSpacing(12)

        self.context_help = ContextHelp(
            [
                "识别门槛越高，结果通常越少，但判断会更严格。",
                "图片、文件夹和视频会使用同一套模型与门槛配置。",
                "测试结果只用于查看，不会写回项目标注。",
            ],
            title="测试提示",
        )
        content_layout.addWidget(self.context_help)

        content_layout.addWidget(self.create_model_card())
        content_layout.addWidget(self.create_source_card())
        content_layout.addWidget(self.create_options_card())
        content_layout.addWidget(self.create_log_section())
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # 运行区不进滚动区：窗口再小，主操作、进度和状态也一直在
        layout.addWidget(self.create_run_card())

        return panel

    def create_model_card(self) -> QFrame:
        """① 用哪个模型。"""
        card, body = make_card("选择训练好的模型")

        self.model_name_label = QLabel("未选择模型")
        self.model_name_label.setObjectName("title")
        self.model_name_label.setWordWrap(True)
        body.addWidget(self.model_name_label)

        # 模型路径这一行不许换行：路径本来就长，一换行就变成
        # 「/Users/huyi/dev/」+「EzYOLO/r…」两截，既断行又带省略号，最难看。
        # 只留一行，装不下就在中间省略（路径的头和尾都是信息，中间才是废话），
        # 完整路径放 tooltip。宽度策略给 Ignored，免得这行的理想宽度反过来把卡片撑宽。
        self.model_source_label = QLabel("")
        self.model_source_label.setObjectName("caption")
        self.model_source_label.setWordWrap(False)
        self.model_source_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.model_source_label.installEventFilter(self)
        body.addWidget(self.model_source_label)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self.btn_default_model = QPushButton("用本项目训练的模型")
        self.btn_default_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_default_model.clicked.connect(self.set_default_model_path)
        btn_row.addWidget(self.btn_default_model)

        self.btn_select_model = QPushButton("选择模型文件…")
        self.btn_select_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_model.clicked.connect(self.select_model)
        btn_row.addWidget(self.btn_select_model)

        btn_row.addStretch()
        body.addLayout(btn_row)

        return card

    def create_source_card(self) -> QFrame:
        """② 拿什么去测。"""
        card, body = make_card("选择图片或视频")

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self.btn_load_image = QPushButton("图片")
        self.btn_load_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_image.clicked.connect(lambda: self.load_data("image"))
        btn_row.addWidget(self.btn_load_image)

        self.btn_load_folder = QPushButton("文件夹")
        self.btn_load_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_folder.clicked.connect(lambda: self.load_data("folder"))
        btn_row.addWidget(self.btn_load_folder)

        self.btn_load_video = QPushButton("视频")
        self.btn_load_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_video.clicked.connect(lambda: self.load_data("video"))
        btn_row.addWidget(self.btn_load_video)

        btn_row.addStretch()
        body.addLayout(btn_row)

        # 也可以直接取项目里已有的图片分组
        group_row = QHBoxLayout()
        group_row.setContentsMargins(0, 0, 0, 0)
        group_row.setSpacing(8)

        group_label = QLabel("或用项目里的分组:")
        group_label.setObjectName("caption")
        group_row.addWidget(group_label)

        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(120)
        group_row.addWidget(self.group_combo, 1)

        self.btn_load_group = QPushButton("加载")
        self.btn_load_group.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_group.clicked.connect(self.load_project_group)
        group_row.addWidget(self.btn_load_group)
        body.addLayout(group_row)

        self.data_list = QListWidget()
        self.data_list.setMaximumHeight(132)
        body.addWidget(self.data_list)

        count_row = QHBoxLayout()
        count_row.setContentsMargins(0, 0, 0, 0)
        count_row.setSpacing(8)

        self.data_count_label = QLabel("还没有选择文件")
        self.data_count_label.setObjectName("caption")
        count_row.addWidget(self.data_count_label)
        count_row.addStretch()

        self.btn_clear_data = QPushButton("清空")
        self.btn_clear_data.setObjectName("ghost")
        self.btn_clear_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_data.clicked.connect(self.clear_data)
        count_row.addWidget(self.btn_clear_data)
        body.addLayout(count_row)

        return card

    def create_options_card(self) -> QFrame:
        """③ 常用两项在外面，其余折叠。"""
        card, body = make_card("设置")

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.01, 1.0)
        self.conf_threshold.setValue(0.25)
        self.conf_threshold.setDecimals(2)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setToolTip("调低会检出更多目标，误检也更多")
        form.addRow("识别门槛:", self.conf_threshold)

        self.inference_device = QComboBox()
        self.inference_device.addItems(["自动选择", "CPU", "CUDA:0", "CUDA:1", "CUDA:2", "CUDA:3"])
        self.inference_device.setToolTip("保持「自动选择」即可")
        form.addRow("运行设备:", self.inference_device)

        body.addLayout(form)

        advanced = CollapsibleSection("高级设置")

        adv_form = QFormLayout()
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.setSpacing(8)

        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.1, 1.0)
        self.iou_threshold.setValue(0.45)
        self.iou_threshold.setDecimals(2)
        self.iou_threshold.setSingleStep(0.05)
        adv_form.addRow("重叠框合并 (NMS IoU):", self.iou_threshold)

        self.inference_size = QSpinBox()
        self.inference_size.setRange(320, 1280)
        self.inference_size.setValue(640)
        self.inference_size.setSingleStep(32)
        adv_form.addRow("推理尺寸:", self.inference_size)

        advanced.content_layout().addLayout(adv_form)

        pretrained_form = QFormLayout()
        pretrained_form.setContentsMargins(0, 0, 0, 0)
        pretrained_form.setSpacing(8)

        self.model_version = QComboBox()
        self.model_version.addItems(sorted(ULTRALYTICS_MODELS.keys()))
        self.model_version.setToolTip("没有选模型时会退回这里选的官方预训练模型")
        self.model_version.currentTextChanged.connect(self.on_version_changed)
        pretrained_form.addRow("版本:", self.model_version)

        self.model_size = QComboBox()
        pretrained_form.addRow("型号:", self.model_size)

        self.task_type = QComboBox()
        self.task_type.setToolTip("决定 ONNX / TensorRT 模型怎么加载")
        pretrained_form.addRow("任务:", self.task_type)

        advanced.content_layout().addLayout(pretrained_form)

        # 先填好型号/任务，再接「选择变了就刷新模型说明」的信号，避免初始化时反复触发
        self._init_model_lists()
        self.model_version.currentTextChanged.connect(self._on_pretrained_changed)
        self.model_size.currentIndexChanged.connect(self._on_pretrained_changed)
        self.task_type.currentIndexChanged.connect(self._on_pretrained_changed)

        advanced.content_layout().addWidget(self.create_class_mapping_block())

        body.addWidget(advanced)

        return card

    def create_class_mapping_block(self) -> QWidget:
        """类别映射：模型的类别号 → 项目里的类别。属于高级项，默认藏起来。"""
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.chk_enable_mapping = QCheckBox("把模型的类别对应到项目类别")
        self.chk_enable_mapping.setChecked(False)
        self.chk_enable_mapping.stateChanged.connect(self.on_enable_mapping_changed)
        layout.addWidget(self.chk_enable_mapping)

        load_row = QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.setSpacing(8)

        self.btn_load_classes = QPushButton("加载模型 classes.txt")
        self.btn_load_classes.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_classes.clicked.connect(self.load_model_classes)
        self.btn_load_classes.setEnabled(False)
        load_row.addWidget(self.btn_load_classes)

        self.model_classes_label = QLabel("未加载")
        self.model_classes_label.setObjectName("caption")
        load_row.addWidget(self.model_classes_label)
        load_row.addStretch()
        layout.addLayout(load_row)

        self.class_mapping_list = QListWidget()
        self.class_mapping_list.setMaximumHeight(96)
        layout.addWidget(self.class_mapping_list)

        self.btn_edit_mapping = QPushButton("编辑类别映射")
        self.btn_edit_mapping.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_mapping.clicked.connect(self.edit_class_mapping)
        self.btn_edit_mapping.setEnabled(False)
        layout.addWidget(self.btn_edit_mapping, 0, Qt.AlignmentFlag.AlignLeft)

        return block

    def create_log_section(self) -> QWidget:
        """运行日志：默认收起，出问题时才展开。"""
        section = CollapsibleSection("运行日志")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        mono = mono_font_family_css()
        self.log_text.setStyleSheet(
            (f"font-family: {mono}; " if mono else "") + "font-size: 12px;"
        )
        section.content_layout().addWidget(self.log_text)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.setObjectName("ghost")
        self.btn_clear_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_log.clicked.connect(self.clear_log)
        btn_row.addWidget(self.btn_clear_log)

        self.btn_save_log = QPushButton("保存日志")
        self.btn_save_log.setObjectName("ghost")
        self.btn_save_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_log.clicked.connect(self.save_log)
        btn_row.addWidget(self.btn_save_log)

        btn_row.addStretch()
        section.content_layout().addLayout(btn_row)

        return section

    def create_run_card(self) -> QFrame:
        """运行区：现在能不能跑、跑到哪了、怎么停。"""
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.status_label = QLabel("先选好模型和图片")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self.btn_start_inference = QPushButton("开始测试")
        self.btn_start_inference.setObjectName("primary")
        self.btn_start_inference.setMinimumHeight(40)
        self.btn_start_inference.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_inference.clicked.connect(self.start_inference)
        btn_row.addWidget(self.btn_start_inference, 1)

        self.btn_stop_inference = QPushButton("停止")
        self.btn_stop_inference.setObjectName("danger")
        self.btn_stop_inference.setMinimumHeight(40)
        self.btn_stop_inference.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_inference.clicked.connect(self.stop_inference)
        self.btn_stop_inference.setVisible(False)
        btn_row.addWidget(self.btn_stop_inference)

        layout.addLayout(btn_row)

        return card

    def create_result_panel(self) -> QWidget:
        """右栏：结果、结果存在哪。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        title = QLabel("测试结果")
        title.setObjectName("h2")
        head.addWidget(title)
        head.addStretch()

        self.btn_open_output = QPushButton("打开结果文件夹")
        self.btn_open_output.setObjectName("ghost")
        self.btn_open_output.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_output.setToolTip(str(self.output_root))
        self.btn_open_output.clicked.connect(self.open_output_dir)
        head.addWidget(self.btn_open_output)
        layout.addLayout(head)

        self.result_stack = QStackedWidget()

        self.result_empty = EmptyState(
            "还没有测试结果",
            "选好模型和图片后点「开始测试」。",
        )
        self.result_stack.addWidget(self.result_empty)      # 0：空状态
        self.result_stack.addWidget(self.create_result_view())  # 1：有结果

        layout.addWidget(self.result_stack, 1)

        return panel

    def create_result_view(self) -> QWidget:
        """有结果时显示：图像/视频 + 这一张检出了什么 + 翻页和导出。"""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.result_tabs = QTabWidget()

        self.image_tab = QWidget()
        image_layout = QVBoxLayout(self.image_tab)
        self.image_viewer = ImageViewer()
        image_layout.addWidget(self.image_viewer)
        self.result_tabs.addTab(self.image_tab, "图像")

        self.video_tab = QWidget()
        video_layout = QVBoxLayout(self.video_tab)
        self.video_player = VideoPlayer()
        video_layout.addWidget(self.video_player)
        self.result_tabs.addTab(self.video_tab, "视频")

        layout.addWidget(self.result_tabs, 1)

        # 检出详情：内容多时自己滚，不把下面的按钮挤出屏幕
        self.result_info = QLabel("等待推理…")
        self.result_info.setWordWrap(True)
        self.result_info.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.result_info.setContentsMargins(12, 10, 12, 10)

        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        info_scroll.setWidget(self.result_info)
        info_scroll.setMinimumHeight(90)
        info_scroll.setMaximumHeight(150)
        info_scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {COLORS['panel']};"
            f" border: 1px solid {COLORS['border']}; border-radius: 6px; }}"
        )
        layout.addWidget(info_scroll)

        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)

        self.btn_prev = QPushButton("◀ 上一个")
        self.btn_prev.setObjectName("ghost")
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev.clicked.connect(self.show_previous_result)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)

        self.result_count_label = QLabel("0 / 0")
        self.result_count_label.setObjectName("caption")
        self.result_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_count_label.setMinimumWidth(60)
        nav_layout.addWidget(self.result_count_label)

        self.btn_next = QPushButton("下一个 ▶")
        self.btn_next.setObjectName("ghost")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.clicked.connect(self.show_next_result)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)

        nav_layout.addStretch()

        self.btn_export = QPushButton("导出结果")
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        nav_layout.addWidget(self.btn_export)

        layout.addLayout(nav_layout)

        return view

    # ==================== 当前选择与状态 ====================

    def _pretrained_model_name(self) -> str:
        """没有自选模型时会退回的官方预训练权重名（与实际推理用的一致）。"""
        version = self.model_version.currentText()
        model_size = self.model_size.currentData()
        task = self.task_type.currentData()

        if not version or not model_size or version not in ULTRALYTICS_MODELS:
            return ""

        prefix = ULTRALYTICS_MODELS[version]['prefix']
        if task == 'detect' or not task:
            return f"{prefix}{model_size}.pt"

        task_suffix_map = {"segment": "seg", "classify": "cls", "pose": "pose", "world": "world"}
        suffix = task_suffix_map.get(task, task)
        return f"{prefix}{model_size}-{suffix}.pt"

    def _on_pretrained_changed(self, *_args):
        """高级设置里换了预训练模型：把「现在会用哪个模型」这句话跟着改。"""
        self._update_model_summary()
        self._update_ready_state()

    def _update_model_summary(self):
        """把「现在会用哪个模型、它是哪来的」写清楚。"""
        if not hasattr(self, 'model_name_label'):
            return

        if self.model_path:
            path = Path(self.model_path)
            self.model_name_label.setText(elide_label_text(self.model_name_label, path.name))
            self.model_name_label.setToolTip(path.name)
            self.model_name_label.setStyleSheet("")
            source = self._model_note or "手动选择"
            # 路径要省略到「这一行减去前缀」还剩下的宽度里，否则前缀把它挤到第二行，
            # 结果就是又断行、又带省略号。省略号打在中间：路径的头（在哪个盘/哪个项目）
            # 和尾（哪个 run 的 best.pt）都是用户要认的，中间的目录才是可以省的。
            metrics = self.model_source_label.fontMetrics()
            prefix_width = metrics.horizontalAdvance(f"{source}　")
            # 留 4px：算得刚好等于宽度时，最后一个字符会贴着边缘被切掉半个
            budget = max(80, self.model_source_label.width() - prefix_width - 4)
            elided_path = metrics.elidedText(
                str(path), Qt.TextElideMode.ElideMiddle, budget
            )
            self.model_source_label.setText(f"{source}　{elided_path}")
            self.model_source_label.setToolTip(str(path))
            return

        fallback = self._pretrained_model_name() or "官方预训练模型"
        self.model_name_label.setText("未选择模型")
        hint = f"未选模型时使用 {fallback}"
        if self._model_note:
            hint = f"{self._model_note}；{hint}"
        self.model_source_label.setText(hint)
        self.model_source_label.setToolTip(
            "官方预训练模型首次运行会自动下载，只认通用类别，不是你训练的模型"
        )

    def _set_status(self, text: str, kind: str = "normal"):
        """运行区那行状态：就绪 / 运行中 / 成功 / 失败，用颜色区分。"""
        self._last_status_kind = kind
        color = {
            "normal": COLORS['text_secondary'],
            "running": COLORS['accent_text'],
            "success": COLORS['success'],
            "error": COLORS['error'],
        }.get(kind, COLORS['text_secondary'])
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 13px;")

    def _update_ready_state(self):
        """主操作能不能点、还差什么——一句话说清。"""
        if not hasattr(self, 'btn_start_inference') or self.is_running:
            return

        self.btn_start_inference.setEnabled(bool(self.data_paths))

        if not self.data_paths:
            self._set_status("还差一步：选择要测试的图片、文件夹或视频")
            return

        count = len(self.data_paths)
        if self.model_path:
            self._set_status(f"准备就绪：用 {Path(self.model_path).name} 测试 {count} 个文件")
        else:
            fallback = self._pretrained_model_name() or "官方预训练模型"
            self._set_status(f"准备就绪：用官方预训练模型 {fallback} 测试 {count} 个文件")

    # ==================== 模型 ====================

    def _init_model_lists(self):
        """初始化型号和任务列表"""
        version = self.model_version.currentText()

        if not version:
            version = sorted(ULTRALYTICS_MODELS.keys())[0]
            self.model_version.setCurrentText(version)

        if version in ULTRALYTICS_MODELS:
            model_info = ULTRALYTICS_MODELS[version]

            # 初始化型号列表
            self.model_size.clear()
            for size in model_info['sizes']:
                display_name = SIZE_NAMES.get(size, size)
                self.model_size.addItem(display_name, size)

            # 初始化任务列表
            self.task_type.clear()
            for task in model_info['tasks']:
                display_name = TASK_NAMES.get(task, task)
                self.task_type.addItem(display_name, task)

    def on_version_changed(self, version: str):
        """版本改变时更新型号和任务"""
        if not version or version not in ULTRALYTICS_MODELS:
            return

        model_info = ULTRALYTICS_MODELS[version]

        # 更新型号列表
        try:
            self.model_size.clear()
            for size in model_info['sizes']:
                display_name = SIZE_NAMES.get(size, size)
                self.model_size.addItem(display_name, size)
        except RuntimeError:
            return

        # 更新任务列表
        try:
            self.task_type.clear()
            for task in model_info['tasks']:
                display_name = TASK_NAMES.get(task, task)
                self.task_type.addItem(display_name, task)
        except RuntimeError:
            return

    def select_model(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "所有模型文件 (*.pt *.pth *.onnx *.engine *.trt);;"
            "PyTorch模型 (*.pt *.pth);;"
            "ONNX模型 (*.onnx);;"
            "TensorRT模型 (*.engine *.trt);;"
            "所有文件 (*.*)"
        )

        if file_path:
            self.model_path = file_path
            self._model_note = "手动选择的模型文件"
            self._model_user_chosen = True
            self._update_model_summary()
            self._update_ready_state()
            self.log_message(f"已选择模型: {file_path}")

            # 根据模型类型给出提示
            if file_path.endswith('.onnx'):
                self.log_message("ℹ 已选择ONNX模型，确保已安装onnxruntime或onnxruntime-gpu")
            elif file_path.endswith('.engine') or file_path.endswith('.trt'):
                self.log_message("ℹ 已选择TensorRT模型，确保已安装tensorrt并配置正确")

    # ==================== 类别映射（高级） ====================

    def on_enable_mapping_changed(self, state):
        """启用映射选项改变"""
        enabled = state == Qt.CheckState.Checked.value
        self.btn_load_classes.setEnabled(enabled)
        self.btn_edit_mapping.setEnabled(enabled and len(self.model_classes) > 0)

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
                    self.model_classes = classes
                    self.model_classes_label.setText(f"已加载 {len(classes)} 个类别")

                    # 自动创建默认映射（一对一）
                    self.class_mapping = {i: i for i in range(len(classes))}
                    self.update_class_mapping_list()

                    self.btn_edit_mapping.setEnabled(True)
                    self.log_message(f"✓ 成功加载类别文件: {file_path}")
                    self.log_message(f"  类别: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
                else:
                    QMessageBox.warning(self, "警告", "classes.txt文件为空")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载classes.txt文件失败: {str(e)}")

    def update_class_mapping_list(self):
        """更新类别映射列表显示"""
        self.class_mapping_list.clear()

        if not self.model_classes:
            return

        # 获取项目类别
        project_classes = []
        if self.current_project:
            classes_json = self.current_project.get('classes', '[]') if isinstance(self.current_project, dict) else '[]'
            project_classes = json.loads(classes_json) if classes_json else []

        project_class_names = {cls['id']: cls['name'] for cls in project_classes}

        for model_id, project_id in self.class_mapping.items():
            if model_id < len(self.model_classes):
                model_name = self.model_classes[model_id]
                project_name = project_class_names.get(project_id, f"ID:{project_id}")
                item_text = f"{model_id}: {model_name} → {project_id}: {project_name}"
                self.class_mapping_list.addItem(item_text)

    def edit_class_mapping(self):
        """编辑类别映射"""
        if not self.model_classes:
            QMessageBox.warning(self, "警告", "请先加载模型classes.txt文件")
            return

        # 获取项目类别
        project_classes = []
        if self.current_project:
            classes_json = self.current_project.get('classes', '[]') if isinstance(self.current_project, dict) else '[]'
            project_classes = json.loads(classes_json) if classes_json else []

        if not project_classes:
            QMessageBox.warning(self, "警告", "当前项目没有类别，请先设置项目类别")
            return

        # 创建对话框
        from PyQt6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("编辑类别映射")
        dialog.setMinimumSize(400, 300)

        layout = QVBoxLayout(dialog)

        # 说明标签
        info_label = QLabel("将模型类别映射到项目类别:")
        layout.addWidget(info_label)

        # 创建映射选择
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        mapping_widgets = []

        for i, model_class in enumerate(self.model_classes):
            row_layout = QHBoxLayout()

            # 模型类别标签
            full_label_text = f"{i}: {model_class}"
            model_label = QLabel()
            model_label.setMinimumWidth(150)
            model_label.setText(elide_label_text(model_label, full_label_text, width_hint=190))
            model_label.setToolTip(full_label_text)
            row_layout.addWidget(model_label)

            row_layout.addWidget(QLabel("→"))

            # 项目类别选择
            project_combo = QComboBox()
            for cls in project_classes:
                project_combo.addItem(f"{cls['id']}: {cls['name']}", cls['id'])

            # 设置当前映射
            current_mapping = self.class_mapping.get(i, i)
            index = project_combo.findData(current_mapping)
            if index >= 0:
                project_combo.setCurrentIndex(index)

            row_layout.addWidget(project_combo)
            mapping_widgets.append((i, project_combo))

            scroll_layout.addLayout(row_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)

        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("primary")
        btn_ok.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_ok)

        layout.addLayout(btn_layout)

        # 显示对话框
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 更新映射
            for model_id, combo in mapping_widgets:
                project_id = combo.currentData()
                self.class_mapping[model_id] = project_id

            self.update_class_mapping_list()
            self.log_message(f"✓ 类别映射已更新: {self.class_mapping}")

    # ==================== 数据来源 ====================

    def load_data(self, data_type: str):
        """加载数据"""
        if data_type == "image":
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择图片", "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*.*)"
            )
            if file_paths:
                self.data_paths.extend(file_paths)
                self.log_message(f"已加载 {len(file_paths)} 张图片")

        elif data_type == "folder":
            folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if folder_path:
                # 获取文件夹中的所有图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(Path(folder_path).glob(f"*{ext}"))
                    image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

                file_paths = [str(f) for f in image_files]
                self.data_paths.extend(file_paths)
                self.log_message(f"已从文件夹加载 {len(file_paths)} 张图片")
                if not file_paths:
                    QMessageBox.information(self, "提示", "这个文件夹里没有找到图片")

        elif data_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
            )
            if file_path:
                self.data_paths.append(file_path)
                self.log_message(f"已加载视频: {file_path}")

        # 更新数据列表显示
        self.update_data_list()

    def refresh_data_group_combo(self):
        """刷新项目分组下拉列表。"""
        if not hasattr(self, 'group_combo'):
            return

        self.group_combo.blockSignals(True)
        self.group_combo.clear()

        if not self.current_project_id:
            self.group_combo.addItem("请先选择项目", None)
            self.group_combo.setEnabled(False)
            self.btn_load_group.setEnabled(False)
            self.group_combo.blockSignals(False)
            return

        counts = db.get_group_image_counts(self.current_project_id)
        groups = db.get_project_image_groups(self.current_project_id)

        ungrouped_count = counts.get(None, 0)
        self.group_combo.addItem(f"未分组 ({ungrouped_count})", UNGROUPED_GROUP_ID)
        for group in groups:
            count = counts.get(group['id'], 0)
            self.group_combo.addItem(f"{group['name']} ({count})", group['id'])

        has_groups = self.group_combo.count() > 0
        self.group_combo.setEnabled(has_groups)
        self.btn_load_group.setEnabled(has_groups)
        self.group_combo.blockSignals(False)

    def load_project_group(self):
        """从当前项目分组加载图片。"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return

        group_id = self.group_combo.currentData()
        if group_id is None:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return

        if group_id == UNGROUPED_GROUP_ID:
            images = db.get_project_images(self.current_project_id, ungrouped_only=True)
            group_label = "未分组"
        else:
            images = db.get_project_images(self.current_project_id, group_id=group_id)
            group = db.get_image_group(group_id)
            group_label = group['name'] if group else str(group_id)

        if not images:
            QMessageBox.information(self, "提示", f"分组「{group_label}」中没有图片")
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        added = 0
        skipped = 0
        existing = set(self.data_paths)

        for image in images:
            storage_path = image.get('storage_path', '')
            if not storage_path or not os.path.exists(storage_path):
                skipped += 1
                continue
            if Path(storage_path).suffix.lower() not in image_extensions:
                skipped += 1
                continue
            if storage_path in existing:
                skipped += 1
                continue
            self.data_paths.append(storage_path)
            existing.add(storage_path)
            added += 1

        self.update_data_list()
        self.log_message(f"已从分组「{group_label}」加载 {added} 张图片")
        if skipped > 0:
            self.log_message(f"  跳过 {skipped} 个（不存在、非图片或已加载）")

    def update_data_list(self):
        """更新数据列表显示"""
        self.data_list.clear()
        for path in self.data_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.data_list.addItem(item)

        count = len(self.data_paths)
        self.data_count_label.setText(f"已选 {count} 个文件" if count else "还没有选择文件")
        self._update_ready_state()

    def clear_data(self):
        """清空数据"""
        self.data_paths.clear()
        self.update_data_list()
        self.log_message("已清空待测试的文件")

    # ==================== 运行 ====================

    def start_inference(self):
        """开始推理"""
        # 检查是否有数据
        if not self.data_paths:
            QMessageBox.warning(self, "还不能开始", "请先选择要测试的图片、文件夹或视频")
            return

        # 清空之前的结果（内存里的，以及本次会覆盖的那几个输出文件）
        self.inference_results.clear()
        self.current_data_index = 0
        self.video_player.clear()
        self._clear_outputs_for_run(self.data_paths)

        # 获取项目类别信息
        project_classes = []
        if self.current_project:
            classes_json = self.current_project.get('classes', '[]') if isinstance(self.current_project, dict) else '[]'
            project_classes = json.loads(classes_json) if classes_json else []

        # 获取任务类型
        task = self.task_type.currentData()

        # 收集配置
        config = {
            'conf': self.conf_threshold.value(),
            'iou': self.iou_threshold.value(),
            'imgsz': self.inference_size.value(),
            'device': self.inference_device.currentText(),
            'task': task,  # 添加任务类型到配置
        }

        # 确定模型路径：没有自选模型时退回官方预训练权重
        model_path = self.model_path
        if not model_path:
            model_path = self._pretrained_model_name() or None
            if model_path:
                self.log_message(f"使用预训练模型: {model_path}")

        # 获取类别映射
        class_mapping = {}
        if self.chk_enable_mapping.isChecked():
            class_mapping = self.class_mapping

        # 创建推理线程
        self.inference_thread = InferenceThread(
            model_path=model_path,
            config=config,
            data_paths=self.data_paths.copy(),
            project_classes=project_classes,
            class_mapping=class_mapping
        )
        self.inference_thread.progress_updated.connect(self.on_progress_updated)
        self.inference_thread.inference_finished.connect(self.on_inference_finished)
        self.inference_thread.log_message.connect(self.on_log_message)
        self.inference_thread.result_ready.connect(self.on_result_ready)
        self.inference_thread.frame_ready.connect(self.on_frame_ready)

        # 更新UI状态：主操作让位给「停止」，进度和状态都在同一张卡里
        self.is_running = True
        self._stopping = False
        self.btn_start_inference.setEnabled(False)
        self.btn_start_inference.setText("测试中…")
        self.btn_stop_inference.setVisible(True)
        self.btn_stop_inference.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.data_paths))
        self.progress_bar.setValue(0)
        self._set_status(f"正在测试… 0/{len(self.data_paths)}", "running")

        # 结果区先回到空状态，第一个结果出来会自动切过去
        self.result_empty.set_text("正在测试…", "第一个结果出来后就会显示在这里。")
        self.result_stack.setCurrentIndex(0)
        self.result_count_label.setText("0 / 0")
        self.btn_export.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

        # 启动推理
        self.inference_thread.start()

        self.log_message("=" * 50)
        self.log_message("开始测试！")
        self.log_message(f"数据数量: {len(self.data_paths)}")
        if class_mapping:
            self.log_message("类别映射: 已启用")
        self.log_message("=" * 50)

    def stop_inference(self):
        """停止推理：只发出停止请求，不在界面线程里等。

        原来这里是 stop() 之后直接 wait()——推理线程正卡在一张大图或一段视频上时，
        wait() 会把界面线程一起冻住：窗口画不动、按钮点不了，看起来就像卡死。
        现在只置停止标志、把按钮改成「正在停止…」，线程自己跑完当前这张就退出，
        收尾交给 inference_finished 回调（那条路径本来就会把状态复位）。
        """
        thread = self.inference_thread
        if thread is None or not thread.isRunning():
            self.reset_ui_state()
            self._set_status("已停止", "normal")
            return

        self._stopping = True
        self.btn_stop_inference.setEnabled(False)
        self.btn_stop_inference.setText("正在停止…")
        self._set_status("正在停止…", "running")
        self.log_message("正在停止测试…（当前这张跑完就停）")
        thread.stop()

    def reset_ui_state(self):
        """回到「可以再跑一次」的状态。"""
        self.is_running = False
        self.btn_start_inference.setText("开始测试")
        self.btn_stop_inference.setText("停止")
        self.btn_stop_inference.setVisible(False)
        self.btn_stop_inference.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._update_ready_state()

    def shutdown(self):
        """关窗前收工：推理线程还在跑就先请它停，再等它退出。

        这里可以等——窗口都要关了，界面卡不卡已经无所谓；反过来，
        QThread 在 run() 没结束时被销毁会直接崩。等待仍然给上限。
        """
        thread = self.inference_thread
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(5000)

    def refresh_theme(self):
        """主题切换后刷新预览区与状态色。"""
        viewer = getattr(self, 'image_viewer', None)
        if viewer is not None and hasattr(viewer, 'image_label'):
            viewer.image_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {COLORS['sidebar']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 4px;
                }}
            """)
            if hasattr(viewer, 'info_label'):
                viewer.info_label.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; font-size: 12px;"
                )
        if hasattr(self, 'status_label'):
            kind = getattr(self, '_last_status_kind', 'normal')
            self._set_status(self.status_label.text(), kind)

    def on_progress_updated(self, current: int, total: int):
        """进度更新"""
        self.progress_bar.setValue(current)
        self._set_status(f"正在测试… {current}/{total}", "running")

    def on_inference_finished(self, success: bool, message: str):
        """推理完成"""
        stopping = self._stopping
        self._stopping = False
        self.reset_ui_state()

        if success:
            self._set_status(f"✓ {message}", "success")
            if self.inference_results:
                self.show_result(0)
        elif stopping:
            self._set_status("已停止", "normal")
        else:
            self._set_status(f"✗ {message}", "error")
            if not self.inference_results:
                self.result_empty.set_text(
                    "这次没有得到结果",
                    message,
                )
                self.result_stack.setCurrentIndex(0)
            QMessageBox.warning(self, "测试结束", message)

        # 更新导航按钮状态
        self.update_nav_buttons()

    def on_log_message(self, message: str):
        """日志消息"""
        self.log_message(message)

    def log_message(self, message: str):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ==================== 结果 ====================

    def on_result_ready(self, result: dict):
        """单张推理结果就绪"""
        self.inference_results.append(result)

        # 显示最新结果
        self.show_result(len(self.inference_results) - 1)

        # 更新导出按钮
        self.btn_export.setEnabled(True)

    def on_frame_ready(self, frame: np.ndarray, detection_info: dict):
        """视频帧就绪"""
        # 保留接口兼容；当前视频流程改为写盘后播放，不再使用内存帧缓存
        if detection_info.get('frame_number') == 0:
            self.result_tabs.setCurrentIndex(1)  # 切换到视频标签页

    def show_result(self, index: int):
        """显示指定索引的结果"""
        if not self.inference_results or index < 0 or index >= len(self.inference_results):
            return

        self.current_data_index = index
        result = self.inference_results[index]

        # 有结果了，从空状态切到结果视图
        self.result_stack.setCurrentIndex(1)

        # 检查是否为视频
        if result.get('is_video'):
            # 切换到视频标签页
            self.result_tabs.setCurrentIndex(1)
            video_to_play = result.get('output_video') or result.get('source_path')
            if video_to_play and os.path.exists(video_to_play):
                self.video_player.load_video(video_to_play)
        else:
            # 切换到图像标签页
            self.result_tabs.setCurrentIndex(0)

            # 显示写盘后的标注图像
            image_to_show = result.get('annotated_image_path') or result.get('source_path')
            self.image_viewer.show_image(image_to_show)

        # 更新结果信息
        info_text = f"""
<b>文件:</b> {result['filename']}<br>
        """

        if result.get('is_video'):
            info_text += f"<b>类型:</b> 视频<br>"
            info_text += f"<b>总帧数:</b> {result.get('total_frames', 0)}<br>"
            info_text += f"<b>已处理:</b> {result.get('processed_frames', 0)} 帧<br>"
        else:
            # 显示检测框信息
            detections = result['detections']
            if detections:
                info_text += f"<b>检测到:</b> {len(detections)} 个目标<br>"
                info_text += "<b>检测详情:</b><br>"
                for i, det in enumerate(detections[:10]):  # 最多显示10个
                    info_text += f"  {i+1}. {det['class_name']} ({det['confidence']:.2f})<br>"
                if len(detections) > 10:
                    info_text += f"  ... 还有 {len(detections) - 10} 个目标<br>"
            else:
                info_text += "<b>这张图没有检测到目标。</b><br>"

            # 显示分割mask信息（如果有）
            if result.get('masks') and len(result['masks']) > 0:
                info_text += f"<br><b>分割区域:</b> {len(result['masks'])} 个<br>"
                info_text += "<b>分割详情:</b><br>"
                for i, mask in enumerate(result['masks'][:10]):  # 最多显示10个
                    point_count = len(mask.get('points', []))
                    info_text += f"  {i+1}. {mask['class_name']} ({mask['confidence']:.2f}) - {point_count} 个点<br>"
                if len(result['masks']) > 10:
                    info_text += f"  ... 还有 {len(result['masks']) - 10} 个分割区域<br>"

        if result.get('speed'):
            speed = result['speed']
            info_text += f"<br><b>推理速度:</b><br>"
            if 'preprocess' in speed:
                info_text += f"  预处理: {speed['preprocess']:.1f}ms<br>"
            if 'inference' in speed:
                info_text += f"  推理: {speed['inference']:.1f}ms<br>"
            if 'postprocess' in speed:
                info_text += f"  后处理: {speed['postprocess']:.1f}ms<br>"

        self.result_info.setText(info_text)

        # 更新计数
        self.result_count_label.setText(f"{index + 1} / {len(self.inference_results)}")

        # 更新导航按钮
        self.update_nav_buttons()

    def update_nav_buttons(self):
        """更新导航按钮状态"""
        total = len(self.inference_results)
        current = self.current_data_index

        self.btn_prev.setEnabled(current > 0)
        self.btn_next.setEnabled(current < total - 1)

    def show_previous_result(self):
        """显示上一个结果"""
        if self.current_data_index > 0:
            self.show_result(self.current_data_index - 1)

    def show_next_result(self):
        """显示下一个结果"""
        if self.current_data_index < len(self.inference_results) - 1:
            self.show_result(self.current_data_index + 1)

    def open_output_dir(self):
        """打开保存结果的文件夹。"""
        self.output_root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_root)))

    def export_results(self):
        """导出推理结果"""
        if not self.inference_results:
            QMessageBox.warning(self, "错误", "没有可导出的结果")
            return

        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return

        try:
            import shutil

            # 创建导出子目录
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = os.path.join(export_dir, f"inference_results_{timestamp}")
            os.makedirs(export_subdir, exist_ok=True)

            # 创建images和labels目录
            images_dir = os.path.join(export_subdir, "images")
            labels_dir = os.path.join(export_subdir, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            exported_count = 0

            for result in self.inference_results:
                filename = result['filename']
                name_without_ext = os.path.splitext(filename)[0]

                if result.get('is_video'):
                    # 导出视频
                    if result.get('annotated_image') is not None:
                        output_path = os.path.join(images_dir, filename)
                        # 这里应该保存视频文件，简化处理
                        self.log_message(f"视频导出暂不支持: {filename}")
                else:
                    # 保存标注后的图像
                    if result.get('annotated_image') is not None:
                        output_path = os.path.join(images_dir, filename)
                        cv2.imwrite(output_path, result['annotated_image'])
                    else:
                        # 复制原图
                        shutil.copy2(result['source_path'], os.path.join(images_dir, filename))

                    # 保存标注文件（YOLO格式）
                    label_file = os.path.join(labels_dir, f"{name_without_ext}.txt")
                    with open(label_file, 'w') as f:
                        for det in result['detections']:
                            bbox = det['bbox']
                            # 转换为YOLO格式 (x_center, y_center, width, height)
                            x1, y1, x2, y2 = bbox
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1

                            # 这里假设图像是640x640，实际应该获取真实尺寸
                            # 简化处理，使用归一化坐标
                            f.write(f"{det['class_id']} {x_center} {y_center} {width} {height}\n")

                exported_count += 1

            # 保存推理日志
            log_file = os.path.join(export_subdir, "inference_log.txt")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())

            QMessageBox.information(
                self,
                "导出成功",
                f"已导出 {exported_count} 个结果到:\n{export_subdir}"
            )
            self.log_message(f"✓ 导出完成: {export_subdir}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出时出错:\n{str(e)}")
            self.log_message(f"✗ 导出失败: {e}")

    def clear_log(self):
        """清空日志"""
        self.log_text.clear()

    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "inference_log.txt",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "保存成功", f"日志已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存日志时出错:\n{str(e)}")
