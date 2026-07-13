# -*- coding: utf-8 -*-
"""
自动打标签核心逻辑模块
负责使用YOLO模型自动生成标注
"""

import os
import time
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from core.model_manager import model_manager
from models.database import db


class AutoLabeler:
    """自动打标签器"""
    
    def __init__(self, model_path: str, model_manager):
        self.model_path = model_path
        self.model_manager = model_manager
        self.current_model = None
        self.model_info = {}
        self.class_mappings = {}
        self.model_task = 'detect'
    
    def load_model(self, config: Dict) -> bool:
        """加载模型
        
        Args:
            config: 模型配置
            
        Returns:
            是否加载成功
        """
        try:
            model_version = config['model_version']
            model_size = config['model_size']
            model_source = config['model_source']
            model_task = config.get('model_task', 'detect')
            custom_model_path = config.get('custom_model_path', '')
            
            # 输出调试信息
            print("=" * 50)
            print("AutoLabeler.load_model 开始加载模型:")
            print(f"  模型版本: {model_version}")
            print(f"  模型大小: {model_size}")
            print(f"  模型来源: {model_source}")
            print(f"  任务类型: {model_task}")
            if model_source == 'custom':
                print(f"  自定义模型路径: {custom_model_path}")
            print("=" * 50)
            
            # 加载模型
            if model_source == 'official':
                self.current_model = model_manager.load_model(
                    model_version, model_size, model_task
                )
            else:
                self.current_model = model_manager.load_custom_model(custom_model_path)
            
            if self.current_model:
                self.model_info = model_manager.get_model_info(self.current_model)
                self.class_mappings = config.get('class_mappings', {})
                self.model_task = model_task
                
                # 输出加载成功信息
                model_name = f"{model_version}-{model_size}-{model_task}" if model_source == 'official' else os.path.basename(custom_model_path)
                print("=" * 50)
                print("AutoLabeler.load_model 模型加载成功:")
                print(f"  模型名称: {model_name}")
                print(f"  任务类型: {self.model_task}")
                print(f"  模型任务 (model.task): {self.model_info.get('task', 'unknown')}")
                print(f"  类别数量: {self.model_info.get('nc', 0)}")
                print("=" * 50)
                return True
            else:
                print("=" * 50)
                print("AutoLabeler.load_model 模型加载失败!")
                print("=" * 50)
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_single_image(self, image_path: str, image_id: int, config: Dict) -> List[Dict]:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            image_id: 图像ID
            config: 推理配置
            
        Returns:
            生成的标注列表
        """
        if not self.current_model:
            return []
        
        # 推理参数
        conf_threshold = config.get('conf_threshold', 0.5)
        iou_threshold = config.get('iou_threshold', 0.45)
        
        # 进行推理
        result = model_manager.infer(
            self.current_model, image_path, conf_threshold, iou_threshold
        )
        
        if not result:
            return []
        
        # 生成标注
        annotations = self._generate_annotations(result, image_id, config)
        return annotations
    
    def _generate_annotations(self, result: object, image_id: int, config: Dict) -> List[Dict]:
        """根据推理结果生成标注
        
        Args:
            result: 推理结果
            image_id: 图像ID
            config: 配置
            
        Returns:
            标注列表
        """
        annotations = []
        
        # 获取任务类型
        model_task = getattr(self, 'model_task', 'detect')
        
        # 遍历检测结果
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 计算宽度和高度
            width = x2 - x1
            height = y2 - y1
            
            # 应用类别映射
            mapped_class_id = self._map_class_id(class_id)
            
            # 添加调试信息
            print(f"DEBUG: model_task = {model_task}")
            print(f"DEBUG: hasattr(result, 'masks') = {hasattr(result, 'masks')}")
            if hasattr(result, 'masks'):
                print(f"DEBUG: result.masks = {result.masks}")
                print(f"DEBUG: hasattr(result.masks, 'xy') = {hasattr(result.masks, 'xy')}")
                if hasattr(result.masks, 'xy'):
                    print(f"DEBUG: len(result.masks.xy) = {len(result.masks.xy)}")
            
            if model_task == 'segment' and hasattr(result, 'masks') and result.masks:
                # 生成segment任务的多边形标注
                print(f"DEBUG: Generating polygon annotation for segment task")
                if hasattr(result.masks, 'xy') and result.masks.xy is not None:
                    # 获取多边形边界坐标
                    # result.masks.xy 是一个列表，每个元素是一个numpy数组，形状为 (n_points, 2)
                    if i < len(result.masks.xy):
                        polygons = result.masks.xy[i]
                        print(f"DEBUG: Polygon points (raw) = {polygons}")
                        
                        # 转换为列表格式，与手动标注保持一致
                        points = []
                        # polygons 是一个numpy数组，形状为 (n_points, 2)
                        # 需要转换为Python列表并提取坐标
                        # 处理numpy数组：如果有tolist方法，使用它
                        if hasattr(polygons, 'tolist'):
                            polygons_list = polygons.tolist()
                        else:
                            polygons_list = polygons
                        
                        # 遍历每个点
                        for point in polygons_list:
                            # point 可能是 [x, y] 列表或 (x, y) 元组或numpy数组
                            # 处理numpy数组
                            if hasattr(point, 'tolist'):
                                point = point.tolist()
                            
                            # 检查是否是有效的点格式
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                points.append({'x': float(point[0]), 'y': float(point[1])})
                        
                        print(f"DEBUG: Converted points = {points}")
                        
                        # 如果points为空，说明转换失败，使用bbox作为备用
                        if not points:
                            print(f"DEBUG: Warning: Failed to convert polygon points, using bbox instead")
                            annotation = {
                                'image_id': image_id,
                                'type': 'bbox',
                                'class_id': mapped_class_id,
                                'confidence': conf,
                                'data': {
                                    'x': x1,
                                    'y': y1,
                                    'width': width,
                                    'height': height
                                }
                            }
                        else:
                            annotation = {
                                'image_id': image_id,
                                'type': 'polygon',
                                'class_id': mapped_class_id,
                                'confidence': conf,
                                'data': {
                                    'points': points
                                }
                            }
                    else:
                        # 索引超出范围，使用bbox
                        print(f"DEBUG: Warning: Mask index {i} out of range, using bbox instead")
                        annotation = {
                            'image_id': image_id,
                            'type': 'bbox',
                            'class_id': mapped_class_id,
                            'confidence': conf,
                            'data': {
                                'x': x1,
                                'y': y1,
                                'width': width,
                                'height': height
                            }
                        }
                else:
                    # 没有masks.xy，使用bbox作为备用
                    print(f"DEBUG: Warning: No masks.xy available, using bbox instead")
                    annotation = {
                        'image_id': image_id,
                        'type': 'bbox',
                        'class_id': mapped_class_id,
                        'confidence': conf,
                        'data': {
                            'x': x1,
                            'y': y1,
                            'width': width,
                            'height': height
                        }
                    }
                print(f"DEBUG: Generated annotation type = {annotation['type']}")
                print(f"DEBUG: Generated annotation data = {annotation['data']}")
            else:
                # 生成detect任务的bbox标注
                annotation = {
                    'image_id': image_id,
                    'type': 'bbox',
                    'class_id': mapped_class_id,
                    'confidence': conf,
                    'data': {
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height
                    }
                }
            annotations.append(annotation)
        
        return annotations
    
    def _map_class_id(self, class_id: int) -> int:
        """映射类别ID
        
        Args:
            class_id: 模型输出的类别ID
            
        Returns:
            映射后的类别ID
        """
        # 检查是否有手动映射
        if class_id in self.class_mappings:
            return self.class_mappings[class_id]
        
        # 如果没有映射，直接返回原始ID
        # 这里可以添加逻辑，处理超出范围的情况
        return class_id
    
    def process_class_id(self, class_id: int, project_classes: list, enable_mapping: bool = False) -> int:
        """处理类别ID，根据映射设置和项目类别列表
        
        Args:
            class_id: 模型输出的类别ID
            project_classes: 项目类别列表
            enable_mapping: 是否启用类别映射
            
        Returns:
            处理后的类别ID
        """
        if enable_mapping:
            # 使用映射
            return self._map_class_id(class_id)
        else:
            # 不使用映射，直接对应或创建新类别
            # 检查类别ID是否在项目类别范围内
            if project_classes:
                max_class_id = max([cls['id'] for cls in project_classes])
                if class_id > max_class_id:
                    # 如果超出范围，返回原始ID（后续会创建新类别）
                    return class_id
            # 如果在范围内，直接返回
            return class_id
    
    def save_annotations(self, annotations: List[Dict], image_id: int, overwrite: bool = False):
        """保存标注
        
        Args:
            annotations: 标注列表
            image_id: 图像ID
            overwrite: 是否覆盖原标注
        """
        if overwrite:
            # 删除原标注
            db.delete_image_annotations(image_id)
        
        # 保存新标注
        for annotation in annotations:
            # 获取项目ID
            image_info = self.get_image_info(image_id)
            if not image_info:
                continue
            
            project_id = image_info.get('project_id', 0)
            
            # 获取类别名称
            class_id = annotation.get('class_id', 0)
            class_name = f"class_{class_id}"
            
            # 保存标注
            db.add_annotation(
                image_id,
                project_id,
                class_id,
                class_name,
                annotation.get('type', 'bbox'),
                annotation.get('data', {})
            )
    
    def get_image_info(self, image_id: int) -> Optional[Dict]:
        """获取图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            图像信息
        """
        return db.get_image(image_id)
    
    def get_unlabeled_images(self, project_id: int) -> List[Dict]:
        """获取未标注的图像
        
        Args:
            project_id: 项目ID
            
        Returns:
            未标注图像列表
        """
        images = db.get_project_images(project_id)
        unlabeled = []
        
        for image in images:
            annotations = db.get_image_annotations(image['id'])
            if not annotations:
                unlabeled.append(image)
        
        return unlabeled
    
    def get_all_images(self, project_id: int) -> List[Dict]:
        """获取所有图像
        
        Args:
            project_id: 项目ID
            
        Returns:
            图像列表
        """
        return db.get_project_images(project_id)
    
    def process_image(self, image_path: str, conf_threshold: float, iou_threshold: float, class_mapping: dict) -> List[Dict]:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            class_mapping: 类别映射
            
        Returns:
            生成的标注列表
        """
        try:
            # 加载模型
            if not self.current_model:
                # 根据model_path判断是官方模型还是自定义模型
                if os.path.exists(self.model_path):
                    # 自定义模型
                    self.current_model = self.model_manager.load_custom_model(self.model_path)
                else:
                    # 官方模型（模型名称）
                    # 尝试从pretrained目录加载
                    # 这里需要解析model_path来获取版本和大小
                    # 假设model_path是类似"yolov8n"这样的格式
                    match = re.match(r'(yolov)(\d+)([a-z])', self.model_path)
                    if match:
                        prefix = match.group(1)
                        version_num = match.group(2)
                        size = match.group(3)
                        # 转换版本格式，例如yolov8 -> YOLOv8
                        version = f"YOLOv{version_num}"
                        # 尝试通过model_manager加载
                        self.current_model = self.model_manager.load_model(version, size)
                    if not self.current_model:
                        # 如果model_manager加载失败，尝试直接加载但设置正确的下载路径
                        from ultralytics import YOLO
                        
                        # 设置环境变量，指定模型下载路径
                        pretrained_dir = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
                        os.makedirs(pretrained_dir, exist_ok=True)
                        
                        original_hub_dir = os.environ.get('YOLO_HUB_DIR')
                        os.environ['YOLO_HUB_DIR'] = pretrained_dir
                        
                        try:
                            self.current_model = YOLO(self.model_path)
                        finally:
                            # 恢复原始环境变量
                            if original_hub_dir:
                                os.environ['YOLO_HUB_DIR'] = original_hub_dir
                            else:
                                del os.environ['YOLO_HUB_DIR']
            
            if not self.current_model:
                return []
            
            # 进行推理
            result = self.model_manager.infer(
                self.current_model, image_path, conf_threshold, iou_threshold
            )
            
            if not result:
                return []
            
            # 生成标注
            annotations = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 计算宽度和高度
                width = x2 - x1
                height = y2 - y1
                
                # 应用类别映射
                mapped_class_id = class_mapping.get(class_id, class_id)
                
                # 创建标注
                annotation = {
                    'type': 'bbox',
                    'class_id': mapped_class_id,
                    'data': {
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height
                    }
                }
                annotations.append(annotation)
            
            return annotations
        except Exception as e:
            print(f"Error processing image: {e}")
            return []
    
    def unload_model(self):
        """卸载模型"""
        model_manager.unload_all_models()
        self.current_model = None
        self.model_info = {}


class BatchLabelingThread(QThread):
    """批量标注线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, int, int)  # 已处理数量（含失败）, 总数量, 已标注数量
    image_processed = pyqtSignal(str, int)  # 图像路径, 生成的标注数量
    # 是否成功, 消息, 是否被取消, 成功处理数, 失败数。取消单独一个位——它不是失败，
    # 以前混在 success=False 里，界面就只能弹一个红色的「批量标注失败」。
    # 两个计数也要跟着走：manager 以前拿 len(images) 当「处理了多少张」上报，
    # 那是「打算处理多少张」，中途失败的、被取消没跑到的，全被算成了成功。
    batch_completed = pyqtSignal(bool, str, bool, int, int)
    
    def __init__(self, plan, model_manager, config: Dict):
        super().__init__()
        self.plan = plan
        self.model_manager = model_manager
        self.images = list(plan.images)
        self.config = config
        # 标注器在 run() 里才建：创建它、把 .pt 读进显存/内存，都是几百毫秒起步的活，
        # 在界面线程上做就是明晃晃的一次卡顿。
        self.auto_labeler = None
        self._is_running = True
        self._is_paused = False
        # 收工时的真实战果，manager 收线程时按这两个数上报
        self.processed_count = 0
        self.failed_count = 0

    def _resolve_local_model_path(self) -> Optional[str]:
        """找出这次要用的本地权重；找不到就返回 None——绝不去网上取。

        批量这条路只认磁盘上已经存在的权重（.pt / .pth 都收——训练导出的检查点
        常常是 .pth，把它挡在门外只会逼用户去改扩展名）。设置里选的如果是
        「yolov8n」这种官方名字，只在本地预训练目录里按名字找同名文件；找不到就
        当作没有，直接报错。

        这里刻意不碰 model_manager.load_model()：那个方法在本地文件不存在时会
        转头执行 `YOTO(model_name)` 去联网下载一个 COCO 通用模型——用户按下的是
        「开始批量标注」，不是「下载一个我没选过的模型再拿它改我的标注」。
        """
        candidate = (self.plan.model_path or '').strip()
        if not candidate:
            return None

        # 已经是一个存在的本地权重：直接用
        if candidate.lower().endswith(('.pt', '.pth')) and os.path.isfile(candidate):
            return candidate

        # 官方名字（yolov8n / yolo11s …）：只做一次纯路径推算，看本地有没有下过
        match = re.match(r'^yolo(?:v)?(\d+)([a-z])$', candidate.lower())
        if match and self.model_manager is not None:
            version = f"YOLOv{match.group(1)}"
            size = match.group(2)
            try:
                local_path = self.model_manager.get_model_path(
                    version, size, self.plan.model_task or 'detect',
                )
            except Exception:
                return None
            if local_path and os.path.isfile(str(local_path)):
                return str(local_path)

        return None

    def _prepare_labeler(self):
        """在后台线程里建标注器并把模型读进来。失败返回 (None, 原因)。"""
        model_path = self._resolve_local_model_path()
        if not model_path:
            return None, (
                f"找不到本地模型文件：{self.plan.model_path or '（未指定）'}。\n"
                "请在「自动标注设置 → YOLO 检测」里选一个已经下载好的 .pt 模型。"
            )

        labeler = AutoLabeler(model_path, self.model_manager)
        # 一律按「自定义模型」加载：这条分支只会从给定路径读文件，不会联网。
        loaded = labeler.load_model({
            'model_version': '',
            'model_size': '',
            'model_source': 'custom',
            'model_task': self.plan.model_task or 'detect',
            'custom_model_path': model_path,
            'class_mappings': self.plan.class_mapping,
        })
        if not loaded or labeler.current_model is None:
            return None, f"模型加载失败：{model_path}"

        return labeler, ""

    def run(self):
        """运行批量标注：建标注器、加载模型、逐图处理，全都在这个后台线程里。"""
        total = len(self.images)
        processed = 0
        failed = 0
        labeled = 0

        try:
            self.auto_labeler, load_error = self._prepare_labeler()

            # 模型加载期间用户就点了取消：别硬着头皮再跑一整批
            if not self._is_running:
                self._finish(True, "已取消，模型还没加载完就停下了", True, 0, 0)
                return

            if self.auto_labeler is None:
                self._finish(False, load_error, False, 0, 0)
                return

            for image in self.images:
                if not self._is_running:
                    break

                # 检查是否暂停
                while self._is_paused:
                    time.sleep(0.1)
                    if not self._is_running:
                        break

                if not self._is_running:
                    break

                # 处理图像
                image_path = image.get('storage_path', '')
                image_id = image.get('id', 0)

                # 文件没了（被移走、被删、盘没挂上）：记一笔失败接着跑下一张。
                # 以前这里是「什么都不做」——既不算成功也不算失败，进度条就永远
                # 停在差几张的地方，收尾还照样报「全部完成」。
                if not image_path or not os.path.exists(image_path):
                    failed += 1
                    self.progress_updated.emit(processed + failed, total, labeled)
                    self.image_processed.emit(image_path, 0)
                    time.sleep(0.01)
                    continue

                annotations = self.auto_labeler.process_single_image(
                    image_path, image_id, self.config
                )

                overwrite = self.config.get('overwrite_labels', False)
                # 一个目标都没检出、又是覆盖模式：这就是「这张图上没东西」的结论，
                # 旧标注得跟着清掉。以前空结果直接跳过保存，用户勾了「覆盖」，
                # 上一轮的框却原封不动留在库里。不覆盖时当然一个字都不许动。
                if annotations or overwrite:
                    self.auto_labeler.save_annotations(
                        annotations, image_id, overwrite
                    )
                    labeled += len(annotations)

                # 发送信号
                processed += 1
                self.progress_updated.emit(processed + failed, total, labeled)
                self.image_processed.emit(image_path, len(annotations))

                # 避免CPU占用过高
                time.sleep(0.01)

            # 完成
            if not self._is_running:
                # 用户主动喊停不是出错：照实说已经做了多少，不报错
                self._finish(
                    True,
                    f"已取消，已处理 {processed}/{total} 张图像，生成了 {labeled} 个标注"
                    + (f"，{failed} 张失败" if failed else ""),
                    True,
                    processed,
                    failed,
                )
            elif failed:
                # 有图片没跑成，就别说「完成」——照实报出成功几张、失败几张
                self._finish(
                    False,
                    f"批量标注结束：成功 {processed}/{total} 张，"
                    f"{failed} 张失败（图片文件不存在），生成了 {labeled} 个标注",
                    False,
                    processed,
                    failed,
                )
            else:
                self._finish(
                    True,
                    f"批量标注完成，处理了 {processed}/{total} 张图像，生成了 {labeled} 个标注",
                    False,
                    processed,
                    failed,
                )
        except Exception as e:
            self._finish(False, f"批量标注出错: {str(e)}", False, processed, failed)

    def _finish(self, success: bool, message: str, cancelled: bool,
                processed: int, failed: int):
        """收尾：把真实计数留在线程上，再连同结果一起发出去。"""
        self.processed_count = processed
        self.failed_count = failed
        self.batch_completed.emit(success, message, cancelled, processed, failed)
    
    def pause(self):
        """暂停"""
        self._is_paused = True
    
    def resume(self):
        """恢复"""
        self._is_paused = False
    
    def stop(self):
        """停止"""
        self._is_running = False
        self._is_paused = False


class BatchLabelingManager(QObject):
    """批量标注管理器：按一份确认过的 BatchPlan 跑一批，一次只跑一批。"""

    # 信号定义
    progress_updated = pyqtSignal(int, int, int, str)  # 进度, 当前, 总数, 图像名称
    batch_completed = pyqtSignal(bool, str, int, bool)  # 成功, 消息, 处理数量, 是否取消

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_thread = None
        self.auto_labeler = None
        self.model_manager = None
        self.images = []
        self.plan = None
        # 上一批的真实战果（线程数出来的，不是「打算处理多少张」）
        self.processed_count = 0
        self.failed_count = 0

    def start_batch_processing(self, plan, model_manager) -> bool:
        """按快照开跑；已经有一批在跑就直接拒绝。返回「这次是不是真的启动了」。

        这个方法是界面线程调的，所以它只做三件轻活：记下快照、接好线程、把线程点着，
        然后立刻返回。**不在这里创建 AutoLabeler，也不在这里 load_model** ——
        读一个几十上百 MB 的 .pt 要几百毫秒到几秒，在界面线程上做就是一次肉眼可见的
        卡死：用户刚点完「开始批量标注」，窗口就白了。那些活全在 BatchLabelingThread.run() 里。

        参数只有一份快照（BatchPlan），不再是七个散装参数——确认框上给用户看的
        和这里交给线程的，从此是同一个对象，不可能对不上。

        `is_running()` 这道闸是防双击的第二层（第一层在页面上）：两批同时往
        同一批图片里写标注，谁覆盖谁完全看运气。
        """
        if self.is_running():
            return False

        try:
            self.model_manager = model_manager
            self.plan = plan
            self.images = list(plan.images)
            self.auto_labeler = None  # 由线程自己建；线程收工时再取回来

            # 覆盖与否只能来自用户确认过的快照。
            # 这里以前硬编码 overwrite_labels=True —— 确认框上写着「保留已有标注」，
            # 实际却把用户手标的框先删了个干净。
            config = {
                'conf_threshold': plan.conf,
                'iou_threshold': plan.iou,
                'class_mappings': plan.class_mapping,
                'overwrite_labels': plan.overwrite,
            }

            self.current_thread = BatchLabelingThread(plan, model_manager, config)

            # 连接信号
            self.current_thread.progress_updated.connect(self.on_progress_updated)
            self.current_thread.batch_completed.connect(self.on_batch_completed)
            # QThread 原生 finished 才代表线程真的退出了：只有那时候放引用才安全
            self.current_thread.finished.connect(self._on_thread_finished)

            # 启动线程
            self.current_thread.start()
            return True
        except Exception as e:
            self.current_thread = None
            self.batch_completed.emit(False, f"批量标注启动失败: {e}", 0, False)
            return False

    def on_progress_updated(self, processed: int, total: int, labeled: int):
        """进度更新回调"""
        current_image = self.images[processed-1] if 0 < processed <= len(self.images) else {}
        image_name = os.path.basename(current_image.get('storage_path', ''))
        progress = int((processed / total) * 100) if total else 0
        self.progress_updated.emit(progress, processed, total, image_name)

    def on_batch_completed(self, success: bool, message: str, cancelled: bool,
                           processed_count: int = 0, failed_count: int = 0):
        """批量完成回调。

        上报的是线程真正数出来的成功张数。以前这里写的是 `len(self.images)` ——
        那是「这一批打算处理多少张」：取消了、文件不在了、跑挂了，都照样按满勤
        上报，界面于是弹出「处理了 30 张图片」，而实际只跑了 3 张。
        """
        self.processed_count = processed_count
        self.failed_count = failed_count
        self.batch_completed.emit(success, message, processed_count, cancelled)

    def _on_thread_finished(self):
        """线程真正退出了，这里才是唯一安全释放引用的地方。"""
        thread = self.sender() or self.current_thread
        # 标注器是线程建的，收工时接回来，cleanup() 还要用它卸载模型
        if thread is not None:
            self.auto_labeler = getattr(thread, 'auto_labeler', None)
        if thread is self.current_thread:
            self.current_thread = None
        if thread is not None:
            thread.deleteLater()

    def request_cancel(self):
        """请求取消：只把标志放下去，立刻返回。

        绝对不能在这里 wait()——这是界面线程调的，一等就是整个窗口卡住，
        而「取消」恰恰是用户嫌它慢才点的。线程会在当前这张图跑完后自己退出。
        """
        if self.current_thread is not None:
            self.current_thread.stop()

    def pause(self):
        """暂停"""
        if self.current_thread:
            self.current_thread.pause()

    def resume(self):
        """恢复"""
        if self.current_thread:
            self.current_thread.resume()

    def stop(self):
        """停止并等线程退出。只给关窗/退出这种收尾路径用，不要在界面线程里调。"""
        if self.current_thread:
            self.current_thread.stop()
            self.current_thread.wait()

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return bool(self.current_thread and self.current_thread.isRunning())

    def cleanup(self):
        """清理资源"""
        self.stop()
        if self.auto_labeler is not None:
            self.auto_labeler.unload_model()
