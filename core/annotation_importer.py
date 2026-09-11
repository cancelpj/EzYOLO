# -*- coding: utf-8 -*-
"""
标注导入器
支持YOLO、COCO、VOC等格式的标注导入
"""

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

import yaml
from models.database import db


class AnnotationImporter:
    """标注导入器"""
    
    def __init__(self, project_id: int, group_id: int = None):
        """
        初始化标注导入器
        
        Args:
            project_id: 项目ID
            group_id: 新导入图片的默认分组
        """
        self.project_id = project_id
        self.group_id = group_id
        self.project = db.get_project(project_id)
        if not self.project:
            raise ValueError(f"项目 {project_id} 不存在")
        
        # 获取项目类别信息
        self.classes = json.loads(self.project['classes']) if self.project['classes'] else []
        self.class_map = {cls['name']: cls['id'] for cls in self.classes}
    
    def import_yolo_annotations(self, labels_dir: str, images_dir: str = None, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入YOLO格式的标注
        
        Args:
            labels_dir: YOLO标签文件夹路径
            images_dir: 对应的图像文件夹路径（可选）
            overwrite: 是否覆盖已有的标注
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            raise ValueError(f"标签文件夹不存在: {labels_dir}")
        
        if images_dir:
            images_dir = Path(images_dir)
        
        imported = 0
        skipped = 0
        
        # 获取所有txt文件
        txt_files = list(labels_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            try:
                # 查找对应的图像文件
                image_file = self._find_corresponding_image(txt_file, images_dir)
                if not image_file:
                    print(f"未找到对应的图像文件: {txt_file}")
                    skipped += 1
                    continue
                
                # 获取图像信息
                image_info = self._get_image_info(str(image_file))
                if not image_info:
                    skipped += 1
                    continue
                
                # 查找数据库中的图像记录
                image_record = self._find_image_record(str(image_file))
                if not image_record:
                    # 如果图像不存在，先导入图像
                    image_record = self._import_image_if_needed(str(image_file))
                    if not image_record:
                        skipped += 1
                        continue
                
                # 检查是否已经有标注
                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    # 如果已经有标注且不覆盖，跳过
                    skipped += 1
                    continue
                
                # 如果需要覆盖，先删除所有原标注
                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])
                
                # 读取YOLO标注
                annotations = self._parse_yolo_file(txt_file, image_info)
                is_empty_label_file = txt_file.read_text(encoding='utf-8').strip() == ""
                
                # 导入标注
                for ann in annotations:
                    # 根据任务类型设置标注类型
                    project_task = self.project.get('type', 'detect')
                    if project_task == 'segment' and 'points' in ann['data']:
                        annotation_type = 'polygon'
                    elif project_task == 'pose' and 'keypoints' in ann['data']:
                        annotation_type = 'keypoint'
                    elif project_task == 'obb' and 'angle' in ann['data']:
                        annotation_type = 'obb'
                    else:
                        annotation_type = 'bbox'
                    
                    db.add_annotation(
                        image_id=image_record['id'],
                        project_id=self.project_id,
                        class_id=ann['class_id'],
                        class_name=ann['class_name'],
                        annotation_type=annotation_type,
                        data=ann['data']
                    )

                # 空标签文件代表负样本，应视为“已标注”而不是继续保持 pending
                if not annotations and is_empty_label_file:
                    db.update_image_status(image_record['id'], 'annotated')
                
                imported += len(annotations)
                
            except Exception as e:
                print(f"导入YOLO标注失败 {txt_file}: {e}")
                skipped += 1
        
        return imported, skipped
    
    def import_coco_annotations(self, coco_file: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入COCO格式的标注
        
        Args:
            coco_file: COCO JSON文件路径
            overwrite: 是否覆盖已有的标注
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        coco_file = Path(coco_file)
        if not coco_file.exists():
            raise ValueError(f"COCO文件不存在: {coco_file}")
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
        except Exception as e:
            raise ValueError(f"读取COCO文件失败: {e}")
        
        imported = 0
        skipped = 0
        
        # 解析COCO数据
        images = {img['id']: img for img in coco_data.get('images', [])}
        categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        annotations = coco_data.get('annotations', [])
        
        for ann in annotations:
            try:
                # 获取图像信息
                image_id = ann.get('image_id')
                if image_id not in images:
                    skipped += 1
                    continue
                
                image_info = images[image_id]
                
                # 查找数据库中的图像记录
                image_record = self._find_image_record_by_filename(image_info['file_name'])
                if not image_record:
                    # 如果图像不存在，先导入图像
                    image_record = self._import_image_if_needed(image_info['file_name'])
                    if not image_record:
                        skipped += 1
                        continue
                
                # 检查是否已经有标注
                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    # 如果已经有标注且不覆盖，跳过
                    skipped += 1
                    continue
                
                # 如果需要覆盖，先删除所有原标注
                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])
                
                # 获取类别信息
                category_id = ann.get('category_id')
                if category_id not in categories:
                    skipped += 1
                    continue
                
                category = categories[category_id]
                
                # 解析标注数据
                if 'bbox' in ann:
                    # 边界框标注
                    bbox = ann['bbox']  # [x, y, width, height]
                    data = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3]
                    }
                    
                    db.add_annotation(
                        image_id=image_record['id'],
                        project_id=self.project_id,
                        class_id=category_id,
                        class_name=category['name'],
                        annotation_type='bbox',
                        data=data
                    )
                    
                    imported += 1
                
                if 'segmentation' in ann:
                    # 分割标注
                    segmentation = ann['segmentation']
                    if isinstance(segmentation, list):
                        # 多边形格式
                        data = {'points': segmentation}
                        
                        db.add_annotation(
                            image_id=image_record['id'],
                            project_id=self.project_id,
                            class_id=category_id,
                            class_name=category['name'],
                            annotation_type='polygon',
                            data=data
                        )
                        
                        imported += 1
                
            except Exception as e:
                print(f"导入COCO标注失败: {e}")
                skipped += 1
        
        return imported, skipped
    
    def import_voc_annotations(self, voc_dir: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入Pascal VOC格式的标注
        
        Args:
            voc_dir: VOC标注文件夹路径
            overwrite: 是否覆盖已有的标注
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        voc_dir = Path(voc_dir)
        if not voc_dir.exists():
            raise ValueError(f"VOC文件夹不存在: {voc_dir}")
        
        imported = 0
        skipped = 0
        
        # 获取所有XML文件
        xml_files = list(voc_dir.glob("*.xml"))
        
        for xml_file in xml_files:
            try:
                # 解析XML文件
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 获取图像文件名
                filename_elem = root.find('filename')
                if filename_elem is None:
                    skipped += 1
                    continue
                
                filename = filename_elem.text
                
                # 查找对应的图像文件
                image_record = self._find_image_record_by_filename(filename)
                if not image_record:
                    # 如果图像不存在，先导入图像
                    image_record = self._import_image_if_needed(filename)
                    if not image_record:
                        skipped += 1
                        continue
                
                # 检查是否已经有标注
                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    # 如果已经有标注且不覆盖，跳过
                    skipped += 1
                    continue
                
                # 如果需要覆盖，先删除所有原标注
                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])
                
                # 获取图像尺寸
                size_elem = root.find('size')
                width = int(size_elem.find('width').text) if size_elem is not None else 0
                height = int(size_elem.find('height').text) if size_elem is not None else 0
                
                # 解析标注对象
                for obj in root.findall('object'):
                    try:
                        # 获取类别信息
                        name_elem = obj.find('name')
                        if name_elem is None:
                            continue
                        
                        class_name = name_elem.text
                        class_id = self.class_map.get(class_name, 0)
                        
                        # 获取边界框信息
                        bbox_elem = obj.find('bndbox')
                        if bbox_elem is None:
                            continue
                        
                        xmin = int(bbox_elem.find('xmin').text)
                        ymin = int(bbox_elem.find('ymin').text)
                        xmax = int(bbox_elem.find('xmax').text)
                        ymax = int(bbox_elem.find('ymax').text)
                        
                        # 转换为YOLO格式
                        data = {
                            'x': xmin,
                            'y': ymin,
                            'width': xmax - xmin,
                            'height': ymax - ymin
                        }
                        
                        db.add_annotation(
                            image_id=image_record['id'],
                            project_id=self.project_id,
                            class_id=class_id,
                            class_name=class_name,
                            annotation_type='bbox',
                            data=data
                        )
                        
                        imported += 1
                        
                    except Exception as e:
                        print(f"解析VOC对象失败: {e}")
                        skipped += 1
                
            except Exception as e:
                print(f"导入VOC标注失败 {xml_file}: {e}")
                skipped += 1
        
        return imported, skipped
    
    def import_yolo_dataset(self, dataset_dir: str, auto_groups: bool = True,
                           overwrite: bool = False) -> Dict:
        """
        导入完整的 YOLO 数据集目录（图片 + 标注 + 分类）一次性导入。

        目录结构（两种都支持）：
            dataset/
                data.yaml            # 含 names: 分类定义
                images/train/*.jpg   images/val/*.jpg
                labels/train/*.txt   labels/val/*.txt
        或扁平形式：
            dataset/images/*.jpg  dataset/labels/*.txt

        Args:
            dataset_dir: 数据集根目录
            auto_groups: 是否按 images 下的子目录（train/val/test）自动建分组
            overwrite:    是否覆盖已存在的标注

        Returns:
            导入统计 dict
        """
        from core.import_manager import ImportManager

        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise ValueError(f"数据集目录不存在: {dataset_dir}")

        # 1) 解析 data.yaml 拿到分类，并合并落库
        new_classes = self._parse_yolo_dataset_yaml(dataset_dir)
        if new_classes is None:
            raise ValueError(
                "在数据集目录未找到 data.yaml，或其中没有可用的 names 分类定义。"
            )
        self._merge_and_save_classes(new_classes)

        # 2) 刷新内存里的类别信息（后续解析标注要靠它做 id->name 映射）
        project = db.get_project(self.project_id)
        self.classes = json.loads(project['classes']) if project['classes'] else []
        self.class_map = {c['name']: c['id'] for c in self.classes}

        # 3) 定位 images/labels 与各 split
        img_root = dataset_dir / 'images'
        lbl_root = dataset_dir / 'labels'
        splits = self._detect_yolo_splits(img_root, lbl_root)
        if not splits:
            raise ValueError(
                f"未在 {img_root} 找到图片。标准结构应为 images/train、images/val 或 images/*.jpg"
            )

        # 项目里已有的图片记录，用于去重，并避免逐张全表扫描（O(n) 而非 O(n²)）
        existing_imgs = db.get_project_images(self.project_id)
        existing_paths = {
            img.get('original_path')
            for img in existing_imgs
            if img.get('original_path')
        }
        record_cache: Dict[str, Dict] = {}
        for img in existing_imgs:
            if img.get('original_path'):
                record_cache[img['original_path']] = img
            record_cache[img['filename']] = img

        def _lookup_record(img_file_path: str, img_name: str):
            if img_file_path in record_cache:
                return record_cache[img_file_path]
            if img_name in record_cache:
                return record_cache[img_name]
            rec = self._find_image_record(img_file_path)
            if rec is not None:
                record_cache[img_file_path] = rec
                record_cache[img_name] = rec
            return rec

        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
        stats = {
            'classes': len(self.classes),
            'images_imported': 0,
            'images_skipped': 0,
            'annotations': 0,
            'labels_missing': 0,
            'splits': {},
        }

        for split_name, img_dir, lbl_dir in splits:
            group_id = None
            if auto_groups and split_name:
                group_id = db.create_image_group(self.project_id, split_name)

            importer = ImportManager(self.project_id, group_id=group_id)
            split_img = split_ann = split_skip = split_missing = 0

            image_files = []
            for ext in image_exts:
                image_files.extend(img_dir.glob(f"*{ext}"))
                image_files.extend(img_dir.glob(f"*{ext.upper()}"))
            image_files = sorted(set(image_files))

            for img_file in image_files:
                # 去重：同一 original_path 已存在则跳过图片导入（但仍可补标注）
                if str(img_file) in existing_paths:
                    image_record = _lookup_record(str(img_file), img_file.name)
                    split_skip += 1
                    stats['images_skipped'] += 1
                else:
                    if not importer.import_single_image(str(img_file)):
                        split_skip += 1
                        stats['images_skipped'] += 1
                        continue
                    image_record = _lookup_record(str(img_file), img_file.name)
                    if image_record:
                        existing_paths.add(str(img_file))
                    split_img += 1
                    stats['images_imported'] += 1

                if not image_record:
                    continue

                # 标注
                label_file = lbl_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    split_missing += 1
                    stats['labels_missing'] += 1
                    continue

                existing_anns = db.get_image_annotations(image_record['id'])
                if existing_anns and not overwrite:
                    # 已有标注且不覆盖：保留，仅标记已标注
                    db.update_image_status(image_record['id'], 'annotated')
                    continue
                if existing_anns and overwrite:
                    db.delete_image_annotations(image_record['id'])

                image_info = {
                    'width': image_record.get('width'),
                    'height': image_record.get('height'),
                }
                if not image_info['width'] or not image_info['height']:
                    info = self._get_image_info(str(img_file))
                    if info:
                        image_info = info

                anns = self._parse_yolo_file(label_file, image_info)
                project_task = self.project.get('type', 'detect')
                for ann in anns:
                    if project_task == 'segment' and 'points' in ann['data']:
                        atype = 'polygon'
                    elif project_task == 'pose' and 'keypoints' in ann['data']:
                        atype = 'keypoint'
                    elif project_task == 'obb' and 'angle' in ann['data']:
                        atype = 'obb'
                    else:
                        atype = 'bbox'
                    db.add_annotation(
                        image_id=image_record['id'],
                        project_id=self.project_id,
                        class_id=ann['class_id'],
                        class_name=ann['class_name'],
                        annotation_type=atype,
                        data=ann['data'],
                    )
                    split_ann += 1
                    stats['annotations'] += 1

                # 有标签文件即视为已标注（含空标签的负样本）
                db.update_image_status(image_record['id'], 'annotated')

            stats['splits'][split_name or 'all'] = {
                'images': split_img,
                'annotations': split_ann,
                'skipped': split_skip,
                'missing_labels': split_missing,
            }

        return stats

    def _parse_yolo_dataset_yaml(self, dataset_dir: Path) -> Optional[List[Dict]]:
        """在数据集目录找 data.yaml 并解析 names: 为类别列表。

        兼容 names 写成 dict（{0: OK, 1: NG}）或 list（[OK, NG]）两种形式。
        找不到 / 解析不出分类时返回 None。
        """
        yaml_path = dataset_dir / 'data.yaml'
        if not yaml_path.exists():
            alt = dataset_dir.parent / 'data.yaml'
            yaml_path = alt if alt.exists() else None
        if yaml_path is None:
            hits = list(dataset_dir.glob('**/data.yaml'))
            yaml_path = hits[0] if hits else None
        if yaml_path is None:
            return None

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"解析 data.yaml 失败: {e}")
            return None

        if not isinstance(data, dict):
            return None
        names = data.get('names')
        if not names:
            return None

        classes: List[Dict] = []
        if isinstance(names, dict):
            for k, v in names.items():
                try:
                    cid = int(k)
                except (ValueError, TypeError):
                    continue
                classes.append({'id': cid, 'name': str(v)})
        elif isinstance(names, list):
            for i, v in enumerate(names):
                classes.append({'id': i, 'name': str(v)})
        else:
            return None

        if not classes:
            return None

        # 补 color（按 id 确定性生成，重复导入颜色稳定）
        for c in classes:
            if 'color' not in c:
                rnd = random.Random(c['id'] + 1)
                c['color'] = f"#{rnd.randint(0, 0xFFFFFF):06x}"
        return classes

    def _merge_and_save_classes(self, new_classes: List[Dict]) -> None:
        """把 yaml 里的分类合并进项目类别并落库。

        按 id 合并：已存在的 id 以 yaml 的 name 为准（保留已有 color），
        新增的 id 直接补上。保证重复导入时类别稳定、不丢已有标注的映射。
        """
        existing = {c['id']: dict(c) for c in self.classes}
        for nc in new_classes:
            cid = nc['id']
            if cid in existing:
                existing[cid]['name'] = nc['name']
            else:
                existing[cid] = {
                    'id': cid,
                    'name': nc['name'],
                    'color': nc.get('color', '#808080'),
                }
        merged = [existing[k] for k in sorted(existing.keys())]
        db.update_project(self.project_id, classes=merged)
        self.classes = merged
        self.class_map = {c['name']: c['id'] for c in merged}

    def _detect_yolo_splits(self, img_root: Path, lbl_root: Path) -> List[Tuple[str, Path, Path]]:
        """探测 images 下的 train/val/test 子目录，或扁平 images/*.jpg。"""
        splits: List[Tuple[str, Path, Path]] = []
        if not img_root.is_dir():
            return splits

        sub_dirs = sorted(d.name for d in img_root.iterdir() if d.is_dir())
        if sub_dirs:
            for name in sub_dirs:
                img_dir = img_root / name
                lbl_dir = lbl_root / name
                if any(img_dir.glob('*.*')):
                    splits.append((name, img_dir, lbl_dir))
        elif any(img_root.glob('*.*')):
            splits.append(('', img_root, lbl_root))
        return splits

    def _find_image_record_by_path(self, image_path: str) -> Optional[Dict]:
        """按 original_path 或文件名查找数据库中的图像记录。"""
        filename = Path(image_path).name
        for img in db.get_project_images(self.project_id):
            if img.get('original_path') == image_path or img['filename'] == filename:
                return img
        return None

    def _parse_yolo_file(self, txt_file: Path, image_info: Dict) -> List[Dict]:
        """
        解析YOLO标注文件
        
        Args:
            txt_file: YOLO标签文件路径
            image_info: 图像信息
            
        Returns:
            标注列表
        """
        annotations = []
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 1:
                    continue
                
                # 解析类别ID
                class_id = int(parts[0])
                class_name = self._get_class_name_by_id(class_id)
                
                # 转换为像素坐标
                img_width = image_info['width']
                img_height = image_info['height']
                
                project_task = self.project.get('type', 'detect')
                
                if project_task == 'classify':
                    # 分类任务：只需要类别ID
                    annotations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'data': {
                            'class_id': class_id
                        }
                    })
                elif len(parts) >= 5:
                    # 解析基本的边界框信息
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    if project_task == 'segment' and len(parts) > 5:
                        # 分割任务：处理多边形点
                        points = []
                        for i in range(5, len(parts), 2):
                            if i + 1 < len(parts):
                                px = float(parts[i]) * img_width
                                py = float(parts[i + 1]) * img_height
                                points.append({'x': px, 'y': py})
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'points': points
                            }
                        })
                    elif project_task == 'pose' and len(parts) > 5:
                        # 姿态估计任务：处理关键点
                        keypoints = []
                        for i in range(5, len(parts), 3):
                            if i + 2 < len(parts):
                                kp_x = float(parts[i]) * img_width
                                kp_y = float(parts[i + 1]) * img_height
                                kp_v = float(parts[i + 2])
                                keypoints.append({
                                    'x': kp_x,
                                    'y': kp_y,
                                    'v': kp_v
                                })
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': (x_center - width / 2) * img_width,
                                'y': (y_center - height / 2) * img_height,
                                'width': width * img_width,
                                'height': height * img_height,
                                'keypoints': keypoints
                            }
                        })
                    elif project_task == 'obb' and len(parts) > 5:
                        # 旋转目标检测任务：处理角度
                        angle = float(parts[5]) if len(parts) > 5 else 0.0
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': (x_center - width / 2) * img_width,
                                'y': (y_center - height / 2) * img_height,
                                'width': width * img_width,
                                'height': height * img_height,
                                'angle': angle
                            }
                        })
                    else:
                        # 检测任务：处理边界框
                        x = (x_center - width / 2) * img_width
                        y = (y_center - height / 2) * img_height
                        w = width * img_width
                        h = height * img_height
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            }
                        })
            
        except Exception as e:
            print(f"解析YOLO文件失败 {txt_file}: {e}")
        
        return annotations
    
    def _find_corresponding_image(self, txt_file: Path, images_dir: Path = None) -> Optional[Path]:
        """
        查找对应的图像文件
        
        Args:
            txt_file: YOLO标签文件路径
            images_dir: 图像文件夹路径
            
        Returns:
            对应的图像文件路径，未找到返回None
        """
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 尝试相同目录
        base_name = txt_file.stem
        for ext in image_extensions:
            image_file = txt_file.parent / f"{base_name}{ext}"
            if image_file.exists():
                return image_file
        
        # 如果指定了图像目录，尝试在图像目录中查找
        if images_dir:
            for ext in image_extensions:
                image_file = images_dir / f"{base_name}{ext}"
                if image_file.exists():
                    return image_file
        
        return None
    
    def _get_image_info(self, image_path: str) -> Optional[Dict]:
        """
        获取图像信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像信息字典，失败返回None
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                return {
                    'width': width,
                    'height': height,
                    'format': img.format.lower() if img.format else 'unknown'
                }
        except Exception as e:
            print(f"获取图像信息失败 {image_path}: {e}")
            return None
    
    def _find_image_record(self, image_path: str) -> Optional[Dict]:
        """
        查找数据库中的图像记录
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像记录，未找到返回None
        """
        filename = Path(image_path).name
        images = db.get_project_images(self.project_id)
        
        for image in images:
            if image['filename'] == filename or image['original_path'] == image_path:
                return image
        
        return None
    
    def _find_image_record_by_filename(self, filename: str) -> Optional[Dict]:
        """
        根据文件名查找数据库中的图像记录
        
        Args:
            filename: 图像文件名
            
        Returns:
            图像记录，未找到返回None
        """
        images = db.get_project_images(self.project_id)
        
        for image in images:
            if image['filename'] == filename:
                return image
        
        return None
    
    def _import_image_if_needed(self, image_path: str) -> Optional[Dict]:
        """
        如果需要，导入图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像记录，失败返回None
        """
        from core.import_manager import ImportManager

        path = Path(image_path)
        if not path.is_file():
            return None

        import_manager = ImportManager(self.project_id, group_id=self.group_id)
        if not import_manager.import_single_image(str(path)):
            return None

        return self._find_image_record(path.name)
    
    def _get_class_name_by_id(self, class_id: int) -> str:
        """
        根据类别ID获取类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            类别名称
        """
        for cls in self.classes:
            if cls['id'] == class_id:
                return cls['name']
        return f"class_{class_id}"
