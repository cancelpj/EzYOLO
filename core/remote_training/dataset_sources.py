"""把 EzYOLO 项目数据库快照转换为远程训练的只读数据源选择。

本模块只读取项目、图片和标注事实；它不调用本机 TrainingThread、不会清空
datasets/，也不写回项目目录。标注文本在内存中受控生成，随后仅写入新的临时快照。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import stat
from typing import Any, Mapping, Protocol, Sequence

from .snapshot import GeneratedSnapshotSource, SnapshotSource


class RemoteDatasetPlanningError(ValueError):
    """现有项目不能安全形成远程训练数据快照。"""


class ProjectDatabaseLike(Protocol):
    def get_project(self, project_id: int) -> Mapping[str, Any] | None: ...

    def get_project_images(self, project_id: int) -> list[Mapping[str, Any]]: ...

    def get_project_images_by_groups(
        self,
        project_id: int,
        group_ids: list[int],
    ) -> list[Mapping[str, Any]]: ...

    def get_image_annotations(self, image_id: int) -> list[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class RemoteDatasetSelection:
    class_names: tuple[str, ...]
    layout: dict[str, str]
    sources: tuple[SnapshotSource | GeneratedSnapshotSource, ...]
    allowed_roots: tuple[Path, ...]
    split_counts: Mapping[str, int]


class RemoteDatasetPlanner:
    """从项目 DB 的当前事实建立确定性的 source selection。"""

    def __init__(self, database: ProjectDatabaseLike) -> None:
        self._database = database

    def build(
        self,
        *,
        project_id: int,
        task_type: str,
        runtime_config: Mapping[str, Any],
    ) -> RemoteDatasetSelection:
        if type(project_id) is not int or project_id <= 0:
            raise RemoteDatasetPlanningError("项目无效，无法准备远程训练数据")
        if task_type not in {"detect", "segment"}:
            raise RemoteDatasetPlanningError("远程训练首版只支持目标检测和实例分割")
        project = self._database.get_project(project_id)
        if not project:
            raise RemoteDatasetPlanningError("找不到当前项目，无法准备远程训练数据")
        storage_root = _project_storage_root(project)
        class_names = _project_class_names(project)
        images = list(self._database.get_project_images(project_id))
        if not images:
            raise RemoteDatasetPlanningError("项目中还没有图片，无法开始远程训练")

        split_images = self._split_images(
            project_id=project_id,
            images=images,
            runtime_config=runtime_config,
        )
        sources: list[SnapshotSource | GeneratedSnapshotSource] = []
        labelled_counts = {"train": 0, "val": 0, "test": 0}
        for split, split_rows in split_images.items():
            for image in split_rows:
                image_id = _positive_image_id(image)
                source_path = _project_image_path(image, storage_root)
                payload_stem = _payload_stem(image_id, source_path)
                sources.append(
                    SnapshotSource(
                        source_path=source_path,
                        payload_path=f"images/{split}/{payload_stem}{source_path.suffix.lower()}",
                    )
                )
                label_bytes = _yolo_label_bytes(
                    annotations=self._database.get_image_annotations(image_id),
                    image=image,
                    task_type=task_type,
                    class_count=len(class_names),
                )
                if label_bytes:
                    labelled_counts[split] += 1
                sources.append(
                    GeneratedSnapshotSource(
                        payload_path=f"labels/{split}/{payload_stem}.txt",
                        content=label_bytes,
                    )
                )

        if not split_images["train"] or not split_images["val"]:
            raise RemoteDatasetPlanningError("训练集和验证集都至少需要一张图片")
        if not labelled_counts["train"] or not labelled_counts["val"]:
            raise RemoteDatasetPlanningError(
                "训练集和验证集都至少需要一张可用于当前任务类型的已标注图片"
            )
        layout = {
            "train": "images/train",
            "val": "images/val",
        }
        if split_images["test"]:
            layout["test"] = "images/test"
        return RemoteDatasetSelection(
            class_names=class_names,
            layout=layout,
            sources=tuple(sources),
            allowed_roots=(storage_root,),
            split_counts={split: len(rows) for split, rows in split_images.items()},
        )

    def _split_images(
        self,
        *,
        project_id: int,
        images: list[Mapping[str, Any]],
        runtime_config: Mapping[str, Any],
    ) -> dict[str, list[Mapping[str, Any]]]:
        split_mode = runtime_config.get("split_mode", "random")
        if split_mode == "group":
            splits = {
                "train": self._database.get_project_images_by_groups(
                    project_id,
                    _group_ids(
                        runtime_config.get("train_group_ids", []),
                        required=True,
                    ),
                ),
                "val": self._database.get_project_images_by_groups(
                    project_id,
                    _group_ids(
                        runtime_config.get("val_group_ids", []),
                        required=True,
                    ),
                ),
                "test": self._database.get_project_images_by_groups(
                    project_id,
                    _group_ids(
                        runtime_config.get("test_group_ids", []),
                        required=False,
                    ),
                ),
            }
            _validate_disjoint_group_splits(splits)
            return splits
        if split_mode != "random":
            raise RemoteDatasetPlanningError("数据划分方式无效")
        return _random_split(images, runtime_config)


def _project_storage_root(project: Mapping[str, Any]) -> Path:
    raw = project.get("storage_path")
    if not isinstance(raw, str) or not raw:
        raise RemoteDatasetPlanningError("项目没有安全的数据目录")
    root = Path(raw)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise RemoteDatasetPlanningError("项目数据目录不存在") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RemoteDatasetPlanningError("项目数据目录不是普通目录")
    return root.resolve(strict=True)


def _project_class_names(project: Mapping[str, Any]) -> tuple[str, ...]:
    raw = project.get("classes", "[]")
    if isinstance(raw, str):
        try:
            classes = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RemoteDatasetPlanningError("项目类别数据无法读取") from exc
    else:
        classes = raw
    if not isinstance(classes, list):
        raise RemoteDatasetPlanningError("项目类别数据格式不正确")
    names = []
    for item in classes:
        if not isinstance(item, Mapping):
            raise RemoteDatasetPlanningError("项目类别数据格式不正确")
        name = item.get("name")
        if (
            not isinstance(name, str)
            or not name.strip()
            or any(ord(char) < 32 for char in name)
        ):
            raise RemoteDatasetPlanningError("项目类别名称无效")
        names.append(name)
    if not names or len(set(names)) != len(names):
        raise RemoteDatasetPlanningError("项目必须包含不重复的有效类别")
    return tuple(names)


def _random_split(
    images: list[Mapping[str, Any]],
    runtime_config: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    ordered = list(images)
    random.Random(42).shuffle(ordered)
    total = len(ordered)
    train_ratio = _percentage(runtime_config.get("train_split", 80), "训练集比例")
    val_ratio = _percentage(runtime_config.get("val_split", 10), "验证集比例")
    test_ratio = _percentage(runtime_config.get("test_split", 10), "测试集比例")
    if train_ratio + val_ratio + test_ratio != 100:
        raise RemoteDatasetPlanningError("数据集划分比例总和必须等于 100%")

    train_count = max(1, int(total * train_ratio / 100))
    val_count = max(1, int(total * val_ratio / 100))
    if total < 3:
        train_count, val_count = total, 0
    elif train_count + val_count > total:
        train_count, val_count = total - 1, 1
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count : train_count + val_count],
        "test": ordered[train_count + val_count :],
    }


def _group_ids(value: object, *, required: bool) -> list[int]:
    if not isinstance(value, list):
        raise RemoteDatasetPlanningError("按分组划分时，分组选择无效")
    if required and not value:
        raise RemoteDatasetPlanningError("按分组划分时，训练集和验证集必须各选至少一个分组")
    if any(type(group_id) is not int or group_id < 0 for group_id in value):
        raise RemoteDatasetPlanningError("分组选择无效")
    return list(value)


def _validate_disjoint_group_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    seen: set[int] = set()
    for split in ("train", "val", "test"):
        for image in splits[split]:
            image_id = _positive_image_id(image)
            if image_id in seen:
                raise RemoteDatasetPlanningError("同一图片不能同时属于多个数据集划分")
            seen.add(image_id)
    if not splits["train"] or not splits["val"]:
        raise RemoteDatasetPlanningError("训练集和验证集都至少需要一张图片")


def _percentage(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > 100:
        raise RemoteDatasetPlanningError(f"{label}无效")
    return value


def _positive_image_id(image: Mapping[str, Any]) -> int:
    image_id = image.get("id")
    if type(image_id) is not int or image_id <= 0:
        raise RemoteDatasetPlanningError("项目图片记录无效")
    return image_id


def _project_image_path(image: Mapping[str, Any], storage_root: Path) -> Path:
    raw = image.get("storage_path")
    if not isinstance(raw, str) or not raw:
        raise RemoteDatasetPlanningError("有图片缺少本机文件路径")
    source = Path(raw)
    try:
        source_stat = source.lstat()
        resolved = source.resolve(strict=True)
        resolved.relative_to(storage_root)
    except (OSError, ValueError) as exc:
        raise RemoteDatasetPlanningError("图片不在当前项目的安全数据目录中") from exc
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise RemoteDatasetPlanningError("项目图片不是普通文件")
    if not source.suffix:
        raise RemoteDatasetPlanningError("项目图片没有可识别的格式")
    return resolved


def _payload_stem(image_id: int, source_path: Path) -> str:
    # 只用数据库 ID，避免相同文件名或显示名中的空白/Unicode 影响远程 payload 路径。
    if not source_path.suffix:
        raise RemoteDatasetPlanningError("项目图片没有可识别的格式")
    return f"{image_id:08d}"


def _yolo_label_bytes(
    *,
    annotations: Sequence[Mapping[str, Any]],
    image: Mapping[str, Any],
    task_type: str,
    class_count: int,
) -> bytes:
    width = _positive_dimension(image.get("width"), "图片宽度")
    height = _positive_dimension(image.get("height"), "图片高度")
    lines = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        class_id = annotation.get("class_id")
        if type(class_id) is not int or not 0 <= class_id < class_count:
            continue
        data = annotation.get("data")
        if not isinstance(data, Mapping):
            continue
        annotation_type = annotation.get("type")
        if task_type == "detect" and annotation_type == "bbox":
            line = _bbox_line(class_id, data, width, height)
        elif task_type == "segment" and annotation_type == "polygon":
            line = _polygon_line(class_id, data, width, height)
        else:
            continue
        if line:
            lines.append(line)
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _positive_dimension(value: object, label: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise RemoteDatasetPlanningError(f"{label}无效")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise RemoteDatasetPlanningError(f"{label}无效")
    return numeric


def _bbox_line(
    class_id: int,
    data: Mapping[str, Any],
    image_width: float,
    image_height: float,
) -> str | None:
    try:
        x = _finite_number(data.get("x"))
        y = _finite_number(data.get("y"))
        width = _finite_number(data.get("width"))
        height = _finite_number(data.get("height"))
    except RemoteDatasetPlanningError:
        return None
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return None
    return (
        f"{class_id} {(x + width / 2) / image_width:.6f} "
        f"{(y + height / 2) / image_height:.6f} "
        f"{width / image_width:.6f} {height / image_height:.6f}"
    )


def _polygon_line(
    class_id: int,
    data: Mapping[str, Any],
    image_width: float,
    image_height: float,
) -> str | None:
    points = data.get("points")
    if not isinstance(points, list) or len(points) < 3:
        return None
    parts = [str(class_id)]
    try:
        for point in points:
            if not isinstance(point, Mapping):
                return None
            x = _finite_number(point.get("x"))
            y = _finite_number(point.get("y"))
            if not 0 <= x <= image_width or not 0 <= y <= image_height:
                return None
            parts.extend((f"{x / image_width:.6f}", f"{y / image_height:.6f}"))
    except RemoteDatasetPlanningError:
        return None
    return " ".join(parts)


def _finite_number(value: object) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise RemoteDatasetPlanningError("标注坐标无效")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RemoteDatasetPlanningError("标注坐标无效")
    return numeric
