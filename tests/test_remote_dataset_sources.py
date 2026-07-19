# -*- coding: utf-8 -*-
"""远程训练项目 source selection 的只读离线测试。"""

import _bootstrap  # noqa: F401

import json
import os
from pathlib import Path
import sys
import tempfile

from core.remote_training.dataset_sources import (  # noqa: E402
    RemoteDatasetPlanner,
    RemoteDatasetPlanningError,
)
from core.remote_training.snapshot import GeneratedSnapshotSource, SnapshotSource  # noqa: E402


class FakeDatabase:
    def __init__(self, project, images, annotations, group_images=None):
        self.project = project
        self.images = images
        self.annotations = annotations
        self.group_images = group_images or {}
        self.calls = []

    def get_project(self, project_id):
        self.calls.append(("project", project_id))
        return self.project

    def get_project_images(self, project_id):
        self.calls.append(("images", project_id))
        return list(self.images)

    def get_project_images_by_groups(self, project_id, group_ids):
        self.calls.append(("groups", tuple(group_ids)))
        rows = []
        for group_id in group_ids:
            rows.extend(self.group_images.get(group_id, []))
        return rows

    def get_image_annotations(self, image_id):
        self.calls.append(("annotations", image_id))
        return list(self.annotations.get(image_id, []))


def make_data(task="detect"):
    root = Path(tempfile.mkdtemp(prefix="ezyolo-remote-source-"))
    images = []
    annotations = {}
    for image_id in (1, 2, 3, 4):
        path = root / f"image-{image_id}.jpg"
        path.write_bytes(f"image-{image_id}".encode())
        images.append(
            {
                "id": image_id,
                "storage_path": str(path),
                "filename": path.name,
                "width": 100,
                "height": 50,
            }
        )
        annotations[image_id] = [
            (
                {
                    "class_id": 0,
                    "type": "bbox",
                    "data": {"x": 10, "y": 5, "width": 30, "height": 20},
                }
                if task == "detect"
                else {
                    "class_id": 0,
                    "type": "polygon",
                    "data": {
                        "points": [
                            {"x": 10, "y": 5},
                            {"x": 40, "y": 5},
                            {"x": 30, "y": 25},
                        ]
                    },
                }
            )
        ]
    project = {
        "storage_path": str(root),
        "classes": json.dumps([{"id": 0, "name": "person"}]),
    }
    return root, project, images, annotations


def assert_rejected(callable_object, *args, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except RemoteDatasetPlanningError:
        return
    raise AssertionError("预期输入被拒绝")


def test_random_selection_is_deterministic_read_only_and_generates_labels_in_memory():
    root, project, images, annotations = make_data()
    database = FakeDatabase(project, images, annotations)
    planner = RemoteDatasetPlanner(database)
    config = {"train_split": 50, "val_split": 25, "test_split": 25}
    first = planner.build(project_id=1, task_type="detect", runtime_config=config)
    second = planner.build(project_id=1, task_type="detect", runtime_config=config)

    assert first.split_counts == second.split_counts == {"train": 2, "val": 1, "test": 1}
    assert first.allowed_roots == (root.resolve(),)
    assert first.layout == {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    assert all(
        isinstance(source, (SnapshotSource, GeneratedSnapshotSource))
        for source in first.sources
    )
    labels = [source for source in first.sources if isinstance(source, GeneratedSnapshotSource)]
    assert labels and all(source.payload_path.startswith("labels/") for source in labels)
    assert all(b"0 " in source.content for source in labels)
    assert all(path.is_file() for path in root.iterdir())


def test_detect_and_segment_only_export_matching_annotation_shapes():
    root, project, images, annotations = make_data(task="segment")
    planner = RemoteDatasetPlanner(FakeDatabase(project, images, annotations))
    selection = planner.build(
        project_id=1,
        task_type="segment",
        runtime_config={"train_split": 50, "val_split": 50, "test_split": 0},
    )
    labels = [source.content.decode() for source in selection.sources if isinstance(source, GeneratedSnapshotSource)]
    assert labels and all(len(line.split()) == 7 for content in labels for line in content.splitlines())

    # 同样一组 polygon 用 detect 时没有可用 bbox，必须拒绝而不是导出错误格式。
    assert_rejected(
        planner.build,
        project_id=1,
        task_type="detect",
        runtime_config={"train_split": 50, "val_split": 50, "test_split": 0},
    )


def test_group_splits_must_be_nonempty_and_disjoint():
    root, project, images, annotations = make_data()
    database = FakeDatabase(
        project,
        images,
        annotations,
        group_images={1: images[:2], 2: images[2:]},
    )
    planner = RemoteDatasetPlanner(database)
    selection = planner.build(
        project_id=1,
        task_type="detect",
        runtime_config={
            "split_mode": "group",
            "train_group_ids": [1],
            "val_group_ids": [2],
            "test_group_ids": [],
        },
    )
    assert selection.split_counts == {"train": 2, "val": 2, "test": 0}

    database.group_images = {1: images[:2], 2: images[:2]}
    assert_rejected(
        planner.build,
        project_id=1,
        task_type="detect",
        runtime_config={
            "split_mode": "group",
            "train_group_ids": [1],
            "val_group_ids": [2],
            "test_group_ids": [],
        },
    )


def test_selection_rejects_outside_symlink_or_unlabelled_project_data():
    root, project, images, annotations = make_data()
    outside = Path(tempfile.mkdtemp(prefix="ezyolo-remote-outside-")) / "outside.jpg"
    outside.write_bytes(b"outside")
    images[0]["storage_path"] = str(outside)
    assert_rejected(
        RemoteDatasetPlanner(FakeDatabase(project, images, annotations)).build,
        project_id=1,
        task_type="detect",
        runtime_config={"train_split": 50, "val_split": 50, "test_split": 0},
    )

    root, project, images, annotations = make_data()
    try:
        os.symlink(root / "image-1.jpg", root / "image-link.jpg")
    except (NotImplementedError, OSError):
        return
    images[0]["storage_path"] = str(root / "image-link.jpg")
    assert_rejected(
        RemoteDatasetPlanner(FakeDatabase(project, images, annotations)).build,
        project_id=1,
        task_type="detect",
        runtime_config={"train_split": 50, "val_split": 50, "test_split": 0},
    )


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
