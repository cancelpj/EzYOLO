# -*- coding: utf-8 -*-
"""U6：created_at 相同时，图片列表 / 显示名称映射的排序必须确定。

覆盖的真实问题：
    批量导入时，一批图片的 created_at 很容易落在同一秒甚至同一微秒。此时纯按
    created_at 排序，SQLite 不保证平局顺序（依赖磁盘物理布局/查询计划），导致
    同一批图片在不同时刻查出的顺序不一样——进而让 build_project_display_names
    生成的「显示名称 <-> 图片」映射跟着漂移，同一张图前后两次显示名称不一致。

修复方式（models/database.py）：
    get_project_images / get_project_images_by_groups /
    get_project_images_by_class / get_negative_sample_images 全部把排序键从
    `created_at` 改成 `created_at, id`。id 是 AUTOINCREMENT，天然反映插入顺序
    且永远唯一，可以在 created_at 平局时提供确定的次级排序。

运行：
    python tests/test_display_name_ordering.py
"""

import _bootstrap  # noqa: F401  必须第一个导入（把 QSettings/db 换成临时的）

import sqlite3
import sys
import tempfile
from pathlib import Path

from gui.display_names import build_project_display_names
from models.database import Database

TIED_TIMESTAMP = "2024-01-01 00:00:00"


def _new_temp_db_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _make_project_with_tied_timestamps(db_path: str, count: int = 6):
    """建一个项目，插入 count 张图片，并强制它们 created_at 完全相同。"""
    db = Database(db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO projects (name, description, type, classes, storage_path) "
            "VALUES (?, ?, ?, ?, ?)",
            ("排序测试项目", "", "detect", "[]", "/tmp/order-test"),
        )
        project_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()

    image_ids = []
    for i in range(count):
        image_id = db.add_image(
            project_id,
            filename=f"img_{i:02d}.jpg",
            storage_path=f"/tmp/order-test/img_{i:02d}.jpg",
        )
        image_ids.append(image_id)

    # 插入之后统一改成同一个 created_at，模拟批量导入落在同一秒内的场景
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE images SET created_at = ? WHERE project_id = ?",
            (TIED_TIMESTAMP, project_id),
        )
        conn.commit()
    finally:
        conn.close()

    return project_id, image_ids


def _make_old_schema_db_with_tied_images(count: int = 6):
    """手搭一个「排序修复之前」的旧库：表结构和字段都是老的，没有 display_name_rule。"""
    db_path = _new_temp_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                type TEXT DEFAULT 'detection',
                classes TEXT DEFAULT '[]',
                status TEXT DEFAULT 'created',
                storage_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                dataset_id INTEGER,
                filename TEXT NOT NULL,
                original_path TEXT,
                storage_path TEXT,
                width INTEGER,
                height INTEGER,
                size INTEGER,
                format TEXT,
                status TEXT DEFAULT 'pending',
                annotated_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT INTO projects (name, description, type, classes, storage_path) "
            "VALUES (?, ?, ?, ?, ?)",
            ("老项目", "迁移前就有的项目", "detect", "[]", "/tmp/old-project"),
        )
        project_id = cursor.lastrowid
        image_ids = []
        for i in range(count):
            cursor.execute(
                "INSERT INTO images (project_id, filename, original_path, storage_path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (project_id, f"old_{i:02d}.jpg", f"/home/user/old_{i:02d}.jpg",
                 f"/tmp/old-project/old_{i:02d}.jpg", TIED_TIMESTAMP),
            )
            image_ids.append(cursor.lastrowid)
        conn.commit()
    finally:
        conn.close()
    return db_path, project_id, image_ids


def test_order_stable_across_repeated_reads_with_tied_created_at():
    """同一个 Database 实例，created_at 全相同时，多次读取顺序必须一致，且等于插入顺序（按 id）。"""
    db_path = _new_temp_db_path()
    project_id, image_ids = _make_project_with_tied_timestamps(db_path)
    db = Database(db_path=db_path)

    first = [img['id'] for img in db.get_project_images(project_id)]
    assert first == image_ids, f"顺序应等于插入顺序（按 id）：{first} != {image_ids}"

    for _ in range(5):
        again = [img['id'] for img in db.get_project_images(project_id)]
        assert again == first, "created_at 相同时，重复读取顺序不应漂移"

    Path(db_path).unlink(missing_ok=True)


def test_order_stable_after_closing_and_reopening_database():
    """关闭数据库（新建指向同一文件的 Database 实例）后，顺序依然稳定且不变。"""
    db_path = _new_temp_db_path()
    project_id, image_ids = _make_project_with_tied_timestamps(db_path)

    db_a = Database(db_path=db_path)
    order_a = [img['id'] for img in db_a.get_project_images(project_id)]

    # 模拟关掉软件重开：新建一个指向同一个文件的 Database 实例
    db_b = Database(db_path=db_path)
    order_b = [img['id'] for img in db_b.get_project_images(project_id)]

    assert order_a == image_ids
    assert order_b == image_ids
    assert order_a == order_b, "关闭重开数据库后，图片顺序不应变化"

    Path(db_path).unlink(missing_ok=True)


def test_display_name_mapping_does_not_drift_when_created_at_ties():
    """created_at 全相同时，显示名称映射（source 模式，按顺序编号）在重复读取/重开数据库后不漂移。"""
    db_path = _new_temp_db_path()
    project_id, image_ids = _make_project_with_tied_timestamps(db_path)
    rule = {"mode": "source"}

    snapshots = []
    for _ in range(3):
        db = Database(db_path=db_path)  # 每次都模拟一次重新打开
        images = db.get_project_images(project_id)
        assert [img['id'] for img in images] == image_ids
        snapshots.append(build_project_display_names(rule, images, project_name="排序测试项目"))

    for snapshot in snapshots[1:]:
        assert snapshot == snapshots[0], (
            f"created_at 相同时，显示名称映射不应漂移：{snapshot} != {snapshots[0]}"
        )

    Path(db_path).unlink(missing_ok=True)


def test_group_and_class_queries_also_stable_on_tied_created_at():
    """get_project_images_by_groups / get_project_images_by_class 同样要在平局时稳定。"""
    db_path = _new_temp_db_path()
    project_id, image_ids = _make_project_with_tied_timestamps(db_path, count=4)
    db = Database(db_path=db_path)

    # 全部未分组：group_ids=[0] 表示只要未分组（group_id IS NULL）的图片
    grouped_first = [img['id'] for img in db.get_project_images_by_groups(project_id, [0])]
    grouped_again = [img['id'] for img in db.get_project_images_by_groups(project_id, [0])]
    assert grouped_first == image_ids
    assert grouped_first == grouped_again

    for image_id in image_ids:
        db.add_annotation(
            image_id, project_id, class_id=0, class_name="cls0",
            annotation_type="bbox", data={"x": 0, "y": 0, "width": 1, "height": 1},
        )

    by_class_first = [img['id'] for img in db.get_project_images_by_class(project_id, 0)]
    by_class_again = [img['id'] for img in db.get_project_images_by_class(project_id, 0)]
    assert by_class_first == image_ids
    assert by_class_first == by_class_again

    Path(db_path).unlink(missing_ok=True)


def test_tied_created_at_order_stable_on_old_schema_database():
    """兼容旧数据库：老库文件（打开前无 display_name_rule 列）里 created_at 平局时，排序同样稳定。"""
    db_path, project_id, image_ids = _make_old_schema_db_with_tied_images()

    db = Database(db_path=db_path)  # 打开旧库会触发列迁移，但排序修复不依赖新列

    first = [img['id'] for img in db.get_project_images(project_id)]
    assert first == image_ids, f"旧库迁移后顺序应等于插入顺序（按 id）：{first} != {image_ids}"

    db_reopened = Database(db_path=db_path)
    again = [img['id'] for img in db_reopened.get_project_images(project_id)]
    assert again == first, "旧库重开后，顺序不应变化"

    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(dict(globals())))
