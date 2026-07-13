# -*- coding: utf-8 -*-
"""U6：projects.display_name_rule 列的数据库迁移。

覆盖的真实问题：
    1. 老数据库文件没有这一列，打开时不能报错、更不能丢项目/图片数据。
    2. 新列要能被 update_project / get_project 正常读写。
    3. 关掉软件重开（重新实例化 Database 指向同一个文件），规则还在。

运行：
    python tests/test_display_name_migration.py
"""

import _bootstrap  # noqa: F401  必须第一个导入（把 QSettings/db 换成临时的）

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from gui.display_names import default_rule, parse_display_name_rule
from models.database import Database


def _make_old_schema_db() -> str:
    """手搭一个「加这一列之前」的数据库文件：projects 表没有 display_name_rule。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
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
        cursor.execute(
            "INSERT INTO images (project_id, filename, original_path, storage_path, width, height) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, "cat.jpg", "/home/user/photos/cat.jpg", "/tmp/old-project/cat.jpg", 640, 480),
        )
        conn.commit()
    finally:
        conn.close()
    return tmp.name


def test_opening_old_database_adds_column_without_losing_data():
    """老库打开一次：projects/images 里原有的行必须原封不动还在。"""
    db_path = _make_old_schema_db()

    db = Database(db_path=db_path)

    projects = db.get_all_projects()
    assert len(projects) == 1, projects
    project = projects[0]
    assert project['name'] == "老项目"
    assert project['description'] == "迁移前就有的项目"
    assert 'display_name_rule' in project, "迁移后新列必须存在"
    assert project['display_name_rule'] is None, "老数据没设置过规则，新列应该是 NULL"

    images = db.get_project_images(project['id'])
    assert len(images) == 1, images
    assert images[0]['filename'] == "cat.jpg"
    assert images[0]['original_path'] == "/home/user/photos/cat.jpg"

    # 旧项目没设置过规则：解析出来必须等价于「保留原名」，行为和升级前完全一样
    assert parse_display_name_rule(project['display_name_rule']) == default_rule()

    Path(db_path).unlink(missing_ok=True)


def test_reopening_old_database_is_idempotent():
    """老库连续打开两次（模拟软件重启两次）不应该报错或重复迁移。"""
    db_path = _make_old_schema_db()

    Database(db_path=db_path)
    db2 = Database(db_path=db_path)

    projects = db2.get_all_projects()
    assert len(projects) == 1
    assert projects[0]['display_name_rule'] is None

    Path(db_path).unlink(missing_ok=True)


def test_update_project_persists_display_name_rule_as_json():
    """update_project 写规则，get_project 读回来是能被 parse_display_name_rule 解析的 JSON。"""
    project_id = _bootstrap.create_temp_project(name="规则测试")
    rule = {"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 6}

    ok = _bootstrap.db.update_project(project_id, display_name_rule=rule)
    assert ok

    stored = _bootstrap.db.get_project(project_id)
    assert isinstance(stored['display_name_rule'], str)
    assert json.loads(stored['display_name_rule']) == rule
    assert parse_display_name_rule(stored['display_name_rule']) == rule

    _bootstrap.db.delete_project(project_id)


def test_rule_survives_closing_and_reopening_the_database():
    """关掉软件重开：新建一个指向同一个文件的 Database 实例，规则还能读出来。"""
    db_path = _make_old_schema_db()
    rule = {"mode": "source"}

    db_a = Database(db_path=db_path)
    project_id = db_a.get_all_projects()[0]['id']
    db_a.update_project(project_id, display_name_rule=rule)

    # 模拟重新启动软件：换一个全新的 Database 实例，指向同一个文件
    db_b = Database(db_path=db_path)
    reopened = db_b.get_project(project_id)
    assert parse_display_name_rule(reopened['display_name_rule']) == rule

    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(dict(globals())))
