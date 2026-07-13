# -*- coding: utf-8 -*-
"""缩略图要的「框 + 版本号」必须来自同一个读快照。

背景：导入页后台线程把整个项目的框和版本号读回来，缩略图按版本号缓存合成好的图。
两条 SELECT 如果各开一条连接、各拿一个快照，中间有人挪了一个框，读到的就是
「旧框 + 新版本号」——缩略图把旧框按新版本号缓存住，之后版本号再也对不上变化，
那张图的框永远停在旧位置。所以这里钉死：拿到的那一对，必须互相对得上。

这个文件只测 models.database.Database，不走 tests/_bootstrap（不需要 Qt，也不该
碰真实 data/EzYOLO.db），自己造临时数据库。数据库开 WAL：只有 WAL 允许写入方在
读事务还开着的时候提交，也只有这样才能真的把「两条 SELECT 中间插一次提交」跑出来。

运行：
    python -m pytest tests/test_bbox_preview_snapshot.py -q
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

APP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(APP_ROOT))

from models.database import Database  # noqa: E402

_OLD_BOX = {'x': 1, 'y': 1, 'width': 5, 'height': 5}
_NEW_BOX = {'x': 20, 'y': 10, 'width': 15, 'height': 8}


def _wal_db():
    """临时 sqlite 文件 + WAL：writer 可以在 reader 的读事务开着的时候提交。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = Database(db_path=tmp.name)
    conn = sqlite3.connect(tmp.name)
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # 落在文件里，之后每条连接都是 WAL
    finally:
        conn.close()
    return db, tmp.name


def _seed_one_boxed_image(db_path):
    """一个项目 / 一张图 / 一个框，直接插表。

    不用 Database.create_project()：它无视 db_path，永远在真实 APP_ROOT/projects
    下 mkdir（见 tests/test_database_sync_safety.py 的同一个坑）。
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO projects (name, description, type, classes, storage_path) "
            "VALUES (?, ?, ?, ?, ?)",
            ("snapshot", "", "detection", "[]", ""),
        )
        project_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO images (project_id, filename, storage_path, width, height, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, "one.jpg", "/nowhere/one.jpg", 40, 20, "annotated"),
        )
        image_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO annotations (image_id, project_id, class_id, class_name, type, data, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 'bbox', ?, ?, ?)",
            (image_id, project_id, 0, "人", json.dumps(_OLD_BOX),
             "2024-01-01T00:00:00", "2024-01-01T00:00:00"),
        )
        annotation_id = cursor.lastrowid
        conn.commit()
        return project_id, image_id, annotation_id
    finally:
        conn.close()


def _move_the_box(db_path, annotation_id):
    """另一条连接：把框挪走、换个类别，然后提交。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE annotations SET data = ?, class_id = ?, class_name = ?, updated_at = ? "
            "WHERE id = ?",
            (json.dumps(_NEW_BOX), 1, "车", "2024-06-01T12:00:00", annotation_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_snapshot_holds_when_a_box_moves_between_the_two_queries():
    """框查完、版本号还没查的那一瞬间有人挪了框：拿回来的必须还是「旧框 + 旧版本」。

    挪框不增减框，框数不变——版本号只靠 updated_at 区分。如果两条 SELECT 各看各的
    库，这里就会拿到旧框配新版本号：缩略图缓存住旧框，之后再也不失效。
    """
    db, db_path = _wal_db()
    project_id, image_id, annotation_id = _seed_one_boxed_image(db_path)

    version_before = db.get_project_annotation_versions(project_id)[image_id]

    real_read_bboxes = Database._read_bbox_previews

    def move_the_box_right_after_reading_them(self, cursor, pid):
        previews = real_read_bboxes(self, cursor, pid)
        _move_the_box(db_path, annotation_id)  # 恰好卡在两条 SELECT 中间
        return previews

    with patch.object(Database, "_read_bbox_previews", move_the_box_right_after_reading_them):
        previews, versions = db.get_project_bbox_preview_snapshot(project_id)

    assert previews[image_id][0]['data'] == _OLD_BOX, "读到的框不是快照那一刻的框"
    assert previews[image_id][0]['class_id'] == 0
    assert versions[image_id] == version_before, (
        "框是旧的、版本号却是新的——缩略图会把旧框按新版本号缓存住，从此不再刷新"
    )

    # 而库里现在确实已经是新的了：这次改动没有丢，下一轮刷新就会看到
    version_now = db.get_project_annotation_versions(project_id)[image_id]
    assert version_now != version_before, "第二条连接的提交根本没落库，这个测试没测到东西"
    assert db.get_project_bbox_previews(project_id)[image_id][0]['data'] == _NEW_BOX

    # 版本号变了 → 下一轮读回来的框也一定是新的，两者仍然配套
    fresh_previews, fresh_versions = db.get_project_bbox_preview_snapshot(project_id)
    assert fresh_previews[image_id][0]['data'] == _NEW_BOX
    assert fresh_versions[image_id] == version_now


def test_snapshot_matches_the_two_single_readers_when_nothing_changes():
    """没人写库的时候，快照读出来的东西跟两个老 API 一模一样。"""
    db, db_path = _wal_db()
    project_id, image_id, _annotation_id = _seed_one_boxed_image(db_path)

    previews, versions = db.get_project_bbox_preview_snapshot(project_id)

    assert previews == db.get_project_bbox_previews(project_id)
    assert versions == db.get_project_annotation_versions(project_id)
    assert previews[image_id][0] == {'type': 'bbox', 'class_id': 0, 'data': _OLD_BOX}


def test_writes_still_go_through_after_a_snapshot_read():
    """读快照用了显式事务：读完必须干净收尾，不能把库锁在那儿。"""
    db, db_path = _wal_db()
    project_id, image_id, annotation_id = _seed_one_boxed_image(db_path)

    db.get_project_bbox_preview_snapshot(project_id)

    assert db.update_annotation(annotation_id, data=_NEW_BOX) is True
    previews, _versions = db.get_project_bbox_preview_snapshot(project_id)
    assert previews[image_id][0]['data'] == _NEW_BOX


if __name__ == "__main__":
    test_snapshot_holds_when_a_box_moves_between_the_two_queries()
    test_snapshot_matches_the_two_single_readers_when_nothing_changes()
    test_writes_still_go_through_after_a_snapshot_read()
    print("bbox 快照一致性测试全部通过")
