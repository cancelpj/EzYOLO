# -*- coding: utf-8 -*-
"""U6：数据导入页“管理 → 图片显示名称…”入口，以及 ImportPage / AnnotatePage 的接入。

覆盖的真实问题：
    1. 入口点得到，点了之后规则真的能落地成两个页面上的界面文字。
    2. 取消、或者规则本身不合法，都不能把规则悄悄写进数据库。
    3. 对话框预览只看前三张图；真正保存前必须对项目全部图片再校验一遍，
       不能让「预览没问题」掩盖「全量会冲突」。
    4. 两个页面共用同一份规则：改了之后互相同步，但标注页不能因此重新加载
       画布、丢掉缩放/锁定状态，也不能把当前选中的图片换掉。
    5. tooltip 永远留一条能找到真实文件的路：文件名、分辨率、来源路径。
    6. 这一切只改界面文字：filename / original_path / storage_path / 磁盘文件 /
       标注一个字节都不能变。

运行：
    python tests/test_display_name_pages.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from gui.display_names import (
    build_project_display_names as real_build_project_display_names,
    parse_display_name_rule,
)
from gui.pages.import_page import ImportPage
from gui.pages.annotate_page import AnnotatePage

_app = _bootstrap.app()
db = _bootstrap.db

_TMP_DIR = Path(tempfile.mkdtemp(prefix="ezyolo-display-name-pages-"))


def _make_project(name="显示名称接入测试项目"):
    return _bootstrap.create_temp_project(name=name, project_type="detect", classes=[])


def _seed_image(project_id, filename, original_path, width=640, height=480):
    return db.add_image(
        project_id, filename, str(_TMP_DIR / filename),
        width=width, height=height, original_path=original_path,
    )


def _open_annotate_page(project_id) -> AnnotatePage:
    """AnnotatePage.set_project 把项目数据加载丢给了一个后台 QThread；等它真正
    跑完（wait）再处理事件，跨线程信号才保证已经派发到位，列表才是满的。
    """
    page = AnnotatePage()
    page.set_project(project_id)
    page.load_thread.wait()
    _app.processEvents()
    return page


def _stub_dialog_class(rule, accepted=True):
    """替掉真正的 DisplayNameRuleDialog：真弹窗的 exec() 会开一个嵌套事件循环，
    测试里没人去点它，会一直卡住。这里直接跳过 UI，模拟用户选好规则后的结果。
    """
    captured = {}

    class _Stub:
        def __init__(self, parent=None, current_rule=None, sample_images=None, project_name=""):
            captured['parent'] = parent
            captured['current_rule'] = current_rule
            captured['sample_images'] = list(sample_images or [])
            captured['project_name'] = project_name

        def exec(self):
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

        def selected_rule(self):
            return rule if accepted else None

    _Stub.captured = captured
    return _Stub


# ==================== 入口 ====================

def test_manage_menu_has_display_name_rule_entry():
    """「管理」菜单里要有这一条，且没有项目时不能点。"""
    project_id = _make_project()
    page = ImportPage()

    assert not page.action_display_name_rule.isEnabled(), "还没有项目时不该能点"

    page.set_project(project_id)
    _app.processEvents()

    labels = [a.text() for a in page.manage_menu.actions() if a.text()]
    assert "图片显示名称…" in labels
    assert page.action_display_name_rule.isEnabled()

    page.stop_image_loading()
    db.delete_project(project_id)


def test_entry_opens_dialog_seeded_with_the_persisted_rule_and_all_images():
    """点开入口：对话框拿到的是当前已保存的规则和项目的全部图片（不是空的）。"""
    project_id = _make_project()
    _seed_image(project_id, "a.jpg", "/data/x/a.jpg")
    _seed_image(project_id, "b.jpg", "/data/x/b.jpg")
    db.update_project(project_id, display_name_rule={"mode": "source"})

    page = ImportPage()
    page.set_project(project_id)

    stub = _stub_dialog_class({"mode": "source"}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub):
        page.action_display_name_rule.trigger()

    assert stub.captured['current_rule'] == {"mode": "source"}
    assert len(stub.captured['sample_images']) == 2
    assert stub.captured['project_name']

    page.stop_image_loading()
    db.delete_project(project_id)


# ==================== 取消 / 校验失败：不写库 ====================

def test_cancel_does_not_persist_anything():
    project_id = _make_project()
    _seed_image(project_id, "cat.jpg", "/data/a/cat.jpg")

    page = ImportPage()
    page.set_project(project_id)

    stub = _stub_dialog_class(None, accepted=False)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub):
        page.open_display_name_rule_dialog()

    assert db.get_project(project_id)['display_name_rule'] is None
    assert page.image_list.item(0).text() == "cat.jpg", "取消了，列表也不该变"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_invalid_rule_does_not_persist_and_warns_instead_of_crashing():
    """就算对话框自己没拦住（比如被绕过测试用的假规则），页面这边也必须再挡一次。"""
    project_id = _make_project()
    _seed_image(project_id, "cat.jpg", "/data/a/cat.jpg")

    page = ImportPage()
    page.set_project(project_id)

    bad_rule = {"mode": "custom", "prefix": "", "start": 1, "digits": 3}
    stub = _stub_dialog_class(bad_rule, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub), \
         patch("gui.pages.import_page.show_warning") as warn:
        page.open_display_name_rule_dialog()

    warn.assert_called_once()
    assert db.get_project(project_id)['display_name_rule'] is None
    assert page.image_list.item(0).text() == "cat.jpg"

    page.stop_image_loading()
    db.delete_project(project_id)


def test_db_update_failure_warns_keeps_list_unchanged_and_does_not_emit_signal():
    """db.update_project 写失败（返回 False）：提示用户，列表原样，不广播一个
    数据库里其实没有的规则出去——不然标注页会跟着刷成一个没保存住的假状态。"""
    project_id = _make_project()
    _seed_image(project_id, "cat.jpg", "/data/a/cat.jpg")

    page = ImportPage()
    page.set_project(project_id)

    emitted = []
    page.display_name_rule_changed.connect(lambda pid: emitted.append(pid))

    stub = _stub_dialog_class({"mode": "source"}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub), \
         patch.object(db, 'update_project', return_value=False), \
         patch("gui.pages.import_page.show_warning") as warn:
        page.open_display_name_rule_dialog()

    warn.assert_called_once()
    assert emitted == [], "写库失败不该发同步信号"
    assert page.image_list.item(0).text() == "cat.jpg", "写库失败不该改列表文字"
    assert db.get_project(project_id)['display_name_rule'] is None

    page.stop_image_loading()
    db.delete_project(project_id)


# ==================== 全量校验 ====================

def test_validation_covers_every_project_image_not_just_the_three_shown_in_preview():
    """预览只截前三张，但保存前的校验必须喂全部图片进去——用一个真实撞车场景验证。

    3 张图同属文件夹 batch_1，第 4 张来自 batch:1——清洗非法字符后前缀撞车，
    这个冲突只有看到全部 4 张才发现得了。
    """
    project_id = _make_project()
    _seed_image(project_id, "a.jpg", "/data/batch_1/a.jpg")
    _seed_image(project_id, "b.jpg", "/data/batch_1/b.jpg")
    _seed_image(project_id, "c.jpg", "/data/batch_1/c.jpg")
    _seed_image(project_id, "d.jpg", "/data/batch:1/d.jpg")

    page = ImportPage()
    page.set_project(project_id)

    stub = _stub_dialog_class({"mode": "source"}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub), \
         patch("gui.pages.import_page.show_warning") as warn:
        page.open_display_name_rule_dialog()

    warn.assert_called_once()
    assert db.get_project(project_id)['display_name_rule'] is None

    page.stop_image_loading()
    db.delete_project(project_id)


def test_validation_is_called_with_the_full_image_list_not_the_preview_sample():
    """更直接地证明一遍：校验函数收到的图片数量等于项目全量，不是预览的 3 张。"""
    project_id = _make_project()
    for i in range(5):
        _seed_image(project_id, f"img_{i}.jpg", f"/data/batch/img_{i}.jpg")

    page = ImportPage()
    page.set_project(project_id)

    all_count = len(db.get_project_images(project_id))
    assert all_count == 5

    stub = _stub_dialog_class({"mode": "original"}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub), \
         patch(
             "gui.pages.import_page.build_project_display_names",
             wraps=real_build_project_display_names,
         ) as spy:
        page.open_display_name_rule_dialog()

    # 保存前的那次校验调用（第一次）必须喂的是全量 5 张，不是预览用的 3 张
    assert spy.call_args_list, "校验函数应该至少被调用一次"
    first_call_images = spy.call_args_list[0].args[1]
    assert len(first_call_images) == 5, first_call_images

    page.stop_image_loading()
    db.delete_project(project_id)


# ==================== 跨页同步 / 选择保持 / 画布不重载 ====================

def test_rule_change_syncs_to_annotate_page_without_touching_canvas_or_selection():
    project_id = _make_project()
    _seed_image(project_id, "20260712_000602_575947_frame_000223.jpg", "/data/videos/v1.mp4")

    import_page = ImportPage()
    import_page.set_project(project_id)

    annotate_page = _open_annotate_page(project_id)
    annotate_page.load_image(annotate_page.images[0]['id'])
    _app.processEvents()

    import_page.display_name_rule_changed.connect(annotate_page.on_display_name_rule_changed)

    # 模拟用户已经缩放/锁定了画布，并且选中了这张图
    annotate_page.canvas.image_scale = 2.5
    annotate_page.canvas.view_locked = True
    image_id_before = annotate_page.current_image_id
    assert image_id_before is not None

    stub = _stub_dialog_class({"mode": "source"}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub), \
         patch.object(annotate_page.canvas, 'load_image') as reload_spy:
        import_page.open_display_name_rule_dialog()
    _app.processEvents()

    # 标注页跟着刷新了文字和顶部信息条
    item = annotate_page.image_list.item(0)
    assert "v1_帧000223" in item.text(), item.text()
    text = str(annotate_page.image_name_label.property('_full_text'))
    assert "v1_帧000223" in text, text

    # 但画布没有被重新加载，缩放/锁定状态和当前选中的图片都没变
    reload_spy.assert_not_called()
    assert annotate_page.canvas.image_scale == 2.5
    assert annotate_page.canvas.view_locked is True
    assert annotate_page.current_image_id == image_id_before

    import_page.stop_image_loading()
    annotate_page.shutdown()
    db.delete_project(project_id)


def test_annotate_page_ignores_rule_change_from_a_different_project():
    """信号带着 project_id：不是当前打开的这个项目，标注页不该跟着刷新。"""
    project_a = _make_project(name="项目A")
    project_b = _make_project(name="项目B")
    _seed_image(project_a, "cat.jpg", "/data/a/cat.jpg")
    _seed_image(project_b, "dog.jpg", "/data/b/dog.jpg")

    annotate_page = _open_annotate_page(project_b)

    with patch.object(annotate_page, 'update_image_list_display') as refresh_spy:
        annotate_page.on_display_name_rule_changed(project_a)

    refresh_spy.assert_not_called()

    annotate_page.shutdown()
    db.delete_project(project_a)
    db.delete_project(project_b)


def test_import_page_keeps_selected_items_by_id_after_rule_change():
    project_id = _make_project()
    id1 = _seed_image(project_id, "a.jpg", "/data/x/a.jpg")
    id2 = _seed_image(project_id, "b.jpg", "/data/x/b.jpg")

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    target_item = next(
        page.image_list.item(i) for i in range(page.image_list.count())
        if page.image_list.item(i).data(Qt.ItemDataRole.UserRole) == id2
    )
    target_item.setSelected(True)
    _app.processEvents()

    stub = _stub_dialog_class({"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 3}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub):
        page.open_display_name_rule_dialog()
    _app.processEvents()

    selected_ids = {item.data(Qt.ItemDataRole.UserRole) for item in page.image_list.selectedItems()}
    assert selected_ids == {id2}, selected_ids

    page.stop_image_loading()
    db.delete_project(project_id)


# ==================== 持久化重开 ====================

def test_rule_persists_across_reopening_both_pages():
    """关掉页面重开（新建页面实例，重新 set_project）：规则还在，两个页面都认。"""
    project_id = _make_project()
    _seed_image(project_id, "20260712_000602_575947_frame_000223.jpg", "/data/videos/v1.mp4")

    page = ImportPage()
    page.set_project(project_id)

    stub = _stub_dialog_class({"mode": "source"}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub):
        page.open_display_name_rule_dialog()
    page.stop_image_loading()

    reopened_import = ImportPage()
    reopened_import.set_project(project_id)
    _app.processEvents()
    assert reopened_import.image_list.item(0).text() == "v1_帧000223"

    reopened_annotate = _open_annotate_page(project_id)
    assert "v1_帧000223" in reopened_annotate.image_list.item(0).text()

    reopened_import.stop_image_loading()
    reopened_annotate.shutdown()
    db.delete_project(project_id)


# ==================== tooltip ====================

def test_import_page_tooltip_always_has_filename_resolution_and_source_path():
    project_id = _make_project()
    _seed_image(project_id, "cat.jpg", "/data/photos/cat.jpg", width=800, height=600)

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    tooltip = page.image_list.item(0).toolTip()
    assert "cat.jpg" in tooltip, tooltip
    assert "分辨率: 800x600" in tooltip, tooltip
    assert "来源: /data/photos/cat.jpg" in tooltip, tooltip

    page.stop_image_loading()
    db.delete_project(project_id)


def test_annotate_page_tooltip_always_has_filename_resolution_and_source_path():
    project_id = _make_project()
    _seed_image(project_id, "cat.jpg", "/data/photos/cat.jpg", width=800, height=600)

    page = _open_annotate_page(project_id)

    tooltip = page.image_list.item(0).toolTip()
    assert "cat.jpg" in tooltip, tooltip
    assert "分辨率: 800x600" in tooltip, tooltip
    assert "来源: /data/photos/cat.jpg" in tooltip, tooltip

    page.shutdown()
    db.delete_project(project_id)


def test_import_page_tooltip_shows_placeholders_when_resolution_and_source_are_missing():
    """width/height/original_path 缺失时，行不能消失，要换成能看懂的占位文案。"""
    project_id = _make_project()
    db.add_image(project_id, "no_meta.jpg", str(_TMP_DIR / "no_meta.jpg"))

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    tooltip = page.image_list.item(0).toolTip()
    assert "no_meta.jpg" in tooltip, tooltip
    assert "分辨率: 未知" in tooltip, tooltip
    assert "来源: 未记录" in tooltip, tooltip

    page.stop_image_loading()
    db.delete_project(project_id)


def test_annotate_page_tooltip_shows_placeholders_when_resolution_and_source_are_missing():
    project_id = _make_project()
    db.add_image(project_id, "no_meta.jpg", str(_TMP_DIR / "no_meta.jpg"))

    page = _open_annotate_page(project_id)

    tooltip = page.image_list.item(0).toolTip()
    assert "no_meta.jpg" in tooltip, tooltip
    assert "分辨率: 未知" in tooltip, tooltip
    assert "来源: 未记录" in tooltip, tooltip

    page.shutdown()
    db.delete_project(project_id)


# ==================== 真实字段 / 磁盘 / 标注不变 ====================

def test_applying_a_rule_never_touches_db_fields_disk_file_or_annotations():
    project_id = _make_project()
    disk_path = _TMP_DIR / f"real-{project_id}.jpg"
    disk_path.write_bytes(b"fake-image-bytes")

    image_id = db.add_image(
        project_id, "cat.jpg", str(disk_path),
        width=800, height=600, original_path="/data/photos/cat.jpg",
    )
    db.add_annotation(
        image_id=image_id, project_id=project_id, class_id=0, class_name="人",
        annotation_type="rectangle", data={'x': 1, 'y': 1, 'width': 5, 'height': 5},
    )

    before_bytes = disk_path.read_bytes()
    before_row = db.get_project_images(project_id)[0]
    before_annotations = db.get_image_annotations(image_id)

    page = ImportPage()
    page.set_project(project_id)

    stub = _stub_dialog_class({"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 3}, accepted=True)
    with patch("gui.pages.import_page.DisplayNameRuleDialog", stub):
        page.open_display_name_rule_dialog()

    after_row = db.get_project_images(project_id)[0]
    after_annotations = db.get_image_annotations(image_id)

    assert after_row['filename'] == before_row['filename'] == "cat.jpg"
    assert after_row['original_path'] == before_row['original_path'] == "/data/photos/cat.jpg"
    assert after_row['storage_path'] == before_row['storage_path'] == str(disk_path)
    assert disk_path.read_bytes() == before_bytes
    assert after_annotations == before_annotations

    # 显示名确实按新规则变了——证明不是规则压根没生效（有标注，格子文字带 ✓ 前缀）
    assert page.image_list.item(0).text() == "✓ IMG_001"
    assert parse_display_name_rule(db.get_project(project_id)['display_name_rule']) == {
        "mode": "custom", "prefix": "IMG_", "start": 1, "digits": 3,
    }

    page.stop_image_loading()
    db.delete_project(project_id)


if __name__ == "__main__":
    exit_code = _bootstrap.run_module_tests(dict(globals()))
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    sys.exit(exit_code)
