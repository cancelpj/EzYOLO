# -*- coding: utf-8 -*-
"""图片显示名：把抽帧生成的文件名念成人话，同时不丢原始文件名。

覆盖的真实问题：
    1. 视频抽帧落库的名字是 20260712_000602_575947_frame_000223.jpg。
       一屏几十行，前 22 个字符完全一样——列表里等于每张图都没有名字。
    2. 只截尾巴不行：两段视频都抽到第 223 帧时，两行会长得一模一样。
    3. 别名只是显示层的事：文件名和数据库不能被改写，完整名字必须还找得到（tooltip）。

运行：
    python tests/test_display_names.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys

from gui.display_names import display_name, display_names, parse_frame_name

_app = _bootstrap.app()
db = _bootstrap.db


# ==================== 纯函数 ====================

def test_generated_frame_name_is_spoken_as_frame_number():
    """抽帧名 → 「帧 223」：前导零去掉，时间戳不念。"""
    assert display_name("20260712_000602_575947_frame_000223.jpg") == "帧 223"
    assert display_name("20260712_000602_575947_frame_000001.jpg") == "帧 1"


def test_all_zero_frame_number_stays_zero_not_empty():
    """第 0 帧是「帧 0」，不能把零全去掉变成「帧 」。"""
    assert display_name("20260712_000602_575947_frame_000000.jpg") == "帧 0"


def test_user_filenames_are_left_alone():
    """普通图片的文件名是用户自己起的，本来就可读——原样返回，不替他改写。"""
    for name in ("cat.jpg", "IMG_2024.png", "安全帽-01.jpeg", "frame_000223.jpg"):
        assert display_name(name) == name, name


def test_unrecognized_names_degrade_to_themselves():
    """认不出来的名字宁可难看也不能念错——退回原样，不猜。"""
    assert display_name("") == ""
    assert display_name("2026_frame_1.jpg") == "2026_frame_1.jpg"
    assert parse_frame_name("cat.jpg") is None


def test_parse_exposes_the_pieces_for_discriminating():
    parsed = parse_frame_name("20260712_000602_575947_frame_000223.jpg")
    assert parsed['date'] == "20260712"
    assert parsed['time'] == "000602"
    assert parsed['frame'] == "000223"


# ==================== 重名 ====================

def test_frames_from_one_video_need_no_discriminator():
    """同一段视频里帧号本来就不会重复，别名保持最短。"""
    names = [
        "20260712_000602_575947_frame_000000.jpg",
        "20260712_000602_575948_frame_000030.jpg",
        "20260712_000602_575949_frame_000060.jpg",
    ]
    assert display_names(names) == ["帧 0", "帧 30", "帧 60"]


def test_same_frame_number_from_different_days_is_split_by_date():
    """两段视频都抽到第 223 帧：先用日期分，别名不能一模一样。"""
    names = [
        "20260712_000602_575947_frame_000223.jpg",
        "20260713_101500_000001_frame_000223.jpg",
    ]
    aliases = display_names(names)

    assert aliases == ["帧 223 · 07-12", "帧 223 · 07-13"], aliases
    assert len(set(aliases)) == 2


def test_same_day_falls_through_to_time():
    """同一天导入的两段视频：日期分不开，用时间分。"""
    names = [
        "20260712_000602_575947_frame_000223.jpg",
        "20260712_183000_000001_frame_000223.jpg",
    ]
    aliases = display_names(names)

    assert aliases == ["帧 223 · 00:06:02", "帧 223 · 18:30:00"], aliases


def test_same_second_falls_back_to_a_stable_ordinal():
    """同一秒也分不开（微秒不同）：给序号。顺序稳定，不随调用次数变。"""
    names = [
        "20260712_000602_575947_frame_000223.jpg",
        "20260712_000602_575999_frame_000223.jpg",
    ]
    aliases = display_names(names)

    assert aliases == ["帧 223 (1)", "帧 223 (2)"], aliases
    assert display_names(names) == aliases, "同样的输入必须给同样的别名"


def test_duplicate_user_filenames_also_get_ordinals():
    """普通文件名重了一样要能分开（没有时间戳可用 → 序号）。"""
    aliases = display_names(["cat.jpg", "cat.jpg", "dog.jpg"])
    assert aliases == ["cat.jpg (1)", "cat.jpg (2)", "dog.jpg"], aliases


def test_aliases_line_up_one_to_one_with_the_input():
    """别名列表和输入列表一一对应——错位就会把 A 的名字挂到 B 头上。"""
    names = [
        "cat.jpg",
        "20260712_000602_575947_frame_000223.jpg",
        "20260713_101500_000001_frame_000223.jpg",
        "dog.png",
    ]
    aliases = display_names(names)

    assert len(aliases) == len(names)
    assert aliases[0] == "cat.jpg"
    assert aliases[3] == "dog.png"
    assert aliases[1].startswith("帧 223") and aliases[2].startswith("帧 223")
    assert aliases[1] != aliases[2]


# ==================== 页面里用起来 ====================

def _seed_frames(project_id, filenames):
    for name in filenames:
        db.add_image(project_id, name, f"/tmp/{name}", width=640, height=480)


def test_import_grid_shows_aliases_but_keeps_the_real_name_in_the_tooltip():
    """导入页网格：格子上写别名，完整文件名和分辨率留在 tooltip 里。"""
    from PyQt6.QtCore import Qt
    from gui.pages.import_page import ImportPage

    project_id = _bootstrap.create_temp_project(name="别名测试", project_type="detect")
    original = "20260712_000602_575947_frame_000223.jpg"
    _seed_frames(project_id, [original])

    page = ImportPage()
    page.set_project(project_id)
    _app.processEvents()

    item = page.image_list.item(0)
    assert item.text() == "帧 223", item.text()
    assert original in item.toolTip(), item.toolTip()
    assert "640x480" in item.toolTip(), item.toolTip()

    # 数据库里存的还是原来那个名字——别名只是显示层的事
    stored = db.get_project_images(project_id)[0]['filename']
    assert stored == original

    page.stop_image_loading()
    db.delete_project(project_id)


def test_annotate_list_and_context_bar_use_aliases_and_put_position_first():
    """标注页：左边列表用别名（保留 ✓/○），顶上先说第几张，再说是哪张。"""
    from gui.pages.annotate_page import AnnotatePage

    project_id = _bootstrap.create_temp_project(name="别名测试2", project_type="detect")
    original = "20260712_000602_575947_frame_000223.jpg"
    _seed_frames(project_id, [original])

    page = AnnotatePage()
    page.current_project_id = project_id
    page.load_image_list()
    _app.processEvents()

    item = page.image_list.item(0)
    assert item.text() == "○ 帧 223", item.text()
    assert original in item.toolTip(), item.toolTip()

    page.load_image(page.images[0]['id'])
    _app.processEvents()

    text = str(page.image_name_label.property('_full_text'))
    assert text.startswith("第 1/1 张"), f"位置要排在最前面：{text!r}"
    assert "帧 223" in text, text
    assert original not in text, f"长文件名不该出现在信息条上：{text!r}"
    # 完整文件名还找得到
    assert original in page.image_name_label.toolTip()

    page.shutdown()
    db.delete_project(project_id)


# ==================== U6：项目级显示名称规则 ====================

from gui.display_names import (
    build_project_display_names,
    default_rule,
    parse_display_name_rule,
    serialize_display_name_rule,
    validate_custom_rule_fields,
    validate_display_name,
    validate_rule,
)


def _frame(original_path, frame_no, micro="000001"):
    return {
        "filename": f"20260712_000602_{micro}_frame_{frame_no:06d}.jpg",
        "original_path": original_path,
    }


def _image(original_path, name="cat.jpg"):
    return {"filename": name, "original_path": original_path}


# ---- original 模式 ----

def test_original_rule_keeps_user_filenames_and_short_frame_names():
    images = [_image(None, "cat.jpg"), _frame("/videos/v1.mp4", 223)]
    names = build_project_display_names(default_rule(), images, project_name="项目")
    assert names == ["cat.jpg", "帧 223"]


# ---- source 模式 ----

def test_source_rule_names_video_frames_by_source_stem_and_frame_number():
    images = [
        _frame("/data/videos/myvideo.mp4", 0),
        _frame("/data/videos/myvideo.mp4", 30),
    ]
    names = build_project_display_names({"mode": "source"}, images, project_name="项目")
    assert names == ["myvideo_帧000000", "myvideo_帧000030"], names


def test_source_rule_names_normal_images_by_parent_folder_and_ordinal():
    images = [
        _image("/data/batch1/a.jpg"),
        _image("/data/batch1/b.jpg"),
        _image("/data/batch2/c.jpg"),
    ]
    names = build_project_display_names({"mode": "source"}, images, project_name="项目")
    assert names == ["batch1_000001", "batch1_000002", "batch2_000001"], names


def test_source_rule_falls_back_to_project_name_when_no_original_path():
    images = [_image(None), _image("")]
    names = build_project_display_names({"mode": "source"}, images, project_name="我的项目")
    assert names == ["我的项目_000001", "我的项目_000002"], names


def test_source_rule_raises_conflict_error_when_two_sources_clean_to_same_name():
    images = [
        _image("/data/A:B/a.jpg"),
        _image("/data/A?B/b.jpg"),  # 两个不同的文件夹名，清洗非法字符后变成同一个前缀
    ]
    try:
        build_project_display_names({"mode": "source"}, images, project_name="项目")
        assert False, "两个来源清洗后撞名应该报错，不能静默覆盖"
    except ValueError as exc:
        assert "撞" in str(exc) or "重复" in str(exc) or "冲突" in str(exc), str(exc)


def test_source_rule_does_not_mutate_or_touch_input_records():
    images = [_image("/data/batch1/a.jpg"), _frame("/data/videos/v1.mp4", 5)]
    snapshot = [dict(img) for img in images]
    build_project_display_names({"mode": "source"}, images, project_name="项目")
    assert images == snapshot


def test_source_rule_600_images_are_stable_and_unique():
    images = [_frame("/data/videos/v1.mp4", i) for i in range(600)]
    names = build_project_display_names({"mode": "source"}, images, project_name="项目")
    assert len(names) == 600
    assert len(set(names)) == 600
    assert build_project_display_names({"mode": "source"}, images, project_name="项目") == names


# ---- custom 模式 ----

def test_custom_rule_builds_prefix_plus_padded_ordinal():
    images = [_image(None, "a.jpg"), _image(None, "b.jpg"), _image(None, "c.jpg")]
    rule = {"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 3}
    names = build_project_display_names(rule, images)
    assert names == ["IMG_001", "IMG_002", "IMG_003"], names


def test_custom_rule_respects_start_and_digits():
    images = [_image(None, "a.jpg"), _image(None, "b.jpg")]
    rule = {"mode": "custom", "prefix": "P", "start": 9, "digits": 2}
    names = build_project_display_names(rule, images)
    assert names == ["P_09", "P_10"], names


def test_custom_rule_inserts_separator_between_prefix_and_ordinal():
    """“阀门” + 起始 1 + 位数 5 必须念成“阀门_00001”，不是“阀门00001”。"""
    images = [_image(None, "a.jpg"), _image(None, "b.jpg")]
    rule = {"mode": "custom", "prefix": "阀门", "start": 1, "digits": 5}
    names = build_project_display_names(rule, images)
    assert names == ["阀门_00001", "阀门_00002"], names


def test_custom_rule_does_not_double_underscore_when_prefix_already_ends_with_one():
    images = [_image(None, "a.jpg")]
    rule = {"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 3}
    names = build_project_display_names(rule, images)
    assert names == ["IMG_001"], names
    assert "__" not in names[0], names[0]


def test_custom_rule_raises_when_prefix_invalid():
    images = [_image(None, "a.jpg")]
    rule = {"mode": "custom", "prefix": "", "start": 1, "digits": 3}
    try:
        build_project_display_names(rule, images)
        assert False, "空前缀应该报错"
    except ValueError:
        pass


def test_custom_rule_600_images_are_stable_and_unique():
    images = [_image(None, f"img{i}.jpg") for i in range(600)]
    rule = {"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 6}
    names = build_project_display_names(rule, images)
    assert len(names) == 600
    assert len(set(names)) == 600
    assert names[0] == "IMG_000001" and names[-1] == "IMG_000600"


def test_original_rule_600_images_are_stable_and_unique():
    images = [_frame("/data/videos/v1.mp4", i) for i in range(600)]
    names = build_project_display_names(default_rule(), images, project_name="项目")
    assert len(names) == 600
    assert len(set(names)) == 600


# ---- 校验 ----

def test_validate_display_name_rejects_illegal_windows_characters():
    for bad in ['a/b', 'a\\b', 'a:b', 'a*b', 'a?b', 'a"b', 'a<b', 'a>b', 'a|b']:
        assert validate_display_name(bad) is not None, bad


def test_validate_display_name_rejects_empty_whitespace_and_trailing_dot():
    assert validate_display_name("") is not None
    assert validate_display_name(" a") is not None
    assert validate_display_name("a ") is not None
    assert validate_display_name("a.") is not None


def test_validate_display_name_rejects_overlong_name():
    assert validate_display_name("a" * 96) is None
    assert validate_display_name("a" * 97) is not None


def test_validate_display_name_accepts_normal_name():
    assert validate_display_name("帧 223") is None
    assert validate_display_name("cat.jpg") is None


def test_validate_custom_rule_fields_rejects_empty_prefix_and_bad_numbers():
    assert validate_custom_rule_fields("", 1, 6) is not None
    assert validate_custom_rule_fields("IMG", -1, 6) is not None
    assert validate_custom_rule_fields("IMG", 1, 0) is not None
    assert validate_custom_rule_fields("IMG", 1, 6) is None


# ---- 规则解析 / 持久化 ----

def test_parse_display_name_rule_defaults_to_original_when_missing():
    assert parse_display_name_rule(None) == default_rule()
    assert parse_display_name_rule("") == default_rule()


def test_parse_display_name_rule_falls_back_to_original_on_bad_json():
    assert parse_display_name_rule("{not json") == default_rule()
    assert parse_display_name_rule("[1, 2, 3]") == default_rule()
    assert parse_display_name_rule('{"mode": "not_a_real_mode"}') == default_rule()
    assert parse_display_name_rule('{"mode": "custom"}') == default_rule()


def test_parse_display_name_rule_round_trips_custom_rule():
    rule = {"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 6}
    raw = serialize_display_name_rule(rule)
    assert parse_display_name_rule(raw) == rule


def test_validate_rule_rejects_unknown_mode():
    assert validate_rule({"mode": "weird"}) is not None
    assert validate_rule({"mode": "original"}) is None


# ==================== 对话框 ====================

def _make_dialog(**kwargs):
    from gui.widgets.display_name_rule_dialog import DisplayNameRuleDialog
    dialog = DisplayNameRuleDialog(**kwargs)
    # isVisible() 只在真正 show() 过之后才反映 setVisible() 的调用结果，
    # 离屏平台下 show() 不会真的弹窗，但足以让可见性状态可测。
    dialog.show()
    _app.processEvents()
    return dialog


def test_dialog_defaults_to_original_and_previews_passthrough_names():
    dialog = _make_dialog(
        sample_images=[_image(None, "cat.jpg"), _frame("/videos/v1.mp4", 223)],
        project_name="测试项目",
    )
    assert dialog.radios["original"].isChecked()
    assert "cat.jpg" in dialog.preview_label.text()
    assert "帧 223" in dialog.preview_label.text()
    assert dialog.apply_btn.isEnabled()
    assert not dialog.error_label.isVisible()
    # 默认焦点落在取消/安全操作上，不是会产生副作用的应用按钮
    assert dialog.cancel_btn.hasFocus() or dialog.cancel_btn.isDefault()
    dialog.close()


def test_dialog_switching_to_custom_updates_preview_live():
    dialog = _make_dialog(
        sample_images=[_image(None, "a.jpg"), _image(None, "b.jpg")],
        project_name="测试项目",
    )
    dialog.radios["custom"].setChecked(True)
    dialog.prefix_input.setText("IMG_")
    dialog.start_input.setValue(1)
    dialog.digits_input.setValue(3)

    assert "IMG_001" in dialog.preview_label.text()
    assert "IMG_002" in dialog.preview_label.text()
    assert dialog.apply_btn.isEnabled()
    dialog.close()


def test_dialog_shows_error_and_disables_apply_for_empty_custom_prefix():
    dialog = _make_dialog(sample_images=[_image(None, "a.jpg")], project_name="测试项目")
    dialog.radios["custom"].setChecked(True)
    dialog.prefix_input.setText("")

    assert dialog.error_label.isVisible()
    assert not dialog.apply_btn.isEnabled()
    dialog.close()


def test_dialog_reset_restores_original_without_applying_or_persisting():
    dialog = _make_dialog(
        current_rule={"mode": "custom", "prefix": "IMG_", "start": 1, "digits": 6},
        sample_images=[_image(None, "a.jpg")],
        project_name="测试项目",
    )
    assert dialog.radios["custom"].isChecked()

    dialog._on_reset()

    assert dialog.radios["original"].isChecked()
    assert "a.jpg" in dialog.preview_label.text()
    # 恢复默认只改控件，没有点“应用”——selected_rule() 仍然是 None
    assert dialog.selected_rule() is None
    dialog.close()


def test_dialog_selected_rule_is_none_until_applied_then_returns_the_rule():
    dialog = _make_dialog(sample_images=[_image(None, "a.jpg")], project_name="测试项目")
    assert dialog.selected_rule() is None

    dialog.radios["source"].setChecked(True)
    dialog._on_apply()

    assert dialog.selected_rule() == {"mode": "source"}
    dialog.close()


def test_dialog_cancel_leaves_selected_rule_none():
    dialog = _make_dialog(sample_images=[_image(None, "a.jpg")], project_name="测试项目")
    dialog.radios["custom"].setChecked(True)
    dialog.prefix_input.setText("IMG_")
    dialog.reject()

    assert dialog.selected_rule() is None
    dialog.close()


def test_dialog_handles_empty_project_with_no_sample_images():
    dialog = _make_dialog(sample_images=[], project_name="空项目")
    assert dialog.apply_btn.isEnabled()
    assert not dialog.error_label.isVisible()
    dialog.close()


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(dict(globals())))
