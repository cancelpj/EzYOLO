# -*- coding: utf-8 -*-
"""缩略图标注框预览：坐标换算与绘制。

这里盯的是一件事——框必须画在对的位置。导入页的缩略图是
`cv2.resize(img, (160, 160))`，非等比缩放，横图被压扁、竖图被拉宽。
所以两个轴要各用各的比例；谁要是哪天改成 letterbox 的算法（同一个比例 + 补边），
下面横图和竖图那两个用例会立刻翻。

运行：
    python -m pytest tests/test_thumbnail_overlay.py -q
    或
    python tests/test_thumbnail_overlay.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys

from PyQt6.QtGui import QColor, QImage, QPixmap

from gui.thumbnail_overlay import (
    DEFAULT_BOX_COLOR,
    bbox_preview_boxes,
    draw_boxes_on_thumbnail,
    scale_bbox,
)

_app = _bootstrap.app()

THUMB = 160  # 缩略图边长，跟 ImageLoadWorker 里的 cv2.resize(img, (160, 160)) 对齐


def _bbox(x, y, w, h):
    return {'x': x, 'y': y, 'width': w, 'height': h}


def _annotation(x, y, w, h, class_id=0, ann_type='bbox'):
    return {'type': ann_type, 'class_id': class_id, 'data': _bbox(x, y, w, h)}


def _color_count(image: QImage, color: QColor) -> int:
    target = QColor(color).rgb()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) == target
    )


def _white_base(size=THUMB) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor('#FFFFFF'))
    return pixmap


# ==================== 坐标换算：横 / 竖 / 方 ====================

def test_square_image_maps_with_a_single_uniform_ratio():
    """方图：两个轴比例相同，就是最朴素的等比缩放。"""
    assert scale_bbox(_bbox(50, 50, 100, 100), 200, 200, THUMB, THUMB) == (40, 40, 80, 80)


def test_landscape_image_stretches_each_axis_independently():
    """横图 640x360：x 轴压 0.25，y 轴压 0.444，比例不一样。

    如果按 letterbox（两轴都用 0.25，再上下补边）算，y 会得到 45/22，
    完全不是这个答案——这条用例就是拿来钉死「非等比」这件事的。
    """
    # 框正好占据图片中心那一块：x 50%~75%，y 50%~75%
    assert scale_bbox(_bbox(320, 180, 160, 90), 640, 360, THUMB, THUMB) == (80, 80, 40, 40)


def test_portrait_image_stretches_each_axis_independently():
    """竖图 360x640：同样占中心 50%~75% 的框，落点必须和横图一模一样。

    非等比缩放把原图硬拉成正方形，所以「归一化位置相同」= 「缩略图上位置相同」，
    跟原图是横是竖无关。这正是用户要的：横图竖图方图都不能偏。
    """
    assert scale_bbox(_bbox(180, 320, 90, 160), 360, 640, THUMB, THUMB) == (80, 80, 40, 40)


def test_extremely_wide_image_still_lands_on_the_right_half():
    """极端宽幅（2000x100）也不能把框挤到图外。"""
    rect = scale_bbox(_bbox(1000, 50, 500, 25), 2000, 100, THUMB, THUMB)
    assert rect == (80, 80, 40, 40)


# ==================== 越界裁剪 ====================

def test_box_larger_than_the_image_is_clamped_to_the_thumbnail():
    """框比图还大：裁到缩略图边界，不画到格子外面去。"""
    assert scale_bbox(_bbox(-20, -20, 200, 200), 100, 100, THUMB, THUMB) == (0, 0, THUMB, THUMB)


def test_box_hanging_off_the_right_edge_is_clamped():
    """框有一半探出右下角：只画留在图里的那一半。"""
    # 100x100 的图，框 x=80..120 —— 后面 20px 在图外
    assert scale_bbox(_bbox(80, 80, 40, 40), 100, 100, THUMB, THUMB) == (128, 128, 32, 32)


def test_box_entirely_outside_the_image_is_dropped():
    """整个框都在图外：不画。"""
    assert scale_bbox(_bbox(120, 120, 10, 10), 100, 100, THUMB, THUMB) is None


def test_tiny_box_survives_as_at_least_one_pixel():
    """小目标缩到不足 1 像素时，也要留 1 像素，不能直接消失。"""
    rect = scale_bbox(_bbox(0, 0, 2, 2), 1600, 1600, THUMB, THUMB)
    assert rect == (0, 0, 1, 1)


def test_clamped_one_pixel_box_stays_inside_the_thumbnail():
    """补到 1 像素之后不能顶出右下边界。"""
    x, y, w, h = scale_bbox(_bbox(1599, 1599, 1, 1), 1600, 1600, THUMB, THUMB)
    assert x + w <= THUMB and y + h <= THUMB


# ==================== 画不出来的输入 ====================

def test_unknown_image_size_yields_no_box():
    """原图尺寸不明（老库里 width/height 可能是 0 或 None）：不画，别乱画。"""
    assert scale_bbox(_bbox(10, 10, 20, 20), 0, 0, THUMB, THUMB) is None
    assert scale_bbox(_bbox(10, 10, 20, 20), 100, 0, THUMB, THUMB) is None


def test_zero_or_negative_sized_box_yields_no_box():
    assert scale_bbox(_bbox(10, 10, 0, 10), 100, 100, THUMB, THUMB) is None
    assert scale_bbox(_bbox(10, 10, 10, -5), 100, 100, THUMB, THUMB) is None


def test_malformed_box_data_yields_no_box():
    """脏数据不能让整页缩略图崩掉。"""
    assert scale_bbox(None, 100, 100, THUMB, THUMB) is None
    assert scale_bbox({'x': 'abc', 'y': 0, 'width': 10, 'height': 10}, 100, 100, THUMB, THUMB) is None
    assert scale_bbox({}, 100, 100, THUMB, THUMB) is None


# ==================== 只挑 bbox ====================

def test_only_bbox_annotations_are_previewed():
    """多边形、关键点、旋转框在 160px 的格子里是一团糊：忽略，不画。"""
    annotations = [
        _annotation(10, 10, 20, 20, class_id=1),
        {'type': 'polygon', 'class_id': 2, 'data': {'points': [[0, 0], [10, 0], [10, 10]]}},
        {'type': 'keypoint', 'class_id': 3, 'data': {'points': [[5, 5]]}},
        {'type': 'obb', 'class_id': 4, 'data': {'points': [[0, 0], [9, 1], [8, 9], [1, 8]]}},
    ]

    boxes = bbox_preview_boxes(annotations, 100, 100, THUMB, THUMB)

    assert len(boxes) == 1
    assert boxes[0][1] == 1, "留下来的必须是那个 bbox，类别 id 要跟着一起带出来"


def test_preview_keeps_class_id_per_box_and_skips_junk():
    annotations = [
        _annotation(0, 0, 10, 10, class_id=0),
        _annotation(50, 50, 10, 10, class_id=7),
        _annotation(200, 200, 10, 10, class_id=9),   # 完全在图外，丢掉
        {'type': 'bbox', 'class_id': 5, 'data': None},  # 脏数据，丢掉
        "不是字典",                                     # 更脏的，也不能炸
    ]

    boxes = bbox_preview_boxes(annotations, 100, 100, THUMB, THUMB)

    assert [class_id for _rect, class_id in boxes] == [0, 7]


def test_no_annotations_means_no_boxes():
    assert bbox_preview_boxes([], 100, 100, THUMB, THUMB) == []
    assert bbox_preview_boxes(None, 100, 100, THUMB, THUMB) == []


# ==================== 绘制 ====================

def test_drawing_leaves_the_base_pixmap_untouched():
    """底图会被反复复用（关掉开关就要拿它回来），绝不能就地画上去。"""
    base = _white_base()
    before = base.toImage()

    composed = draw_boxes_on_thumbnail(base, [((10, 10, 50, 50), 0)], {0: '#FF0000'})

    assert composed is not base, "必须返回新 pixmap"
    assert base.toImage() == before, "底图被就地改掉了"
    assert _color_count(base.toImage(), QColor('#FF0000')) == 0


def test_boxes_are_drawn_in_the_project_class_colour():
    base = _white_base()

    composed = draw_boxes_on_thumbnail(base, [((10, 10, 50, 50), 0)], {0: '#FF0000'})

    assert _color_count(composed.toImage(), QColor('#FF0000')) > 0


def test_each_class_keeps_its_own_colour():
    """同一张图里的两个类别，画出来是两种颜色。"""
    base = _white_base()
    boxes = [((10, 10, 40, 40), 0), ((90, 90, 40, 40), 1)]

    composed = draw_boxes_on_thumbnail(base, boxes, {0: '#FF0000', 1: '#00FF00'})

    image = composed.toImage()
    assert _color_count(image, QColor('#FF0000')) > 0
    assert _color_count(image, QColor('#00FF00')) > 0


def test_class_without_a_colour_falls_back_to_grey():
    """类别没配颜色时用兜底灰，跟标注页 canvas 的兜底一致——不能不画。"""
    base = _white_base()

    composed = draw_boxes_on_thumbnail(base, [((10, 10, 50, 50), 42)], {})

    assert _color_count(composed.toImage(), QColor(DEFAULT_BOX_COLOR)) > 0


def test_no_boxes_returns_the_base_itself():
    """没有框就别白白复制一张 pixmap——600 张图的时候这点开销是要还的。"""
    base = _white_base()
    assert draw_boxes_on_thumbnail(base, [], {0: '#FF0000'}) is base


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(globals()))
