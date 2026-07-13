# -*- coding: utf-8 -*-
"""缩略图上的标注框预览：坐标换算 + 画框。

只做换算和绘制：不读磁盘、不查数据库、不碰列表控件。换算部分是纯函数，
不需要 QApplication 就能测。

坐标为什么必须用两个比例：导入页的缩略图是 `cv2.resize(img, (160, 160))` 出来的，
这是**非等比**缩放——横图被压扁、竖图被拉宽，没有 letterbox 留白。所以
sx = thumb_w / image_w 和 sy = thumb_h / image_h 必须分开算。按 letterbox 那样
取同一个比例再补边，画出来的框会整体偏移，横图和竖图偏得最狠。
"""

from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap

# 缩略图坐标系里的一个框：(x, y, width, height)
ThumbBox = Tuple[int, int, int, int]

# 类别没配颜色时用的灰，跟标注页 canvas 的兜底色一致
DEFAULT_BOX_COLOR = '#808080'

BOX_PEN_WIDTH = 2


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale_bbox(data: Dict, image_w: int, image_h: int,
               thumb_w: int, thumb_h: int) -> Optional[ThumbBox]:
    """把一个 bbox 从原图坐标换算到缩略图坐标；换不出来就返回 None。

    换不出来的三种情况：原图尺寸不明（0 或负数）、框本身没有面积、框完全落在图外。
    超出图片边界的框会被裁进缩略图范围——标注时框可以画到贴边甚至溢出一点，
    照原样映射会画到缩略图外面去。
    """
    if not isinstance(data, dict):
        return None
    if image_w <= 0 or image_h <= 0 or thumb_w <= 0 or thumb_h <= 0:
        return None

    try:
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        width = float(data.get('width', 0))
        height = float(data.get('height', 0))
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    sx = thumb_w / image_w
    sy = thumb_h / image_h

    left = _clamp(x * sx, 0, thumb_w)
    right = _clamp((x + width) * sx, 0, thumb_w)
    top = _clamp(y * sy, 0, thumb_h)
    bottom = _clamp((y + height) * sy, 0, thumb_h)

    # 裁完没面积了：框整个在图外，不画
    if right - left <= 0 or bottom - top <= 0:
        return None

    ix = int(round(left))
    iy = int(round(top))
    # 小目标映射到 160px 上可能不足 1 像素，四舍五入直接变成 0 宽而消失。
    # 至少给 1 像素，让「这里有个框」还看得见。
    iw = max(1, int(round(right)) - ix)
    ih = max(1, int(round(bottom)) - iy)

    # 补到 1 像素后可能又顶出边界，往回挪
    ix = max(0, min(ix, thumb_w - iw))
    iy = max(0, min(iy, thumb_h - ih))

    return (ix, iy, iw, ih)


def bbox_preview_boxes(annotations: Sequence[Dict], image_w: int, image_h: int,
                       thumb_w: int, thumb_h: int) -> List[Tuple[ThumbBox, int]]:
    """一张图的标注 → 缩略图上要画的 [(框, 类别 id)]。

    只认 type == 'bbox'。多边形、关键点、旋转框在 160px 的格子里画出来是一团糊，
    这里直接忽略——导入页要回答的是「标没标、标在哪」，不是把标注页复刻一遍。
    """
    boxes: List[Tuple[ThumbBox, int]] = []

    for annotation in annotations or ():
        if not isinstance(annotation, dict):
            continue
        if annotation.get('type') != 'bbox':
            continue

        rect = scale_bbox(annotation.get('data'), image_w, image_h, thumb_w, thumb_h)
        if rect is None:
            continue

        boxes.append((rect, annotation.get('class_id', 0)))

    return boxes


def draw_boxes_on_thumbnail(base: QPixmap, boxes: Sequence[Tuple[ThumbBox, int]],
                            class_colors: Dict[int, str] = None) -> QPixmap:
    """在底图的副本上画框，返回新 QPixmap；底图本身不动。

    底图是缓存里那张「没有框的」缩略图，会被反复复用（开关一关就得把它拿回来），
    所以绝不能就地画上去。
    """
    if base is None or base.isNull() or not boxes:
        return base

    class_colors = class_colors or {}
    composed = base.copy()

    # 不开抗锯齿：160px 的小图上，2px 的框糊开之后反而看不清边
    painter = QPainter(composed)
    try:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for (x, y, width, height), class_id in boxes:
            color = class_colors.get(class_id, DEFAULT_BOX_COLOR)
            if isinstance(color, str):
                color = QColor(color)
            if not isinstance(color, QColor) or not color.isValid():
                color = QColor(DEFAULT_BOX_COLOR)

            pen = QPen(color)
            pen.setWidth(BOX_PEN_WIDTH)
            painter.setPen(pen)
            painter.drawRect(x, y, width, height)
    finally:
        painter.end()

    return composed
