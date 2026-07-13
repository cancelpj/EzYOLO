# -*- coding: utf-8 -*-
"""
数据导入页面（第 1 步）

页面只负责一件事：把图片弄进当前项目。
项目的选择/新建/删除入口在主窗口侧边栏，这里不再重复放一个项目下拉框。
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QFileDialog, QProgressBar,
    QMenu, QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QDialog, QStackedWidget, QSizePolicy, QToolButton, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon
import cv2
import numpy as np
import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import os

from gui.styles import COLORS, set_menu_indicator
from gui.display_names import (
    build_project_display_names, display_name, display_names, parse_display_name_rule,
)
from gui.thumbnail_overlay import bbox_preview_boxes, draw_boxes_on_thumbnail
from models.database import db
from core.import_manager import ImportManager, VIDEO_MODE_INTERVAL, VIDEO_MODE_RANDOM
from core.annotation_importer import AnnotationImporter
from gui.widgets.context_help import ContextHelp
from gui.widgets.loading_dialog import LoadingOverlay
from gui.widgets.group_select_dialog import GroupSelectDialog, ask_import_group
from gui.widgets.task_type_dialog import ask_task_type, task_type_label
from gui.widgets.workflow_widgets import EmptyState
from gui.widgets.app_dialog import (
    ask_text, confirm, confirm_destructive, show_info, show_warning,
)
from gui.widgets.video_extract_dialog import ask_video_extract_plan
from gui.widgets.display_name_rule_dialog import DisplayNameRuleDialog


def short_task_label(task_type: str) -> str:
    """工具栏胶囊上只放中文那半截：「目标检测 detect」→「目标检测」。

    共用的任务类型对话框仍然显示带英文的完整标签——在那里，detect / segment
    这些词要和 YOLO 的术语对得上，是有用的信息；而工具栏上它只是把胶囊撑长。
    """
    return task_type_label(task_type).split(' ')[0]


# 数据导入线程：文件夹 / 多图 / 视频，实际工作全部委托给 core.import_manager，
# 主线程不做逐文件复制、解码或数据库写入。
class ImportWorkerThread(QThread):
    """通用后台导入线程"""

    progress_updated = pyqtSignal(int, str)
    # success, error_message, cancelled, imported, skipped
    # 注意：不能叫 finished —— QThread 自带同名信号（线程真正退出时才发），
    # 定义同名信号会把它遮住，导致没人能再监听「线程真的跑完了」这件事，
    # 从而在还没运行结束时就被提前释放引用，触发
    # "QThread: Destroyed while thread is still running"。
    result_ready = pyqtSignal(bool, str, bool, int, int)

    def __init__(self, project_id, group_id, kind, source, frame_interval=1,
                 video_mode=VIDEO_MODE_INTERVAL, sample_count=None):
        super().__init__()
        self.project_id = project_id
        self.group_id = group_id
        self.kind = kind  # 'folder' | 'images' | 'video'
        self.source = source
        self.frame_interval = frame_interval
        self.video_mode = video_mode        # 'interval' | 'random'
        self.sample_count = sample_count    # 随机模式要抽的张数
        self._cancel_event = threading.Event()

    def cancel(self):
        """请求取消：最迟在下一个文件/帧边界停止。"""
        self._cancel_event.set()

    def run(self):
        """运行导入"""
        try:
            import_manager = ImportManager(self.project_id, group_id=self.group_id)

            def progress_callback(progress, message):
                self.progress_updated.emit(progress, message)

            if self.kind == 'folder':
                imported, skipped = import_manager.import_folder(
                    self.source, progress_callback=progress_callback,
                    cancel_event=self._cancel_event,
                )
            elif self.kind == 'images':
                imported, skipped = import_manager.import_images(
                    self.source, progress_callback=progress_callback,
                    cancel_event=self._cancel_event,
                )
            elif self.kind == 'video':
                imported, skipped = import_manager.import_video(
                    self.source, frame_interval=self.frame_interval,
                    progress_callback=progress_callback,
                    cancel_event=self._cancel_event,
                    mode=self.video_mode,
                    sample_count=self.sample_count,
                )
            else:
                raise ValueError(f"未知导入类型: {self.kind}")

            self.result_ready.emit(True, "", self._cancel_event.is_set(), imported, skipped)
        except Exception as e:
            self.result_ready.emit(False, str(e), False, 0, 0)


class ImageLoadWorker(QThread):
    """图片加载工作线程"""
    
    # 信号：进度更新、单个图片加载完成、全部完成
    progress = pyqtSignal(int, int)  # 当前进度, 总数
    image_loaded = pyqtSignal(int, object, str)  # 索引, 缩略图, 存储路径
    finished_loading = pyqtSignal()
    
    def __init__(self, image_tasks: List[Tuple[int, Dict]]):
        super().__init__()
        self.image_tasks = image_tasks
        self._is_running = True
    
    def run(self):
        """在后台线程中加载图片"""
        total = len(self.image_tasks)
        
        for task_index, (row_index, image_data) in enumerate(self.image_tasks):
            if not self._is_running:
                break
            
            storage_path = image_data.get('storage_path', '')
            pixmap = None
            
            if storage_path and os.path.exists(storage_path):
                try:
                    # 使用OpenCV加载，比QPixmap更快
                    img = cv2.imread(storage_path)
                    if img is not None:
                        # 直接缩小到缩略图尺寸，减少内存占用
                        img = cv2.resize(img, (160, 160))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                except Exception:
                    pass
            
            # 如果加载失败，创建空白图
            if pixmap is None or pixmap.isNull():
                pixmap = QPixmap(160, 160)
                pixmap.fill(QColor(COLORS['sidebar']))
            
            # 发送信号到主线程更新UI
            self.image_loaded.emit(row_index, pixmap, storage_path)
            self.progress.emit(task_index + 1, total)
            
            # 每加载10张图片休眠一下，让UI有机会更新
            if task_index % 10 == 0:
                self.msleep(1)
        
        self.finished_loading.emit()
    
    def stop(self):
        """停止加载"""
        self._is_running = False


class _ResponsiveThumbnailGrid(QListWidget):
    """图片缩略图网格：列数跟着视口宽度走，不留一整列的空白。

    固定 gridSize 在窗口宽度和「整数个格子」对不上时，要么挤出横向滚动条，
    要么在最右边留一条不够放下一格的空白（比如 890px 宽只塞得下 4 个 184px
    格子，剩下 154px 就那么空着）。这里在视口变化时重算列数，让 usable
    宽度正好被列数整除，余数摊薄到每一格里，而不是攒成一条空白。
    """

    _PREFERRED_CELL_WIDTH = 184
    _MIN_CELL_WIDTH = 156
    _MAX_CELL_WIDTH = 220
    _VIEWPORT_INSET = 4
    _MIN_ICON_EXTENT = 120
    _MAX_ICON_EXTENT = 160
    _ICON_HORIZONTAL_ALLOWANCE = 24
    _CELL_VERTICAL_ALLOWANCE = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        # 用定时器把重算推迟到下一轮事件循环：resizeEvent 里直接 setGridSize
        # 可能马上再触发一次 resizeEvent，级联下去；排队执行、且只在视口真的
        # 变化时才动，就不会递归。
        self._grid_update_timer = QTimer(self)
        self._grid_update_timer.setSingleShot(True)
        self._grid_update_timer.timeout.connect(self._recalculate_grid)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._grid_update_timer.start(0)

    def showEvent(self, event):
        super().showEvent(event)
        self._grid_update_timer.start(0)

    def _recalculate_grid(self):
        viewport = self.viewport()
        if viewport is None:
            return
        usable = max(1, viewport.width() - self._VIEWPORT_INSET)

        columns = max(1, round(usable / self._PREFERRED_CELL_WIDTH))
        cell_width = usable // columns

        # 太宽：多分几列；分到会低于下限就停，不做无意义的震荡
        while cell_width > self._MAX_CELL_WIDTH:
            next_columns = columns + 1
            next_cell_width = usable // next_columns
            if next_cell_width < self._MIN_CELL_WIDTH:
                break
            columns = next_columns
            cell_width = next_cell_width

        # 太窄：少分几列；退回 1 列前，先看看会不会又超过上限
        while cell_width < self._MIN_CELL_WIDTH and columns > 1:
            next_columns = columns - 1
            next_cell_width = usable // next_columns
            if next_cell_width > self._MAX_CELL_WIDTH:
                break
            columns = next_columns
            cell_width = next_cell_width

        icon_extent = max(
            self._MIN_ICON_EXTENT,
            min(self._MAX_ICON_EXTENT, cell_width - self._ICON_HORIZONTAL_ALLOWANCE),
        )
        font_height = self.fontMetrics().height()
        grid_size = QSize(cell_width, icon_extent + font_height + self._CELL_VERTICAL_ALLOWANCE)
        icon_size = QSize(icon_extent, icon_extent)

        if grid_size != self.gridSize():
            self.setGridSize(grid_size)
        if icon_size != self.iconSize():
            self.setIconSize(icon_size)


class ImportPage(QWidget):
    """数据导入页面"""

    # 项目列表变了（新建 / 删除），附带之后应该选中的项目 id（没有则 None）
    projects_changed = pyqtSignal(object)
    # 当前项目的图片或标注数量变了，主窗口据此刷新流程进度
    project_data_changed = pyqtSignal()
    # 项目的图片显示名称规则变了（附带 project_id）：标注页据此只重刷文字/tooltip，
    # 不重新加载画布或标注状态
    display_name_rule_changed = pyqtSignal(int)
    # 后台线程把整批标注框/版本号查回来了：载荷是一个纯 Python dict，
    # 跨线程只传数据，不传任何 Qt 控件或 QPixmap
    _annotation_previews_ready = pyqtSignal(object)

    # 整批重画缩略图时，一轮事件循环里最多画这么多张、最多花这么长时间，
    # 剩下的排到下一轮。600 张一次性画完会把界面按住不动好几秒。
    _ICON_CHUNK_SIZE = 12
    _ICON_CHUNK_BUDGET = 0.008  # 秒

    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.images = []
        # 缓存的是「没有框」的底图（key 为 storage_path）：它对应磁盘上的那张图，
        # 只有图片本身变了才需要重读。标注变了只用重画框，不用再读一次盘。
        self.thumbnail_cache = {}
        self.load_worker = None
        self._image_load_generation = 0
        self.thumbnail_widgets = []  # 存储缩略图控件引用

        # 缩略图上的标注框预览（默认开着，用户可以临时关掉纯看图片）
        self.show_annotation_boxes = True
        self._bbox_previews = {}        # image_id -> [bbox 标注]，整批查回来，不逐图查
        self._annotation_versions = {}  # image_id -> 版本号，用来判断哪张图的框变了
        self._class_colors = {}         # class_id -> 颜色，跟标注页同一份项目类别配色
        # image_id -> (版本号, 合成好的带框缩略图)。版本号对得上就直接复用，
        # 所以改一张图的标注只会重画那一张，其余的连碰都不碰。
        self._overlay_cache = {}
        # 查框/版本号的后台世代：切项目、再刷一次都会 +1，回来的过期结果直接丢
        self._preview_generation = 0
        self._previews_pending = False
        # 整批重画缩略图的分块状态：世代用来作废旧队列，游标记录画到第几个格子
        self._icon_refresh_generation = 0
        self._icon_refresh_cursor = 0
        self._icon_refresh_pending = False
        self._annotation_previews_ready.connect(self._on_annotation_previews_ready)

        # 导入任务状态：与上面的缩略图加载状态（load_worker/_image_load_generation）
        # 完全分离，切页、刷新缩略图都不应该影响这一组状态
        self._active_import_thread = None
        # 被取消/作废但线程还没真正跑完的（比如切项目）：必须继续持有引用，
        # 直到 QThread 原生 finished 信号确认线程真的退出了才能放手，
        # 否则会在线程仍在运行时被 GC，触发 "QThread: Destroyed while thread is still running"。
        self._retired_import_threads = []
        self._import_busy = False
        self._import_generation = 0
        self._import_finalize_pending = False
        self._import_finalize_summary = ""
        self._import_finalize_cancelled = False
        self._import_before_ids = set()

        self.init_ui()
        self.refresh_view_filter_options()
        self.update_project_controls()
        self.update_view_mode()

    def _remove_cached_thumbnails(self, storage_paths):
        """按路径移除缩略图缓存。"""
        for path in storage_paths:
            if path in self.thumbnail_cache:
                del self.thumbnail_cache[path]

    def stop_image_loading(self, reset_progress: bool = True):
        """停止当前缩略图加载线程，并可选清理进度状态。"""
        self._image_load_generation += 1
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.stop()
            self.load_worker.wait()
        self.load_worker = None
        if reset_progress and hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 18, 24, 18)
        main_layout.setSpacing(14)

        # 项目信息条没了：当前项目在左侧流程栏里选、也一直显示在那儿，
        # 这里再放一张写着项目名和「01」的卡片，只是把同一件事说第二遍，
        # 顺带把整整一行高度从图片区里拿走。任务类型改挂到工具栏右侧。

        # 工具栏：左边加数据，右边管数据
        self.toolbar = self.create_toolbar()
        main_layout.addWidget(self.toolbar)

        # 轻量帮助：收起时只在右侧留一个小入口
        self.context_help = ContextHelp(
            [
                "“显示标注框”只改变缩略图预览，不会修改任何标注。",
                "导入标注前先确认格式与图片能够一一对应。",
                "清空图片或删除项目会连同标注一起删除，且不可撤销。",
            ],
            risk_steps=[3],
            title="导入提示",
        )
        main_layout.addWidget(self.context_help)

        # 导入任务状态条（默认隐藏）：只反映“导入任务”本身，
        # 跟下面缩略图加载用的 progress_bar 是两套独立状态
        self.import_status_frame = self.create_import_status_bar()
        self.import_status_frame.setVisible(False)
        main_layout.addWidget(self.import_status_frame)

        # 缩略图加载进度条（默认隐藏）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 图片区：没项目 / 没图片 / 图片网格，三选一
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.create_no_project_state())   # 0
        self.view_stack.addWidget(self.create_no_image_state())     # 1
        self.view_stack.addWidget(self.create_image_grid())         # 2
        main_layout.addWidget(self.view_stack, 1)

        # 状态栏
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)

    def create_toolbar(self) -> QFrame:
        """工具栏：左边「把数据弄进来」，右边「这批数据怎么看、怎么管」。

        右边一排以前是筛选 + 移动分组 + 删除选中 + 清空，四个按钮平铺，
        「清空」和「导入文件夹」一样大——用一次的和每次都用的抢同样的注意力。
        现在只留任务类型和筛选，其余收进「管理」菜单，破坏性的那两个单独成一段。
        """
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(14, 10, 14, 10)
        # 2 而不是 4：统一箭头视觉后下拉框的箭头专用区变宽（padding-right 28→34），
        # 1100px（支持的最窄窗口）下右边这一排又要开始切「任务：目标检测」的字，
        # 再收紧 2px × 9 个间隙补上。
        layout.setSpacing(2)

        # 主操作：绝大多数人是导入一个文件夹
        self.btn_import_folder = QPushButton("导入文件夹")
        self.btn_import_folder.setObjectName("primary")
        self.btn_import_folder.setToolTip("把一个文件夹里的图片一次性全部导入")
        self.btn_import_folder.clicked.connect(self.import_folder)
        layout.addWidget(self.btn_import_folder)

        self.btn_import_images = QPushButton("导入图片")
        self.btn_import_images.setToolTip("选择单张或多张图片导入")
        self.btn_import_images.clicked.connect(self.import_images)
        layout.addWidget(self.btn_import_images)

        self.btn_import_video = QPushButton("导入视频")
        self.btn_import_video.setToolTip("从视频里每隔若干帧抽一张图片导入")
        self.btn_import_video.clicked.connect(self.import_video)
        layout.addWidget(self.btn_import_video)

        self.btn_import_annotations = QPushButton("导入标注")
        self.btn_import_annotations.setToolTip("导入已标注数据（YOLO / COCO / VOC 格式）")
        self.btn_import_annotations.clicked.connect(self.import_annotations)
        layout.addWidget(self.btn_import_annotations)

        layout.addStretch()

        # 任务类型：一个能点的胶囊，点开就是原来那个选择框
        self.btn_task_type = QPushButton("任务：未设置")
        self.btn_task_type.setToolTip("决定标注工具和训练方式，一般选「目标检测」")
        self.btn_task_type.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_task_type.clicked.connect(self.change_task_type)
        layout.addWidget(self.btn_task_type)

        # 缩略图上画不画标注框。只是个视图开关：不动数据库，也不会重读磁盘。
        self.chk_show_boxes = QCheckBox("显示标注框")
        self.chk_show_boxes.setChecked(self.show_annotation_boxes)
        self.chk_show_boxes.setToolTip("在缩略图上画出已有的标注框（只是预览，不会改动标注）")
        self.chk_show_boxes.toggled.connect(self.on_toggle_annotation_boxes)

        # 勾选框外面套一层定宽的壳，壳才进工具栏的布局。
        #
        # 为什么非套不可：样式表挂在 MainWindow 上，导入页构造时还没被 parent 进去，
        # 样式还没级联下来。工具栏布局就在这个时候量了一次勾选框的宽度——量到的是
        # 「没上样式」的原生尺寸（indicator 14px，合计 88px），并按 88px 定下了
        # 「筛选」的位置。等样式级联下来，indicator 变成 17px、勾选框自己认 98px，
        # 于是它在那个 88px 的槽里居中撑开，两边各溢出 5px，右边这 5px 正好压在
        # 「筛选」上（1280x720 + 完整样式必现）。
        #
        # Qt 每个 widget 只有一个 QWidgetItem，那份 88px 是缓存住的：事后改 sizeHint、
        # minimumWidth、setFixedWidth，或者 invalidate / 重新 addWidget，槽宽都不回头
        # ——控件只会在原槽里越撑越宽，压得更狠。所以不跟缓存较劲：进布局的换成一个
        # 普通 QWidget，它的尺寸不随样式级联变化，构造时定死多少，布局量到的就是多少。
        # 98px = 完整样式下实测所需宽度（indicator 17 + spacing 8 + 文字 65 + 8 余量）。
        boxes_toggle_holder = QWidget()
        boxes_toggle_holder.setFixedWidth(98)
        # 壳只负责占住宽度，别自己长出一块底色来（默认会顶着一块面板灰）
        boxes_toggle_holder.setStyleSheet("background: transparent;")
        holder_layout = QHBoxLayout(boxes_toggle_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setSpacing(0)
        holder_layout.addWidget(self.chk_show_boxes)
        layout.addWidget(boxes_toggle_holder)

        filter_label = QLabel("筛选")
        filter_label.setObjectName("caption")
        layout.addWidget(filter_label)

        self.view_combo = QComboBox()
        self.view_combo.setMinimumWidth(110)
        self.view_combo.currentTextChanged.connect(self.filter_images)
        layout.addWidget(self.view_combo)

        layout.addWidget(self.create_manage_button())

        return toolbar

    def create_manage_button(self) -> QToolButton:
        """「管理」：不常用的和会删东西的都收在这里。

        破坏性的两个（清空图片、删除项目）单独放在一段里，前面一个红点——
        Qt 的菜单项没法单独染色（QAction 不是控件，样式表选不中它），
        图标是唯一能把「这一条会删东西」标出来的位置。
        """
        self.btn_manage = QToolButton()
        self.btn_manage.setText("管理")
        self.btn_manage.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_manage.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_manage.setToolTip("移动分组、删除图片、删除项目")
        set_menu_indicator(self.btn_manage)

        self.manage_menu = QMenu(self)

        self.action_move_group = self.manage_menu.addAction("移动分组")
        self.action_move_group.setToolTip("把选中图片移到指定分组")
        self.action_move_group.triggered.connect(self.move_selected_to_group)

        self.action_delete_selected = self.manage_menu.addAction("删除选中的图片")
        self.action_delete_selected.triggered.connect(self.delete_selected_images)

        self.manage_menu.addSeparator()

        self.action_display_name_rule = self.manage_menu.addAction("图片显示名称…")
        self.action_display_name_rule.setToolTip(
            "设置这个项目里图片显示成什么名字，不改文件名、不影响标注和训练"
        )
        self.action_display_name_rule.triggered.connect(self.open_display_name_rule_dialog)

        self.manage_menu.addSeparator()

        # 一个点不动的标题，把下面两条框成「危险」的那一段
        danger_caption = self.manage_menu.addAction("危险操作")
        danger_caption.setEnabled(False)

        self.action_clear = self.manage_menu.addAction(self._danger_icon(), "清空全部图片")
        self.action_clear.setToolTip("删除当前项目里的全部图片")
        self.action_clear.triggered.connect(self.clear_all_images)

        self.action_delete_project = self.manage_menu.addAction(self._danger_icon(), "删除项目")
        self.action_delete_project.setToolTip("连同项目里的图片和标注一起删除，不可恢复")
        self.action_delete_project.triggered.connect(self.delete_current_project)

        self.destructive_actions = [self.action_clear, self.action_delete_project]
        for action in self.destructive_actions:
            action.setProperty('destructive', True)

        # 菜单弹出前再算一次：选中状态可能是在菜单关着的时候变的
        self.manage_menu.aboutToShow.connect(self.update_manage_action_state)

        self.btn_manage.setMenu(self.manage_menu)
        return self.btn_manage

    def update_manage_action_state(self):
        """管理菜单里每一条现在能不能点。

        「移动分组」和「删除选中的图片」没有选中图片时就是不能用的——
        以前它们一直亮着，点下去只弹一句「还没选图片」，等于用一个弹窗
        代替了本来一眼就该看出来的状态。「清空全部图片」同理：没有图片可清。
        """
        has_project = bool(self.current_project_id)
        has_selection = bool(self.image_list.selectedItems())
        has_images = bool(self.images)

        self.action_move_group.setEnabled(has_project and has_selection)
        self.action_delete_selected.setEnabled(has_project and has_selection)
        self.action_display_name_rule.setEnabled(has_project)

        # 破坏性的两个在导入进行中一律关掉：正在往里写图片的时候不能把项目端了。
        # 移动/删除选中不受影响——那是对已有图片的操作，导入中照样可以做。
        busy = self._import_busy
        self.action_clear.setEnabled(has_project and has_images and not busy)
        self.action_delete_project.setEnabled(has_project and not busy)

    def _danger_icon(self) -> QIcon:
        """破坏性菜单项前面那个红点。自己画的，不引第三方图标。"""
        pixmap = QPixmap(10, 10)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS['error_fill']))
        painter.drawEllipse(1, 1, 8, 8)
        painter.end()

        return QIcon(pixmap)

    def create_import_status_bar(self) -> QFrame:
        """导入任务状态条：状态文字 + 进度条 + 取消按钮。

        只在有导入任务时出现，跟缩略图加载进度条（self.progress_bar）
        是两套独立状态，互不清空对方。
        """
        frame = QFrame()
        frame.setObjectName("card")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self.import_status_label = QLabel("")
        self.import_status_label.setObjectName("caption")
        top_row.addWidget(self.import_status_label, 1)

        self.btn_cancel_import = QPushButton("取消导入")
        self.btn_cancel_import.setObjectName("danger")
        self.btn_cancel_import.setVisible(False)
        self.btn_cancel_import.clicked.connect(self._cancel_active_import)
        top_row.addWidget(self.btn_cancel_import)

        layout.addLayout(top_row)

        self.import_progress_bar = QProgressBar()
        self.import_progress_bar.setTextVisible(False)
        layout.addWidget(self.import_progress_bar)

        return frame

    def create_no_project_state(self) -> EmptyState:
        """还没有项目时的引导。"""
        state = EmptyState(
            "还没有项目",
            "新建一个项目开始。",
        )
        state.add_action("新建项目", self.create_new_project, primary=True)
        return state

    def create_no_image_state(self) -> EmptyState:
        """项目里还没有图片时的引导。"""
        state = EmptyState(
            "还没有图片",
            "",
        )
        state.add_action("导入文件夹", self.import_folder, primary=True)
        state.add_action("导入图片", self.import_images)
        state.add_action("导入视频", self.import_video)
        return state

    def create_image_grid(self) -> QListWidget:
        """图片缩略图网格。"""
        self.image_list = _ResponsiveThumbnailGrid()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setSpacing(10)
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setMovement(QListWidget.Movement.Static)
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.setUniformItemSizes(True)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.itemClicked.connect(self.on_image_clicked)
        # 选中哪些图片，决定了「移动分组 / 删除选中」能不能点
        self.image_list.itemSelectionChanged.connect(self.update_manage_action_state)
        # 注意：这里不写 item 的 background-color，
        # 让代码里 setBackground 设的「已标注」底色能显示出来。
        # 选中态的 color 必须显式写死：Fusion 的 HighlightedText 是白色，而格子
        # 不铺蓝底，默认白字会在浅色主题里消失。主题切换时继续由统一方法重算。
        self.image_list.setStyleSheet(self._image_list_stylesheet())
        return self.image_list

    def _image_list_stylesheet(self) -> str:
        return f"""
            QListWidget {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
            QListWidget::item {{
                margin: 4px;
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px;
            }}
            QListWidget::item:selected {{
                border: 1px solid {COLORS['primary']};
                color: {COLORS['text_primary']};
            }}
        """

    def refresh_theme(self):
        """主题切换后刷新列表与状态色。"""
        if hasattr(self, 'image_list'):
            self.image_list.setStyleSheet(self._image_list_stylesheet())
        if hasattr(self, 'status_annotated'):
            self.status_annotated.setStyleSheet(f"color: {COLORS['success']};")
        if hasattr(self, 'status_pending'):
            self.status_pending.setStyleSheet(f"color: {COLORS['text_secondary']};")
        if hasattr(self, 'image_list') and hasattr(self, 'images'):
            self._refresh_item_labels()

    def _refresh_image_display_names(self):
        """整批算显示名：按项目的显示名称规则来（默认等价于旧的「帧号化名」逻辑）。

        必须整批算——「这个帧号/编号在这批图里是不是独一份」只有看全列表才知道。
        """
        project = db.get_project(self.current_project_id) if self.current_project_id else None
        rule = parse_display_name_rule((project or {}).get('display_name_rule'))
        project_name = (project or {}).get('name', '')
        try:
            aliases = build_project_display_names(rule, self.images, project_name)
        except ValueError:
            # 规则本该在保存前就校验过全量图片；万一还是生成失败，退回「保留原名」，
            # 不能让整页刷不出来。
            aliases = display_names([img.get('filename', '') for img in self.images])
        self._image_display_names = {
            img['id']: alias for img, alias in zip(self.images, aliases)
        }

    def _image_display_name(self, image_data: Dict) -> str:
        cached = getattr(self, '_image_display_names', {}).get(image_data.get('id'))
        return cached or display_name(image_data.get('filename', ''))

    def _image_tooltip(self, image_data: Dict, group_name: Optional[str] = None) -> str:
        """提示文字跟显示名规则无关：永远是原始文件名、分辨率、来源路径这三行。

        显示名可能被规则改写成完全认不出的样子（比如「阀门_00001」），这里必须
        留一个用户随时能找到真实文件的地方——哪怕某项元数据缺失，也只把这一项
        换成「未知/未记录」，不能整行消失，不然用户会以为软件漏读了数据。
        """
        width = image_data.get('width')
        height = image_data.get('height')
        resolution = f"{width}x{height}" if width and height else "未知"
        original_path = image_data.get('original_path') or "未记录"
        lines = [
            image_data.get('filename', ''),
            f"分辨率: {resolution}",
            f"来源: {original_path}",
        ]
        if group_name is not None:
            lines.append(f"分组: {group_name}")
        return "\n".join(lines)

    # ==================== 缩略图上的标注框预览 ====================

    def _invalidate_thumbnail_overlays(self, image_ids=None):
        """丢掉合成好的「带框缩略图」；不传 image_ids 就全丢。

        底图（thumbnail_cache）一律不动——那是从磁盘读出来的，重画框不需要再读一次盘。
        """
        if image_ids is None:
            self._overlay_cache.clear()
            return
        for image_id in image_ids:
            self._overlay_cache.pop(image_id, None)

    def _load_class_colors(self, project_id) -> Dict[int, str]:
        """项目类别的配色，跟标注页读的是同一份，所以同一类在两个页面永远同色。

        project_id 是参数不是 self.current_project_id：这个方法也在后台线程里跑，
        必须查「发起时那个项目」，不能读一个随时会被主线程改掉的字段。
        """
        project = db.get_project(project_id) if project_id else None
        raw = (project or {}).get('classes') or '[]'
        try:
            classes = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (TypeError, ValueError):
            return {}

        return {
            cls['id']: cls.get('color', '#808080')
            for cls in classes
            if isinstance(cls, dict) and 'id' in cls
        }

    def _load_annotation_previews(self):
        """整批读一次当前项目的框和版本号：600 张图也只查两次库，不是 600 次。

        查库放在后台线程里：两次查询本身不算慢，但它们卡在 GUI 线程上时，用户
        点开一个大项目就是「界面先僵一下」。sqlite 每次调用都开一条新连接，跨线程
        安全；后台只组一个纯 Python dict 回来，一个 QWidget / QPixmap 都不碰。

        结果带着 project_id 和世代号回来：切了项目、或者又刷了一次，过期的那份
        就直接丢掉，不会把旧项目的框盖到新项目的图上。
        """
        self._preview_generation += 1

        if not self.current_project_id:
            self._previews_pending = False
            self._bbox_previews = {}
            self._annotation_versions = {}
            self._class_colors = {}
            self._invalidate_thumbnail_overlays()
            return

        # 结果回来之前先当作「还不知道有没有框」：格子上先摆没框的底图，
        # 不拿上一批的框去合成——版本号一旦对不上，合出来的就是错的框。
        self._previews_pending = True
        self._bbox_previews = {}
        self._annotation_versions = {}

        threading.Thread(
            target=self._fetch_annotation_previews,
            args=(self.current_project_id, self._preview_generation),
            daemon=True,
        ).start()

    def _fetch_annotation_previews(self, project_id: int, generation: int):
        """后台线程：只查库、只组纯 Python dict，不碰任何 Qt 控件。

        框和版本号一次读回来：分两次读会读到「旧框 + 新版本号」，缩略图会把旧框
        按新版本号缓存住，之后再也不刷新（见 get_project_bbox_preview_snapshot）。
        """
        bbox_previews, annotation_versions = db.get_project_bbox_preview_snapshot(project_id)
        self._annotation_previews_ready.emit({
            'project_id': project_id,
            'generation': generation,
            'bbox_previews': bbox_previews,
            'annotation_versions': annotation_versions,
            'class_colors': self._load_class_colors(project_id),
        })

    def _on_annotation_previews_ready(self, payload: Dict):
        """回到 GUI 线程：过期的结果（切了项目 / 又刷了一次）直接丢掉。

        丢掉过期结果时不清 _previews_pending：过期只说明「有一份更新的还在路上」，
        这时候声称「查完了」是假的。
        """
        if (payload['project_id'] != self.current_project_id
                or payload['generation'] != self._preview_generation):
            return

        self._previews_pending = False
        self._bbox_previews = payload['bbox_previews']
        self._annotation_versions = payload['annotation_versions']

        if payload['class_colors'] != self._class_colors:
            # 配色变了，已经合成好的那些框颜色就是错的，只能全部重画
            self._invalidate_thumbnail_overlays()
            self._class_colors = payload['class_colors']

        # 框现在才知道，所以现在才叠上去——分块叠，不在一个事件循环里啃完
        self._schedule_icon_refresh()

    def _thumbnail_icon(self, pixmap: QPixmap) -> QIcon:
        """normal / selected / active 用同一张图。

        QIcon 在没给 Selected 态图片时会自己刷一层蓝——那层蓝正好盖在缩略图上，
        把要看的标注框压得看不清。选中态由卡片边框表达，不靠给图片蒙一层色。
        """
        icon = QIcon()
        icon.addPixmap(pixmap, QIcon.Mode.Normal)
        icon.addPixmap(pixmap, QIcon.Mode.Selected)
        icon.addPixmap(pixmap, QIcon.Mode.Active)
        return icon

    def _display_pixmap(self, image_data: Dict) -> Optional[QPixmap]:
        """这张图现在该显示成什么样：底图，或者底图叠上框。没有底图就返回 None。"""
        base = self.thumbnail_cache.get(image_data.get('storage_path', ''))
        if base is None or base.isNull():
            return None

        if not self.show_annotation_boxes:
            return base

        image_id = image_data.get('id')
        version = self._annotation_versions.get(image_id)
        if version is None:
            return base  # 没标注：不画空框

        cached = self._overlay_cache.get(image_id)
        if cached is not None and cached[0] == version:
            return cached[1]  # 这张图的标注没变，直接用上次画好的

        boxes = bbox_preview_boxes(
            self._bbox_previews.get(image_id, []),
            image_data.get('width') or 0,
            image_data.get('height') or 0,
            base.width(), base.height(),
        )
        composed = draw_boxes_on_thumbnail(base, boxes, self._class_colors) if boxes else base
        self._overlay_cache[image_id] = (version, composed)
        return composed

    def _apply_item_icon(self, item: QListWidgetItem, image_data: Dict) -> bool:
        """给格子设置图标；底图还没读出来（没缓存）时返回 False，交给后台线程去读。"""
        pixmap = self._display_pixmap(image_data)
        if pixmap is None:
            return False
        item.setIcon(self._thumbnail_icon(pixmap))
        return True

    def _apply_base_icon(self, item: QListWidgetItem, image_data: Dict) -> bool:
        """只贴没有框的底图（缓存里那张）；没缓存返回 False，交给后台线程去读盘。

        建列表的循环专用：600 个格子要在一次调用里建完，这里绝不能顺手合成框
        ——那是 600 次 QPainter，界面会直接僵住。框等后台把标注查回来之后再分块叠。
        """
        base = self.thumbnail_cache.get(image_data.get('storage_path', ''))
        if base is None or base.isNull():
            return False
        item.setIcon(self._thumbnail_icon(base))
        return True

    def _image_by_id(self, image_id) -> Optional[Dict]:
        return next((img for img in self.images if img.get('id') == image_id), None)

    def on_toggle_annotation_boxes(self, checked: bool):
        """「显示标注框」开关：只换图，不写数据库、不重读磁盘。

        点一下要立刻回到事件循环——所以这里只排队，不在这一次调用里把 600 张
        缩略图重画完（那样开关会「按下去半天弹不起来」）。
        """
        self.show_annotation_boxes = checked
        self._schedule_icon_refresh()

    def _cancel_icon_refresh(self):
        """让还排着队的分块重画作废：列表马上要清掉或者换一批图了。"""
        self._icon_refresh_generation += 1
        self._icon_refresh_pending = False

    def _schedule_icon_refresh(self):
        """按当前开关和标注，把列表里的格子重画一遍——分块跑，不一口气啃完。

        底图都在缓存里，全程不碰磁盘、不查数据库；每一块之间把控制权还给事件
        循环，所以刷新期间界面照样能滚动、能点。中途再来一次刷新（或者切了项目）
        会换一个世代号，排在队里的旧块自己就退出了，不会拿旧数据盖掉新状态。
        """
        self._icon_refresh_generation += 1
        self._icon_refresh_cursor = 0
        self._icon_refresh_pending = True
        generation = self._icon_refresh_generation
        QTimer.singleShot(0, lambda: self._refresh_icon_chunk(generation))

    def _refresh_icon_chunk(self, generation: int):
        """重画一块格子；没画完就把自己排到下一轮事件循环。"""
        if generation != self._icon_refresh_generation:
            return  # 过期的队列：期间又刷了一次，或者列表已经换了一批图

        by_id = {img['id']: img for img in self.images}
        deadline = time.monotonic() + self._ICON_CHUNK_BUDGET
        drawn = 0

        while self._icon_refresh_cursor < self.image_list.count():
            item = self.image_list.item(self._icon_refresh_cursor)
            self._icon_refresh_cursor += 1
            if item is not None:
                image_data = by_id.get(item.data(Qt.ItemDataRole.UserRole))
                if image_data:
                    self._apply_item_icon(item, image_data)
            drawn += 1
            # 先画再看预算：哪怕预算已经花光，一块也至少画一张，否则一张都画不出来
            if drawn >= self._ICON_CHUNK_SIZE or time.monotonic() >= deadline:
                break

        if self._icon_refresh_cursor < self.image_list.count():
            QTimer.singleShot(0, lambda: self._refresh_icon_chunk(generation))
        else:
            self._icon_refresh_pending = False

    def _apply_item_status(self, item: QListWidgetItem, image_data: Dict):
        """让「哪些图已经标过」在网格里一眼看得出来：勾号 + 绿字，不只靠颜色。

        格子底下写的是显示名（完整文件名在 tooltip 里）：一网格全是
        20260712_000602_575947_frame_000223.jpg 的话，每个格子只放得下前面那段
        一模一样的时间戳，等于每张图都没有名字。
        """
        annotated = image_data.get('status') == 'annotated'
        alias = self._image_display_name(image_data)

        if annotated:
            item.setText(f"✓ {alias}")
            item.setForeground(QColor(COLORS['success']))
            item.setBackground(QColor(COLORS['success_soft']))
        else:
            item.setText(alias)
            item.setForeground(QColor(COLORS['text_secondary']))
            item.setBackground(QColor(COLORS['panel']))

    def create_status_bar(self) -> QFrame:
        """创建状态栏"""
        status_bar = QFrame()
        status_bar.setObjectName("card")

        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)

        self.status_total = QLabel("共 0 张图片")
        layout.addWidget(self.status_total)

        self.status_annotated = QLabel("已标注 0")
        self.status_annotated.setStyleSheet(f"color: {COLORS['success']};")
        layout.addWidget(self.status_annotated)

        self.status_pending = QLabel("未标注 0")
        self.status_pending.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.status_pending)

        layout.addStretch()

        self.btn_refresh_status = QPushButton("刷新")
        self.btn_refresh_status.setObjectName("ghost")
        self.btn_refresh_status.setToolTip("重新读取当前项目的图片与标注状态")
        self.btn_refresh_status.clicked.connect(self.force_refresh_images)
        layout.addWidget(self.btn_refresh_status)

        return status_bar

    # ==================== 项目 ====================

    def set_project(self, project_id):
        """由主窗口调用：切换当前项目。"""
        self.stop_image_loading()
        if self._import_busy:
            # 项目context变了，旧项目的导入任务不再有意义：只请求取消、
            # 把这一批回调标记作废，但线程本身可能还在跑（cancel() 只是设个
            # 事件，最迟在下一个文件/帧边界才会真正退出）——不能在这里就丢掉
            # 引用，否则 QThread 对象会在仍在运行时被 GC，触发
            # "QThread: Destroyed while thread is still running"。
            # 把它转移到「已作废但还没退出」的列表里继续持有，真正退出
            # （原生 finished 信号）时再释放，见 _on_import_thread_actually_finished。
            if self._active_import_thread is not None:
                self._active_import_thread.cancel()
                self._retired_import_threads.append(self._active_import_thread)
                self._active_import_thread = None
            self._import_generation += 1
            self._end_import_ui()
        self.current_project_id = project_id
        self.images = []
        # 列表马上要被清空：排在队里的分块重画得先作废，不然它们会去画一批
        # 已经不存在的格子
        self._cancel_icon_refresh()
        self.image_list.clear()
        self.thumbnail_widgets.clear()
        # 换项目：合成好的带框缩略图全作废（底图按 storage_path 缓存，可以留着）
        self._invalidate_thumbnail_overlays()
        self._bbox_previews = {}
        self._annotation_versions = {}

        self.refresh_view_filter_options()
        self.update_project_controls()

        if project_id:
            self.load_project_images()
        else:
            self.update_status_bar()
            self.update_view_mode()

    def update_project_controls(self):
        """刷新任务类型胶囊与各处可用性（没有项目就全灰掉）。"""
        has_project = bool(self.current_project_id)

        project = db.get_project(self.current_project_id) if has_project else None
        if project:
            self.btn_task_type.setText(f"任务：{short_task_label(project.get('type'))}")
        else:
            self.btn_task_type.setText("任务：未设置")

        for control in (
            self.btn_task_type,
            self.btn_import_folder, self.btn_import_images,
            self.btn_import_video, self.btn_import_annotations,
            self.btn_manage, self.btn_refresh_status, self.view_combo,
            self.chk_show_boxes,
        ):
            control.setEnabled(has_project)

        self.update_manage_action_state()

        # 导入任务进行中：即使有项目，也不能再启动新导入或破坏当前项目
        if self._import_busy:
            self._set_import_controls_enabled(False)

    def update_view_mode(self):
        """在「没项目 / 没图片 / 有图片」三种状态之间切换。"""
        has_project = bool(self.current_project_id)

        # 没有项目时，工具栏和状态栏全是灰的、没意义，直接收起来，只留一句引导
        self.toolbar.setVisible(has_project)
        self.status_bar.setVisible(has_project)

        if not has_project:
            self.view_stack.setCurrentIndex(0)
        elif not self.images:
            self.view_stack.setCurrentIndex(1)
        else:
            self.view_stack.setCurrentIndex(2)

    def force_refresh_images(self):
        """刷新按钮：强制重读一次。"""
        if self.current_project_id:
            self.load_project_images()

    def refresh_project_images(self):
        """由主窗口在进入本页时调用：数据变了才重建列表。

        「变了」不能只看 status：在标注页把框拖到别处、改成另一个类别，图片状态
        还是 annotated，只比 status 的话这里会认为什么都没发生，缩略图就永远停在
        旧的框上。所以签名里带上每张图的标注版本号（框数 + 最大标注 id + 改动时间）。

        类别配色同理：把「人」改成另一个颜色，标注一个字节都没动，但框该换色了。

        真要重载也不会去读盘：底图按 storage_path 缓存着，重建列表只是把框重画一遍，
        而且只重画版本号对不上的那几张。
        """
        if not self.current_project_id:
            return

        latest = db.get_project_images(self.current_project_id)
        latest_versions = db.get_project_annotation_versions(self.current_project_id)

        latest_signature = [
            (img['id'], img.get('status'), latest_versions.get(img['id']))
            for img in latest
        ]
        current_signature = [
            (img['id'], img.get('status'), self._annotation_versions.get(img['id']))
            for img in self.images
        ]
        if (latest_signature == current_signature
                and self._load_class_colors(self.current_project_id) == self._class_colors):
            return

        self.load_project_images()

    def change_task_type(self):
        """修改当前项目的任务类型。"""
        if not self.current_project_id:
            return

        project = db.get_project(self.current_project_id)
        current = (project or {}).get('type') or 'detect'

        task_type = ask_task_type(self, current=current, title="修改任务类型")
        if not task_type:
            return

        db.update_project(self.current_project_id, type=task_type)
        self.update_project_controls()

    def create_new_project(self):
        """创建新项目：先起名字，再选任务类型。"""
        name = ask_text(
            self, "新建项目",
            "给项目起个名字，之后图片、标注和训练结果都存在这个项目里。",
            placeholder="比如：安全帽检测",
            confirm_text="创建项目",
        )
        # 取消，或者名字是空的 / 只有空格：ask_text 都返回 None
        if not name:
            return

        task_type = ask_task_type(self, current='detect', title="这个项目要做什么")
        if not task_type:
            return

        project_id = db.create_project(
            name=name,
            description="",
            project_type=task_type,
            classes=[]
        )
        # 交给主窗口去刷新下拉框并选中新项目
        self.projects_changed.emit(project_id)

    def delete_current_project(self):
        """删除当前项目。"""
        if not self.current_project_id:
            return

        project = db.get_project(self.current_project_id)
        project_name = (project or {}).get('name', '')

        if not confirm_destructive(
            self,
            "删除项目",
            f"项目「{project_name}」里的所有图片和标注都会一起删除，无法恢复。",
            detail=f"当前项目有 {len(self.images)} 张图片。",
            confirm_text="删除项目",
        ):
            return

        try:
            self.stop_image_loading()
            project_storage_path = (project or {}).get('storage_path', '')

            db.delete_project(self.current_project_id)

            if project_storage_path and os.path.exists(project_storage_path):
                try:
                    import shutil
                    shutil.rmtree(project_storage_path)
                except Exception as e:
                    # 文件夹删不掉不影响项目已经删除的事实
                    print(f"删除项目文件夹失败: {e}")

            removed_storage_paths = [img.get('storage_path', '') for img in self.images]
            self.current_project_id = None
            self._cancel_icon_refresh()
            self.image_list.clear()
            self.images = []
            self.thumbnail_widgets.clear()
            self._remove_cached_thumbnails(removed_storage_paths)
            self._invalidate_thumbnail_overlays()

            self.update_status_bar()
            self.update_project_controls()
            self.update_view_mode()
            self.projects_changed.emit(None)

        except Exception as e:
            show_warning(self, "删除项目失败", str(e))

    def load_project_images(self):
        """加载项目图像 - 使用多线程"""
        if not self.current_project_id:
            return

        # 全量重载会把最新数据（含刚导入的）一次性读回来，之前那个
        # 「导入后台正在整理缩略图」的收尾就不用再等了，直接结束掉，
        # 避免它的世代被下面的 stop_image_loading 作废后再也等不到回调、卡住不收尾。
        if self._import_finalize_pending:
            self._end_import_ui(self._import_finalize_summary, cancelled=self._import_finalize_cancelled)

        # 停止之前的加载，并创建新的加载世代
        self.stop_image_loading(reset_progress=False)

        # 清空列表（先作废排队中的分块重画：它们画的是马上要没的那批格子）
        self._cancel_icon_refresh()
        self.image_list.clear()
        self.thumbnail_widgets.clear()

        # 从数据库获取图片列表（很快）
        self.images = db.get_project_images(self.current_project_id)
        self._refresh_image_display_names()
        # 标注框：整批读，而且是在后台线程里读——回来之后才分块叠到格子上
        self._load_annotation_previews()
        self.update_status_bar()
        self.update_view_mode()

        if not self.images:
            self.progress_bar.setVisible(False)
            return

        # 先创建所有列表项（显示占位符）
        uncached_tasks = []
        for index, image_data in enumerate(self.images):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, image_data['id'])
            item.setToolTip(self._image_tooltip(image_data))

            self._apply_item_status(item, image_data)

            # 设置项目大小提示，确保即使没有图标也有足够高度
            item.setSizeHint(QSize(180, 200))

            # 底图已经在缓存里就先按原样贴上（框等后台查回来再分块叠），
            # 否则丢给后台线程去读盘
            if not self._apply_base_icon(item, image_data):
                uncached_tasks.append((index, image_data))

            self.image_list.addItem(item)

        self.filter_images(self.view_combo.currentText())

        if not uncached_tasks:
            self.progress_bar.setVisible(False)
            return

        self._start_thumbnail_worker(uncached_tasks)

    def _start_thumbnail_worker(self, tasks: List[Tuple[int, Dict]], on_finished: Callable[[], None] = None):
        """启动缩略图后台加载线程。

        跟导入任务状态完全独立：这里只负责把 tasks（(行号, 图片记录) 列表）
        对应的缩略图在后台生成好，不做任何导入相关判断。
        """
        self.stop_image_loading(reset_progress=False)
        current_generation = self._image_load_generation

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(tasks))
        self.progress_bar.setValue(0)

        self.load_worker = ImageLoadWorker(tasks)
        self.load_worker.image_loaded.connect(
            lambda index, pixmap, storage_path, generation=current_generation: self.on_image_loaded(
                generation, index, pixmap, storage_path
            )
        )
        self.load_worker.progress.connect(
            lambda current, total, generation=current_generation: self.on_load_progress(
                generation, current, total
            )
        )
        self.load_worker.finished_loading.connect(
            lambda generation=current_generation, cb=on_finished: self.on_load_finished(generation, cb)
        )
        self.load_worker.start()

    def on_image_loaded(self, generation: int, index: int, pixmap: QPixmap, storage_path: str):
        """单个图片加载完成回调（在主线程执行）"""
        if generation != self._image_load_generation:
            return

        # 先进缓存，再画格子：_apply_item_icon 要从缓存里取底图。
        # 缓存里存的永远是后台线程读出来的原图（不带框）。
        self.thumbnail_cache[storage_path] = pixmap

        if index < self.image_list.count():
            item = self.image_list.item(index)
            if item:
                image_data = self._image_by_id(item.data(Qt.ItemDataRole.UserRole))
                if image_data is not None:
                    self._apply_item_icon(item, image_data)
                else:
                    item.setIcon(self._thumbnail_icon(pixmap))

    def on_load_progress(self, generation: int, current: int, total: int):
        """加载进度回调"""
        if generation != self._image_load_generation:
            return
        self.progress_bar.setValue(current)

    def on_load_finished(self, generation: int, on_finished: Callable[[], None] = None):
        """加载完成回调。

        只收自己这条线的尾：进度条是缩略图加载的，收掉；loading_overlay 不是——
        那是「正在导入 YOLO/COCO/VOC 标注」的遮罩，由 on_annotation_import_finished
        收。缩略图加载在标注导入期间本来就会被触发（导入完要重建列表），以前这里
        顺手把它删掉，用户看到的就是「导入提示刚亮起来就没了」，而标注其实还在导。
        """
        if generation != self._image_load_generation:
            return
        worker = self.load_worker
        self.load_worker = None
        if worker is not None:
            # finished_loading 跟 ImportWorkerThread 的 result_ready 是同一类问题：
            # 在 run() 返回前手动 emit，不代表线程已经真正退出。这里同步等一下
            # （此时线程本来就快跑完了，代价可以忽略），确保丢引用前线程已经
            # 真正结束，避免它在还在运行时被 GC 掉。
            worker.wait()
        self.progress_bar.setVisible(False)

        if on_finished:
            on_finished()

    def _build_image_list_item(self, image_data: Dict) -> QListWidgetItem:
        """创建图片列表项（占位图标）"""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, image_data['id'])

        group_id = image_data.get('group_id')
        group_name = "未分组"
        if group_id:
            group = db.get_image_group(group_id)
            if group:
                group_name = group['name']

        item.setToolTip(self._image_tooltip(image_data, group_name=group_name))

        self._apply_item_status(item, image_data)

        item.setSizeHint(QSize(180, 200))
        return item

    def _append_imported_images(self, before_image_ids: set,
                                 on_thumbnails_ready: Callable[[], None] = None) -> int:
        """导入后仅增量追加新图片，返回追加数量。

        缩略图改为丢给后台的 ImageLoadWorker 生成，不在 UI 线程同步 cv2.imread；
        `on_thumbnails_ready` 保证恰好被调用一次——没有新图/缩略图全部命中缓存时
        同步调用，否则等后台线程真正做完再调用。
        """
        if not self.current_project_id:
            if on_thumbnails_ready:
                on_thumbnails_ready()
            return 0

        latest_images = db.get_project_images(self.current_project_id)
        # 同时排除 before_image_ids（导入前已有的）和 self.images 里已存在的
        # （可能被中途的一次全量重载提前补上了），避免同一张图重复出现
        existing_ids = {img.get('id') for img in self.images}
        new_images = [
            img for img in latest_images
            if img.get('id') not in before_image_ids and img.get('id') not in existing_ids
        ]
        if not new_images:
            if on_thumbnails_ready:
                on_thumbnails_ready()
            return 0

        self.images.extend(new_images)
        self._refresh_image_display_names()
        # 导入标注（YOLO/COCO/VOC）走的也是导入路径，新图可能一进来就带框
        self._load_annotation_previews()

        uncached_tasks = []
        for image_data in new_images:
            item = self._build_image_list_item(image_data)
            self.image_list.addItem(item)
            row_index = self.image_list.count() - 1

            # 同样只贴底图：一次导入可能进来几百张，框留给分块刷新
            if not self._apply_base_icon(item, image_data):
                uncached_tasks.append((row_index, image_data))

        # 新导入的图可能和已有的重名（另一段视频的同一个帧号），
        # 那样老格子也得补上区分信息——所以整列表重刷一遍文字
        self._refresh_item_labels()

        self.update_status_bar()
        self.update_view_mode()
        self.filter_images(self.view_combo.currentText())

        if uncached_tasks:
            self._start_thumbnail_worker(uncached_tasks, on_finished=on_thumbnails_ready)
        elif on_thumbnails_ready:
            on_thumbnails_ready()

        return len(new_images)

    def _refresh_item_labels(self):
        """按当前的显示名，把所有格子的文字和提示重刷一遍。"""
        by_id = {img['id']: img for img in self.images}
        group_names = {
            group['id']: group['name']
            for group in db.get_project_image_groups(self.current_project_id)
        } if self.current_project_id else {}
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_data = by_id.get(item.data(Qt.ItemDataRole.UserRole))
            if image_data:
                self._apply_item_status(item, image_data)
                group_name = group_names.get(image_data.get('group_id'), "未分组")
                item.setToolTip(self._image_tooltip(image_data, group_name=group_name))

    # ==================== 图片显示名称规则（U6） ====================

    def open_display_name_rule_dialog(self):
        """管理 → 图片显示名称…：整个项目切换显示名规则，只改界面上的字，不改文件名。"""
        if not self.current_project_id:
            return

        project = db.get_project(self.current_project_id)
        if not project:
            return

        project_name = project.get('name', '')
        current_rule = parse_display_name_rule(project.get('display_name_rule'))
        all_images = db.get_project_images(self.current_project_id)

        dialog = DisplayNameRuleDialog(
            self,
            current_rule=current_rule,
            sample_images=all_images,
            project_name=project_name,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        rule = dialog.selected_rule()
        if rule is None:
            return

        # 对话框只预览了前三张；真正保存前必须对项目全部图片做一次完整的
        # 合法性/唯一性校验，冲突了就不保存，把原因原样告诉用户。
        try:
            build_project_display_names(rule, all_images, project_name)
        except ValueError as exc:
            show_warning(self, "显示名称规则不合法", str(exc))
            return

        if not db.update_project(self.current_project_id, display_name_rule=rule):
            # 没写进去就不能假装成功：列表保持原样，也不能通知标注页去刷一个
            # 数据库里其实没有的规则。
            show_warning(self, "保存失败", "显示名称规则没有写入成功，请重试。")
            return

        self._refresh_image_display_names()
        self._refresh_item_labels()
        self.display_name_rule_changed.emit(self.current_project_id)

    def refresh_view_filter_options(self):
        """刷新筛选下拉框（含分组列表）。"""
        current_filter = self.view_combo.currentText() if hasattr(self, 'view_combo') else "全部"
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("全部", "all")
        self.view_combo.addItem("未标注", "pending")
        self.view_combo.addItem("已标注", "annotated")
        self.view_combo.addItem("未分组", "ungrouped")

        if self.current_project_id:
            groups = db.get_project_image_groups(self.current_project_id)
            counts = db.get_group_image_counts(self.current_project_id)
            for group in groups:
                count = counts.get(group['id'], 0)
                label = f"分组: {group['name']} ({count})"
                self.view_combo.addItem(label, group['id'])

        restored = False
        for i in range(self.view_combo.count()):
            if self.view_combo.itemText(i) == current_filter:
                self.view_combo.setCurrentIndex(i)
                restored = True
                break
        if not restored:
            self.view_combo.setCurrentIndex(0)
        self.view_combo.blockSignals(False)

    
    def filter_images(self, filter_text: str):
        """筛选图像"""
        filter_data = self.view_combo.currentData()
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_id = item.data(Qt.ItemDataRole.UserRole)
            
            image_data = next((img for img in self.images if img['id'] == image_id), None)
            if not image_data:
                continue
            
            status = image_data.get('status', 'pending')
            group_id = image_data.get('group_id')
            visible = True

            if filter_data == "all":
                visible = True
            elif filter_data == "pending":
                visible = status == 'pending'
            elif filter_data == "annotated":
                visible = status == 'annotated'
            elif filter_data == "ungrouped":
                visible = group_id is None
            elif isinstance(filter_data, int):
                visible = group_id == filter_data
            else:
                if filter_text == "全部":
                    visible = True
                elif filter_text == "未标注":
                    visible = status == 'pending'
                elif filter_text == "已标注":
                    visible = status == 'annotated'
                elif filter_text == "未分组":
                    visible = group_id is None
                elif filter_text.startswith("分组:"):
                    visible = False
                    if group_id:
                        group = db.get_image_group(group_id)
                        if group and f"分组: {group['name']}" in filter_text:
                            visible = True

            item.setHidden(not visible)
    
    def on_image_clicked(self, item: QListWidgetItem):
        """图像点击事件"""
        image_id = item.data(Qt.ItemDataRole.UserRole)
        # TODO: 实现图像预览或编辑
        pass
    
    def update_status_bar(self):
        """更新状态栏，并把进度变化告诉主窗口"""
        total = len(self.images)
        annotated = sum(1 for img in self.images if img.get('status') == 'annotated')
        pending = total - annotated

        self.status_total.setText(f"共 {total} 张图片")
        self.status_annotated.setText(f"已标注 {annotated}")
        self.status_pending.setText(f"未标注 {pending}")

        # 图片数变了：「清空全部图片」在没有图片时就该是灰的
        self.update_manage_action_state()

        self.project_data_changed.emit()

    # ==================== 导入任务状态 ====================
    # 下面这组方法管理「导入任务」本身的状态（进行中 / 取消 / 完成），
    # 跟缩略图加载状态（load_project_images / on_load_finished 那组）完全分开。

    def _set_import_controls_enabled(self, enabled: bool):
        """导入中要禁掉的，只有会启动重复导入、或者会破坏当前项目的操作。"""
        for button in (
            self.btn_import_folder, self.btn_import_images,
            self.btn_import_video, self.btn_import_annotations,
        ):
            button.setEnabled(enabled)

        # 破坏性的两个现在在「管理」菜单里：禁的是菜单项，不是整个菜单——
        # 导入中仍然允许移动分组、删除选中的图片。具体谁能点由这里统一算，
        # 它读的是 self._import_busy，所以调用顺序上必须先把 busy 标志改好。
        self.update_manage_action_state()

    def _start_import_ui(self, message: str, indeterminate: bool = True):
        """进入「导入中」状态：用户点确认导入后，一个事件循环内就要看到这个。"""
        self._import_busy = True
        self._import_finalize_pending = False

        self.import_status_frame.setVisible(True)
        self.import_status_label.setText(message)
        if indeterminate:
            self.import_progress_bar.setRange(0, 0)  # 总量未知：忙碌态
        else:
            self.import_progress_bar.setRange(0, 100)
            self.import_progress_bar.setValue(0)

        self.btn_cancel_import.setVisible(True)
        self.btn_cancel_import.setEnabled(True)
        self._set_import_controls_enabled(False)

    def _set_import_progress(self, progress: int, message: str = None):
        """更新导入状态文字 / 进度；progress < 0 表示总量未知，切到忙碌态。"""
        if message is not None:
            self.import_status_label.setText(message)
        if progress is None or progress < 0:
            self.import_progress_bar.setRange(0, 0)
        else:
            self.import_progress_bar.setRange(0, 100)
            self.import_progress_bar.setValue(min(max(progress, 0), 100))

    def _end_import_ui(self, summary: str = None, cancelled: bool = False):
        """结束导入（成功/取消/失败都会走这里）：恢复控件，取消按钮收起。

        `summary` 有值时，把结果留在页面里（页内状态），不用弹窗打断操作；
        没有值（比如失败场景，调用方会另外弹错误框）时直接把状态条收起来。
        `cancelled` 为真时不把进度条拉满到 100%——取消是半途而废，不是
        「普通完成」，进度条应该停在中断时的真实进度上，不能误导成已完成。

        注意：这里不清空 self._active_import_thread —— 线程是否真的退出了
        由 QThread 原生 finished 信号决定（见 _on_import_thread_actually_finished），
        这里只是 UI 收尾，跟线程生命周期分开管理。
        """
        self._import_busy = False
        self._import_finalize_pending = False

        self.btn_cancel_import.setVisible(False)
        if summary:
            self.import_status_label.setText(summary)
            if not cancelled:
                self.import_progress_bar.setRange(0, 100)
                self.import_progress_bar.setValue(100)
            self.import_status_frame.setVisible(True)
        else:
            self.import_status_frame.setVisible(False)

        self.update_project_controls()

    def _cancel_active_import(self):
        """取消按钮：请求后台线程停止，最迟在下一个文件/帧边界生效。"""
        if self._active_import_thread is not None:
            self._active_import_thread.cancel()
            self.btn_cancel_import.setEnabled(False)
            self.import_status_label.setText("正在取消…")

    def _on_import_thread_actually_finished(self, thread):
        """QThread 原生 finished：线程真的退出了，这里才是唯一安全释放引用的地方。

        跟 result_ready（业务结果，run() 返回前手动 emit）分开：无论正常完成、
        失败、取消，还是切项目导致的作废，最终都会走到这里——只有这里能保证
        `thread.isRunning()` 已经是 False，不会出现线程还在跑就被 GC / deleteLater
        的情况。
        """
        if self._active_import_thread is thread:
            self._active_import_thread = None
        if thread in self._retired_import_threads:
            self._retired_import_threads.remove(thread)
        # finished 信号发出时线程已经在退出的路上，wait() 在这里只是确保万无
        # 一失（此时通常立即返回），之后再安全地交给 Qt 事件循环销毁对象。
        thread.wait()
        thread.deleteLater()

    def _start_data_import(self, kind: str, source, group_id,
                            frame_interval: int = 1, initial_message: str = "",
                            indeterminate: bool = True,
                            video_mode: str = VIDEO_MODE_INTERVAL,
                            sample_count: int = None):
        """统一入口：文件夹 / 多图 / 视频导入都从这里起后台线程。"""
        if not self.current_project_id:
            return
        if self._import_busy:
            show_info(self, "已有导入任务在进行", "等这一个跑完再开始下一个。")
            return

        self._import_generation += 1
        generation = self._import_generation
        self._import_before_ids = {img.get('id') for img in self.images}

        self._start_import_ui(initial_message, indeterminate=indeterminate)

        thread = ImportWorkerThread(
            self.current_project_id, group_id, kind, source,
            frame_interval=frame_interval,
            video_mode=video_mode, sample_count=sample_count,
        )
        thread.progress_updated.connect(
            lambda progress, message, g=generation: self._on_import_progress(g, progress, message)
        )
        thread.result_ready.connect(
            lambda success, error, cancelled, imported, skipped, g=generation, t=thread:
            self._on_import_worker_finished(g, t, success, error, cancelled, imported, skipped)
        )
        # QThread 原生 finished：只有它才代表线程真的退出了（result_ready 是我们
        # 自己在 run() 返回前手动 emit 的业务结果，不代表 OS 线程已经结束）。
        # 只有在这里才真正释放引用，避免 "Destroyed while thread is still running"。
        thread.finished.connect(lambda t=thread: self._on_import_thread_actually_finished(t))
        self._active_import_thread = thread
        thread.start()

    def _on_import_progress(self, generation: int, progress: int, message: str):
        if generation != self._import_generation:
            return
        self._set_import_progress(progress, message)

    def _on_import_worker_finished(self, generation: int, thread, success: bool, error: str,
                                    cancelled: bool, imported: int, skipped: int):
        # result_ready 是线程自己在 run() 返回前手动 emit 的，这一刻 run() 几乎
        # 已经跑完但严格意义上还没退出（QThread 原生 finished/isRunning() 变
        # False 要稍后才会发生）。同步 wait() 一下，把这个窗口关掉——此时线程
        # 本来就即将结束，等待成本可以忽略不计，但能确保调用方（包括后面可能
        # 立刻释放 self 的场景）拿到结果时线程已经真正退出，不会有人在这个
        # 窗口期把仍在运行的 QThread 销毁掉。
        thread.wait()

        if generation != self._import_generation:
            return

        if not success:
            self._end_import_ui()
            show_warning(self, "导入失败", "导入过程中出错，没有改动项目里的图片。", detail=error)
            return

        prefix = "已取消" if cancelled else "导入完成"
        summary = f"{prefix}：成功导入 {imported} 张，跳过 {skipped} 张"

        if imported > 0:
            self._import_finalize_pending = True
            self._import_finalize_summary = summary
            self._import_finalize_cancelled = cancelled
            self._set_import_progress(-1, "正在整理缩略图…")
            appended = self._append_imported_images(
                self._import_before_ids,
                on_thumbnails_ready=lambda g=generation, s=summary, c=cancelled: self._finish_import_finalization(g, s, c),
            )
            if appended == 0:
                # 兜底：数据库里应该有新图但没识别出来，回退全量重载保证界面和数据库一致
                # （on_thumbnails_ready 在这条分支里已经同步跑过，导入态已经收尾了）
                self.load_project_images()
        else:
            self._end_import_ui(summary, cancelled=cancelled)

        self.refresh_view_filter_options()

    def _finish_import_finalization(self, generation: int, summary: str, cancelled: bool = False):
        """后台缩略图整理真正完成后才收尾——这时才算导入任务完全结束。"""
        if generation != self._import_generation:
            return
        self._end_import_ui(summary, cancelled=cancelled)

    def import_folder(self):
        """导入文件夹"""
        if not self.current_project_id:
            show_warning(self, "还没有项目", "先在左边选一个项目，或者新建一个。")
            return
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            proceed, group_id = ask_import_group(self, self.current_project_id)
            if not proceed:
                return
            self.process_folder_import(folder_path, group_id=group_id)
    
    def import_images(self):
        """导入单张或多张图片"""
        if not self.current_project_id:
            show_warning(self, "还没有项目", "先在左边选一个项目，或者新建一个。")
            return
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*.*)"
        )
        
        if file_paths:
            proceed, group_id = ask_import_group(self, self.current_project_id)
            if not proceed:
                return
            self.process_image_import(file_paths, group_id=group_id)
    
    def import_video(self):
        """导入视频"""
        if not self.current_project_id:
            show_warning(self, "还没有项目", "先在左边选一个项目，或者新建一个。")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if file_path:
            proceed, group_id = ask_import_group(self, self.current_project_id)
            if not proceed:
                return
            self.process_video_import(file_path, group_id=group_id)
    
    def process_video_import(self, file_path: str, group_id: int = None):
        """处理视频导入：抽帧方案问完，后台线程负责剩下的一切。

        元信息（总帧数 / fps / 时长）在弹窗里就读好了——只读属性，不解码，
        所以「打开一个大视频看看能抽几张」不会卡住界面，也不会先导入一堆帧。
        """
        if not self.current_project_id:
            return

        try:
            plan = ask_video_extract_plan(self, file_path)
        except ValueError as exc:
            show_warning(self, "打不开这个视频", str(exc))
            return

        if not plan:
            return

        name = Path(file_path).name
        if plan['mode'] == VIDEO_MODE_RANDOM:
            initial_message = f"正在打开视频: {name}（随机抽取 {plan['sample_count']} 张）"
        else:
            initial_message = f"正在打开视频: {name}（每 {plan['frame_interval']} 帧取 1 张）"

        self._start_data_import(
            'video', file_path, group_id,
            frame_interval=plan['frame_interval'],
            video_mode=plan['mode'],
            sample_count=plan['sample_count'],
            initial_message=initial_message,
        )

    def import_annotations(self):
        """导入已有标注"""
        if not self.current_project_id:
            show_warning(self, "还没有项目", "先在左边选一个项目，或者新建一个。")
            return
        
        # 检查项目是否有任务标签
        project = db.get_project(self.current_project_id)
        if not project:
            show_warning(self, "读不到项目信息", "项目可能已经被删除，换一个试试。")
            return
        
        task_type = project.get('type')
        if not task_type or task_type not in ['detect', 'segment', 'pose', 'classify']:
            task_type = ask_task_type(self, current='detect', title="这个项目要做什么")
            if not task_type:
                return
            db.update_project(self.current_project_id, type=task_type)
            self.update_project_controls()

        # 选择标注格式
        from PyQt6.QtWidgets import QRadioButton

        dialog = QDialog(self)
        dialog.setWindowTitle("选择标注格式")
        dialog.setMinimumWidth(360)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        label = QLabel("你的标注文件是哪种格式？")
        label.setObjectName("subtitle")
        layout.addWidget(label)

        yolo_radio = QRadioButton("YOLO 格式")
        yolo_radio.setToolTip("一张图片配一个 .txt")
        yolo_radio.setChecked(True)
        layout.addWidget(yolo_radio)

        coco_radio = QRadioButton("COCO 格式")
        coco_radio.setToolTip("整个数据集一个 .json")
        layout.addWidget(coco_radio)

        voc_radio = QRadioButton("Pascal VOC 格式")
        voc_radio.setToolTip("一张图片配一个 .xml")
        layout.addWidget(voc_radio)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 0)
        btn_layout.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        ok_btn = QPushButton("确定")
        ok_btn.setObjectName("primary")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        proceed, group_id = ask_import_group(self, self.current_project_id)
        if not proceed:
            return
        
        if yolo_radio.isChecked():
            self.import_yolo_annotations(group_id=group_id)
        elif coco_radio.isChecked():
            self.import_coco_annotations(group_id=group_id)
        elif voc_radio.isChecked():
            self.import_voc_annotations(group_id=group_id)
    
    def import_yolo_annotations(self, group_id: int = None):
        """导入YOLO标注"""
        labels_dir = QFileDialog.getExistingDirectory(
            self, "选择YOLO标签文件夹 (labels)", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not labels_dir:
            return
        
        need_images_dir = confirm(
            self, "图像文件夹",
            "标签文件和图片不在同一个目录时，要另外指定图片所在的文件夹。",
            detail="如果 .txt 和图片就放在一起，选「不用」。",
            confirm_text="去选择",
            cancel_text="不用",
        )

        images_dir = None
        if need_images_dir:
            images_dir = QFileDialog.getExistingDirectory(
                self, "选择图像文件夹 (images)", "",
                QFileDialog.Option.ShowDirsOnly
            )
        
        # 检查项目是否已经有标注
        project_images = db.get_project_images(self.current_project_id)
        has_annotations = False
        for image in project_images:
            annotations = db.get_image_annotations(image['id'])
            if annotations:
                has_annotations = True
                break
        
        # 如果有标注，提示是否覆盖
        overwrite = False
        if has_annotations:
            overwrite = confirm_destructive(
                self, "覆盖已有标注",
                "项目里已经有标注了。继续导入会用新文件里的标注覆盖它们，覆盖后无法恢复。",
                detail="两种选择都会继续导入；选「保留现有标注」只是不动已经标好的那些图片。",
                confirm_text="覆盖",
                cancel_text="保留现有标注",
            )
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在导入YOLO标注...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来执行导入操作
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class AnnotationImportThread(QThread):
            """标注导入线程"""
            
            finished = pyqtSignal(bool, str, int, int)
            
            def __init__(self, project_id, labels_dir, images_dir, overwrite, group_id=None):
                super().__init__()
                self.project_id = project_id
                self.labels_dir = labels_dir
                self.images_dir = images_dir
                self.overwrite = overwrite
                self.group_id = group_id
            
            def run(self):
                """运行导入"""
                try:
                    from core.annotation_importer import AnnotationImporter
                    importer = AnnotationImporter(self.project_id, group_id=self.group_id)
                    imported, skipped = importer.import_yolo_annotations(
                        self.labels_dir, self.images_dir, self.overwrite
                    )
                    self.finished.emit(True, "导入成功", imported, skipped)
                except Exception as e:
                    self.finished.emit(False, f"导入失败: {e}", 0, 0)
        
        # 创建并启动线程
        self.import_thread = AnnotationImportThread(
            self.current_project_id, labels_dir, images_dir, overwrite, group_id=group_id
        )
        self.import_thread.finished.connect(self.on_annotation_import_finished)
        self.import_thread.start()
    
    def on_annotation_import_finished(self, success, message, imported, skipped):
        """标注导入完成回调"""
        # 隐藏加载动画
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')
        
        # 重新加载项目图片
        self.load_project_images()
        self.refresh_view_filter_options()
        
        # 显示结果
        if success:
            show_info(self, "标注导入完成", f"导入了 {imported} 个标注，跳过 {skipped} 个。")
        else:
            show_warning(self, "标注导入失败", message)
    
    def import_coco_annotations(self, group_id: int = None):
        """导入COCO标注"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择COCO标注文件", "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        # 检查项目是否已经有标注
        project_images = db.get_project_images(self.current_project_id)
        has_annotations = False
        for image in project_images:
            annotations = db.get_image_annotations(image['id'])
            if annotations:
                has_annotations = True
                break
        
        # 如果有标注，提示是否覆盖
        overwrite = False
        if has_annotations:
            overwrite = confirm_destructive(
                self, "覆盖已有标注",
                "项目里已经有标注了。继续导入会用新文件里的标注覆盖它们，覆盖后无法恢复。",
                detail="两种选择都会继续导入；选「保留现有标注」只是不动已经标好的那些图片。",
                confirm_text="覆盖",
                cancel_text="保留现有标注",
            )
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在导入COCO标注...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来执行导入操作
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class AnnotationImportThread(QThread):
            """标注导入线程"""
            
            finished = pyqtSignal(bool, str, int, int)
            
            def __init__(self, project_id, file_path, overwrite, group_id=None):
                super().__init__()
                self.project_id = project_id
                self.file_path = file_path
                self.overwrite = overwrite
                self.group_id = group_id
            
            def run(self):
                """运行导入"""
                try:
                    from core.annotation_importer import AnnotationImporter
                    importer = AnnotationImporter(self.project_id, group_id=self.group_id)
                    imported, skipped = importer.import_coco_annotations(
                        self.file_path, self.overwrite
                    )
                    self.finished.emit(True, "导入成功", imported, skipped)
                except Exception as e:
                    self.finished.emit(False, f"导入失败: {e}", 0, 0)
        
        # 创建并启动线程
        self.import_thread = AnnotationImportThread(
            self.current_project_id, file_path, overwrite, group_id=group_id
        )
        self.import_thread.finished.connect(self.on_annotation_import_finished)
        self.import_thread.start()
    
    def import_voc_annotations(self, group_id: int = None):
        """导入VOC标注"""
        voc_dir = QFileDialog.getExistingDirectory(
            self, "选择VOC标注文件夹 (Annotations)", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not voc_dir:
            return
        
        # 检查项目是否已经有标注
        project_images = db.get_project_images(self.current_project_id)
        has_annotations = False
        for image in project_images:
            annotations = db.get_image_annotations(image['id'])
            if annotations:
                has_annotations = True
                break
        
        # 如果有标注，提示是否覆盖
        overwrite = False
        if has_annotations:
            overwrite = confirm_destructive(
                self, "覆盖已有标注",
                "项目里已经有标注了。继续导入会用新文件里的标注覆盖它们，覆盖后无法恢复。",
                detail="两种选择都会继续导入；选「保留现有标注」只是不动已经标好的那些图片。",
                confirm_text="覆盖",
                cancel_text="保留现有标注",
            )
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在导入VOC标注...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来执行导入操作
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class AnnotationImportThread(QThread):
            """标注导入线程"""
            
            finished = pyqtSignal(bool, str, int, int)
            
            def __init__(self, project_id, voc_dir, overwrite, group_id=None):
                super().__init__()
                self.project_id = project_id
                self.voc_dir = voc_dir
                self.overwrite = overwrite
                self.group_id = group_id
            
            def run(self):
                """运行导入"""
                try:
                    from core.annotation_importer import AnnotationImporter
                    importer = AnnotationImporter(self.project_id, group_id=self.group_id)
                    imported, skipped = importer.import_voc_annotations(
                        self.voc_dir, self.overwrite
                    )
                    self.finished.emit(True, "导入成功", imported, skipped)
                except Exception as e:
                    self.finished.emit(False, f"导入失败: {e}", 0, 0)
        
        # 创建并启动线程
        self.import_thread = AnnotationImportThread(
            self.current_project_id, voc_dir, overwrite, group_id=group_id
        )
        self.import_thread.finished.connect(self.on_annotation_import_finished)
        self.import_thread.start()
    
    def process_folder_import(self, folder_path: str, group_id: int = None):
        """处理文件夹导入：扫描、复制、写库全部丢给后台线程。"""
        if not self.current_project_id:
            return

        folder_name = Path(folder_path).name or folder_path
        self._start_data_import(
            'folder', folder_path, group_id,
            initial_message=f"正在准备导入文件夹: {folder_name}",
        )

    def process_image_import(self, file_paths: List[str], group_id: int = None):
        """处理图像导入：复制、写库全部丢给后台线程。"""
        if not self.current_project_id:
            return

        total = len(file_paths)
        self._start_data_import(
            'images', file_paths, group_id,
            initial_message=f"正在导入 0/{total} 张图片",
            indeterminate=False,
        )

    def clear_all_images(self):
        """清空所有图像"""
        if not self.images:
            return
        
        total = len(self.images)
        if not confirm_destructive(
            self, "清空图片",
            "当前项目里的图片会全部删除，标注也一起没了，无法恢复。",
            detail=f"共 {total} 张图片。",
            confirm_text=f"删除 {total} 张",
        ):
            return

        self.stop_image_loading()
        deleted = 0
        failed = 0

        # 使用副本迭代，避免删除过程中修改原列表导致遍历异常
        images_snapshot = list(self.images)
        for image in images_snapshot:
            if db.delete_image(image['id']):
                deleted += 1
            else:
                failed += 1

        # 全部删除成功时，直接本地清空，避免触发整页重载
        if failed == 0:
            removed_storage_paths = [img.get('storage_path', '') for img in self.images]
            self.images.clear()
            self._cancel_icon_refresh()
            self.image_list.clear()
            self.thumbnail_widgets.clear()
            self._remove_cached_thumbnails(removed_storage_paths)
            self._invalidate_thumbnail_overlays()
            self.update_status_bar()
            self.update_view_mode()
            show_info(self, "已清空", f"删除了 {deleted} 张图片。")
        else:
            # 部分失败时回退到全量重载，确保UI与数据库一致
            self.load_project_images()
            show_warning(self, "没有全部删掉", f"成功删除 {deleted} 张，失败 {failed} 张。")
    
    def move_selected_to_group(self):
        """将选中图片移动到指定分组"""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            show_info(self, "还没选图片", "先在下面的图片里选中要移动的那几张。")
            return
        if not self.current_project_id:
            return

        dialog = GroupSelectDialog(
            self,
            self.current_project_id,
            title="移动分组",
            allow_ungroup=True,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.was_cancelled():
            return

        group_id = dialog.get_selected_group_id()
        image_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        updated = db.assign_images_to_group(image_ids, group_id)

        for image_id in image_ids:
            image_data = next((img for img in self.images if img['id'] == image_id), None)
            if image_data:
                image_data['group_id'] = group_id

        group_label = "未分组"
        if group_id is not None:
            group = db.get_image_group(group_id)
            if group:
                group_label = group['name']

        for item in selected_items:
            image_id = item.data(Qt.ItemDataRole.UserRole)
            image_data = next((img for img in self.images if img['id'] == image_id), None)
            if image_data:
                item.setToolTip(self._image_tooltip(image_data, group_name=group_label))

        self.refresh_view_filter_options()
        self.filter_images(self.view_combo.currentText())

        show_info(self, "移动完成", f"已把 {updated} 张图片移到「{group_label}」。")

    def delete_selected_images(self):
        """删除选中的图片"""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            show_info(self, "还没选图片", "先在下面的图片里选中要删除的那几张。")
            return

        count = len(selected_items)
        if not confirm_destructive(
            self, "删除选中的图片",
            "选中的图片和它们的标注都会删掉，无法恢复。",
            detail=f"选中了 {count} 张图片。",
            confirm_text=f"删除 {count} 张",
        ):
            return

        self.stop_image_loading()
        deleted = 0
        failed = 0
        deleted_ids = []
        
        for item in selected_items:
            image_id = item.data(Qt.ItemDataRole.UserRole)
            if db.delete_image(image_id):
                deleted += 1
                deleted_ids.append(image_id)
            else:
                failed += 1

        # 仅移除已成功删除的项，避免每次删除都整页重载
        if deleted_ids:
            deleted_id_set = set(deleted_ids)

            # 先更新内存数据
            removed_storage_paths = {
                img.get('storage_path', '')
                for img in self.images
                if img.get('id') in deleted_id_set
            }
            self.images = [img for img in self.images if img.get('id') not in deleted_id_set]

            # 清理缩略图缓存
            for path in removed_storage_paths:
                if path in self.thumbnail_cache:
                    del self.thumbnail_cache[path]
            self._invalidate_thumbnail_overlays(deleted_id_set)

            # 再移除列表项（倒序删除避免索引变化）
            rows_to_remove = []
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) in deleted_id_set:
                    rows_to_remove.append(i)
            for row in reversed(rows_to_remove):
                self.image_list.takeItem(row)

            self.update_status_bar()
            self.update_view_mode()

        if failed == 0:
            show_info(self, "已删除", f"删除了 {deleted} 张图片。")
        else:
            show_warning(self, "没有全部删掉", f"成功删除 {deleted} 张，失败 {failed} 张。")
