# -*- coding: utf-8 -*-
"""批量标注的确认步骤（交付单元 2）。

盯的是一件事：一整批推理不能因为一次普通点击就开始跑。
围绕它的几条硬约束：

    取消 = 什么都没发生      不建线程、不写库、不改图片状态
    确认 = 只发生一次        双击、重复点，都只能起一个任务
    看到的 = 跑的            确认框上的范围/参数，就是真正交给线程的那一份快照
    覆盖听用户的            以前 BatchLabelingManager 里 overwrite 硬编码 True
    取消不是错误            用户喊停不该弹红色的「失败」
    没配模型不许瞎猜        以前会偷偷退回 yolov8n（还会联网下载）

全程不加载真实模型、不发网络请求：AutoLabeler / LLM 都换成假的。

运行：
    python -m pytest tests/test_batch_confirm.py -q
    或
    python tests/test_batch_confirm.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QDialog, QMenu  # noqa: E402

import core.auto_labeler as auto_labeler_module  # noqa: E402
from core.auto_labeler import BatchLabelingManager  # noqa: E402
from gui.pages import annotate_page as annotate_module  # noqa: E402
from gui.pages.annotate_page import AnnotatePage  # noqa: E402
from gui.styles import get_full_stylesheet  # noqa: E402
from gui.widgets.batch_confirm_dialog import (  # noqa: E402
    NOT_APPLICABLE, BatchConfirmDialog, BatchPlan, SCOPE_ALL, SCOPE_RANGE,
    SCOPE_UNLABELED,
)

_app = _bootstrap.app()
db = _bootstrap.db

_TMP = Path(tempfile.mkdtemp(prefix="ezyolo-batch-confirm-"))

# 批量这条路只认磁盘上已经存在的 .pt（不存在就报错，绝不联网下载）。
# 所以夹具里得真的躺着一个文件——内容无所谓，模型加载全程是假的。
_LOCAL_PT = _TMP / "fake_best.pt"
_LOCAL_PT.write_bytes(b"not-a-real-checkpoint")


# ==================== 夹具 ====================

def _make_project(image_count: int = 4, annotated: int = 1) -> tuple:
    """一个有图片的项目；前 `annotated` 张是已经标好的。"""
    project_id = _bootstrap.create_temp_project(
        name="批量确认测试", project_type="detect",
        classes=[{'id': 0, 'name': '人', 'color': '#FF0000'}],
    )
    image_ids = []
    for i in range(image_count):
        path = _TMP / f"p{project_id}_{i}.jpg"
        path.write_bytes(b"not-a-real-jpeg")
        image_id = db.add_image(
            project_id, path.name, str(path), width=64, height=48,
        )
        image_ids.append(image_id)

    for image_id in image_ids[:annotated]:
        db.add_annotation(
            image_id=image_id, project_id=project_id, class_id=0, class_name='人',
            annotation_type='bbox', data={'x': 1, 'y': 1, 'width': 5, 'height': 5},
        )
    return project_id, image_ids


def _plan(project_id, images, **kwargs) -> BatchPlan:
    defaults = dict(
        project_id=project_id,
        engine='yolo',
        images=images,
        scope=SCOPE_UNLABELED,
        model_label='fake_best.pt',
        model_path=str(_LOCAL_PT),
        conf=0.5,
        iou=0.45,
        overwrite=False,
    )
    defaults.update(kwargs)
    return BatchPlan(**defaults)


def _image_rows(project_id):
    return db.get_project_images(project_id)


def _annotation_count(project_id) -> int:
    return sum(
        len(db.get_image_annotations(img['id']))
        for img in db.get_project_images(project_id)
    )


def _page(project_id=None, images=None, settings=None) -> AnnotatePage:
    """一个真的标注页，只是把项目/图片/设置直接塞进去，不走后台加载线程。

    用真页面而不是替身：这些用例要验的正是页面自己那几道闸（防双击、
    引导去配置、状态保持），拿替身测等于什么都没测。
    """
    page = AnnotatePage()
    page.current_project_id = project_id
    page.images = list(images or [])
    page.auto_label_settings = settings
    # 菜单行高、宽度这些是全局样式表给的——真实的主窗口就是这么套上去的
    page.setStyleSheet(get_full_stylesheet())
    return page


# ==================== 快照本身 ====================

def test_plan_is_a_frozen_snapshot_of_its_inputs():
    """快照要真的冻住：调用方后面再动自己那份列表，影响不到已经确认的计划。"""
    images = [{'id': 1, 'storage_path': '/a.jpg', 'filename': 'a.jpg'}]
    mapping = {0: 1}

    plan = _plan(7, images, class_mapping=mapping)

    images.append({'id': 2})
    mapping[9] = 9
    images[0]['id'] = 999

    assert plan.count == 1, "快照跟着调用方的列表一起变了"
    assert plan.image_ids == [1], "快照里的图片被外面改掉了"
    assert plan.class_mapping == {0: 1}, "快照里的类别映射被外面改掉了"


def test_llm_plan_says_not_applicable_instead_of_showing_a_blank():
    """LLM 没有 conf / IoU：照实写「不适用」，不留空格子让人猜。"""
    plan = _plan(1, [{'id': 1}], engine='llm', conf=None, iou=None, scope=SCOPE_RANGE)

    assert plan.conf_text() == NOT_APPLICABLE
    assert plan.iou_text() == NOT_APPLICABLE
    assert plan.overwrite_text() == "否（保留已有标注）"


def test_overwrite_wording_spells_out_the_damage():
    plan = _plan(1, [{'id': 1}], overwrite=True)
    assert "不可撤销" in plan.overwrite_text()


# ==================== 确认框 ====================

def test_confirm_dialog_shows_everything_the_user_must_know():
    """处理张数、范围、模型、置信度、IoU、是否覆盖——六样都得在。"""
    images = [{'id': i, 'filename': f"{i}.jpg"} for i in range(3)]
    plan = _plan(1, images, scope=SCOPE_ALL, model_label='yolov8s', conf=0.35, iou=0.6)

    dialog = BatchConfirmDialog(None, plan)

    assert dialog.lbl_count.text() == "3 张"
    assert dialog.lbl_scope.text() == "全部图片"
    assert dialog.lbl_model.text() == "yolov8s"
    assert dialog.lbl_conf.text() == "0.35"
    assert dialog.lbl_iou.text() == "0.60"
    assert dialog.lbl_overwrite.text() == "否（保留已有标注）"
    assert dialog.btn_confirm.text() == "开始批量标注"
    assert dialog.btn_cancel.text() == "取消"


def test_cancel_is_the_default_button_and_has_focus():
    """默认焦点在取消：一整批推理不该是一个回车就开始的事。"""
    plan = _plan(1, [{'id': 1}])
    dialog = BatchConfirmDialog(None, plan)

    assert dialog.btn_cancel.isDefault(), "默认按钮不是取消"
    assert not dialog.btn_confirm.isDefault(), "确认按钮不该是默认按钮"


def test_range_selection_is_merged_into_the_confirm_dialog():
    """LLM 的起止范围并进确认框：一个框看全，不是先一个范围弹窗再一个确认弹窗。"""
    images = [{'id': i} for i in range(1, 7)]
    plan = _plan(1, images, engine='llm', conf=None, iou=None, scope=SCOPE_RANGE)

    dialog = BatchConfirmDialog(None, plan, allow_range=True)
    assert dialog.lbl_count.text() == "6 张"

    dialog.spin_start.setValue(2)
    dialog.spin_end.setValue(4)

    assert dialog.lbl_count.text() == "3 张", "改了范围，处理张数没跟着变"

    chosen = dialog.selected_plan()
    assert chosen.image_ids == [2, 3, 4], "确认框返回的快照跟界面上选的范围对不上"
    assert chosen.project_id == plan.project_id
    assert chosen.class_name == plan.class_name


def test_range_start_cannot_pass_the_end():
    plan = _plan(1, [{'id': i} for i in range(1, 5)], scope=SCOPE_RANGE)
    dialog = BatchConfirmDialog(None, plan, allow_range=True)

    dialog.spin_start.setValue(4)
    dialog.spin_end.setValue(2)

    assert dialog.spin_start.value() <= dialog.spin_end.value()
    assert dialog.lbl_count.text().endswith("张")


# ==================== 取消这条路上什么都不许发生 ====================

def test_cancelling_the_confirmation_creates_no_thread_no_write_no_status_change():
    """点取消：线程数不变、标注数不变、图片状态不变。"""
    project_id, _ = _make_project(image_count=4, annotated=1)

    page = _page(project_id, _image_rows(project_id), settings={
        'model_path': 'best.pt', 'model_task': 'detect',
        'conf_threshold': 0.5, 'iou_threshold': 0.45,
        'class_mapping': {}, 'only_unlabeled': True, 'overwrite_labels': False,
    })

    annotations_before = _annotation_count(project_id)
    statuses_before = {img['id']: img['status'] for img in _image_rows(project_id)}
    started = []

    plan = page._build_yolo_batch_plan(page.auto_label_settings)
    assert plan is not None and plan.count == 3, "未标注的应该是 3 张"

    # 确认框一律返回「取消」
    with patch.object(annotate_module, "confirm_batch_plan", return_value=None), \
         patch.object(AnnotatePage, "_start_yolo_batch", lambda self, p: started.append(p)):
        page.run_batch_inference()

    assert started == [], "取消之后仍然启动了批量任务"
    assert page.batch_labeling_manager is None, "取消却把批量管理器/线程建起来了"
    assert page._batch_running is False, "取消却把页面切进了「跑批中」"
    assert _annotation_count(project_id) == annotations_before, "取消却写了标注"
    assert {img['id']: img['status'] for img in _image_rows(project_id)} == statuses_before, \
        "取消却改了图片状态"

    page.deleteLater()
    db.delete_project(project_id)


# ==================== 确认之后：范围一致、只跑一次 ====================

def test_confirmed_scope_matches_the_image_ids_actually_executed():
    """确认框上说的范围，跟真正交给 manager 的 image id，必须一个不差。"""
    project_id, image_ids = _make_project(image_count=5, annotated=2)

    page = _page(project_id, _image_rows(project_id), settings={
        'model_path': 'best.pt', 'model_task': 'detect',
        'conf_threshold': 0.4, 'iou_threshold': 0.5,
        'class_mapping': {}, 'only_unlabeled': True, 'overwrite_labels': False,
    })

    plan = page._build_yolo_batch_plan(page.auto_label_settings)

    # 「仅未标注」= 后 3 张
    assert plan.scope == SCOPE_UNLABELED
    assert plan.image_ids == image_ids[2:], "范围算错了"

    dialog = BatchConfirmDialog(None, plan)
    assert dialog.lbl_count.text() == "3 张"
    assert dialog.lbl_scope.text() == "仅未标注的图片"

    # 确认框返回的快照，就是要执行的那一份
    executed = dialog.selected_plan()
    assert executed.image_ids == plan.image_ids
    assert executed is plan, "没改范围时应该原样把同一份快照还回去"

    page.deleteLater()
    db.delete_project(project_id)


def test_all_scope_includes_already_annotated_images():
    project_id, image_ids = _make_project(image_count=4, annotated=2)
    page = _page(project_id, _image_rows(project_id), settings={
        'model_path': 'best.pt', 'only_unlabeled': False, 'overwrite_labels': True,
    })

    plan = page._build_yolo_batch_plan(page.auto_label_settings)

    assert plan.scope == SCOPE_ALL
    assert plan.image_ids == image_ids
    assert plan.overwrite is True

    page.deleteLater()
    db.delete_project(project_id)


# ==================== manager：overwrite 透传、防重复、非阻塞取消 ====================

class _SlowFakeLabeler:
    """假的标注器：不读真模型、不推理，只是慢吞吞地返回一个框。"""

    seen_config = None
    load_delay = 0.0       # 模拟「读一个大 .pt 要花的时间」
    load_succeeds = True
    loaded_paths = []

    def __init__(self, model_path, model_manager):
        self.model_path = model_path
        self.current_model = None      # 线程要靠它判断模型到底加载上没有

    def load_model(self, config):
        _SlowFakeLabeler.loaded_paths.append(config.get('custom_model_path'))
        assert config.get('model_source') == 'custom', \
            "批量加载走了 official 分支——那条路本地没有就会联网下载"
        if _SlowFakeLabeler.load_delay:
            time.sleep(_SlowFakeLabeler.load_delay)
        if not _SlowFakeLabeler.load_succeeds:
            return False
        self.current_model = object()
        return True

    def process_single_image(self, image_path, image_id, config):
        _SlowFakeLabeler.seen_config = config
        time.sleep(0.05)
        return [{'class_id': 0, 'type': 'bbox', 'data': {'x': 1, 'y': 1, 'width': 2, 'height': 2}}]

    def save_annotations(self, annotations, image_id, overwrite=False):
        _SlowFakeLabeler.saved_overwrite = overwrite

    def unload_model(self):
        pass


def _pump_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app.processEvents()
        if predicate():
            return True
    return False


def _reset_fake_labeler():
    _SlowFakeLabeler.seen_config = None
    _SlowFakeLabeler.saved_overwrite = None
    _SlowFakeLabeler.load_delay = 0.0
    _SlowFakeLabeler.load_succeeds = True
    _SlowFakeLabeler.loaded_paths = []


def _wait_for_batch(manager, timeout=15.0):
    """等这一批真的收工。

    注意不能用 `not manager.is_running()` 当等待条件：start() 现在是立刻返回的，
    QThread 要过一小会儿才真的跑起来，isRunning() 在那个窗口里是 False——
    那样等于「还没开始」被误判成「已经结束」，测试会在线程动手之前就去断言。
    只有 batch_completed 才代表这一批真的完了。
    """
    done = []
    manager.batch_completed.connect(lambda *args: done.append(args))
    assert _pump_until(lambda: bool(done), timeout=timeout), "没等到 batch_completed"
    _pump_until(lambda: not manager.is_running(), timeout=timeout)
    return done[0]   # (success, message, count, cancelled)


def test_manager_passes_the_plans_overwrite_flag_through_to_saving():
    """overwrite 必须来自快照。这里以前硬编码 True——确认框说「保留」，实际全删。"""
    project_id, _ = _make_project(image_count=2, annotated=0)
    images = _image_rows(project_id)

    for overwrite in (False, True):
        _reset_fake_labeler()
        manager = BatchLabelingManager()
        plan = _plan(project_id, images, overwrite=overwrite)

        with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
            assert manager.start_batch_processing(plan, model_manager=None) is True
            success, _message, _count, _cancelled = _wait_for_batch(manager)

        assert success is True
        assert _SlowFakeLabeler.seen_config['overwrite_labels'] is overwrite, \
            "config 里的 overwrite 不是快照里那个"
        assert _SlowFakeLabeler.saved_overwrite is overwrite, \
            "save_annotations 收到的 overwrite 不是快照里那个"

    db.delete_project(project_id)


def test_manager_refuses_to_start_a_second_batch_while_one_is_running():
    """防双击第二层：已经在跑就直接拒绝，返回 False。"""
    project_id, _ = _make_project(image_count=4, annotated=0)
    _reset_fake_labeler()
    _SlowFakeLabeler.load_delay = 0.2   # 撑开「已经启动、还在加载模型」这段窗口

    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id))

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
        first = manager.start_batch_processing(plan, model_manager=None)
        second = manager.start_batch_processing(plan, model_manager=None)
        third = manager.start_batch_processing(plan, model_manager=None)

        assert first is True, "第一次应该真的启动"
        assert second is False and third is False, "跑着的时候不能再起第二个任务"

        _wait_for_batch(manager)

    _reset_fake_labeler()
    db.delete_project(project_id)


def test_request_cancel_does_not_block_the_calling_thread():
    """取消是「请求」：立刻返回，不在界面线程里等线程退出。"""
    project_id, _ = _make_project(image_count=30, annotated=0)
    _reset_fake_labeler()
    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id))

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
        assert manager.start_batch_processing(plan, model_manager=None) is True

        start = time.monotonic()
        manager.request_cancel()
        elapsed = time.monotonic() - start

        assert elapsed < 0.02, f"request_cancel 阻塞了调用线程 {elapsed:.3f}s —— 界面会卡住"

        _wait_for_batch(manager)

    db.delete_project(project_id)


def test_cancelling_a_batch_is_not_reported_as_a_failure():
    """取消不是错误：success 仍然为真，cancelled 单独一位。"""
    project_id, _ = _make_project(image_count=30, annotated=0)
    _reset_fake_labeler()
    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id))

    results = []
    manager.batch_completed.connect(
        lambda success, message, count, cancelled: results.append((success, cancelled, message))
    )

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
        manager.start_batch_processing(plan, model_manager=None)
        # 先让它真的跑掉几张，再喊停——否则可能停在「模型还没加载完」那一档
        assert _pump_until(lambda: _SlowFakeLabeler.seen_config is not None, timeout=10.0)
        manager.request_cancel()
        assert _pump_until(lambda: bool(results), timeout=10.0), "没等到收尾信号"
        _pump_until(lambda: not manager.is_running(), timeout=10.0)

    success, cancelled, message = results[0]
    assert cancelled is True, "取消没有被标成 cancelled"
    assert success is True, "取消被当成失败上报了——界面会弹一个红色的「批量标注失败」"
    assert "取消" in message

    db.delete_project(project_id)


# ==================== 模型加载必须在后台，且绝不联网 ====================

class _NoDownloadModelManager:
    """假的模型管理器。

    `load_model` 是官方模型那条路——本地没有就会 `YOLO("yolov8n")` 去网上下载。
    批量标注绝不该走到那里，所以这里直接炸，让任何一次误用都当场现形。
    """

    def __init__(self, pretrained_dir: Path):
        self.pretrained_dir = pretrained_dir
        self.custom_paths = []
        self.official_calls = []

    def get_model_path(self, version, size, task='detect'):
        # 纯路径推算，不碰网络（真实实现也是如此）
        return self.pretrained_dir / f"{version.lower()}{size}.pt"

    def load_model(self, version, size, task='detect'):
        self.official_calls.append((version, size, task))
        raise AssertionError(
            "批量标注走到了官方模型加载分支——本地没有时它会联网下载！"
        )

    def load_custom_model(self, path):
        self.custom_paths.append(str(path))
        return object()          # 一个「模型」，够用了

    def get_model_info(self, model):
        return {'task': 'detect', 'nc': 1}


def test_start_returns_immediately_even_while_the_model_is_still_loading():
    """blocker：加载模型不能在界面线程上做。

    把「读一个大 .pt」模拟成 0.6 秒。start_batch_processing 是界面线程调的，
    它必须立刻返回——否则用户点完「开始批量标注」，窗口就白 0.6 秒（真实的
    YOLOv8x 权重可以卡好几秒）。
    """
    project_id, _ = _make_project(image_count=2, annotated=0)
    _reset_fake_labeler()
    _SlowFakeLabeler.load_delay = 0.6

    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id))

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
        start = time.monotonic()
        started = manager.start_batch_processing(plan, model_manager=None)
        elapsed = time.monotonic() - start

        assert started is True
        assert elapsed < 0.1, (
            f"start_batch_processing 在界面线程上卡了 {elapsed:.3f}s —— "
            "模型加载还留在 GUI 线程里"
        )
        # 此刻模型还在后台读，一张图都还没处理
        assert _SlowFakeLabeler.seen_config is None, "返回时不该已经开始处理图片"

        success, _message, _count, cancelled = _wait_for_batch(manager)

    assert success is True and cancelled is False
    assert _SlowFakeLabeler.seen_config is not None, "后台最终还是要把图片跑完的"
    assert _SlowFakeLabeler.loaded_paths == [str(_LOCAL_PT)], \
        "加载的不是快照里那个本地 .pt"

    _reset_fake_labeler()
    db.delete_project(project_id)


def test_cancelling_while_the_model_loads_stops_before_touching_any_image():
    """模型还在加载时点取消：一张图都不该被处理，而且要按「已取消」收尾。"""
    project_id, _ = _make_project(image_count=5, annotated=0)
    _reset_fake_labeler()
    _SlowFakeLabeler.load_delay = 0.5

    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id))

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
        manager.start_batch_processing(plan, model_manager=None)
        manager.request_cancel()          # 模型还没读完就喊停
        success, message, _count, cancelled = _wait_for_batch(manager)

    assert cancelled is True, "加载期间取消没有被标成 cancelled"
    assert success is True, "取消不是失败"
    assert "取消" in message
    assert _SlowFakeLabeler.seen_config is None, "已经喊停了，却还是处理了图片"
    assert _annotation_count(project_id) == 0, "已经喊停了，却还是写了标注"

    _reset_fake_labeler()
    db.delete_project(project_id)


def test_batch_never_hands_an_official_model_name_to_the_downloader():
    """快照里是「yolov8n」这种官方名字、本地又没有这个文件时：报错，不下载。

    真实的 ModelManager.load_model 在本地找不到时会执行 `YOLO("yolov8n")`，
    那一句会联网把 COCO 通用模型拽下来。批量这条路绝不能碰它。
    """
    project_id, _ = _make_project(image_count=2, annotated=0)
    empty_dir = Path(tempfile.mkdtemp(prefix="ezyolo-no-pretrained-"))
    fake_manager = _NoDownloadModelManager(empty_dir)

    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id),
                 model_path='yolov8n', model_label='yolov8n')

    # 连真实的 AutoLabeler 一起用：要测的正是它那条加载路
    with patch.object(auto_labeler_module, "model_manager", fake_manager):
        assert manager.start_batch_processing(plan, fake_manager) is True
        success, message, _count, cancelled = _wait_for_batch(manager)

    assert fake_manager.official_calls == [], "批量标注去调了会联网下载的官方加载分支"
    assert fake_manager.custom_paths == [], "本地根本没有这个文件，不该有任何加载动作"
    assert success is False, "找不到模型却报告成功"
    assert cancelled is False, "这是失败，不是取消"
    assert "找不到本地模型文件" in message, message
    assert not manager.is_running(), "失败之后线程状态没有释放"

    db.delete_project(project_id)


def test_an_already_downloaded_official_model_is_loaded_from_disk_not_the_network():
    """本地预训练目录里已经有 yolov8n.pt：直接读文件，仍然不碰下载分支。"""
    project_id, _ = _make_project(image_count=1, annotated=0)
    pretrained = Path(tempfile.mkdtemp(prefix="ezyolo-pretrained-"))
    (pretrained / "yolov8n.pt").write_bytes(b"local-weights")
    fake_manager = _NoDownloadModelManager(pretrained)

    manager = BatchLabelingManager()
    plan = _plan(project_id, _image_rows(project_id),
                 model_path='yolov8n', model_label='yolov8n')

    with patch.object(auto_labeler_module, "model_manager", fake_manager):
        assert manager.start_batch_processing(plan, fake_manager) is True
        _wait_for_batch(manager)

    assert fake_manager.official_calls == [], "走了会联网的官方加载分支"
    assert fake_manager.custom_paths == [str(pretrained / "yolov8n.pt")], \
        "没有从本地预训练目录按路径把模型读进来"

    db.delete_project(project_id)


def test_model_load_failure_is_reported_and_releases_the_page_state():
    """模型加载失败：如实回报，并且把页面从「跑批中」放出来（按钮、状态栏都得复原）。"""
    project_id, _ = _make_project(image_count=2, annotated=0)
    _reset_fake_labeler()
    _SlowFakeLabeler.load_succeeds = False      # 文件在，但读不进来（损坏 / 版本不对）

    page = _page(project_id, _image_rows(project_id), settings={
        'model_path': str(_LOCAL_PT), 'model_task': 'detect',
        'conf_threshold': 0.5, 'iou_threshold': 0.45,
        'class_mapping': {}, 'only_unlabeled': True, 'overwrite_labels': False,
    })

    plan = page._build_yolo_batch_plan(page.auto_label_settings)

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler), \
         patch.object(annotate_module, "QMessageBox") as message_box:
        assert page._start_yolo_batch(plan) is True
        assert page._batch_running is True, "启动后应该先进入「跑批中」"

        assert _pump_until(lambda: not page._batch_running, timeout=10.0), \
            "加载失败之后页面一直卡在「跑批中」，按钮再也点不动了"

    assert message_box.critical.called, "加载失败要如实告诉用户"
    reported = str(message_box.critical.call_args)
    assert "模型加载失败" in reported, reported

    # 状态释放干净：状态栏空了、取消按钮收起、AI 入口恢复可点
    assert page.status_batch.text() == ""
    assert page.btn_cancel_batch.isHidden()
    assert page.btn_auto_label.isEnabled()
    assert page._active_batch_plan is None
    assert _annotation_count(project_id) == 0, "模型都没加载上，不该写任何标注"

    _reset_fake_labeler()
    page.deleteLater()
    db.delete_project(project_id)


# ==================== 没配模型：不许偷偷用 yolov8n ====================

def test_single_inference_without_saved_settings_never_falls_back_to_yolov8n():
    """没保存过 YOLO 设置时，要把设置页推给用户，而不是拿 yolov8n 去联网下载。"""
    page = _page(project_id=1, settings=None)
    page.current_image_data = {'id': 1, 'storage_path': '/x.jpg'}

    offered = []

    def fake_offer(self, title, message, section):
        offered.append(section)
        return False  # 用户在设置页点了取消

    class _ExplodingLabeler:
        def __init__(self, model_path, model_manager):
            raise AssertionError(f"不该在没有配置的情况下加载模型：{model_path}")

    with patch.object(AnnotatePage, "_offer_auto_label_config", fake_offer), \
         patch.object(annotate_module, "QMessageBox"), \
         patch.object(auto_labeler_module, "AutoLabeler", _ExplodingLabeler):
        page.run_single_inference()   # _ExplodingLabeler 会炸；没炸就说明根本没去加载模型

    assert offered == ['yolo'], "没配模型时应该把用户引导到 YOLO 设置页"

    page.deleteLater()


def test_batch_inference_without_saved_settings_never_falls_back_to_yolov8n():
    project_id, _ = _make_project(image_count=2, annotated=0)
    page = _page(project_id, _image_rows(project_id), settings=None)

    offered = []
    started = []

    def fake_offer(self, title, message, section):
        offered.append(section)
        return False

    with patch.object(AnnotatePage, "_offer_auto_label_config", fake_offer), \
         patch.object(AnnotatePage, "_start_yolo_batch", lambda self, p: started.append(p)), \
         patch.object(annotate_module, "confirm_batch_plan") as confirm, \
         patch.object(annotate_module, "QMessageBox"):
        page.run_batch_inference()

    assert offered == ['yolo'], "没配模型时应该把用户引导到 YOLO 设置页"
    assert not confirm.called, "模型都没有，不该弹确认框"
    assert started == [], "没配模型却启动了批量任务"

    page.deleteLater()
    db.delete_project(project_id)


def test_saved_settings_are_used_verbatim_without_any_default_model():
    """存过设置就用存的那份，一个字都不改。"""
    settings = {
        'model_path': '/models/helmet.pt', 'model_task': 'detect',
        'conf_threshold': 0.7, 'iou_threshold': 0.3,
        'class_mapping': {0: 2}, 'only_unlabeled': False, 'overwrite_labels': True,
    }
    page = _page(project_id=1, settings=settings)

    resolved = page._require_yolo_settings()

    assert resolved is settings
    assert resolved['model_path'] == '/models/helmet.pt'
    assert 'yolov8n' not in str(resolved)

    page.deleteLater()


# ==================== 菜单 ====================

def _menu_texts(menu: QMenu):
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def test_menus_offer_settings_single_and_batch_with_an_ellipsis():
    """两个菜单的顺序和文字都被冻住了：设置 / 标注当前图片 / 批量标注…

    省略号不是装饰：它承诺「点了还有一步」。批量项以前直连一个会开跑的函数，
    就是这次要修的那个 bug。
    """
    page = _page()

    for menu in (page.btn_auto_label.menu(), page.btn_llm_label.menu()):
        assert _menu_texts(menu) == ["设置", "标注当前图片", "批量标注…"], _menu_texts(menu)
        assert menu.objectName() == "actionMenu", "菜单没挂上 U2 的样式"

        for action in menu.actions():
            assert not action.icon().isNull(), f"「{action.text()}」缺少 16px 图标"

    page.deleteLater()


def test_clicking_the_batch_item_opens_a_confirmation_instead_of_starting():
    """普通点击「批量标注…」只会打开确认框，绝不会直接开跑。"""
    project_id, _ = _make_project(image_count=3, annotated=0)

    page = _page(project_id, _image_rows(project_id), settings={
        'model_path': 'best.pt', 'model_task': 'detect',
        'conf_threshold': 0.5, 'iou_threshold': 0.45,
        'class_mapping': {}, 'only_unlabeled': True, 'overwrite_labels': False,
    })

    started = []
    with patch.object(annotate_module, "confirm_batch_plan", return_value=None) as confirm, \
         patch.object(AnnotatePage, "_start_yolo_batch", lambda self, p: started.append(p)), \
         patch.object(annotate_module, "QMessageBox"):
        batch_action = page.auto_label_actions['batch']
        batch_action.trigger()

    assert confirm.called, "点「批量标注…」没有弹确认框"
    assert started == [], "点一下菜单项就直接开跑了"

    page.deleteLater()
    db.delete_project(project_id)


def test_action_menu_rows_are_tall_enough_to_hit():
    """单项 40px 上下：默认 QMenu 是 20 出头，那是给命令列表用的，不是操作面板。"""
    page = _page()
    menu = page.btn_auto_label.menu()
    menu.show()
    _app.processEvents()

    for action in menu.actions():
        height = menu.actionGeometry(action).height()
        assert 38 <= height <= 48, f"「{action.text()}」行高 {height}px，不在 40–44px 这一档"

    menu.hide()
    page.deleteLater()


def test_action_menu_is_at_least_as_wide_as_its_button():
    """菜单不能比入口按钮还窄，否则看着像点歪了弹出来的系统菜单。"""
    page = _page()
    page.resize(1200, 800)
    page.show()
    _app.processEvents()

    for button in (page.btn_auto_label, page.btn_llm_label):
        menu = button.menu()
        menu.aboutToShow.emit()   # 宽度是在弹出前才量的
        assert menu.minimumWidth() >= button.width(), (
            f"菜单 {menu.minimumWidth()}px 比按钮 {button.width()}px 还窄"
        )

    page.close()
    page.deleteLater()


def test_menu_text_is_not_clipped_at_125_and_150_percent_font_scaling():
    """125% / 150% 字号下菜单项不能裁字。"""
    page = _page()
    menu = page.btn_auto_label.menu()

    base_font = menu.font()
    base_size = base_font.pointSizeF() if base_font.pointSizeF() > 0 else 13.0

    for scale in (1.25, 1.5):
        font = menu.font()
        font.setPointSizeF(base_size * scale)
        menu.setFont(font)
        menu.show()
        _app.processEvents()

        metrics = menu.fontMetrics()
        for action in menu.actions():
            geometry = menu.actionGeometry(action)
            needed = metrics.horizontalAdvance(action.text())
            assert geometry.width() >= needed, (
                f"{int(scale * 100)}% 字号下「{action.text()}」被裁："
                f"{geometry.width()}px < {needed}px"
            )
            assert geometry.height() >= metrics.height(), (
                f"{int(scale * 100)}% 字号下「{action.text()}」行高压住了文字"
            )
        menu.hide()

    page.deleteLater()


# ==================== 页面：状态保持与防双击 ====================

def test_running_batch_survives_a_regular_status_refresh():
    """跑着的时候标个框、切张图（都会调 update_status_bar），进度和取消入口还得在。"""
    page = _page()

    page._set_batch_status("批量标注: 2/9 · a.jpg", True)
    assert not page.btn_cancel_batch.isHidden()

    page.update_status_bar()

    assert page.status_batch.text().startswith("批量标注: 2/9"), "常规刷新把进度抹掉了"
    assert not page.btn_cancel_batch.isHidden(), "常规刷新把取消入口抹掉了"

    # AI 入口在跑批的时候一律禁掉——这是防重复启动的一环
    assert not page.btn_auto_label.isEnabled()
    assert not page.btn_llm_label.isEnabled()

    page._finish_batch_ui()
    assert page.status_batch.text() == ""
    assert page.btn_cancel_batch.isHidden()

    page.deleteLater()


def test_page_refuses_a_second_batch_while_one_is_running():
    """防双击第一层：页面这边先拦。确认框都不该弹第二次。"""
    project_id, _ = _make_project(image_count=3, annotated=0)

    page = _page(project_id, _image_rows(project_id), settings={
        'model_path': 'best.pt', 'only_unlabeled': True, 'overwrite_labels': False,
    })
    page._set_batch_status("批量标注: 1/3", True)   # 假装已经在跑

    with patch.object(annotate_module, "confirm_batch_plan") as confirm, \
         patch.object(annotate_module, "QMessageBox") as message_box:
        page.run_batch_inference()

    assert not confirm.called, "已经在跑了，还弹第二个确认框"
    assert message_box.information.called, "该告诉用户上一批还在跑"

    page._finish_batch_ui()
    page.deleteLater()
    db.delete_project(project_id)


def test_cancel_button_requests_cancel_without_blocking():
    """状态栏那个取消按钮：只发请求，不等线程。"""
    page = _page()

    class _FakeManager:
        def __init__(self):
            self.cancelled = False
            self.waited = False

        def is_running(self):
            return True

        def request_cancel(self):
            self.cancelled = True

        def stop(self):
            self.waited = True   # 这条路会 wait()，界面线程绝不能走到

    manager = _FakeManager()
    page.batch_labeling_manager = manager
    page._set_batch_status("批量标注: 1/9", True)

    page.btn_cancel_batch.click()

    assert manager.cancelled is True, "取消按钮没有把取消请求发下去"
    assert manager.waited is False, "取消走了会 wait() 的那条路——界面会卡住"
    assert "取消" in page.status_batch.text()
    assert not page.btn_cancel_batch.isEnabled(), "取消按钮该立刻变灰，避免连点"

    page._finish_batch_ui()
    page.deleteLater()


def test_llm_batch_writes_use_the_snapshot_project_not_the_current_page_project():
    """跑批期间用户切了项目：标注仍要写进快照里那个项目，不能跟着页面跑。"""
    project_a, image_ids_a = _make_project(image_count=2, annotated=0)
    project_b, _ = _make_project(image_count=1, annotated=0)

    page = _page(project_a)
    page.classes = [{'id': 0, 'name': '人', 'color': '#FF0000'}]

    plan = BatchPlan(
        project_id=project_a, engine='llm',
        images=[dict(img) for img in _image_rows(project_a)],
        scope=SCOPE_RANGE, model_label='gpt-x',
        conf=None, iou=None, overwrite=False,
        class_id=0, class_name='人',
    )
    page._active_batch_plan = plan

    # 任务跑到一半，用户切去了另一个项目
    page.current_project_id = project_b

    page.on_llm_batch_image_done(
        image_ids_a[0],
        [{'bbox': [1, 2, 11, 12]}],
        "",
    )

    annotations = db.get_image_annotations(image_ids_a[0])
    assert len(annotations) == 1, "标注没写进去"
    assert annotations[0]['project_id'] == project_a, \
        "标注被写进了用户后来切过去的那个项目"
    assert _annotation_count(project_b) == 0, "污染了另一个项目"

    page.deleteLater()
    db.delete_project(project_a)
    db.delete_project(project_b)


# ==================== 关窗收尾 ====================
#
# MainWindow.closeEvent 会调 AnnotatePage.shutdown()。它以前只收 LLM 线程，
# 完全没管 YOLO 的 BatchLabelingManager：批量推理跑着的时候关窗，manager 跟着
# 页面一起销毁，它手上那个还在跑的 QThread 就撞上「destroyed while running」——
# 进程直接崩，而不是干净退出。

class _RecordingBatchManager:
    """假 manager：不起线程，只记下谁被调了、什么顺序。"""

    def __init__(self, events, running=True):
        self.events = events
        self._running = running

    def request_cancel(self):
        self.events.append('yolo:request_cancel')

    def cleanup(self):
        self.events.append('yolo:cleanup')
        self._running = False       # cleanup 里 stop() + wait()，回来线程就该停了

    def is_running(self):
        return self._running


class _RecordingLlmWorker:
    """假 LLM 线程：同样只记调用，不真的跑。"""

    def __init__(self, events, running=True):
        self.events = events
        self._running = running

    def isRunning(self):
        return self._running

    def cancel(self):
        self.events.append('llm:cancel')

    def wait(self, msec=None):
        self.events.append('llm:wait')
        self._running = False
        return True


def test_shutdown_cancels_the_yolo_batch_before_it_waits_on_anything():
    """先把两边的取消都发出去，再去等——顺序本身就是要求。

    取消是放个标志就返回，等待才真的堵着。要是反过来先站在 LLM 那儿等满 5 秒，
    YOLO 线程这 5 秒里还在一张张往下推图，全是白跑的活。
    """
    events = []
    page = _page()
    page.batch_labeling_manager = _RecordingBatchManager(events)
    page.llm_batch_worker = _RecordingLlmWorker(events)

    page.shutdown()

    assert events == [
        'yolo:request_cancel',   # 先发 YOLO 的取消
        'llm:cancel',            # 再发 LLM 的取消（两边都已经收到停的指令）
        'yolo:cleanup',          # 然后才开始等：YOLO 线程退出 + 卸载模型
        'llm:wait',              # 最后等 LLM 线程退出
    ], f"关窗收尾的顺序不对：{events}"

    page.deleteLater()


def test_shutdown_is_safe_when_no_batch_was_ever_started():
    """从没跑过批量的页面（manager 还是 None），关窗不能报错；重复调也不能。"""
    page = _page()
    assert page.batch_labeling_manager is None, "夹具前提变了：manager 本该还没建"

    page.shutdown()
    page.shutdown()   # 幂等：closeEvent 万一走两遍也得安然无恙

    page.deleteLater()


def test_shutdown_still_finishes_the_llm_workers_it_always_did():
    """别为了接上 YOLO 就把原来的 LLM 收尾弄丢了：单张、批量、退休的线程都要收。"""
    events = []
    page = _page()
    single = _RecordingLlmWorker(events)
    batch = _RecordingLlmWorker(events)
    retired = _RecordingLlmWorker(events)
    page.llm_worker = single
    page.llm_batch_worker = batch
    page._retired_llm_workers = [retired]

    page.shutdown()

    for worker, name in ((single, "单张"), (batch, "批量"), (retired, "已退休")):
        assert not worker.isRunning(), f"{name} LLM 线程没等到它退出"   # wait() 才会把它放倒
    assert events.count('llm:cancel') == 3, "三个 LLM 线程都得收到取消"
    assert events.count('llm:wait') == 3, "三个 LLM 线程都得等到它退出"

    page.deleteLater()


def test_shutdown_actually_stops_a_running_yolo_batch_thread():
    """真线程、真 manager：关窗之后不能再有还在跑的线程引用。

    这条是这次修复的正主。假 manager 只能证明「调了 request_cancel / cleanup」，
    证明不了那两个调用真的把线程摁下去了——所以这里起一个真的
    BatchLabelingManager（只把推理换成假的），跑起来，然后关窗。
    """
    project_id, _ = _make_project(image_count=40, annotated=0)
    _reset_fake_labeler()

    manager = BatchLabelingManager()
    page = _page(project_id, _image_rows(project_id))
    page.batch_labeling_manager = manager
    plan = _plan(project_id, _image_rows(project_id))

    with patch.object(auto_labeler_module, "AutoLabeler", _SlowFakeLabeler):
        assert manager.start_batch_processing(plan, model_manager=None) is True
        assert _pump_until(lambda: manager.is_running()), "线程根本没跑起来"

        page.shutdown()      # ← 关窗。cleanup() 里 stop() + wait()，回来线程必须已经退出

        assert not manager.is_running(), \
            "shutdown 之后线程还在跑——QThread 正要在这个状态下被销毁"

        # wait() 是在界面线程里堵着等的，thread.finished 那会儿投递不出去；
        # 泵一轮事件，让 manager 把引用真正放掉。
        assert _pump_until(lambda: manager.current_thread is None), \
            "线程退出了，manager 手上还攥着它的引用"

    done = _annotation_count(project_id)
    assert done < 40, "取消没起作用：40 张全跑完了，说明它是自然跑完而不是被停下的"

    page.deleteLater()
    _reset_fake_labeler()
    db.delete_project(project_id)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(globals()))
