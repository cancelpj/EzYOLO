# -*- coding: utf-8 -*-
"""批量标注执行时的三条边界。

盯的都是「跑起来之后」那一段，确认框已经点过了：

    .pth 也是权重          训练导出的检查点常是 .pth，不该被扩展名挡在门外
    空结果也是结论         勾了覆盖、这一轮一个目标都没检出 → 旧标注得清掉
    文件没了不是没发生     图片被移走/删掉：算失败、接着跑、进度照样走到头

全程不加载真实模型、不发网络请求：推理换成假的，模型加载分支一旦碰到官方
下载那条路就当场炸。

运行：
    python -m pytest tests/test_batch_execution_edges.py -q
    或
    python tests/test_batch_execution_edges.py
"""

import _bootstrap  # noqa: F401  必须第一个导入

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import core.auto_labeler as auto_labeler_module  # noqa: E402
from core.auto_labeler import BatchLabelingManager  # noqa: E402
from gui.widgets.batch_confirm_dialog import BatchPlan, SCOPE_ALL  # noqa: E402

_app = _bootstrap.app()
db = _bootstrap.db

_TMP = Path(tempfile.mkdtemp(prefix="ezyolo-batch-edges-"))


# ==================== 夹具 ====================

class _NoDownloadModelManager:
    """假的模型管理器：只从磁盘读，推理结果由测试指定。

    `load_model` 是官方模型那条路——本地没有就会 `YOLO("yolov8n")` 去联网下载。
    批量标注绝不该走到那里，所以这里直接炸，让任何一次误用当场现形。
    """

    def __init__(self, infer_result=None):
        self.infer_result = infer_result     # None = 一个目标都没检出
        self.custom_paths = []
        self.official_calls = []
        self.inferred_paths = []

    def get_model_path(self, version, size, task='detect'):
        return _TMP / f"{version.lower()}{size}.pt"

    def load_model(self, version, size, task='detect'):
        self.official_calls.append((version, size, task))
        raise AssertionError("批量标注走到了官方模型加载分支——本地没有时它会联网下载！")

    def load_custom_model(self, path):
        self.custom_paths.append(str(path))
        return object()          # 一个「模型」，够用了

    def get_model_info(self, model):
        return {'task': 'detect', 'nc': 1}

    def infer(self, model, image_path, conf, iou):
        self.inferred_paths.append(image_path)
        return self.infer_result

    def unload_all_models(self):
        pass


def _make_project(image_count=1, annotated=0, missing_from=None):
    """一个有图片的项目。`missing_from` 之后的图片只进库、不落盘（模拟文件被移走）。"""
    project_id = _bootstrap.create_temp_project(
        name="批量边界测试", project_type="detect",
        classes=[{'id': 0, 'name': '人', 'color': '#FF0000'}],
    )
    image_ids = []
    for i in range(image_count):
        path = _TMP / f"p{project_id}_{i}.jpg"
        if missing_from is None or i < missing_from:
            path.write_bytes(b"not-a-real-jpeg")
        image_ids.append(
            db.add_image(project_id, path.name, str(path), width=64, height=48)
        )

    for image_id in image_ids[:annotated]:
        db.add_annotation(
            image_id=image_id, project_id=project_id, class_id=0, class_name='人',
            annotation_type='bbox', data={'x': 1, 'y': 1, 'width': 5, 'height': 5},
        )
    return project_id, image_ids


def _plan(project_id, images, model_path, **kwargs) -> BatchPlan:
    defaults = dict(
        project_id=project_id,
        engine='yolo',
        images=images,
        scope=SCOPE_ALL,
        model_label=Path(model_path).name,
        model_path=str(model_path),
        conf=0.5,
        iou=0.45,
        overwrite=False,
    )
    defaults.update(kwargs)
    return BatchPlan(**defaults)


def _pump_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app.processEvents()
        if predicate():
            return True
    return False


def _run_batch(plan, fake_manager, timeout=15.0):
    """跑完一整批，返回 (manager, 完成信号参数, 进度信号列表)。"""
    manager = BatchLabelingManager()
    done = []
    progress = []
    manager.batch_completed.connect(lambda *args: done.append(args))
    manager.progress_updated.connect(lambda *args: progress.append(args))

    # 连真实的 AutoLabeler 一起用：要测的正是它那条加载 / 保存路径
    with patch.object(auto_labeler_module, "model_manager", fake_manager):
        assert manager.start_batch_processing(plan, fake_manager) is True
        assert _pump_until(lambda: bool(done), timeout=timeout), "没等到 batch_completed"
        _pump_until(lambda: not manager.is_running(), timeout=timeout)

    assert fake_manager.official_calls == [], "批量标注去调了会联网下载的官方加载分支"
    return manager, done[0], progress


def _annotation_count(image_id) -> int:
    return len(db.get_image_annotations(image_id))


# ==================== .pth 也是本地权重 ====================

def test_a_local_pth_checkpoint_is_accepted_and_never_downloaded():
    """磁盘上躺着一个 .pth：照常读它，不联网。

    以前只认 .pt，用户拿训练导出的 .pth 就会被顶回「找不到本地模型文件」——
    文件明明就在那儿。
    """
    project_id, _ = _make_project(image_count=1)
    checkpoint = _TMP / "checkpoint_best.pth"
    checkpoint.write_bytes(b"not-a-real-checkpoint")

    fake_manager = _NoDownloadModelManager()
    plan = _plan(project_id, db.get_project_images(project_id), checkpoint)

    _manager, (success, message, _count, cancelled), _progress = _run_batch(plan, fake_manager)

    assert success is True, f"本地 .pth 没被当成有效权重：{message}"
    assert cancelled is False
    assert fake_manager.custom_paths == [str(checkpoint)], \
        "没有按路径把那个 .pth 读进来"

    db.delete_project(project_id)


def test_a_model_path_that_is_not_on_disk_is_refused_without_touching_the_network():
    """路径指着一个不存在的 .pth：报错，不去网上找替代品。"""
    project_id, _ = _make_project(image_count=1)
    fake_manager = _NoDownloadModelManager()
    plan = _plan(project_id, db.get_project_images(project_id), _TMP / "not_here.pth")

    _manager, (success, message, _count, cancelled), _progress = _run_batch(plan, fake_manager)

    assert success is False and cancelled is False
    assert "找不到本地模型文件" in message, message
    assert fake_manager.custom_paths == [], "文件根本不存在，不该有任何加载动作"

    db.delete_project(project_id)


# ==================== 覆盖模式下的空结果 ====================

def test_overwrite_clears_old_annotations_when_the_model_detects_nothing():
    """勾了「覆盖」、这一轮一个目标都没检出：旧标注必须清掉。

    以前空结果直接跳过保存——用户按覆盖跑了一遍，库里留着的还是上一轮的框，
    界面上看不出任何变化，等于这次标注被静默吞了。
    """
    project_id, image_ids = _make_project(image_count=1, annotated=1)
    weights = _TMP / "empty_result.pt"
    weights.write_bytes(b"not-a-real-checkpoint")

    assert _annotation_count(image_ids[0]) == 1, "夹具没准备好旧标注"

    fake_manager = _NoDownloadModelManager(infer_result=None)   # 零检出
    plan = _plan(project_id, db.get_project_images(project_id), weights, overwrite=True)

    _manager, (success, _message, _count, _cancelled), _progress = _run_batch(plan, fake_manager)

    assert success is True
    assert _annotation_count(image_ids[0]) == 0, \
        "覆盖模式下推理为空，旧标注却还留在库里"

    db.delete_project(project_id)


def test_an_empty_result_deletes_nothing_when_overwrite_is_off():
    """没勾「覆盖」：哪怕这轮什么都没检出，也一个字都不许动用户的旧标注。"""
    project_id, image_ids = _make_project(image_count=1, annotated=1)
    weights = _TMP / "empty_result.pt"
    weights.write_bytes(b"not-a-real-checkpoint")

    fake_manager = _NoDownloadModelManager(infer_result=None)
    plan = _plan(project_id, db.get_project_images(project_id), weights, overwrite=False)

    _manager, (success, _message, _count, _cancelled), _progress = _run_batch(plan, fake_manager)

    assert success is True
    assert _annotation_count(image_ids[0]) == 1, "没勾覆盖，旧标注却被删了"

    db.delete_project(project_id)


# ==================== 图片文件不在了 ====================

def test_a_missing_image_counts_as_a_failure_and_the_batch_keeps_going():
    """两张图，第二张的文件被移走了。

    以前这张图既不算成功也不算失败：循环里直接跳过，进度条永远停在 1/2，
    收尾却还照报「批量标注完成」，而且 manager 拿 len(images) 上报「处理了 2 张」。
    现在要求：算一次失败、接着跑、进度走到 2/2、成功数是 1、失败数是 1。
    """
    project_id, image_ids = _make_project(image_count=2, missing_from=1)
    weights = _TMP / "keeps_going.pt"
    weights.write_bytes(b"not-a-real-checkpoint")

    images = db.get_project_images(project_id)
    assert Path(images[0]['storage_path']).exists()
    assert not Path(images[1]['storage_path']).exists(), "第二张图应该是「文件不在」的"

    fake_manager = _NoDownloadModelManager(infer_result=None)
    plan = _plan(project_id, images, weights)

    manager, (success, message, processed_count, cancelled), progress = _run_batch(
        plan, fake_manager
    )

    # 缺文件的那张没被推理，存在的那张跑过了
    assert fake_manager.inferred_paths == [images[0]['storage_path']]

    # 进度走到头：最后一次进度是 2/2、100%
    assert progress, "一条进度都没发"
    last_percent, last_current, last_total, _name = progress[-1]
    assert (last_current, last_total) == (2, 2), \
        f"进度没走到总数，停在 {last_current}/{last_total}"
    assert last_percent == 100

    # 计数诚实：成功 1、失败 1，缺文件那张不算成功
    assert processed_count == 1, f"缺文件的图片被算成了成功：processed_count={processed_count}"
    assert manager.failed_count == 1, f"缺文件的图片没被算成失败：{manager.failed_count}"
    assert cancelled is False, "这不是取消"
    assert success is False, "有图片没跑成，却报告整批成功"
    assert "1" in message and "失败" in message, f"消息没如实说明失败：{message}"

    db.delete_project(project_id)


if __name__ == "__main__":
    sys.exit(_bootstrap.run_module_tests(dict(globals())))
