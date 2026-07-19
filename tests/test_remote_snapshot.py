# -*- coding: utf-8 -*-
"""远程训练数据快照的离线安全测试。"""

import _bootstrap  # noqa: F401  保持测试路径与现有测试一致

import ast
import os
from pathlib import Path
import sys
import tempfile

from core.remote_training.snapshot import (  # noqa: E402
    DatasetSnapshotBuilder,
    GeneratedSnapshotSource,
    SnapshotError,
    SnapshotIntegrityError,
    SnapshotSource,
    verify_remote_payload,
    write_job_spec,
)
from remote_protocol.v1 import JobSpec, REMOTE_PROTOCOL_VERSION  # noqa: E402


JOB_ID = "1" * 32


def assert_rejected(callable_object, *args, expected=SnapshotError, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except expected:
        return
    raise AssertionError("预期输入被拒绝")


def make_source_tree():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-snapshot-source-"))
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}-1.jpg").write_bytes(
            f"{split}-image".encode("utf-8")
        )
        (root / "labels" / split / f"{split}-1.txt").write_text(
            "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
        )
    return root


def build_sources(root):
    return [
        SnapshotSource(root / "images/train/train-1.jpg", "images/train/train-1.jpg"),
        SnapshotSource(root / "labels/train/train-1.txt", "labels/train/train-1.txt"),
        SnapshotSource(root / "images/val/val-1.jpg", "images/val/val-1.jpg"),
        SnapshotSource(root / "labels/val/val-1.txt", "labels/val/val-1.txt"),
    ]


def make_builder(source_root, snapshot_parent=None):
    return DatasetSnapshotBuilder(
        snapshot_parent or Path(tempfile.mkdtemp(prefix="ezyolo-snapshot-staging-")),
        allowed_roots=[source_root],
    )


def build_snapshot(builder, source_root):
    return builder.build(
        job_id=JOB_ID,
        task_type="detect",
        class_names=("person",),
        layout={"train": "images/train", "val": "images/val"},
        sources=build_sources(source_root),
    )


def test_estimate_is_read_only_and_snapshot_has_complete_manifest():
    source_root = make_source_tree()
    staging_root = Path(tempfile.mkdtemp(prefix="ezyolo-snapshot-staging-"))
    builder = make_builder(source_root, staging_root)
    source = source_root / "images/train/train-1.jpg"
    before = (source.read_bytes(), source.stat().st_mtime_ns, source.stat().st_size)

    estimate = builder.estimate(build_sources(source_root))

    assert estimate.file_count == 4
    assert estimate.total_bytes > 0
    assert not any(staging_root.iterdir())
    assert (source.read_bytes(), source.stat().st_mtime_ns, source.stat().st_size) == before

    snapshot = build_snapshot(builder, source_root)
    assert snapshot.manifest_path.name == "manifest.json"
    assert snapshot.manifest.total_bytes == estimate.total_bytes
    assert not (snapshot.root / "data.yaml").exists()
    assert (source.read_bytes(), source.stat().st_mtime_ns, source.stat().st_size) == before
    verified = builder.verify(snapshot)
    assert verified == snapshot.manifest


def test_snapshot_rejects_symlink_escape_without_following_it():
    source_root = make_source_tree()
    outside = Path(tempfile.mkdtemp(prefix="ezyolo-snapshot-outside-")) / "outside.jpg"
    outside.write_bytes(b"outside")
    link = source_root / "images/train/outside.jpg"
    try:
        os.symlink(outside, link)
    except (NotImplementedError, OSError):
        print("SKIP symlink unsupported")
        return
    builder = make_builder(source_root)
    sources = build_sources(source_root)
    sources[0] = SnapshotSource(link, "images/train/outside.jpg")
    assert_rejected(builder.estimate, sources)


def test_snapshot_rejects_forbidden_payload_and_duplicate_destination():
    source_root = make_source_tree()
    builder = make_builder(source_root)
    forbidden = source_root / "images/train/runner.py"
    forbidden.write_text("print('not payload')", encoding="utf-8")
    assert_rejected(
        builder.estimate,
        [SnapshotSource(forbidden, "images/train/runner.py")],
    )
    duplicate = build_sources(source_root)
    duplicate.append(
        SnapshotSource(
            source_root / "images/val/val-1.jpg",
            "images/train/train-1.jpg",
        )
    )
    assert_rejected(builder.estimate, duplicate)


def test_snapshot_allows_only_controlled_generated_label_text():
    source_root = make_source_tree()
    builder = make_builder(source_root)
    sources = build_sources(source_root)
    sources[1] = GeneratedSnapshotSource(
        "labels/train/train-1.txt",
        b"0 0.5 0.5 0.3 0.3\n",
    )
    snapshot = builder.build(
        job_id=JOB_ID,
        task_type="detect",
        class_names=("person",),
        layout={"train": "images/train", "val": "images/val"},
        sources=sources,
    )
    assert (
        snapshot.root / "labels/train/train-1.txt"
    ).read_text(encoding="utf-8") == "0 0.5 0.5 0.3 0.3\n"
    assert_rejected(
        builder.estimate,
        [GeneratedSnapshotSource("images/train/not-allowed.jpg", b"image")],
    )


def test_snapshot_rejects_missing_split_and_never_overwrites_job_directory():
    source_root = make_source_tree()
    staging_root = Path(tempfile.mkdtemp(prefix="ezyolo-snapshot-staging-"))
    builder = make_builder(source_root, staging_root)
    assert_rejected(
        builder.build,
        job_id=JOB_ID,
        task_type="detect",
        class_names=("person",),
        layout={"train": "images/train"},
        sources=build_sources(source_root),
    )
    build_snapshot(builder, source_root)
    assert_rejected(
        build_snapshot,
        builder,
        source_root,
    )


def test_verify_detects_missing_hash_mismatch_and_extra_file():
    source_root = make_source_tree()
    builder = make_builder(source_root)
    snapshot = build_snapshot(builder, source_root)

    missing = snapshot.root / "labels/train/train-1.txt"
    missing.unlink()
    assert_rejected(builder.verify, snapshot, expected=SnapshotIntegrityError)

    # 重新建一个独立快照分别测试哈希和额外文件，避免一次篡改影响另一次断言。
    builder = make_builder(source_root)
    snapshot = build_snapshot(builder, source_root)
    target = snapshot.root / "images/train/train-1.jpg"
    target.write_bytes(b"tampered")
    assert_rejected(builder.verify, snapshot, expected=SnapshotIntegrityError)

    builder = make_builder(source_root)
    snapshot = build_snapshot(builder, source_root)
    (snapshot.root / "images/train/extra.jpg").write_bytes(b"extra")
    assert_rejected(builder.verify, snapshot, expected=SnapshotIntegrityError)


def test_remote_payload_requires_matching_job_spec_and_rejects_extra_metadata():
    source_root = make_source_tree()
    builder = make_builder(source_root)
    snapshot = build_snapshot(builder, source_root)
    spec = JobSpec(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        task_type="detect",
        model_symbol="yolov10n",
        epochs=100,
        batch=8,
        imgsz=640,
        class_names=("person",),
        snapshot_hash=snapshot.manifest.snapshot_hash,
    )
    write_job_spec(snapshot, spec)
    manifest, restored_spec = verify_remote_payload(snapshot.root)
    assert manifest == snapshot.manifest
    assert restored_spec == spec
    assert_rejected(builder.verify, snapshot, expected=SnapshotIntegrityError)

    (snapshot.root / "unexpected.json").write_text("{}", encoding="utf-8")
    assert_rejected(verify_remote_payload, snapshot.root, expected=SnapshotIntegrityError)


def test_job_spec_cannot_claim_a_different_snapshot():
    source_root = make_source_tree()
    builder = make_builder(source_root)
    snapshot = build_snapshot(builder, source_root)
    mismatched = JobSpec(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        task_type="detect",
        model_symbol="yolov10n",
        epochs=100,
        batch=8,
        imgsz=640,
        class_names=("person",),
        snapshot_hash="a" * 64,
    )
    assert_rejected(write_job_spec, snapshot, mismatched)


def test_snapshot_module_does_not_depend_on_local_training_thread():
    source = (
        Path(__file__).parent.parent / "core" / "remote_training" / "snapshot.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(module == "gui" or module.startswith("gui.") for module in imports)


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
