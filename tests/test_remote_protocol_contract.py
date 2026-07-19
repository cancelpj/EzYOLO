# -*- coding: utf-8 -*-
"""ezyolo-remote/v1 共享 wire contract 的离线测试。

运行：
    PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \\
      ./.venv/bin/python tests/test_remote_protocol_contract.py
"""

import _bootstrap  # noqa: F401  保持测试路径与现有测试一致

import ast
import sys
from pathlib import Path

from remote_protocol.v1 import (  # noqa: E402
    CAPABILITIES_FIELDS,
    JOB_SPEC_FIELDS,
    MANIFEST_ENTRY_FIELDS,
    MANIFEST_FIELDS,
    RESULT_MANIFEST_FIELDS,
    RESULT_RECEIPT_FIELDS,
    RUNNER_FAILURE_FIELDS,
    STATUS_FIELDS,
    REMOTE_PROTOCOL_VERSION,
    DatasetManifest,
    FailureCode,
    JobSpec,
    JobStatus,
    ManifestEntry,
    ProtocolStateError,
    ProtocolValidationError,
    RemoteStatus,
    RunnerFailureEnvelope,
    ResultManifest,
    ResultReceipt,
    ServerCapabilities,
    advance_local_status,
    begin_attach,
    complete_collection,
    mark_unknown,
    resolve_runner_status,
    validate_job_id,
)


JOB_ID = "a" * 32
SHA256 = "b" * 64


def assert_rejected(callable_object, *args, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except (ProtocolValidationError, ProtocolStateError):
        return
    raise AssertionError("预期输入被拒绝")


def sample_capabilities():
    return ServerCapabilities(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        canonical_remote_root="/srv/ezyolo/alice",
        supported_tasks=("detect", "segment"),
        model_symbols=("yolov10n",),
        max_epochs=300,
        max_runtime_seconds=3600,
        max_payload_bytes=10_000_000,
        max_result_bytes=5_000_000,
    )


def sample_spec():
    return JobSpec(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        task_type="detect",
        model_symbol="yolov10n",
        epochs=100,
        batch=8,
        imgsz=640,
        class_names=("person", "car"),
        snapshot_hash=SHA256,
    )


def sample_manifest():
    entry = ManifestEntry(path="images/train/image-001.jpg", size=42, sha256=SHA256)
    return DatasetManifest(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        task_type="detect",
        class_names=("person",),
        layout=(("train", "images/train"), ("val", "images/val")),
        entries=(entry,),
        total_bytes=42,
        snapshot_hash=SHA256,
    )


def sample_result_manifest():
    entry = ManifestEntry(path="weights/best.pt", size=99, sha256=SHA256)
    return ResultManifest(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        entries=(entry,),
        total_bytes=99,
    )


def test_wire_field_names_are_stable():
    assert CAPABILITIES_FIELDS == (
        "protocol_version", "canonical_remote_root", "supported_tasks", "model_symbols",
        "max_epochs", "max_runtime_seconds", "max_payload_bytes", "max_result_bytes",
    )
    assert JOB_SPEC_FIELDS == (
        "protocol_version", "job_id", "task_type", "model_symbol", "epochs", "batch",
        "imgsz", "class_names", "snapshot_hash",
    )
    assert MANIFEST_ENTRY_FIELDS == ("path", "size", "sha256")
    assert MANIFEST_FIELDS == (
        "protocol_version", "job_id", "task_type", "class_names", "layout", "entries",
        "total_bytes", "snapshot_hash",
    )
    assert STATUS_FIELDS == (
        "protocol_version", "job_id", "status", "failure_code", "message", "epoch",
    )
    assert RESULT_RECEIPT_FIELDS == (
        "protocol_version", "job_id", "result_manifest_hash", "result_count", "result_bytes",
    )
    assert RESULT_MANIFEST_FIELDS == (
        "protocol_version", "job_id", "entries", "total_bytes",
    )
    assert RUNNER_FAILURE_FIELDS == (
        "protocol_version", "ok", "failure_code", "message",
    )


def test_contract_types_round_trip_strictly():
    capabilities = sample_capabilities()
    assert ServerCapabilities.from_wire(capabilities.to_wire()) == capabilities

    spec = sample_spec()
    assert JobSpec.from_wire(spec.to_wire()) == spec

    manifest = sample_manifest()
    assert DatasetManifest.from_wire(manifest.to_wire()) == manifest

    result_manifest = sample_result_manifest()
    assert ResultManifest.from_wire(result_manifest.to_wire()) == result_manifest

    status = RemoteStatus(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        status=JobStatus.RUNNING,
        message="第 4 轮",
        epoch=4,
    )
    assert RemoteStatus.from_wire(status.to_wire()) == status

    failed = RemoteStatus(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        status=JobStatus.FAILED,
        failure_code=FailureCode.REMOTE_CRASHED,
    )
    assert RemoteStatus.from_wire(failed.to_wire()) == failed

    receipt = ResultReceipt(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        result_manifest_hash=SHA256,
        result_count=3,
        result_bytes=99,
    )
    assert ResultReceipt.from_wire(receipt.to_wire()) == receipt

    runner_failure = RunnerFailureEnvelope(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        ok=False,
        failure_code=FailureCode.PRECHECK_FAILED,
        message="服务器拒绝请求",
    )
    assert RunnerFailureEnvelope.from_wire(runner_failure.to_wire()) == runner_failure


def test_contract_rejects_version_and_unknown_fields():
    payload = sample_spec().to_wire()
    payload["protocol_version"] = 99
    assert_rejected(JobSpec.from_wire, payload)

    payload = sample_spec().to_wire()
    payload["runner_command"] = "not-allowed"
    assert_rejected(JobSpec.from_wire, payload)

    payload = sample_manifest().to_wire()
    payload["entries"][0]["extra"] = True
    assert_rejected(DatasetManifest.from_wire, payload)

    payload = sample_result_manifest().to_wire()
    payload["unexpected"] = True
    assert_rejected(ResultManifest.from_wire, payload)

    payload = RunnerFailureEnvelope(
        REMOTE_PROTOCOL_VERSION,
        False,
        FailureCode.PRECHECK_FAILED,
        "服务器拒绝请求",
    ).to_wire()
    payload["debug"] = "not-allowed"
    assert_rejected(RunnerFailureEnvelope.from_wire, payload)
    payload = RunnerFailureEnvelope(
        REMOTE_PROTOCOL_VERSION,
        False,
        FailureCode.PRECHECK_FAILED,
        "服务器拒绝请求",
    ).to_wire()
    payload["ok"] = True
    assert_rejected(RunnerFailureEnvelope.from_wire, payload)

    assert_rejected(
        ServerCapabilities,
        REMOTE_PROTOCOL_VERSION,
        "/root/ezyolo",
        ("detect",),
        ("yolov10n",),
        1,
        1,
        1,
        1,
    )


def test_job_and_payload_validation_rejects_escape_values():
    assert validate_job_id(JOB_ID) == JOB_ID
    for bad_id in ("A" * 32, "a" * 31, "a" * 33, "a" * 31 + ";"):
        assert_rejected(validate_job_id, bad_id)

    assert_rejected(ManifestEntry, "../image.jpg", 1, SHA256)
    assert_rejected(ManifestEntry, "/absolute.jpg", 1, SHA256)
    assert_rejected(ManifestEntry, "images/-unsafe.jpg", 1, SHA256)
    assert_rejected(ManifestEntry, "images/line\\n.jpg", 1, SHA256)
    assert_rejected(ManifestEntry, "images/a.jpg", -1, SHA256)
    assert_rejected(ResultManifest, REMOTE_PROTOCOL_VERSION, JOB_ID, (), 0)


def test_failed_status_requires_known_failure_code():
    assert_rejected(
        RemoteStatus,
        REMOTE_PROTOCOL_VERSION,
        JOB_ID,
        JobStatus.FAILED,
    )
    assert_rejected(
        RemoteStatus,
        REMOTE_PROTOCOL_VERSION,
        JOB_ID,
        JobStatus.RUNNING,
        FailureCode.TRAINING_FAILED,
    )


def test_unknown_is_only_resolved_by_runner_fact():
    unknown = mark_unknown(JobStatus.RUNNING)
    assert unknown == JobStatus.UNKNOWN
    assert_rejected(advance_local_status, unknown, JobStatus.RUNNING)
    assert resolve_runner_status(unknown, JobStatus.RUNNING) == JobStatus.RUNNING

    attaching = begin_attach()
    assert attaching == JobStatus.ATTACHING
    assert_rejected(advance_local_status, attaching, JobStatus.RUNNING)
    assert resolve_runner_status(attaching, JobStatus.CANCELLED) == JobStatus.CANCELLED


def test_success_requires_local_collection_and_atomic_promote():
    collecting = advance_local_status(
        JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
        JobStatus.COLLECTING,
    )
    assert_rejected(complete_collection, collecting, promoted=False)
    assert complete_collection(collecting, promoted=True) == JobStatus.SUCCEEDED
    assert_rejected(advance_local_status, JobStatus.COLLECTING, JobStatus.SUCCEEDED)


def test_protocol_source_stays_pure_python():
    source = (Path(__file__).parent.parent / "remote_protocol" / "v1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    for forbidden_prefix in ("core", "gui", "PyQt", "ultralytics"):
        assert not any(
            module == forbidden_prefix or module.startswith(forbidden_prefix + ".")
            for module in imported_modules
        )


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
