# -*- coding: utf-8 -*-
"""远程训练 job record 与本机原子落地的离线测试。"""

import _bootstrap

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from PyQt6.QtCore import QSettings  # noqa: E402

from core.remote_training.jobs import (  # noqa: E402
    REMOTE_TRAINING_JOBS_KEY,
    RemoteTrainingJobError,
    RemoteTrainingJobRecord,
    RemoteTrainingJobStore,
    ResultPromotionError,
    ResultStager,
)
from remote_protocol.v1 import (  # noqa: E402
    REMOTE_PROTOCOL_VERSION,
    FailureCode,
    JobStatus,
    RemoteStatus,
    ResultReceipt,
)


JOB_ID = "2" * 32
PROFILE_ID = "3" * 32
SHA256 = "a" * 64
NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def assert_rejected(callable_object, *args, expected=RemoteTrainingJobError, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except expected:
        return
    raise AssertionError("预期输入被拒绝")


def make_record(job_id=JOB_ID):
    return RemoteTrainingJobRecord.new(
        job_id=job_id,
        project_id=7,
        profile_id=PROFILE_ID,
        canonical_remote_root="/srv/ezyolo/alice",
        snapshot_hash=SHA256,
        now=NOW,
    )


def make_settings(name):
    settings = QSettings("EzYOLO", name)
    settings.clear()
    settings.sync()
    return settings


def test_job_record_round_trip_and_store_rejects_duplicates():
    store = RemoteTrainingJobStore(make_settings("remote-job-store"))
    record = make_record()
    store.create(record)
    assert store.get(JOB_ID) == record
    assert store.list() == [record]
    assert_rejected(store.create, record)

    payload = record.to_dict()
    assert RemoteTrainingJobRecord.from_dict(payload) == record
    payload["password"] = "must-not-be-saved"
    assert_rejected(RemoteTrainingJobRecord.from_dict, payload)


def test_malformed_store_fails_closed():
    settings = make_settings("remote-job-bad-store")
    settings.setValue(REMOTE_TRAINING_JOBS_KEY, '{"not": "a list"}')
    settings.sync()
    assert_rejected(RemoteTrainingJobStore(settings).list)


def test_job_record_rejects_root_owned_remote_root_even_if_settings_are_hand_edited():
    payload = make_record().to_dict()
    payload["canonical_remote_root"] = "/root/ezyolo"
    assert_rejected(RemoteTrainingJobRecord.from_dict, payload)


def test_runner_status_unknown_remote_crash_cancel_and_collection_failure():
    record = make_record()
    running = record.with_runner_status(
        RemoteStatus(
            REMOTE_PROTOCOL_VERSION,
            JOB_ID,
            JobStatus.RUNNING,
        ),
        now=NOW,
    )
    unknown = RemoteTrainingJobRecord(
        **{**running.to_dict(), "last_status": JobStatus.UNKNOWN.value, "failure_code": None}
    )
    restored = unknown.with_runner_status(
        RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.RUNNING),
        now=NOW,
    )
    assert restored.last_status == JobStatus.RUNNING

    crashed = restored.with_runner_status(
        RemoteStatus(
            REMOTE_PROTOCOL_VERSION,
            JOB_ID,
            JobStatus.FAILED,
            FailureCode.REMOTE_CRASHED,
        ),
        now=NOW,
    )
    assert crashed.last_status == JobStatus.FAILED
    assert crashed.failure_code == FailureCode.REMOTE_CRASHED

    cancel_requested = running.with_runner_status(
        RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.CANCEL_REQUESTED),
        now=NOW,
    )
    cancelled = cancel_requested.with_runner_status(
        RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.CANCELLED),
        now=NOW,
    )
    assert cancelled.last_status == JobStatus.CANCELLED

    remote_success = running.with_runner_status(
        RemoteStatus(
            REMOTE_PROTOCOL_VERSION,
            JOB_ID,
            JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
        ),
        now=NOW,
    )
    collecting = remote_success.begin_collection(now=NOW)
    collection_failed = collecting.fail_collection(now=NOW)
    assert collection_failed.last_status == JobStatus.FAILED
    assert collection_failed.failure_code == FailureCode.COLLECTION_FAILED


def test_job_record_local_progress_failure_and_unknown_are_persistable():
    record = make_record()
    snapshotting = record.advance_local(JobStatus.SNAPSHOTTING, now=NOW)
    uploading = snapshotting.advance_local(JobStatus.UPLOADING, now=NOW)
    assert uploading.last_status == JobStatus.UPLOADING
    unknown = uploading.mark_unknown(now=NOW)
    assert unknown.last_status == JobStatus.UNKNOWN
    assert_rejected(unknown.advance_local, JobStatus.UPLOADING)

    failed = uploading.fail(FailureCode.UPLOAD_INTEGRITY, now=NOW)
    assert failed.last_status == JobStatus.FAILED
    assert failed.failure_code == FailureCode.UPLOAD_INTEGRITY
    assert_rejected(uploading.advance_local, JobStatus.FAILED)


def test_result_promote_is_same_filesystem_atomic_and_never_deserializes_weights():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-result-staging-"))
    stager = ResultStager(root / "staging", root / "runs" / "train")
    staging = stager.create_staging(JOB_ID)
    weights = staging / "weights"
    weights.mkdir()
    (weights / "best.pt").write_bytes(b"not-a-pickle-and-never-loaded")
    (staging / "metrics.json").write_text("{}", encoding="utf-8")

    final_dir = stager.promote(staging, project_id=7, job_id=JOB_ID)
    assert final_dir.name == "exp_7_remote_22222222"
    assert (final_dir / "weights" / "best.pt").read_bytes() == b"not-a-pickle-and-never-loaded"
    assert not staging.exists()


def test_result_promote_refuses_existing_destination_and_different_filesystem():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-result-staging-"))
    stager = ResultStager(root / "staging", root / "runs" / "train")
    staging = stager.create_staging(JOB_ID)
    final_dir = stager.final_dir_for(7, JOB_ID)
    final_dir.parent.mkdir(parents=True)
    final_dir.mkdir()
    marker = final_dir / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    assert_rejected(stager.promote, staging, project_id=7, job_id=JOB_ID, expected=ResultPromotionError)
    assert marker.read_text(encoding="utf-8") == "do not overwrite"
    assert staging.exists()

    other_job = "4" * 32
    staging = stager.create_staging(other_job)
    target = stager.runs_train_root
    real_stat = __import__("os").stat

    def fake_stat(path):
        result = real_stat(path)
        device = 1 if Path(path).resolve().is_relative_to(stager.staging_parent.resolve()) else 2
        return SimpleNamespace(st_dev=device, st_mode=result.st_mode)

    different_fs = ResultStager(
        stager.staging_parent,
        target,
        stat_function=fake_stat,
    )
    assert_rejected(
        different_fs.promote,
        staging,
        project_id=7,
        job_id=other_job,
        expected=ResultPromotionError,
    )
    assert staging.exists()


def test_finish_collection_requires_receipt_and_promoted_result_directory():
    record = make_record().with_runner_status(
        RemoteStatus(
            REMOTE_PROTOCOL_VERSION,
            JOB_ID,
            JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
        ),
        now=NOW,
    ).begin_collection(now=NOW)
    receipt = ResultReceipt(
        REMOTE_PROTOCOL_VERSION,
        JOB_ID,
        SHA256,
        result_count=2,
        result_bytes=42,
    )
    completed = record.finish_collection(
        receipt=receipt,
        local_result_dir=Path("/tmp/results"),
        now=NOW,
    )
    assert completed.last_status == JobStatus.SUCCEEDED
    assert completed.local_result_dir == "/tmp/results"


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
