# -*- coding: utf-8 -*-
"""RemoteTrainingThread 的 fake backend 离线状态机测试。"""

import _bootstrap  # noqa: F401  必须先隔离 Qt 与用户设置

from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import MappingProxyType

from core.remote_training.jobs import (  # noqa: E402
    RemoteTrainingJobRecord,
    ResultStager,
)
from core.remote_training.results import verify_result_bundle  # noqa: E402
from core.remote_training.launch import RemoteLaunchPlan  # noqa: E402
from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from core.remote_training.snapshot import DatasetSnapshotBuilder, SnapshotSource  # noqa: E402
from core.remote_training.transport import (  # noqa: E402
    RemoteRunnerReportedFailure,
    RemoteTransportError,
)
from gui.remote_training_thread import (  # noqa: E402
    RemoteConnectionTestThread,
    RemoteTrainingRequest,
    RemoteTrainingThread,
)
from remote_protocol.v1 import (  # noqa: E402
    REMOTE_PROTOCOL_VERSION,
    FailureCode,
    JobStatus,
    ManifestEntry,
    RemoteStatus,
    ResultManifest,
    ResultReceipt,
    ServerCapabilities,
)
from test_remote_training_profiles import _ed25519_key  # noqa: E402


JOB_ID = "9" * 32
PROFILE_ID = "a" * 32
SHA256 = "b" * 64


class MemoryJobStore:
    def __init__(self):
        self.records = {}

    def create(self, record):
        if record.job_id in self.records:
            raise AssertionError("duplicate job")
        self.records[record.job_id] = record

    def replace(self, record):
        if record.job_id not in self.records:
            raise AssertionError("missing job")
        self.records[record.job_id] = record


class FakeBackend:
    def __init__(self, *, poll_states=None, upload_error=None, cancel_state=None):
        self.calls = []
        self.poll_states = list(
            poll_states
            or [
                JobStatus.RUNNING,
                JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
            ]
        )
        self.upload_error = upload_error
        self.cancel_state = cancel_state or JobStatus.CANCELLED

    def preflight(self, profile):
        self.calls.append("preflight")
        return ServerCapabilities(
            protocol_version=REMOTE_PROTOCOL_VERSION,
            canonical_remote_root=profile.remote_root,
            supported_tasks=("detect", "segment"),
            model_symbols=("yolov10n",),
            max_epochs=300,
            max_runtime_seconds=3600,
            max_payload_bytes=10_000_000,
            max_result_bytes=10_000_000,
        )

    def upload(self, profile, job_id, snapshot_root):
        self.calls.append("upload")
        if self.upload_error:
            raise self.upload_error

    def verify_upload(self, profile, job_id):
        self.calls.append("verify-upload")
        return self._status(job_id, JobStatus.VERIFYING_UPLOAD)

    def start(self, profile, job_id):
        self.calls.append("start")
        return self._status(job_id, JobStatus.RUNNING)

    def poll(self, profile, job_id):
        self.calls.append("poll")
        return self._status(job_id, self.poll_states.pop(0))

    def cancel(self, profile, job_id):
        self.calls.append("cancel")
        return self._status(job_id, self.cancel_state)

    def collect_manifest(self, profile, job_id):
        self.calls.append("collect-manifest")
        manifest_bytes, result_bytes = _result_manifest_bytes(job_id)
        return ResultReceipt(
            REMOTE_PROTOCOL_VERSION,
            job_id,
            hashlib.sha256(manifest_bytes).hexdigest(),
            result_count=1,
            result_bytes=len(result_bytes),
        )

    def download_results(self, profile, job_id, staging_dir):
        self.calls.append("download-results")
        manifest_bytes, result_bytes = _result_manifest_bytes(job_id)
        weights = Path(staging_dir) / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(result_bytes)
        (Path(staging_dir) / "manifest.json").write_bytes(manifest_bytes)

    @staticmethod
    def _status(job_id, status):
        return RemoteStatus(
            REMOTE_PROTOCOL_VERSION,
            job_id,
            status,
            failure_code=FailureCode.TRAINING_FAILED if status == JobStatus.FAILED else None,
        )


def make_source_root():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-thread-source-"))
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}.jpg").write_bytes(split.encode())
        (root / "labels" / split / f"{split}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    return root


def _result_manifest_bytes(job_id):
    result_bytes = b"result-bytes-are-only-hashed"
    manifest = ResultManifest(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=job_id,
        entries=(
            ManifestEntry(
                path="weights/best.pt",
                size=len(result_bytes),
                sha256=hashlib.sha256(result_bytes).hexdigest(),
            ),
        ),
        total_bytes=len(result_bytes),
    )
    return (
        (
            json.dumps(
                manifest.to_wire(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
        result_bytes,
    )


def make_request(source_root):
    profile = RemoteTrainingProfile(
        name="实验室 A100",
        host="train-lab",
        port=22,
        username="trainer",
        remote_root="/srv/ezyolo/trainer",
        host_public_key=_ed25519_key(),
        id=PROFILE_ID,
    )
    plan = RemoteLaunchPlan(
        project_id=3,
        task_type="detect",
        model_symbol="yolov10n",
        profile=profile,
        runtime_config=MappingProxyType(
            {"epochs": 10, "batch_size": 2, "img_size": 640}
        ),
    )
    sources = []
    for split in ("train", "val"):
        sources.extend(
            [
                SnapshotSource(
                    source_root / "images" / split / f"{split}.jpg",
                    f"images/{split}/{split}.jpg",
                ),
                SnapshotSource(
                    source_root / "labels" / split / f"{split}.txt",
                    f"labels/{split}/{split}.txt",
                ),
            ]
        )
    return RemoteTrainingRequest(
        launch_plan=plan,
        class_names=("person",),
        layout={"train": "images/train", "val": "images/val"},
        sources=sources,
    )


def make_thread(backend, verifier=None, **thread_options):
    source_root = make_source_root()
    root = Path(tempfile.mkdtemp(prefix="ezyolo-thread-work-"))
    store = MemoryJobStore()
    verifier = verifier or verify_result_bundle
    thread = RemoteTrainingThread(
        request=make_request(source_root),
        backend=backend,
        job_store=store,
        snapshot_builder=DatasetSnapshotBuilder(
            root / "snapshots",
            allowed_roots=[source_root],
        ),
        result_stager=ResultStager(root / "staging", root / "runs" / "train"),
        result_verifier=verifier,
        poll_interval_seconds=0,
        sleep=lambda _seconds: None,
        job_id_factory=lambda: JOB_ID,
        **thread_options,
    )
    return thread, store, root


def test_full_fake_workflow_persists_success_only_after_result_verify_and_promote():
    _bootstrap.app()
    backend = FakeBackend()
    thread, store, root = make_thread(backend)
    finished = []
    thread.training_finished.connect(lambda ok, message: finished.append((ok, message)))
    thread.run()

    record = store.records[JOB_ID]
    assert record.last_status == JobStatus.SUCCEEDED
    assert record.local_result_dir
    assert (Path(record.local_result_dir) / "weights" / "best.pt").read_bytes() == b"result-bytes-are-only-hashed"
    assert backend.calls == [
        "preflight",
        "upload",
        "verify-upload",
        "start",
        "poll",
        "poll",
        "collect-manifest",
        "download-results",
    ]
    assert finished[-1][0] is True
    assert not (root / "staging" / JOB_ID).exists()


def test_bad_collected_result_never_promotes_and_marks_local_failure():
    _bootstrap.app()
    backend = FakeBackend()
    thread, store, root = make_thread(
        backend,
        verifier=lambda staging, receipt: (_ for _ in ()).throw(ValueError("hash mismatch")),
    )
    thread.run()
    record = store.records[JOB_ID]
    assert record.last_status == JobStatus.FAILED
    assert record.failure_code == FailureCode.LOCAL_VERIFY_FAILED
    assert not (root / "runs" / "train" / "exp_3_remote_99999999").exists()


def test_transport_error_after_job_created_persists_unknown_for_reconnect():
    _bootstrap.app()
    backend = FakeBackend(
        upload_error=RemoteTransportError(
            FailureCode.UPLOAD_INTEGRITY,
            "connection failed",
        )
    )
    thread, store, _root = make_thread(backend)
    thread.run()
    assert store.records[JOB_ID].last_status == JobStatus.UNKNOWN


def test_declared_result_above_preflight_limit_is_never_downloaded_or_promoted():
    _bootstrap.app()
    backend = FakeBackend()

    def oversized_receipt(_profile, job_id):
        backend.calls.append("collect-manifest")
        return ResultReceipt(
            REMOTE_PROTOCOL_VERSION,
            job_id,
            SHA256,
            result_count=1,
            result_bytes=10_000_001,
        )

    backend.collect_manifest = oversized_receipt
    thread, store, root = make_thread(backend)
    thread.run()

    record = store.records[JOB_ID]
    assert record.last_status == JobStatus.FAILED
    assert record.failure_code == FailureCode.COLLECTION_FAILED
    assert "download-results" not in backend.calls
    assert not (root / "staging" / JOB_ID).exists()


def test_foreign_result_receipt_is_rejected_before_download_or_staging():
    _bootstrap.app()
    backend = FakeBackend()
    foreign_job_id = "d" * 32

    def foreign_receipt(_profile, _job_id):
        backend.calls.append("collect-manifest")
        return ResultReceipt(
            REMOTE_PROTOCOL_VERSION,
            foreign_job_id,
            SHA256,
            result_count=1,
            result_bytes=1,
        )

    backend.collect_manifest = foreign_receipt
    thread, store, root = make_thread(backend)
    thread.run()

    record = store.records[JOB_ID]
    assert record.last_status == JobStatus.FAILED
    assert record.failure_code == FailureCode.COLLECTION_FAILED
    assert "download-results" not in backend.calls
    assert not (root / "staging" / JOB_ID).exists()


def test_cancel_waits_for_runner_confirmation_before_terminal_state():
    _bootstrap.app()
    backend = FakeBackend(cancel_state=JobStatus.CANCELLED)
    thread, store, _root = make_thread(backend)
    thread.request_cancel()
    thread.run()
    assert backend.calls[-1] == "cancel"
    assert store.records[JOB_ID].last_status == JobStatus.CANCELLED


def test_cancel_requested_is_polled_until_runner_confirms_cancelled():
    _bootstrap.app()
    backend = FakeBackend(
        cancel_state=JobStatus.CANCEL_REQUESTED,
        poll_states=[JobStatus.CANCELLED],
    )
    thread, store, _root = make_thread(backend)
    thread.request_cancel()
    thread.run()

    assert backend.calls[-2:] == ["cancel", "poll"]
    assert store.records[JOB_ID].last_status == JobStatus.CANCELLED


def test_cancel_timeout_stops_local_wait_without_claiming_remote_was_stopped():
    _bootstrap.app()
    backend = FakeBackend(
        cancel_state=JobStatus.CANCEL_REQUESTED,
        poll_states=[JobStatus.CANCEL_REQUESTED],
    )
    ticks = iter((0.0, 61.0))
    thread, store, _root = make_thread(
        backend,
        cancel_timeout_seconds=60,
        clock=lambda: next(ticks),
    )
    thread.request_cancel()
    thread.run()

    record = store.records[JOB_ID]
    assert record.last_status == JobStatus.FAILED
    assert record.failure_code == FailureCode.CANCEL_TIMEOUT
    assert backend.calls.count("cancel") == 1
    assert "poll" not in backend.calls


def test_runner_reported_failure_marks_created_job_failed_not_unknown():
    _bootstrap.app()
    backend = FakeBackend()
    secret = "server-internal-path-should-not-reach-ui"

    def reported_failure(_profile, _job_id):
        backend.calls.append("verify-upload")
        raise RemoteRunnerReportedFailure(FailureCode.PRECHECK_FAILED, secret)

    backend.verify_upload = reported_failure
    thread, store, _root = make_thread(backend)
    finished = []
    thread.training_finished.connect(lambda ok, message: finished.append((ok, message)))
    thread.run()

    record = store.records[JOB_ID]
    assert record.last_status == JobStatus.FAILED
    assert record.failure_code == FailureCode.PRECHECK_FAILED
    assert secret not in finished[-1][1]


def test_connection_test_thread_only_calls_preflight_and_never_creates_a_job():
    _bootstrap.app()
    backend = FakeBackend()
    profile = make_request(make_source_root()).launch_plan.profile
    thread = RemoteConnectionTestThread(profile=profile, backend=backend)
    results = []
    thread.preflight_finished.connect(lambda ok, message, capabilities: results.append((ok, message, capabilities)))

    thread.run()

    assert backend.calls == ["preflight"]
    assert results[-1][0] is True
    assert profile.name in results[-1][1]
    assert "未上传数据" in results[-1][1]
    assert results[-1][2].canonical_remote_root == profile.remote_root


def test_connection_test_thread_redacts_transport_failure_details():
    _bootstrap.app()

    class FailingBackend:
        def preflight(self, _profile):
            raise RemoteTransportError(
                FailureCode.PRECHECK_FAILED,
                "private-key=/very-secret-path",
            )

    profile = make_request(make_source_root()).launch_plan.profile
    thread = RemoteConnectionTestThread(profile=profile, backend=FailingBackend())
    results = []
    thread.preflight_finished.connect(lambda ok, message, capabilities: results.append((ok, message, capabilities)))
    thread.run()

    assert results[-1][0] is False
    assert "very-secret" not in results[-1][1]
    assert results[-1][2] is None


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
