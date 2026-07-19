# -*- coding: utf-8 -*-
"""remote_runner 的纯离线协议与安全边界测试。

运行：
    PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
      ./.venv/bin/python tests/test_remote_runner_protocol.py
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from remote_protocol.v1 import (  # noqa: E402
    REMOTE_PROTOCOL_VERSION,
    DatasetManifest,
    FailureCode,
    JobSpec,
    JobStatus,
    ManifestEntry,
    RemoteStatus,
    RunnerFailureEnvelope,
    ResultReceipt,
)
import remote_runner.cli as runner_cli  # noqa: E402
from remote_runner.cli import dispatch, parse_action  # noqa: E402
from remote_runner.config import ConfigError, server_config_from_mapping  # noqa: E402
from remote_runner.jobs import (  # noqa: E402
    DATASET_MANIFEST_FILENAME,
    JOB_SPEC_FILENAME,
    RESULT_MANIFEST_FILENAME,
    ProcessIdentity,
    Runner,
    RunnerFailure,
    RunnerPaths,
)
from remote_runner.trainer import Watchdog  # noqa: E402


JOB_ID = "a" * 32
SHA256 = "b" * 64


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def assert_rejected(callable_object, *args, code=None, expected=(RunnerFailure, ConfigError, ValueError), **kwargs):
    try:
        callable_object(*args, **kwargs)
    except expected as exc:
        if code is not None:
            assert isinstance(exc, RunnerFailure), exc
            assert exc.code == code, exc.code
        return exc
    raise AssertionError("预期输入被拒绝")


class FakeInspector:
    def __init__(self):
        self.current_boot_id = "boot-1"
        self.processes = {
            101: {"alive": True, "pgid": 101, "marker": "start-101", "group": True}
        }

    def boot_id(self):
        return self.current_boot_id

    def is_alive(self, pid):
        process = self.processes.get(pid)
        return process["alive"] if process is not None else False

    def pgid(self, pid):
        process = self.processes.get(pid)
        return process["pgid"] if process is not None else None

    def start_marker(self, pid):
        process = self.processes.get(pid)
        return process["marker"] if process is not None else None

    def group_alive(self, pgid):
        for process in self.processes.values():
            if process["pgid"] == pgid:
                return process["group"]
        return False

    def identity_for(self, job_id, pid):
        if self.is_alive(pid) is not True:
            return None
        return ProcessIdentity(
            job_id=job_id,
            pid=pid,
            pgid=self.pgid(pid),
            boot_id=self.boot_id(),
            start_marker=self.start_marker(pid),
        )


class FakeSpawner:
    def __init__(self, pid=101):
        self.pid = pid
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((tuple(command), kwargs))
        return SimpleNamespace(pid=self.pid)


class PreflightInspector(FakeInspector):
    def identity_for(self, job_id, pid):
        if pid == 101:
            return super().identity_for(job_id, pid)
        return ProcessIdentity(job_id, pid, pid, self.boot_id(), "current-process")


class FakeDisk:
    def __init__(self, free=10_000_000):
        self.free = free

    def __call__(self, _path):
        return SimpleNamespace(free=self.free)


class FakeDispatchRunner:
    def __init__(self, *, cancel_status, receipt):
        self.cancel_status = cancel_status
        self.receipt = receipt

    def cancel(self, _job_id):
        return self.cancel_status, True

    def collect_manifest(self, _job_id):
        return {"not": "exposed"}, self.receipt


def make_config(tmp: Path, **overrides):
    root = (tmp / "remote-root").resolve()
    root.mkdir()
    for name in ("incoming", "jobs", "results"):
        (root / name).mkdir(mode=0o700)
    weights = (tmp / "weights.pt").resolve()
    weights.write_bytes(b"preinstalled weight bytes")
    launcher = (tmp / "remote-launcher").resolve()
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o700)
    payload = {
        "protocol_version": REMOTE_PROTOCOL_VERSION,
        "canonical_remote_root": str(root),
        "model_allowlist": {"yolo11n": str(weights)},
        "runtime": {"launcher": str(launcher)},
        "max_epochs": 20,
        "max_runtime_seconds": 10,
        "max_payload_bytes": 100_000,
        "max_jobs_bytes": 1_000_000,
        "max_results_bytes": 100_000,
        "min_free_disk_bytes": 100,
        "single_task": True,
    }
    payload.update(overrides)
    return server_config_from_mapping(payload)


def make_runner(tmp: Path, *, inspector=None, disk=None, killer=None, config=None, spawner=None):
    config = config or make_config(tmp)
    return Runner(
        config,
        inspector=inspector or FakeInspector(),
        disk_usage=disk or FakeDisk(),
        process_group_killer=killer,
        supervisor_spawner=spawner,
    )


def write_upload(
    runner: Runner,
    *,
    job_id=JOB_ID,
    model_symbol="yolo11n",
    epochs=2,
    class_names=("car", "person"),
):
    incoming = runner.paths.incoming_dir(job_id)
    incoming.mkdir(parents=True)
    train = b"train-image"
    val = b"val-image"
    (incoming / "images" / "train").mkdir(parents=True)
    (incoming / "images" / "val").mkdir(parents=True)
    (incoming / "images" / "train" / "a.jpg").write_bytes(train)
    (incoming / "images" / "val" / "b.jpg").write_bytes(val)
    entries = (
        ManifestEntry("images/train/a.jpg", len(train), sha256(train)),
        ManifestEntry("images/val/b.jpg", len(val), sha256(val)),
    )
    manifest = DatasetManifest(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=job_id,
        task_type="detect",
        class_names=tuple(class_names),
        layout=(("train", "images/train"), ("val", "images/val")),
        entries=entries,
        total_bytes=sum(entry.size for entry in entries),
        snapshot_hash=SHA256,
    )
    spec = JobSpec(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=job_id,
        task_type="detect",
        model_symbol=model_symbol,
        epochs=epochs,
        batch=2,
        imgsz=640,
        class_names=tuple(class_names),
        snapshot_hash=SHA256,
    )
    write_json(incoming / DATASET_MANIFEST_FILENAME, manifest.to_wire())
    write_json(incoming / JOB_SPEC_FILENAME, spec.to_wire())
    return incoming, spec, manifest


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def verified_runner(tmp: Path, **runner_kwargs):
    runner = make_runner(tmp, **runner_kwargs)
    write_upload(runner)
    status = runner.verify_upload(JOB_ID)
    assert status.status == JobStatus.VERIFYING_UPLOAD
    return runner


def test_protocol_version_and_config_fields_fail_closed():
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        config = make_config(tmp)
        assert config.capabilities().protocol_version == REMOTE_PROTOCOL_VERSION

        raw = {
            "protocol_version": 2,
            "canonical_remote_root": str(config.canonical_remote_root),
            "model_allowlist": {"yolo11n": str(next(iter(config.model_allowlist.values())))},
            "runtime": {"launcher": str(config.runtime.launcher)},
            "max_epochs": 1,
            "max_runtime_seconds": 1,
            "max_payload_bytes": 1,
            "max_jobs_bytes": 1,
            "max_results_bytes": 1,
            "min_free_disk_bytes": 0,
            "single_task": True,
        }
        assert_rejected(server_config_from_mapping, raw)
        raw["protocol_version"] = REMOTE_PROTOCOL_VERSION
        raw["ssh_password"] = "must-not-be-accepted"
        assert_rejected(server_config_from_mapping, raw)


def test_config_rejects_noncanonical_root_and_symlinked_weight():
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        config = make_config(tmp)
        raw = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "canonical_remote_root": "/",
            "model_allowlist": {"yolo11n": str(next(iter(config.model_allowlist.values())))},
            "runtime": {"launcher": str(config.runtime.launcher)},
            "max_epochs": 1,
            "max_runtime_seconds": 1,
            "max_payload_bytes": 1,
            "max_jobs_bytes": 1,
            "max_results_bytes": 1,
            "min_free_disk_bytes": 0,
            "single_task": True,
        }
        assert_rejected(server_config_from_mapping, raw)

        weight_link = tmp / "weight-link.pt"
        weight_link.symlink_to(next(iter(config.model_allowlist.values())))
        raw["canonical_remote_root"] = str(config.canonical_remote_root)
        raw["model_allowlist"] = {"yolo11n": str(weight_link)}
        assert_rejected(server_config_from_mapping, raw)


def test_config_and_preflight_reject_group_or_world_writable_runner_paths():
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        config = make_config(tmp)
        raw = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "canonical_remote_root": str(config.canonical_remote_root),
            "model_allowlist": {"yolo11n": str(next(iter(config.model_allowlist.values())))},
            "runtime": {"launcher": str(config.runtime.launcher)},
            "max_epochs": 1,
            "max_runtime_seconds": 1,
            "max_payload_bytes": 1,
            "max_jobs_bytes": 1,
            "max_results_bytes": 1,
            "min_free_disk_bytes": 0,
            "single_task": True,
        }
        config.canonical_remote_root.chmod(0o777)
        assert_rejected(server_config_from_mapping, raw)

    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name), inspector=PreflightInspector())
        runner.paths.incoming_root.chmod(0o777)
        assert_rejected(runner.preflight, code=FailureCode.RUNNER_PROTOCOL)

    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name), inspector=PreflightInspector())
        next(iter(runner.config.model_allowlist.values())).chmod(0o666)
        assert_rejected(runner.preflight, code=FailureCode.RUNNER_PROTOCOL)


def test_cli_accepts_only_frozen_actions_and_job_id_shape():
    assert parse_action(["preflight"]) == ("preflight", None)
    assert parse_action(["status", JOB_ID]) == ("status", JOB_ID)
    assert_rejected(parse_action, ["preflight", JOB_ID])
    assert_rejected(parse_action, ["status", "A" * 32])
    assert_rejected(parse_action, ["shell", JOB_ID])
    assert_rejected(parse_action, ["start", JOB_ID, "--gpu", "0"])


def test_cli_dispatch_cancel_and_collect_emit_only_direct_protocol_wires():
    cancelled = RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.CANCELLED)
    receipt = ResultReceipt(
        REMOTE_PROTOCOL_VERSION,
        JOB_ID,
        "c" * 64,
        result_count=2,
        result_bytes=42,
    )
    runner = FakeDispatchRunner(cancel_status=cancelled, receipt=receipt)
    assert dispatch("cancel", JOB_ID, runner) == cancelled.to_wire()
    assert dispatch("collect-manifest", JOB_ID, runner) == receipt.to_wire()

    missing_status = FakeDispatchRunner(cancel_status=None, receipt=receipt)
    assert_rejected(
        dispatch,
        "cancel",
        JOB_ID,
        missing_status,
        code=FailureCode.RUNNER_PROTOCOL,
    )


def test_cli_failure_uses_strict_envelope_and_does_not_echo_config_error():
    original_load = runner_cli.load_server_config
    buffer = io.StringIO()

    def failing_config():
        raise ConfigError("private-config-path-must-not-be-echoed")

    runner_cli.load_server_config = failing_config
    try:
        with redirect_stdout(buffer):
            assert runner_cli.main(["preflight"]) == 2
    finally:
        runner_cli.load_server_config = original_load

    payload = json.loads(buffer.getvalue())
    envelope = RunnerFailureEnvelope.from_wire(payload)
    assert envelope.failure_code == FailureCode.RUNNER_PROTOCOL
    assert "private-config-path" not in envelope.message


def test_preflight_returns_protocol_capabilities_and_checks_disk_without_installing():
    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name), inspector=PreflightInspector())
        capabilities = runner.preflight()
        assert capabilities.protocol_version == REMOTE_PROTOCOL_VERSION
        assert capabilities.canonical_remote_root == str(runner.config.canonical_remote_root)
        assert capabilities.model_symbols == ("yolo11n",)

    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(
            Path(name), inspector=PreflightInspector(), disk=FakeDisk(free=0)
        )
        assert_rejected(runner.preflight, code=FailureCode.LOW_DISK_SPACE)


def test_preflight_requires_admin_preprovisioned_layout_and_never_creates_it():
    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name), inspector=PreflightInspector())
        missing = runner.config.canonical_remote_root / "incoming"
        missing.rmdir()

        assert_rejected(runner.preflight, code=FailureCode.RUNNER_PROTOCOL)
        assert not missing.exists(), "只读 preflight 不能偷偷创建远程目录"


def test_post_preflight_actions_also_refuse_missing_admin_layout_without_recreating_it():
    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name), inspector=PreflightInspector())
        missing = runner.config.canonical_remote_root / "incoming"
        missing.rmdir()

        assert_rejected(runner.verify_upload, JOB_ID, code=FailureCode.RUNNER_PROTOCOL)
        assert not missing.exists(), "runner 不应在上传核验时补建管理员目录"


def test_paths_stay_under_canonical_root():
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        runner = make_runner(tmp)
        paths = RunnerPaths(runner.config.canonical_remote_root)
        assert paths.job_dir(JOB_ID).relative_to(runner.config.canonical_remote_root)
        assert_rejected(paths.job_dir, "../" + JOB_ID)
        assert_rejected(paths.incoming_dir, "A" * 32)


def test_verify_upload_requires_exact_wire_file_set_and_hashes():
    cases = ("missing", "extra", "hash", "symlink", "unsafe")
    for case in cases:
        with tempfile.TemporaryDirectory() as name:
            runner = make_runner(Path(name))
            incoming, _spec, manifest = write_upload(runner)
            if case == "missing":
                (incoming / JOB_SPEC_FILENAME).unlink()
            elif case == "extra":
                (incoming / "notes.txt").write_text("extra", encoding="utf-8")
            elif case == "hash":
                (incoming / "images" / "train" / "a.jpg").write_bytes(b"tampered")
            elif case == "symlink":
                image = incoming / "images" / "train" / "a.jpg"
                target = incoming.parent / "outside.jpg"
                target.write_bytes(b"train-image")
                image.unlink()
                image.symlink_to(target)
            else:
                image = incoming / "images" / "train" / "a.jpg"
                image.rename(incoming / "data.yaml")
                raw = manifest.to_wire()
                raw["entries"][0]["path"] = "data.yaml"
                raw["layout"] = {"train": "images/train", "val": "images/val"}
                write_json(incoming / DATASET_MANIFEST_FILENAME, raw)
            assert_rejected(runner.verify_upload, JOB_ID, code=FailureCode.UPLOAD_INTEGRITY)
            assert incoming.exists(), case
            assert not runner.paths.job_dir(JOB_ID).exists(), case


def test_verify_upload_checks_job_spec_against_manifest_and_renames_atomically():
    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name))
        incoming, spec, _manifest = write_upload(runner)
        raw = spec.to_wire()
        raw["snapshot_hash"] = "c" * 64
        write_json(incoming / JOB_SPEC_FILENAME, raw)
        assert_rejected(runner.verify_upload, JOB_ID, code=FailureCode.UPLOAD_INTEGRITY)

    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name))
        incoming, _spec, manifest = write_upload(runner)
        raw = manifest.to_wire()
        raw["protocol_version"] = REMOTE_PROTOCOL_VERSION + 1
        write_json(incoming / DATASET_MANIFEST_FILENAME, raw)
        assert_rejected(runner.verify_upload, JOB_ID, code=FailureCode.RUNNER_PROTOCOL)

    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name))
        incoming, _spec, _manifest = write_upload(runner)
        status = runner.verify_upload(JOB_ID)
        assert status.status == JobStatus.VERIFYING_UPLOAD
        assert not incoming.exists()
        assert runner.paths.job_dir(JOB_ID).exists()
        assert runner.read_status(JOB_ID) == status


def test_status_write_is_atomic_and_running_identity_loss_is_remote_crashed():
    with tempfile.TemporaryDirectory() as name:
        inspector = FakeInspector()
        runner = verified_runner(
            Path(name),
            inspector=inspector,
            killer=lambda _pgid, _signal: None,
        )
        running = RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.RUNNING)
        runner.write_status(running)
        assert runner.read_status(JOB_ID) == running
        assert not list(runner.paths.job_dir(JOB_ID).glob(".*.tmp"))

        identity = inspector.identity_for(JOB_ID, 101)
        runner.lock.acquire(identity)
        inspector.processes[101]["group"] = False
        crashed = runner.status(JOB_ID)
        assert crashed.status == JobStatus.FAILED
        assert crashed.failure_code == FailureCode.REMOTE_CRASHED


def test_status_confirms_delayed_cancel_only_after_the_verified_group_exits():
    with tempfile.TemporaryDirectory() as name:
        inspector = FakeInspector()
        runner = verified_runner(
            Path(name),
            inspector=inspector,
            killer=lambda _pgid, _signal: None,
        )
        identity = inspector.identity_for(JOB_ID, 101)
        runner.lock.acquire(identity)
        runner.write_status(RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.RUNNING))

        requested, changed = runner.cancel(JOB_ID)
        assert changed is True
        assert requested.status == JobStatus.CANCEL_REQUESTED
        assert runner.status(JOB_ID).status == JobStatus.CANCEL_REQUESTED

        inspector.processes[101]["alive"] = False
        inspector.processes[101]["group"] = False
        confirmed = runner.status(JOB_ID)
        assert confirmed.status == JobStatus.CANCELLED
        assert runner.lock.read() is None


def test_single_task_lock_busy_and_only_proven_stale_records_are_reclaimed():
    with tempfile.TemporaryDirectory() as name:
        inspector = FakeInspector()
        runner = make_runner(Path(name), inspector=inspector)
        identity = inspector.identity_for(JOB_ID, 101)
        runner.lock.acquire(identity)
        assert_rejected(runner.lock.assert_available, code=FailureCode.SERVER_BUSY)

        inspector.processes[101]["alive"] = False
        runner.lock.assert_available()
        assert runner.lock.read() is None
        inspector.processes[101]["alive"] = True
        runner.lock.acquire(identity)
        inspector.processes[101]["marker"] = "different-start-marker"
        assert runner.lock.is_stale(identity)


def test_cancel_requires_full_identity_and_never_targets_unverified_processes():
    with tempfile.TemporaryDirectory() as name:
        killed = []
        inspector = FakeInspector()

        def killer(pgid, sig):
            killed.append((pgid, sig))
            inspector.processes[101]["alive"] = False

        runner = verified_runner(Path(name), inspector=inspector, killer=killer)
        runner.write_status(RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.RUNNING))
        identity = inspector.identity_for(JOB_ID, 101)
        runner.lock.acquire(identity)
        status, cancelled = runner.cancel(JOB_ID)
        assert cancelled is True
        assert status.status == JobStatus.CANCELLED
        assert killed and killed[0][0] == 101

    with tempfile.TemporaryDirectory() as name:
        killed = []
        inspector = FakeInspector()
        runner = verified_runner(Path(name), inspector=inspector, killer=lambda *args: killed.append(args))
        runner.write_status(RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.RUNNING))
        identity = inspector.identity_for(JOB_ID, 101)
        runner.lock.acquire(identity)
        inspector.processes[101]["marker"] = "reused-pid-marker"
        status, cancelled = runner.cancel(JOB_ID)
        assert cancelled is False
        assert status.status == JobStatus.RUNNING
        assert killed == []


def test_start_capacity_and_allowlist_failures_have_protocol_failure_codes():
    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name), spawner=FakeSpawner())
        write_upload(runner, epochs=21)
        runner.verify_upload(JOB_ID)
        failed = runner.start(JOB_ID)
        assert failed.failure_code == FailureCode.PRECHECK_FAILED

    with tempfile.TemporaryDirectory() as name:
        config = make_config(Path(name))
        runner = make_runner(Path(name), config=replace(config, model_allowlist={}), spawner=FakeSpawner())
        write_upload(runner)
        # allowlist 在 verify 阶段有防线；此处模拟管理员误删模型映射后的 start 复核。
        job_dir = runner.paths.incoming_dir(JOB_ID)
        assert_rejected(runner.verify_upload, JOB_ID, code=FailureCode.PRECHECK_FAILED)
        assert job_dir.exists()

    with tempfile.TemporaryDirectory() as name:
        config = make_config(Path(name), max_jobs_bytes=1)
        runner = make_runner(Path(name), config=config, spawner=FakeSpawner())
        write_upload(runner)
        runner.verify_upload(JOB_ID)
        failed = runner.start(JOB_ID)
        assert failed.failure_code == FailureCode.JOB_DISK_LIMIT

    with tempfile.TemporaryDirectory() as name:
        disk = FakeDisk(free=0)
        runner = make_runner(Path(name), disk=disk, spawner=FakeSpawner())
        write_upload(runner)
        runner.verify_upload(JOB_ID)
        failed = runner.start(JOB_ID)
        assert failed.failure_code == FailureCode.LOW_DISK_SPACE


def test_start_uses_a_fake_independent_supervisor_and_persists_the_matching_lock():
    with tempfile.TemporaryDirectory() as name:
        inspector = FakeInspector()
        spawner = FakeSpawner()
        runner = verified_runner(Path(name), inspector=inspector, spawner=spawner)
        status = runner.start(JOB_ID)
        assert status.status == JobStatus.RUNNING
        command, kwargs = spawner.calls[0]
        assert command[-1] == JOB_ID
        assert command[-2] == "remote_runner.trainer"
        assert kwargs["start_new_session"] is True
        assert kwargs["close_fds"] is True
        assert set(kwargs["env"]) == {"HOME", "PATH", "LANG", "LC_ALL"}
        assert runner.lock.read() == inspector.identity_for(JOB_ID, 101)


def test_data_yaml_is_runner_generated_json_quoted_and_has_no_client_download_or_paths():
    with tempfile.TemporaryDirectory() as name:
        runner = make_runner(Path(name))
        write_upload(runner, class_names=('car" : !!python/object', "person"))
        runner.verify_upload(JOB_ID)
        data_path = runner.build_data_yaml(JOB_ID)
        content = data_path.read_text(encoding="utf-8")
        assert "download:" not in content
        assert "../" not in content
        assert "path: /" not in content
        assert '0: "car\\\" : !!python/object"' in content
        assert not (runner.paths.job_dir(JOB_ID) / "client-data.yaml").exists()


def test_watchdog_only_stops_verified_job_process_group_and_reports_limits():
    with tempfile.TemporaryDirectory() as name:
        inspector = FakeInspector()
        killed = []
        runner = verified_runner(
            Path(name),
            inspector=inspector,
            killer=lambda pgid, sig: killed.append((pgid, sig)),
        )
        runner.write_status(RemoteStatus(REMOTE_PROTOCOL_VERSION, JOB_ID, JobStatus.RUNNING))
        runner.lock.acquire(inspector.identity_for(JOB_ID, 101))
        ticks = iter((0.0, 11.0))
        watchdog = Watchdog(runner, JOB_ID, clock=lambda: next(ticks))
        assert watchdog.check() == FailureCode.MAX_RUNTIME
        assert killed and killed[0][0] == 101
        assert runner.read_status(JOB_ID).failure_code == FailureCode.MAX_RUNTIME

    with tempfile.TemporaryDirectory() as name:
        config = make_config(Path(name), max_jobs_bytes=1)
        runner = verified_runner(Path(name), config=config)
        assert runner.watchdog_failure(JOB_ID, 0) == FailureCode.JOB_DISK_LIMIT

    with tempfile.TemporaryDirectory() as name:
        runner = verified_runner(Path(name), disk=FakeDisk(free=0))
        assert runner.watchdog_failure(JOB_ID, 0) == FailureCode.LOW_DISK_SPACE


def test_result_manifest_and_receipt_are_controlled_and_never_load_weights():
    with tempfile.TemporaryDirectory() as name:
        runner = verified_runner(Path(name))
        result_dir = runner._create_result_dir(JOB_ID)
        (result_dir / "weights").mkdir()
        (result_dir / "weights" / "best.pt").write_bytes(b"not a pickle and never loaded")
        (result_dir / "metrics.json").write_text("{}", encoding="utf-8")
        declared = runner.declare_results(JOB_ID)
        manifest_path = result_dir / RESULT_MANIFEST_FILENAME
        assert manifest_path.exists()
        assert set(declared) == {"protocol_version", "job_id", "entries", "total_bytes"}
        assert all(set(entry) == {"path", "size", "sha256"} for entry in declared["entries"])
        assert RESULT_MANIFEST_FILENAME not in {entry["path"] for entry in declared["entries"]}
        assert manifest_path.read_bytes() == (
            json.dumps(
                declared, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        runner.write_status(
            RemoteStatus(
                REMOTE_PROTOCOL_VERSION,
                JOB_ID,
                JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
            )
        )
        manifest, receipt = runner.collect_manifest(JOB_ID)
        assert manifest == declared
        assert receipt.result_count == 2
        assert receipt.result_bytes == sum(entry["size"] for entry in declared["entries"])
        assert receipt.result_manifest_hash == sha256(
            manifest_path.read_bytes()
        )

        (result_dir / "unexpected.txt").write_text("not declared", encoding="utf-8")
        assert_rejected(runner.collect_manifest, JOB_ID, code=FailureCode.COLLECTION_FAILED)

    with tempfile.TemporaryDirectory() as name:
        runner = verified_runner(Path(name))
        result_dir = runner._create_result_dir(JOB_ID)
        external = Path(name) / "outside-metrics.json"
        external.write_text("{}", encoding="utf-8")
        (result_dir / "metrics.json").symlink_to(external)
        assert_rejected(runner.declare_results, JOB_ID)


def test_runner_source_stays_stdlib_plus_remote_protocol_only():
    forbidden_prefixes = ("core", "gui", "PyQt", "QSettings", "ultralytics", "yaml", "sqlalchemy")
    for path in (APP_ROOT / "remote_runner").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for forbidden in forbidden_prefixes:
            assert not any(
                module == forbidden or module.startswith(forbidden + ".")
                for module in modules
            ), (path, forbidden, modules)


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
