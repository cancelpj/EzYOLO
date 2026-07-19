# -*- coding: utf-8 -*-
"""传输工具发现与 fake SshRsyncBackend 的离线测试。"""

import _bootstrap  # noqa: F401  保持测试路径与现有测试一致

from pathlib import Path
import subprocess
import sys
import tempfile

from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from core.remote_training.transport import (  # noqa: E402
    ClientTransportResolver,
    ClientTransportTools,
    ClientTransportUnavailable,
    HostTrustStore,
    RemoteCommandBuilder,
    RemoteRunnerReportedFailure,
    RemoteTransportError,
    SshRsyncBackend,
    null_device_for_platform,
)
from remote_protocol.v1 import (  # noqa: E402
    REMOTE_PROTOCOL_VERSION,
    FailureCode,
    JobStatus,
    RemoteStatus,
    RunnerFailureEnvelope,
    ResultReceipt,
)
from test_remote_training_profiles import _ed25519_key  # noqa: E402


PROFILE_ID = "7" * 32
JOB_ID = "8" * 32
SHA256 = "c" * 64


def make_profile():
    return RemoteTrainingProfile(
        name="实验室 A100",
        host="train-lab",
        port=22,
        username="trainer",
        remote_root="/srv/ezyolo/trainer",
        host_public_key=_ed25519_key(),
        id=PROFILE_ID,
    )


def make_backend(fake):
    root = Path(tempfile.mkdtemp(prefix="ezyolo-transport-"))
    tools = ClientTransportTools("/usr/bin/ssh", "/usr/bin/rsync", "/dev/null")
    commands = RemoteCommandBuilder(tools, HostTrustStore(root / "known-hosts"))

    def runner(argv, *, timeout_seconds):
        return fake(argv, timeout_seconds=timeout_seconds)

    return SshRsyncBackend(commands, run_process=runner), root


def assert_rejected(callable_object, *args, expected=RemoteTransportError, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except expected:
        return
    raise AssertionError("预期输入被拒绝")


def test_resolver_uses_only_path_and_reports_missing_tool_without_wsl_or_install():
    seen = []

    def finder(name):
        seen.append(name)
        return {"ssh": "/system/ssh"}.get(name)

    resolver = ClientTransportResolver(which=finder, platform_name="Windows")
    assert_rejected(resolver.resolve, expected=ClientTransportUnavailable)
    assert seen == ["ssh", "rsync", "rsync.exe"]
    assert null_device_for_platform("Windows") == "NUL"
    assert null_device_for_platform("Darwin") == "/dev/null"


def test_preflight_requires_exact_canonical_remote_root_and_protocol():
    def fake(argv, *, timeout_seconds):
        payload = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "canonical_remote_root": "/srv/ezyolo/trainer",
            "supported_tasks": ["detect", "segment"],
            "model_symbols": ["yolov10n"],
            "max_epochs": 300,
            "max_runtime_seconds": 3600,
            "max_payload_bytes": 1000,
            "max_result_bytes": 1000,
        }
        return subprocess.CompletedProcess(argv, 0, __import__("json").dumps(payload), "")

    backend, _root = make_backend(fake)
    capabilities = backend.preflight(make_profile())
    assert capabilities.canonical_remote_root == "/srv/ezyolo/trainer"

    def mismatch(argv, *, timeout_seconds):
        result = fake(argv, timeout_seconds=timeout_seconds)
        result.stdout = result.stdout.replace("/srv/ezyolo/trainer", "/other")
        return result

    backend, _root = make_backend(mismatch)
    assert_rejected(backend.preflight, make_profile())


def test_fake_backend_uses_json_protocol_and_no_real_ssh_or_rsync():
    calls = []

    def fake(argv, *, timeout_seconds):
        calls.append(tuple(argv))
        if argv[0] == "/usr/bin/rsync":
            return subprocess.CompletedProcess(argv, 0, "", "")
        action = next(
            value for value in argv if value in {"verify-upload", "start", "status", "cancel", "collect-manifest"}
        )
        if action == "collect-manifest":
            payload = {
                "protocol_version": REMOTE_PROTOCOL_VERSION,
                "job_id": JOB_ID,
                "result_manifest_hash": SHA256,
                "result_count": 2,
                "result_bytes": 42,
            }
        else:
            payload = {
                "protocol_version": REMOTE_PROTOCOL_VERSION,
                "job_id": JOB_ID,
                "status": JobStatus.RUNNING.value,
                "failure_code": None,
                "message": None,
                "epoch": 1,
            }
        return subprocess.CompletedProcess(argv, 0, __import__("json").dumps(payload), "")

    backend, root = make_backend(fake)
    profile = make_profile()
    assert backend.start(profile, JOB_ID).status == JobStatus.RUNNING
    assert backend.poll(profile, JOB_ID).epoch == 1
    assert backend.cancel(profile, JOB_ID).status == JobStatus.RUNNING
    receipt = backend.collect_manifest(profile, JOB_ID)
    assert isinstance(receipt, ResultReceipt)

    snapshot = root / "snapshot"
    staging = root / "staging"
    snapshot.mkdir()
    staging.mkdir()
    backend.upload(profile, JOB_ID, snapshot)
    backend.download_results(profile, JOB_ID, staging)
    assert calls
    assert any(call[0] == "/usr/bin/ssh" for call in calls)
    assert any(call[0] == "/usr/bin/rsync" for call in calls)


def test_process_failure_becomes_typed_error_without_returning_raw_stderr():
    def fake(argv, *, timeout_seconds):
        return subprocess.CompletedProcess(argv, 255, "", "password or private data")

    backend, _root = make_backend(fake)
    try:
        backend.start(make_profile(), JOB_ID)
    except RemoteTransportError as exc:
        assert "private data" not in str(exc)
    else:
        raise AssertionError("非零退出码必须被阻止")


def test_legacy_wrapped_runner_responses_are_rejected_fail_closed():
    def fake(argv, *, timeout_seconds):
        action = next(
            value for value in argv if value in {"cancel", "collect-manifest"}
        )
        if action == "cancel":
            payload = {
                "status": {
                    "protocol_version": REMOTE_PROTOCOL_VERSION,
                    "job_id": JOB_ID,
                    "status": JobStatus.CANCELLED.value,
                    "failure_code": None,
                    "message": None,
                    "epoch": None,
                },
                "cancelled": True,
            }
        else:
            payload = {
                "receipt": {
                    "protocol_version": REMOTE_PROTOCOL_VERSION,
                    "job_id": JOB_ID,
                    "result_manifest_hash": SHA256,
                    "result_count": 1,
                    "result_bytes": 1,
                }
            }
        return subprocess.CompletedProcess(argv, 0, __import__("json").dumps(payload), "")

    backend, _root = make_backend(fake)
    assert_rejected(backend.cancel, make_profile(), JOB_ID)
    assert_rejected(backend.collect_manifest, make_profile(), JOB_ID)


def test_runner_failure_envelope_is_typed_without_exposing_server_message_even_if_exit_code_is_zero():
    secret = "private-path-must-not-reach-ui"

    def fake(argv, *, timeout_seconds):
        envelope = RunnerFailureEnvelope(
            protocol_version=REMOTE_PROTOCOL_VERSION,
            ok=False,
            failure_code=FailureCode.PRECHECK_FAILED,
            message=secret,
        )
        return subprocess.CompletedProcess(argv, 0, __import__("json").dumps(envelope.to_wire()), "")

    backend, _root = make_backend(fake)
    try:
        backend.start(make_profile(), JOB_ID)
    except RemoteRunnerReportedFailure as exc:
        assert exc.code == FailureCode.PRECHECK_FAILED
        assert secret not in str(exc)
    else:
        raise AssertionError("runner 明确失败必须保留枚举错误码")


def test_ssh_control_uses_short_timeout_but_rsync_transfers_are_not_capped():
    calls = []

    def fake(argv, *, timeout_seconds):
        calls.append((tuple(argv), timeout_seconds))
        if argv[0] == "/usr/bin/rsync":
            return subprocess.CompletedProcess(argv, 0, "", "")
        payload = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "canonical_remote_root": "/srv/ezyolo/trainer",
            "supported_tasks": ["detect", "segment"],
            "model_symbols": ["yolov10n"],
            "max_epochs": 300,
            "max_runtime_seconds": 3600,
            "max_payload_bytes": 1000,
            "max_result_bytes": 1000,
        }
        return subprocess.CompletedProcess(argv, 0, __import__("json").dumps(payload), "")

    backend, root = make_backend(fake)
    profile = make_profile()
    backend.preflight(profile)
    snapshot = root / "snapshot"
    staging = root / "staging"
    snapshot.mkdir()
    staging.mkdir()
    backend.upload(profile, JOB_ID, snapshot)
    backend.download_results(profile, JOB_ID, staging)

    ssh_timeouts = [timeout for argv, timeout in calls if argv[0] == "/usr/bin/ssh"]
    rsync_timeouts = [timeout for argv, timeout in calls if argv[0] == "/usr/bin/rsync"]
    assert ssh_timeouts == [30]
    assert rsync_timeouts == [None, None]


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
