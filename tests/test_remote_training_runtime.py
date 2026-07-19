# -*- coding: utf-8 -*-
"""远程训练桌面运行时装配的纯离线测试。"""

import _bootstrap  # noqa: F401

from pathlib import Path
import sys
import tempfile

from core.remote_training.transport import ClientTransportTools, SshRsyncBackend  # noqa: E402
from gui.remote_training_runtime import (  # noqa: E402
    RemoteTrainingRuntimeError,
    build_system_remote_backend,
    resolve_remote_training_runtime_paths,
)


def _rejected(callable_object, *args, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except RemoteTrainingRuntimeError:
        return
    raise AssertionError("预期没有运行时目录时拒绝")


class _FakeResolver:
    def resolve(self):
        return ClientTransportTools("/usr/bin/ssh", "/usr/bin/rsync", "/dev/null")


def test_runtime_paths_are_private_and_result_staging_shares_runs_parent():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = resolve_remote_training_runtime_paths(
            app_data_location=str(root / "app-data"),
            app_root=root / "EzYOLO",
        )
        assert paths.known_hosts_dir == root / "app-data" / "remote-training-v1" / "known-hosts"
        assert paths.snapshot_parent == root / "app-data" / "remote-training-v1" / "snapshots"
        assert paths.result_staging_parent.parent == paths.runs_train_root.parent
        assert not paths.known_hosts_dir.exists()
        assert not paths.snapshot_parent.exists()


def test_runtime_rejects_empty_or_relative_location_and_only_builds_backend():
    _rejected(
        resolve_remote_training_runtime_paths,
        app_data_location="",
        app_root="/tmp/EzYOLO",
    )
    _rejected(
        resolve_remote_training_runtime_paths,
        app_data_location="relative",
        app_root="/tmp/EzYOLO",
    )

    paths = resolve_remote_training_runtime_paths(
        app_data_location="/tmp/ezyolo-state",
        app_root="/tmp/EzYOLO",
    )
    backend = build_system_remote_backend(paths, resolver=_FakeResolver())
    assert isinstance(backend, SshRsyncBackend)
    assert not paths.known_hosts_dir.exists()


if __name__ == "__main__":
    raise SystemExit(_bootstrap.run_module_tests(globals()))
