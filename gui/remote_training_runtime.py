"""桌面端远程训练运行时路径和系统传输 backend 的装配。

模块只把已经冻结的核心组件连接起来：它不执行网络、不创建目录，也不保存认证
秘密。真正的文件写入在用户确认后，由后台线程中的快照、known_hosts 和结果
staging 边界分别完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QStandardPaths

from core.remote_training.transport import (
    ClientTransportResolver,
    HostTrustStore,
    RemoteCommandBuilder,
    SshRsyncBackend,
)


class RemoteTrainingRuntimeError(ValueError):
    """本机没有安全可用的应用数据位置。"""


@dataclass(frozen=True)
class RemoteTrainingRuntimePaths:
    known_hosts_dir: Path
    snapshot_parent: Path
    result_staging_parent: Path
    runs_train_root: Path


def resolve_remote_training_runtime_paths(
    *,
    app_data_location: str,
    app_root: Path | str,
) -> RemoteTrainingRuntimePaths:
    """从已知目录推导受控路径；本函数不创建任何目录。"""
    if not isinstance(app_data_location, str) or not app_data_location.strip():
        raise RemoteTrainingRuntimeError("本机没有可用的应用数据目录，无法安全准备远程训练")
    state_root = Path(app_data_location).expanduser()
    repository_root = Path(app_root).expanduser()
    if not state_root.is_absolute() or not repository_root.is_absolute():
        raise RemoteTrainingRuntimeError("远程训练运行时目录必须是绝对路径")
    client_root = state_root / "remote-training-v1"
    runs_root = repository_root / "runs"
    return RemoteTrainingRuntimePaths(
        known_hosts_dir=client_root / "known-hosts",
        snapshot_parent=client_root / "snapshots",
        result_staging_parent=runs_root / ".remote-staging",
        runs_train_root=runs_root / "train",
    )


def current_remote_training_runtime_paths(app_root: Path | str) -> RemoteTrainingRuntimePaths:
    """读取 Qt 提供的用户级应用数据位置；空值 fail closed。"""
    return resolve_remote_training_runtime_paths(
        app_data_location=QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        ),
        app_root=app_root,
    )


def build_system_remote_backend(
    paths: RemoteTrainingRuntimePaths,
    *,
    resolver: ClientTransportResolver | None = None,
) -> SshRsyncBackend:
    """只发现系统 PATH 中已有的 OpenSSH / rsync，不安装也不连接。"""
    tools = (resolver or ClientTransportResolver()).resolve()
    return SshRsyncBackend(
        RemoteCommandBuilder(tools, HostTrustStore(paths.known_hosts_dir))
    )


__all__ = [
    "RemoteTrainingRuntimeError",
    "RemoteTrainingRuntimePaths",
    "build_system_remote_backend",
    "current_remote_training_runtime_paths",
    "resolve_remote_training_runtime_paths",
]
