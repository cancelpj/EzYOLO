"""v1 远程训练的 OpenSSH + rsync 传输边界。

这里所有本机命令均使用 argv 列表，并且只能运行系统 PATH 中已有的 ssh/rsync。
本模块不保存认证秘密、不读取用户 SSH 配置、不执行任意远程命令，也不会自动
创建服务器目录、安装软件或启动真实训练。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import platform as platform_module
import re
import secrets
import shlex
import shutil
import stat
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

from remote_protocol.v1 import (
    FailureCode,
    ProtocolValidationError,
    RemoteStatus,
    RunnerFailureEnvelope,
    ResultReceipt,
    ServerCapabilities,
    validate_job_id,
)

from .profiles import RemoteTrainingProfile


RUNNER_ENTRYPOINT = "ezyolo-remote-runner"
RUNNER_ACTIONS = frozenset(
    {"preflight", "verify-upload", "start", "status", "cancel", "collect-manifest"}
)
JOB_ACTIONS = RUNNER_ACTIONS - {"preflight"}
_HOST_ALIAS_PREFIX = "ezyolo-"
_REMOTE_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]+\Z")


class RemoteTransportError(RuntimeError):
    """本机传输边界失败；消息不回显密码、私钥或任意命令。"""

    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class RemoteRunnerReportedFailure(RemoteTransportError):
    """runner 已按共享协议报告的失败，而不是网络中断。"""


class ClientTransportUnavailable(RemoteTransportError):
    """当前操作系统 PATH 没有安全可用的系统 ssh 或 rsync。"""

    def __init__(self, message: str) -> None:
        super().__init__(FailureCode.CLIENT_TRANSPORT_UNAVAILABLE, message)


class ProcessRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float | None,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class ClientTransportTools:
    ssh: str
    rsync: str
    null_device: str


class ClientTransportResolver:
    """只查受控 PATH；不安装、不下载、不调用 WSL 或其它替代层。"""

    def __init__(
        self,
        *,
        which: Callable[[str], str | None] = shutil.which,
        platform_name: str | None = None,
    ) -> None:
        self._which = which
        self._platform_name = platform_name or platform_module.system()

    def resolve(self) -> ClientTransportTools:
        ssh = self._find("ssh")
        rsync = self._find("rsync")
        missing = [
            label
            for label, value in (("OpenSSH", ssh), ("rsync", rsync))
            if value is None
        ]
        if missing:
            names = "、".join(missing)
            raise ClientTransportUnavailable(
                f"这台电脑还不能远程训练：PATH 中缺少 {names}。"
                "请由用户或管理员先安装并配置系统工具；EzYOLO 不会自动安装或启动 WSL。"
            )
        return ClientTransportTools(
            ssh=ssh,
            rsync=rsync,
            null_device=null_device_for_platform(self._platform_name),
        )

    def _find(self, name: str) -> str | None:
        candidates = (name, f"{name}.exe") if self._platform_name == "Windows" else (name,)
        for candidate in candidates:
            value = self._which(candidate)
            if value:
                return value
        return None


def null_device_for_platform(platform_name: str | None = None) -> str:
    """便于跨平台测试；真实运行时与当前系统 os.devnull 一致。"""
    name = platform_name or platform_module.system()
    return "NUL" if name == "Windows" else os.devnull


def host_key_alias(profile: RemoteTrainingProfile) -> str:
    return f"{_HOST_ALIAS_PREFIX}{profile.id}"


class HostTrustStore:
    """profile 专属 known_hosts pin：每次连接前重写唯一条目，绝不 TOFU。"""

    def __init__(self, base_dir: Path | str) -> None:
        self._base_dir = Path(base_dir)

    def path_for(self, profile: RemoteTrainingProfile) -> Path:
        return self._base_dir / f"{profile.id}.known_hosts"

    def prepare(self, profile: RemoteTrainingProfile) -> Path:
        self._base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _chmod_private(self._base_dir, directory=True)
        destination = self.path_for(profile)
        content = f"{host_key_alias(profile)} {profile.host_public_key}\n"
        temporary = self._base_dir / (
            f".{profile.id}.{secrets.token_hex(8)}.known_hosts.tmp"
        )
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
                descriptor = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _chmod_private(destination, directory=False)
        except OSError as exc:
            raise RemoteTransportError(
                FailureCode.PRECHECK_FAILED,
                "无法准备远程服务器的主机密钥校验文件",
            ) from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return destination


class RemoteCommandBuilder:
    """把已校验 profile 变成精确、固定、可审计的 ssh/rsync argv。"""

    def __init__(
        self,
        tools: ClientTransportTools,
        trust_store: HostTrustStore,
        *,
        connect_timeout_seconds: int = 10,
        server_alive_interval_seconds: int = 15,
        server_alive_count_max: int = 3,
    ) -> None:
        if (
            type(connect_timeout_seconds) is not int
            or connect_timeout_seconds <= 0
            or type(server_alive_interval_seconds) is not int
            or server_alive_interval_seconds <= 0
            or type(server_alive_count_max) is not int
            or server_alive_count_max <= 0
        ):
            raise ValueError("SSH timeout 配置必须是正整数")
        self._tools = tools
        self._trust_store = trust_store
        self._connect_timeout_seconds = connect_timeout_seconds
        self._server_alive_interval_seconds = server_alive_interval_seconds
        self._server_alive_count_max = server_alive_count_max

    def ssh_argv(
        self,
        profile: RemoteTrainingProfile,
        action: str,
        job_id: str | None = None,
    ) -> list[str]:
        remote_command = self._runner_command(action, job_id)
        known_hosts = self._trust_store.prepare(profile)
        return [
            self._tools.ssh,
            "-F",
            self._tools.null_device,
            *self._ssh_options(profile, known_hosts),
            "-p",
            str(profile.port),
            "--",
            _ssh_destination(profile),
            *remote_command,
        ]

    def rsync_upload_argv(
        self,
        profile: RemoteTrainingProfile,
        *,
        job_id: str,
        snapshot_root: Path | str,
    ) -> list[str]:
        validate_job_id(job_id)
        source = _safe_local_directory(snapshot_root, "数据快照")
        known_hosts = self._trust_store.prepare(profile)
        remote_path = _remote_job_path(profile, "incoming", job_id)
        return [
            self._tools.rsync,
            "-a",
            "--protect-args",
            "--safe-links",
            "-e",
            self._rsync_remote_shell(profile, known_hosts),
            "--",
            _rsync_source_directory(source),
            _rsync_destination(profile, remote_path),
        ]

    def rsync_download_results_argv(
        self,
        profile: RemoteTrainingProfile,
        *,
        job_id: str,
        staging_dir: Path | str,
    ) -> list[str]:
        validate_job_id(job_id)
        destination = _safe_local_directory(staging_dir, "结果 staging")
        known_hosts = self._trust_store.prepare(profile)
        remote_path = _remote_job_path(profile, "results", job_id)
        return [
            self._tools.rsync,
            "-a",
            "--protect-args",
            "--safe-links",
            "-e",
            self._rsync_remote_shell(profile, known_hosts),
            "--",
            _rsync_destination(profile, remote_path) + "/",
            _rsync_source_directory(destination),
        ]

    def _runner_command(self, action: str, job_id: str | None) -> tuple[str, ...]:
        if action not in RUNNER_ACTIONS:
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "远程训练动作无效，已阻止发送",
            )
        if action in JOB_ACTIONS:
            if job_id is None:
                raise RemoteTransportError(
                    FailureCode.RUNNER_PROTOCOL,
                    "远程训练动作缺少任务标识，已阻止发送",
                )
            try:
                validate_job_id(job_id)
            except ProtocolValidationError as exc:
                raise RemoteTransportError(
                    FailureCode.RUNNER_PROTOCOL,
                    "远程任务标识无效，已阻止发送",
                ) from exc
            return (RUNNER_ENTRYPOINT, action, job_id)
        if job_id is not None:
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "预检动作不接受任务标识",
            )
        return (RUNNER_ENTRYPOINT, action)

    def _ssh_options(
        self,
        profile: RemoteTrainingProfile,
        known_hosts: Path,
    ) -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "ControlPersist=no",
            "-o",
            "ProxyCommand=none",
            "-o",
            "ProxyJump=none",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "LocalCommand=none",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "NumberOfPasswordPrompts=0",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            f"GlobalKnownHostsFile={self._tools.null_device}",
            "-o",
            f"HostKeyAlias={host_key_alias(profile)}",
            "-o",
            f"ConnectTimeout={self._connect_timeout_seconds}",
            "-o",
            f"ServerAliveInterval={self._server_alive_interval_seconds}",
            "-o",
            f"ServerAliveCountMax={self._server_alive_count_max}",
        ]

    def _rsync_remote_shell(
        self,
        profile: RemoteTrainingProfile,
        known_hosts: Path,
    ) -> str:
        argv = [
            self._tools.ssh,
            "-F",
            self._tools.null_device,
            *self._ssh_options(profile, known_hosts),
            "-p",
            str(profile.port),
        ]
        # rsync 的 -e 接受的是一个 shell 字符串。POSIX rsync 要 shlex 风格；
        # Windows 的原生 rsync.exe 则按 CreateProcess 规则解析，单引号不会被当成
        # 引号。用工具集携带的 NUL 标记走 Windows list2cmdline，避免要求 WSL。
        if self._tools.null_device == "NUL":
            return subprocess.list2cmdline(argv)
        return shlex.join(argv)


class RemoteTrainingBackend(ABC):
    """远程训练传输协议接口；Qt 页面和线程不直接拼接命令。"""

    @abstractmethod
    def preflight(self, profile: RemoteTrainingProfile) -> ServerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def upload(self, profile: RemoteTrainingProfile, job_id: str, snapshot_root: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def verify_upload(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        raise NotImplementedError

    @abstractmethod
    def start(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        raise NotImplementedError

    @abstractmethod
    def poll(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        raise NotImplementedError

    @abstractmethod
    def collect_manifest(self, profile: RemoteTrainingProfile, job_id: str) -> ResultReceipt:
        raise NotImplementedError

    @abstractmethod
    def download_results(
        self,
        profile: RemoteTrainingProfile,
        job_id: str,
        staging_dir: Path,
    ) -> None:
        raise NotImplementedError


class SshRsyncBackend(RemoteTrainingBackend):
    """使用系统 OpenSSH 与 rsync 的 v1 backend；依赖可注入 command runner。"""

    def __init__(
        self,
        command_builder: RemoteCommandBuilder,
        *,
        run_process: ProcessRunner | None = None,
    ) -> None:
        self._commands = command_builder
        self._run_process = run_process or _default_run_process

    def preflight(self, profile: RemoteTrainingProfile) -> ServerCapabilities:
        payload = self._run_runner_json(profile, "preflight")
        try:
            capabilities = ServerCapabilities.from_wire(payload)
        except ProtocolValidationError as exc:
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "服务器预检返回了不兼容的数据",
            ) from exc
        if capabilities.canonical_remote_root != profile.remote_root:
            raise RemoteTransportError(
                FailureCode.PRECHECK_FAILED,
                "服务器返回的训练目录与已保存档案不一致，已阻止继续",
            )
        return capabilities

    def upload(
        self,
        profile: RemoteTrainingProfile,
        job_id: str,
        snapshot_root: Path,
    ) -> None:
        self._run_rsync(
            self._commands.rsync_upload_argv(
                profile,
                job_id=job_id,
                snapshot_root=snapshot_root,
            ),
            failure_code=FailureCode.UPLOAD_INTEGRITY,
        )

    def verify_upload(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        return self._run_status(profile, "verify-upload", job_id)

    def start(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        return self._run_status(profile, "start", job_id)

    def poll(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        return self._run_status(profile, "status", job_id)

    def cancel(self, profile: RemoteTrainingProfile, job_id: str) -> RemoteStatus:
        return self._run_status(profile, "cancel", job_id)

    def collect_manifest(self, profile: RemoteTrainingProfile, job_id: str) -> ResultReceipt:
        payload = self._run_runner_json(profile, "collect-manifest", job_id)
        try:
            return ResultReceipt.from_wire(payload)
        except ProtocolValidationError as exc:
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "服务器结果清单不兼容，已阻止下载",
            ) from exc

    def download_results(
        self,
        profile: RemoteTrainingProfile,
        job_id: str,
        staging_dir: Path,
    ) -> None:
        self._run_rsync(
            self._commands.rsync_download_results_argv(
                profile,
                job_id=job_id,
                staging_dir=staging_dir,
            ),
            failure_code=FailureCode.COLLECTION_FAILED,
        )

    def _run_status(
        self,
        profile: RemoteTrainingProfile,
        action: str,
        job_id: str,
    ) -> RemoteStatus:
        payload = self._run_runner_json(profile, action, job_id)
        try:
            return RemoteStatus.from_wire(payload)
        except ProtocolValidationError as exc:
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "服务器任务状态不兼容，需重新连接核验",
            ) from exc

    def _run_runner_json(
        self,
        profile: RemoteTrainingProfile,
        action: str,
        job_id: str | None = None,
    ) -> Mapping[str, Any]:
        result = self._invoke(
            self._commands.ssh_argv(profile, action, job_id),
            failure_code=FailureCode.PRECHECK_FAILED if action == "preflight" else FailureCode.RUNNER_PROTOCOL,
            timeout_seconds=30,
        )
        payload = self._decode_runner_payload(result)
        try:
            envelope = RunnerFailureEnvelope.from_wire(payload)
        except ProtocolValidationError:
            envelope = None
        if envelope is not None:
            raise RemoteRunnerReportedFailure(
                envelope.failure_code,
                "服务器拒绝了本次远程训练请求，未将任务标记为完成或已停止",
            )
        if result.returncode != 0:
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "服务器拒绝请求时没有返回可验证的远程训练协议错误",
            )
        return payload

    def _run_rsync(self, argv: Sequence[str], *, failure_code: FailureCode) -> None:
        # 数据集和权重回传都可能远超 30 秒；SSH 的 ServerAlive 选项负责失联探测，
        # 这里不能把正常的大文件传输误判为超时。
        result = self._invoke(argv, failure_code=failure_code, timeout_seconds=None)
        if result.returncode != 0:
            raise RemoteTransportError(
                failure_code,
                "远程训练连接或传输未完成；请稍后重新连接核验任务状态",
            )

    def _invoke(
        self,
        argv: Sequence[str],
        *,
        failure_code: FailureCode,
        timeout_seconds: float | None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._run_process(tuple(argv), timeout_seconds=timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RemoteTransportError(
                failure_code,
                "远程训练连接或传输未完成；请稍后重新连接核验任务状态",
            ) from exc

    @staticmethod
    def _decode_runner_payload(result: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
        try:
            payload = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError, ProtocolValidationError):
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "服务器没有返回有效的远程训练协议数据",
            ) from None
        if not isinstance(payload, Mapping):
            raise RemoteTransportError(
                FailureCode.RUNNER_PROTOCOL,
                "服务器返回的远程训练协议数据格式不正确",
            )
        return payload


def _default_run_process(
    argv: Sequence[str],
    *,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        shell=False,
        check=False,
    )


def _safe_local_directory(value: Path | str, label: str) -> Path:
    path = Path(value)
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise RemoteTransportError(
            FailureCode.UPLOAD_INTEGRITY,
            f"{label}不存在，已阻止传输",
        ) from exc
    if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISDIR(stat_result.st_mode):
        raise RemoteTransportError(
            FailureCode.UPLOAD_INTEGRITY,
            f"{label}不是普通目录，已阻止传输",
        )
    return path.resolve(strict=True)


def _remote_job_path(
    profile: RemoteTrainingProfile,
    category: str,
    job_id: str,
) -> str:
    validate_job_id(job_id)
    if category not in {"incoming", "jobs", "results"}:
        raise RemoteTransportError(FailureCode.RUNNER_PROTOCOL, "远程目录类别无效")
    root = PurePosixPath(profile.remote_root)
    candidate = root / category / job_id
    _assert_remote_path_in_root(profile.remote_root, candidate)
    return candidate.as_posix()


def _assert_remote_path_in_root(root_value: str, candidate: PurePosixPath) -> None:
    root = PurePosixPath(root_value)
    if not root.is_absolute() or root == PurePosixPath("/"):
        raise RemoteTransportError(FailureCode.RUNNER_PROTOCOL, "远程根目录无效")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RemoteTransportError(FailureCode.RUNNER_PROTOCOL, "远程路径越界") from exc
    parts = relative.parts
    if not parts or any(
        part in {"", ".", ".."} or part.startswith("-") or not _REMOTE_SEGMENT_RE.fullmatch(part)
        for part in parts
    ):
        raise RemoteTransportError(FailureCode.RUNNER_PROTOCOL, "远程路径不安全")


def _ssh_destination(profile: RemoteTrainingProfile) -> str:
    return f"{profile.username}@{profile.host}"


def _rsync_destination(profile: RemoteTrainingProfile, remote_path: str) -> str:
    host = f"[{profile.host}]" if ":" in profile.host else profile.host
    return f"{profile.username}@{host}:{remote_path}"


def _rsync_source_directory(path: Path) -> str:
    # rsync 的「目录内容」语义由末尾 / 决定；固定用 /，避免 Windows 的末尾 \
    # 被 rsync -e 的命令行解析成转义字符。
    return str(path).rstrip("/\\") + "/"


def _chmod_private(path: Path, *, directory: bool) -> None:
    # Windows ACL 不等价于 POSIX mode；chmod 在其上是 best effort，而 macOS/Linux
    # 会得到要求的 0700 / 0600。无论哪种平台都不会写入用户 ~/.ssh。
    try:
        os.chmod(path, 0o700 if directory else 0o600)
    except OSError as exc:
        raise RemoteTransportError(
            FailureCode.PRECHECK_FAILED,
            "无法保护远程服务器主机密钥校验文件",
        ) from exc
