"""受限远程训练 job 的文件、状态、锁与结果边界。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import signal
import stat
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

from remote_protocol.v1 import (
    REMOTE_PROTOCOL_VERSION,
    DatasetManifest,
    FailureCode,
    JobSpec,
    JobStatus,
    ManifestEntry,
    ProtocolValidationError,
    RemoteStatus,
    ResultReceipt,
    validate_job_id,
)

from .config import ServerConfig


JOB_SPEC_FILENAME = "job-spec.json"
DATASET_MANIFEST_FILENAME = "manifest.json"
STATUS_FILENAME = "status.json"
RUNNER_DATA_FILENAME = "data.yaml"
RESULT_MANIFEST_FILENAME = "manifest.json"
LOCK_FILENAME = ".single-task.lock"
START_GATE_FILENAME = ".start.gate"
MAX_METADATA_BYTES = 1024 * 1024

_RESULT_MANIFEST_FIELDS = frozenset(
    {"protocol_version", "job_id", "entries", "total_bytes"}
)
_LOCK_FIELDS = frozenset({"job_id", "pid", "pgid", "boot_id", "start_marker"})
_FORBIDDEN_PAYLOAD_SUFFIXES = frozenset(
    {".pt", ".pth", ".pkl", ".py", ".sh", ".bat", ".command"}
)
_RESERVED_UPLOAD_PATHS = frozenset(
    {JOB_SPEC_FILENAME, DATASET_MANIFEST_FILENAME, STATUS_FILENAME, RUNNER_DATA_FILENAME}
)
_CONTROLLED_RESULT_PATHS = frozenset(
    {
        "metrics.json",
        "results.csv",
        "weights/best.pt",
        "weights/last.pt",
    }
)


class RunnerFailure(RuntimeError):
    """可以安全返回给桌面端的 runner 失败。"""

    def __init__(self, code: FailureCode, message: str):
        super().__init__(message)
        self.code = FailureCode(code)
        self.message = message


@dataclass(frozen=True)
class ProcessIdentity:
    job_id: str
    pid: int
    pgid: int
    boot_id: str
    start_marker: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pid": self.pid,
            "pgid": self.pgid,
            "boot_id": self.boot_id,
            "start_marker": self.start_marker,
        }


class ProcessInspector:
    """Linux ``/proc`` 与 POSIX 信号的最小身份检查器。

    ``None`` 表示无法证明，不会被当成进程死亡或可安全回收。
    """

    def boot_id(self) -> str | None:
        try:
            value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        except OSError:
            return None
        return value or None

    def is_alive(self, pid: int) -> bool | None:
        if type(pid) is not int or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
        except OSError:
            return None
        state = self._linux_process_state(pid)
        return False if state == "Z" else True

    def start_marker(self, pid: int) -> str | None:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            _before, after = raw.rsplit(")", 1)
            fields = after.split()
        except (OSError, ValueError):
            return None
        # ``stat`` 的 starttime 是第 22 列；after 从第 3 列 state 开始。
        return fields[19] if len(fields) > 19 and fields[19].isdigit() else None

    def pgid(self, pid: int) -> int | None:
        try:
            value = os.getpgid(pid)
        except OSError:
            return None
        return value if value > 0 else None

    def group_alive(self, pgid: int) -> bool | None:
        if type(pgid) is not int or pgid <= 0:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
        except OSError:
            return None
        return True

    def identity_for(self, job_id: str, pid: int) -> ProcessIdentity | None:
        boot_id = self.boot_id()
        if self.is_alive(pid) is not True or boot_id is None:
            return None
        pgid = self.pgid(pid)
        marker = self.start_marker(pid)
        if pgid is None or marker is None:
            return None
        return ProcessIdentity(job_id, pid, pgid, boot_id, marker)

    @staticmethod
    def _linux_process_state(pid: int) -> str | None:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            _before, after = raw.rsplit(")", 1)
            fields = after.split()
        except (OSError, ValueError):
            return None
        return fields[0] if fields else None


class RunnerPaths:
    """canonical remote root 下唯一允许的 incoming/jobs/results 派生路径。"""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def incoming_root(self) -> Path:
        return self._under_root("incoming")

    @property
    def jobs_root(self) -> Path:
        return self._under_root("jobs")

    @property
    def results_root(self) -> Path:
        return self._under_root("results")

    def incoming_dir(self, job_id: str) -> Path:
        return self._under_root("incoming", validate_job_id(job_id))

    def job_dir(self, job_id: str) -> Path:
        return self._under_root("jobs", validate_job_id(job_id))

    def result_dir(self, job_id: str) -> Path:
        return self._under_root("results", validate_job_id(job_id))

    def assert_layout_ready(self) -> None:
        """只读确认上传所需的目录已经由管理员预置。

        桌面端的 preflight 承诺「失败时不上传」，因此这里不能顺手 mkdir；否则首次
        点击测试连接会在服务器留下状态。缺少目录时明确拒绝，让管理员先完成部署。
        """
        self._assert_root()
        for path in (self.incoming_root, self.jobs_root, self.results_root):
            _assert_owned_private_directory(path, "runner 预置目录")

    def _assert_root(self) -> None:
        _assert_owned_private_directory(self.root, "canonical remote root")
        try:
            resolved = self.root.resolve(strict=True)
        except OSError as exc:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "remote root 不可解析") from exc
        if resolved != self.root:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "remote root 不是 canonical path")

    def _under_root(self, *parts: str) -> Path:
        self._assert_root()
        path = self.root.joinpath(*parts)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "派生路径越出 remote root") from exc
        # 只接受本模块构造的路径，避免已存在软链接把后续写入带出根目录。
        current = self.root
        for part in parts:
            current = current / part
            if current.exists() or current.is_symlink():
                _assert_not_symlink(current, "派生路径")
        return path


class SingleTaskLock:
    """单任务锁；只有被事实证明 stale 的记录才允许回收。"""

    def __init__(self, paths: RunnerPaths, inspector: ProcessInspector):
        self.paths = paths
        self.inspector = inspector

    @property
    def path(self) -> Path:
        return self.paths.jobs_root / LOCK_FILENAME

    @contextmanager
    def start_gate(self) -> Iterator[None]:
        """串行化 Popen 与锁原子落地，避免两个 start 同时越过 admission。"""

        self.paths.assert_layout_ready()
        gate = self.paths.jobs_root / START_GATE_FILENAME
        _assert_managed_file_or_missing(gate, "start gate")
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Linux 部署前置条件
            raise RunnerFailure(FailureCode.ENVIRONMENT_MISMATCH, "runner 需要 POSIX 文件锁") from exc
        with gate.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def assert_available(self) -> None:
        record = self.read()
        if record is None:
            return
        if self.is_stale(record):
            self._unlink_exact(record)
            return
        raise RunnerFailure(FailureCode.SERVER_BUSY, "已有任务持有 single-task 锁")

    def acquire(self, identity: ProcessIdentity) -> None:
        validate_job_id(identity.job_id)
        self.paths.assert_layout_ready()
        _assert_managed_file_or_missing(self.path, "single-task lock")
        payload = _canonical_json(identity.to_wire())
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as exc:
            raise RunnerFailure(FailureCode.SERVER_BUSY, "已有任务持有 single-task 锁") from exc
        try:
            _write_all_and_sync(descriptor, payload)
        finally:
            os.close(descriptor)

    def read(self) -> ProcessIdentity | None:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "无法检查 single-task 锁") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "single-task 锁不是普通文件")
        try:
            payload = _read_json_object(self.path, "single-task 锁")
            data = _require_exact_fields(payload, _LOCK_FIELDS, "single-task 锁")
            identity = ProcessIdentity(
                job_id=validate_job_id(data["job_id"]),
                pid=_positive_int(data["pid"], "pid"),
                pgid=_positive_int(data["pgid"], "pgid"),
                boot_id=_plain_marker(data["boot_id"], "boot_id"),
                start_marker=_plain_marker(data["start_marker"], "start_marker"),
            )
        except (ProtocolValidationError, ValueError, TypeError) as exc:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "single-task 锁内容不可信") from exc
        return identity

    def is_stale(self, record: ProcessIdentity) -> bool:
        boot_id = self.inspector.boot_id()
        if boot_id is not None and boot_id != record.boot_id:
            return True
        alive = self.inspector.is_alive(record.pid)
        if alive is False:
            return True
        if alive is not True:
            return False
        marker = self.inspector.start_marker(record.pid)
        return marker is not None and marker != record.start_marker

    def verification(self, record: ProcessIdentity) -> str:
        """返回 valid/dead/mismatch/group_dead/unknown，不用未知信息猜测死亡。"""

        boot_id = self.inspector.boot_id()
        if boot_id is None:
            return "unknown"
        if boot_id != record.boot_id:
            return "dead"
        alive = self.inspector.is_alive(record.pid)
        if alive is False:
            return "dead"
        if alive is not True:
            return "unknown"
        marker = self.inspector.start_marker(record.pid)
        if marker is None:
            return "unknown"
        if marker != record.start_marker:
            return "mismatch"
        pgid = self.inspector.pgid(record.pid)
        if pgid is None:
            return "unknown"
        if pgid != record.pgid:
            return "mismatch"
        group_alive = self.inspector.group_alive(record.pgid)
        if group_alive is False:
            return "group_dead"
        return "valid" if group_alive is True else "unknown"

    def release_if_matches(self, identity: ProcessIdentity) -> bool:
        record = self.read()
        if record != identity:
            return False
        self._unlink_exact(record)
        return True

    def _unlink_exact(self, expected: ProcessIdentity) -> None:
        current = self.read()
        if current != expected:
            raise RunnerFailure(FailureCode.SERVER_BUSY, "single-task 锁在检查期间变化")
        try:
            self.path.unlink()
        except OSError as exc:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "无法回收 single-task 锁") from exc


class Runner:
    """固定 action 使用的服务端事实层。"""

    def __init__(
        self,
        config: ServerConfig,
        *,
        inspector: ProcessInspector | None = None,
        disk_usage: Callable[[str | Path], Any] | None = None,
        process_group_killer: Callable[[int, int], None] | None = None,
        supervisor_spawner: Callable[..., Any] | None = None,
    ):
        if config.protocol_version != REMOTE_PROTOCOL_VERSION:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "runner 与配置协议版本不匹配")
        self.config = config
        self.paths = RunnerPaths(config.canonical_remote_root)
        self.inspector = inspector or ProcessInspector()
        self.lock = SingleTaskLock(self.paths, self.inspector)
        self._disk_usage = disk_usage or __import__("shutil").disk_usage
        self._killpg = process_group_killer or os.killpg
        self._spawn_supervisor = supervisor_spawner or subprocess.Popen

    def preflight(self):
        """验证本机 runner 的轻量前置条件，不创建环境、不安装任何内容。"""

        self.paths.assert_layout_ready()
        _assert_safe_regular_file(self.config.runtime.launcher, "runtime launcher")
        for path in self.config.model_allowlist.values():
            _assert_safe_regular_file(path, "allowlist 权重")
        if sys.version_info < (3, 10):
            raise RunnerFailure(FailureCode.ENVIRONMENT_MISMATCH, "runner 需要 Python 3.10+")
        if self.inspector.identity_for("0" * 32, os.getpid()) is None:
            raise RunnerFailure(FailureCode.ENVIRONMENT_MISMATCH, "runner 需要可用的 Linux /proc 身份信息")
        if self._free_disk_bytes() < self.config.min_free_disk_bytes:
            raise RunnerFailure(FailureCode.LOW_DISK_SPACE, "可用磁盘低于服务器下限")
        return self.config.capabilities()

    def verify_upload(self, job_id: str) -> RemoteStatus:
        job_id = validate_job_id(job_id)
        self.paths.assert_layout_ready()
        incoming = self.paths.incoming_dir(job_id)
        destination = self.paths.job_dir(job_id)
        if destination.exists() or destination.is_symlink():
            raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "目标 job 已存在，拒绝覆盖")
        _assert_directory(incoming, "incoming job")
        spec, manifest = self._load_upload(incoming, job_id)
        self._validate_upload_tree(incoming, manifest)
        if manifest.total_bytes > self.config.max_payload_bytes:
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "上传数据超过 max_payload_bytes")
        if spec.model_symbol not in self.config.model_allowlist:
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "模型符号不在服务器 allowlist")
        try:
            os.rename(incoming, destination)
        except FileExistsError as exc:
            raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "目标 job 已存在，拒绝覆盖") from exc
        except OSError as exc:
            raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "incoming job 无法原子进入 jobs") from exc
        status = RemoteStatus(REMOTE_PROTOCOL_VERSION, job_id, JobStatus.VERIFYING_UPLOAD)
        self.write_status(status)
        return status

    def start(self, job_id: str) -> RemoteStatus:
        job_id = validate_job_id(job_id)
        try:
            status = self.read_status(job_id)
            if status.status in {JobStatus.RUNNING, JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION}:
                return status
            if status.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
                return status
            if status.status != JobStatus.VERIFYING_UPLOAD:
                raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "job 未处于已验证可启动状态")

            spec, manifest = self._load_verified_job(job_id)
            self._check_start_limits(spec, manifest, job_id)
            with self.lock.start_gate():
                self.lock.assert_available()
                self._create_result_dir(job_id)
                self.build_data_yaml(job_id, manifest)
                process = self._spawn_supervisor(
                    self._supervisor_command(job_id),
                    cwd=str(self.paths.job_dir(job_id)),
                    start_new_session=True,
                    close_fds=True,
                    env=controlled_environment(),
                )
                identity = self.inspector.identity_for(job_id, int(process.pid))
                if identity is None or identity.pgid != identity.pid:
                    raise RunnerFailure(FailureCode.REMOTE_CRASHED, "新启动进程没有可核验的独立进程组")
                self.lock.acquire(identity)
                running = RemoteStatus(REMOTE_PROTOCOL_VERSION, job_id, JobStatus.RUNNING)
                self.write_status(running)
                return running
        except RunnerFailure as exc:
            return self._write_failed(job_id, exc.code, exc.message)

    def status(self, job_id: str) -> RemoteStatus:
        job_id = validate_job_id(job_id)
        status = self.read_status(job_id)
        if status.status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            return status
        record = self.lock.read()
        if record is None or record.job_id != job_id:
            return self._write_failed(
                job_id, FailureCode.REMOTE_CRASHED, "活动 job 没有匹配的进程锁"
            )
        check = self.lock.verification(record)
        if status.status == JobStatus.CANCEL_REQUESTED:
            if check == "valid":
                return status
            if check in {"dead", "group_dead"}:
                cancelled = RemoteStatus(REMOTE_PROTOCOL_VERSION, job_id, JobStatus.CANCELLED)
                self.write_status(cancelled)
                self.lock.release_if_matches(record)
                return cancelled
            if check == "mismatch":
                return self._write_failed(
                    job_id,
                    FailureCode.REMOTE_CRASHED,
                    "停止请求后 job 进程身份不再可信",
                )
            # 无法证明进程已退出时，仍保持 CANCEL_REQUESTED；客户端会继续轮询，
            # 不能把未知状态误写成已停止。
            return status
        if check in {"dead", "mismatch", "group_dead"}:
            return self._write_failed(
                job_id, FailureCode.REMOTE_CRASHED, "RUNNING job 的进程身份或进程组已消失"
            )
        return status

    def cancel(self, job_id: str) -> tuple[RemoteStatus | None, bool]:
        """只取消已核验、属于本 job 的进程组；其余情形安全 no-op。"""

        job_id = validate_job_id(job_id)
        try:
            status = self.read_status(job_id)
        except RunnerFailure:
            return None, False
        if status.status in {
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
        }:
            return status, False
        record = self.lock.read()
        if record is None or record.job_id != job_id or self.lock.verification(record) != "valid":
            return status, False
        requested = RemoteStatus(REMOTE_PROTOCOL_VERSION, job_id, JobStatus.CANCEL_REQUESTED)
        self.write_status(requested)
        self._killpg(record.pgid, signal.SIGTERM)
        if self.inspector.is_alive(record.pid) is False:
            cancelled = RemoteStatus(REMOTE_PROTOCOL_VERSION, job_id, JobStatus.CANCELLED)
            self.write_status(cancelled)
            self.lock.release_if_matches(record)
            return cancelled, True
        return requested, True

    def collect_manifest(self, job_id: str) -> tuple[dict[str, Any], ResultReceipt]:
        job_id = validate_job_id(job_id)
        status = self.read_status(job_id)
        if status.status != JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION:
            raise RunnerFailure(FailureCode.COLLECTION_FAILED, "结果尚未由 runner 声明完成")
        manifest_path = self.paths.result_dir(job_id) / RESULT_MANIFEST_FILENAME
        payload = _read_json_object(manifest_path, "result manifest")
        entries, total_bytes = self._validate_result_manifest(job_id, payload)
        raw = manifest_path.read_bytes()
        receipt = ResultReceipt(
            protocol_version=REMOTE_PROTOCOL_VERSION,
            job_id=job_id,
            result_manifest_hash=_sha256_bytes(raw),
            result_count=len(entries),
            result_bytes=total_bytes,
        )
        return payload, receipt

    def declare_results(self, job_id: str) -> dict[str, Any]:
        """训练 supervisor 退出成功后，以受控 allowlist 声明结果清单。"""

        job_id = validate_job_id(job_id)
        result_dir = self.paths.result_dir(job_id)
        _assert_directory(result_dir, "result job")
        dirs, files = _walk_regular_tree(result_dir)
        if RESULT_MANIFEST_FILENAME in files:
            raise RunnerFailure(FailureCode.COLLECTION_FAILED, "结果清单已存在，拒绝覆盖")
        expected_dirs = {"weights"} if any(path.startswith("weights/") for path in files) else set()
        if dirs != expected_dirs or not files.issubset(_CONTROLLED_RESULT_PATHS):
            raise RunnerFailure(FailureCode.COLLECTION_FAILED, "训练结果包含未声明的文件或目录")
        entries: list[ManifestEntry] = []
        for relative in sorted(files):
            path = result_dir / PurePosixPath(relative)
            info = _regular_file_info(path, "结果文件")
            entries.append(
                ManifestEntry(path=relative, size=info.st_size, sha256=_sha256_file(path))
            )
        total_bytes = sum(entry.size for entry in entries)
        if total_bytes > self.config.max_results_bytes:
            raise RunnerFailure(FailureCode.JOB_DISK_LIMIT, "结果大小超过 max_results_bytes")
        payload = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "job_id": job_id,
            "entries": [entry.to_wire() for entry in entries],
            "total_bytes": total_bytes,
        }
        _atomic_write(result_dir / RESULT_MANIFEST_FILENAME, _canonical_json(payload))
        return payload

    def build_data_yaml(self, job_id: str, manifest: DatasetManifest | None = None) -> Path:
        """只由受控 layout 和 class_names 生成 JSON-compatible YAML。"""

        job_id = validate_job_id(job_id)
        if manifest is None:
            _spec, manifest = self._load_verified_job(job_id)
        layout = dict(manifest.layout)
        lines: list[str] = []
        for split in ("train", "val", "test"):
            if split in layout:
                lines.append(f"{split}: {json.dumps(layout[split], ensure_ascii=False)}")
        lines.append("names:")
        for index, name in enumerate(manifest.class_names):
            lines.append(f"  {index}: {json.dumps(name, ensure_ascii=False)}")
        content = ("\n".join(lines) + "\n").encode("utf-8")
        target = self.paths.job_dir(job_id) / RUNNER_DATA_FILENAME
        _atomic_write(target, content)
        return target

    def job_bytes(self, job_id: str) -> int:
        _dirs, files = _walk_regular_tree(self.paths.job_dir(job_id))
        return sum(_regular_file_info(self.paths.job_dir(job_id) / PurePosixPath(path), "job 文件").st_size for path in files)

    def result_bytes(self, job_id: str) -> int:
        result_dir = self.paths.result_dir(job_id)
        if not result_dir.exists():
            return 0
        _dirs, files = _walk_regular_tree(result_dir)
        return sum(_regular_file_info(result_dir / PurePosixPath(path), "结果文件").st_size for path in files)

    def watchdog_failure(self, job_id: str, elapsed_seconds: float) -> FailureCode | None:
        if elapsed_seconds > self.config.max_runtime_seconds:
            return FailureCode.MAX_RUNTIME
        if self.job_bytes(job_id) > self.config.max_jobs_bytes:
            return FailureCode.JOB_DISK_LIMIT
        if self.result_bytes(job_id) > self.config.max_results_bytes:
            return FailureCode.JOB_DISK_LIMIT
        if self._free_disk_bytes() < self.config.min_free_disk_bytes:
            return FailureCode.LOW_DISK_SPACE
        return None

    def stop_verified_job(self, job_id: str, failure_code: FailureCode, message: str) -> bool:
        """写出失败状态后，仅停止当前 job 已核验的进程组。"""

        job_id = validate_job_id(job_id)
        record = self.lock.read()
        if record is None or record.job_id != job_id or self.lock.verification(record) != "valid":
            return False
        self._write_failed(job_id, failure_code, message)
        self._killpg(record.pgid, signal.SIGTERM)
        return True

    def read_status(self, job_id: str) -> RemoteStatus:
        path = self.paths.job_dir(validate_job_id(job_id)) / STATUS_FILENAME
        try:
            return RemoteStatus.from_wire(_read_json_object(path, "status"))
        except ProtocolValidationError as exc:
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "status 与共享协议不匹配") from exc

    def write_status(self, status: RemoteStatus) -> None:
        job_id = validate_job_id(status.job_id)
        target = self.paths.job_dir(job_id) / STATUS_FILENAME
        _assert_directory(target.parent, "job 目录")
        _atomic_write(target, _canonical_json(status.to_wire()))

    def _load_upload(self, incoming: Path, job_id: str) -> tuple[JobSpec, DatasetManifest]:
        return self._load_job_metadata(incoming, job_id, FailureCode.UPLOAD_INTEGRITY)

    def _load_verified_job(self, job_id: str) -> tuple[JobSpec, DatasetManifest]:
        return self._load_job_metadata(
            self.paths.job_dir(job_id), job_id, FailureCode.RUNNER_PROTOCOL
        )

    @staticmethod
    def _load_job_metadata(
        directory: Path, job_id: str, failure_code: FailureCode
    ) -> tuple[JobSpec, DatasetManifest]:
        try:
            spec_wire = _read_json_object(directory / JOB_SPEC_FILENAME, "job spec")
            manifest_wire = _read_json_object(
                directory / DATASET_MANIFEST_FILENAME, "dataset manifest"
            )
        except RunnerFailure as exc:
            raise RunnerFailure(failure_code, "上传的 job/manifest 与协议不匹配") from exc
        if (
            spec_wire.get("protocol_version") != REMOTE_PROTOCOL_VERSION
            or manifest_wire.get("protocol_version") != REMOTE_PROTOCOL_VERSION
        ):
            raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "上传协议版本不匹配")
        try:
            spec = JobSpec.from_wire(spec_wire)
            manifest = DatasetManifest.from_wire(manifest_wire)
        except ProtocolValidationError as exc:
            raise RunnerFailure(failure_code, "上传的 job/manifest 与协议不匹配") from exc
        if spec.job_id != job_id or manifest.job_id != job_id:
            raise RunnerFailure(failure_code, "job id 与上传目录不匹配")
        if spec.task_type != manifest.task_type:
            raise RunnerFailure(failure_code, "job 与 manifest task_type 不匹配")
        if spec.class_names != manifest.class_names:
            raise RunnerFailure(failure_code, "job 与 manifest class_names 不匹配")
        if spec.snapshot_hash != manifest.snapshot_hash:
            raise RunnerFailure(failure_code, "job 与 manifest snapshot_hash 不匹配")
        return spec, manifest

    def _validate_upload_tree(self, incoming: Path, manifest: DatasetManifest) -> None:
        dirs, files = _walk_regular_tree(incoming)
        expected_files = {JOB_SPEC_FILENAME, DATASET_MANIFEST_FILENAME}
        expected_dirs: set[str] = set()
        for entry in manifest.entries:
            relative = PurePosixPath(entry.path)
            self._validate_payload_entry_path(relative)
            expected_files.add(relative.as_posix())
            parent = relative.parent
            while str(parent) != ".":
                expected_dirs.add(parent.as_posix())
                parent = parent.parent
        if files != expected_files or dirs != expected_dirs:
            raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "上传目录包含缺失或 manifest 外文件")
        for entry in manifest.entries:
            file_path = incoming / PurePosixPath(entry.path)
            info = _regular_file_info(file_path, "payload 文件")
            if info.st_size != entry.size or _sha256_file(file_path) != entry.sha256:
                raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "payload 文件大小或 hash 不匹配")

    @staticmethod
    def _validate_payload_entry_path(path: PurePosixPath) -> None:
        if (
            path.as_posix() in _RESERVED_UPLOAD_PATHS
            or path.name.lower() == "data.yaml"
            or path.suffix.lower() in _FORBIDDEN_PAYLOAD_SUFFIXES
        ):
            raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "payload 含有不允许的客户端文件")
        if path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in path.parts):
            raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "payload 路径不安全")

    def _check_start_limits(
        self, spec: JobSpec, manifest: DatasetManifest, job_id: str
    ) -> None:
        if spec.model_symbol not in self.config.model_allowlist:
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "模型符号不在服务器 allowlist")
        if spec.epochs > self.config.max_epochs:
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "epochs 超过服务器上限")
        if manifest.total_bytes > self.config.max_payload_bytes:
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "payload 超过服务器上限")
        if self.job_bytes(job_id) > self.config.max_jobs_bytes:
            raise RunnerFailure(FailureCode.JOB_DISK_LIMIT, "job 目录超过 max_jobs_bytes")
        if self._free_disk_bytes() < self.config.min_free_disk_bytes:
            raise RunnerFailure(FailureCode.LOW_DISK_SPACE, "可用磁盘低于服务器下限")

    def _create_result_dir(self, job_id: str) -> Path:
        result_dir = self.paths.result_dir(job_id)
        if result_dir.exists() or result_dir.is_symlink():
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "结果目录已存在，拒绝覆盖")
        try:
            result_dir.mkdir(mode=0o700)
        except OSError as exc:
            raise RunnerFailure(FailureCode.PRECHECK_FAILED, "无法创建受控结果目录") from exc
        _assert_directory(result_dir, "结果目录")
        return result_dir

    def _supervisor_command(self, job_id: str) -> Sequence[str]:
        return [sys.executable, "-m", "remote_runner.trainer", validate_job_id(job_id)]

    def _free_disk_bytes(self) -> int:
        try:
            value = self._disk_usage(self.config.canonical_remote_root).free
        except OSError as exc:
            raise RunnerFailure(FailureCode.LOW_DISK_SPACE, "无法读取 remote root 可用磁盘") from exc
        if type(value) is not int or value < 0:
            raise RunnerFailure(FailureCode.LOW_DISK_SPACE, "可用磁盘数据不可信")
        return value

    def _write_failed(
        self, job_id: str, code: FailureCode, message: str
    ) -> RemoteStatus:
        failed = RemoteStatus(
            REMOTE_PROTOCOL_VERSION,
            validate_job_id(job_id),
            JobStatus.FAILED,
            FailureCode(code),
            message[:500],
        )
        self.write_status(failed)
        return failed

    def _validate_result_manifest(
        self, job_id: str, payload: Mapping[str, Any]
    ) -> tuple[tuple[ManifestEntry, ...], int]:
        try:
            data = _require_exact_fields(payload, _RESULT_MANIFEST_FIELDS, "result manifest")
            if data["protocol_version"] != REMOTE_PROTOCOL_VERSION or data["job_id"] != job_id:
                raise ValueError("result manifest 协议或 job id 不匹配")
            if not isinstance(data["entries"], list):
                raise ValueError("result manifest entries 不合法")
            entries = tuple(ManifestEntry.from_wire(item) for item in data["entries"])
            if len({entry.path for entry in entries}) != len(entries):
                raise ValueError("result manifest 有重复路径")
            total = _nonnegative_int(data["total_bytes"], "result total_bytes")
            if total != sum(entry.size for entry in entries):
                raise ValueError("result manifest 大小不匹配")
        except (ProtocolValidationError, ValueError, TypeError) as exc:
            raise RunnerFailure(FailureCode.COLLECTION_FAILED, "result manifest 不可信") from exc
        if any(entry.path not in _CONTROLLED_RESULT_PATHS for entry in entries):
            raise RunnerFailure(FailureCode.COLLECTION_FAILED, "result manifest 暴露了未受控结果")
        if total > self.config.max_results_bytes:
            raise RunnerFailure(FailureCode.JOB_DISK_LIMIT, "结果大小超过 max_results_bytes")
        result_dir = self.paths.result_dir(job_id)
        dirs, files = _walk_regular_tree(result_dir)
        expected_dirs = {"weights"} if any(path.startswith("weights/") for path in files if path != RESULT_MANIFEST_FILENAME) else set()
        expected_files = {RESULT_MANIFEST_FILENAME, *(entry.path for entry in entries)}
        if dirs != expected_dirs or files != expected_files:
            raise RunnerFailure(FailureCode.COLLECTION_FAILED, "结果目录偏离受控清单")
        for entry in entries:
            path = result_dir / PurePosixPath(entry.path)
            info = _regular_file_info(path, "结果文件")
            if info.st_size != entry.size or _sha256_file(path) != entry.sha256:
                raise RunnerFailure(FailureCode.COLLECTION_FAILED, "结果文件大小或 hash 不匹配")
        return entries, total


def _assert_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 必须是普通目录")


def _assert_owned_private_directory(path: Path, label: str) -> None:
    _assert_directory(path, label)
    try:
        info = path.lstat()
    except OSError as exc:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 无法检查") from exc
    if info.st_uid != os.getuid():
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 必须属于 runner 普通账号")
    if info.st_mode & 0o022:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 不能允许组或其他账号写入")


def _assert_regular_file(path: Path, label: str) -> None:
    _regular_file_info(path, label)


def _assert_safe_regular_file(path: Path, label: str) -> None:
    info = _regular_file_info(path, label)
    if info.st_mode & 0o022:
        raise RunnerFailure(
            FailureCode.RUNNER_PROTOCOL,
            f"{label} 不能允许组或其他账号写入",
        )


def _regular_file_info(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, f"{label} 不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, f"{label} 必须是普通文件")
    return info


def _assert_not_symlink(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 无法检查") from exc
    if stat.S_ISLNK(info.st_mode):
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 不能是软链接")


def _assert_managed_file_or_missing(path: Path, label: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        info = path.lstat()
    except OSError as exc:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 无法检查") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, f"{label} 不是普通文件")


def _walk_regular_tree(root: Path) -> tuple[set[str], set[str]]:
    _assert_directory(root, "受控目录")
    dirs: set[str] = set()
    files: set[str] = set()
    for raw_current, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(raw_current)
        for name in list(dirnames):
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "目录树含有软链接或异常目录")
            dirs.add(path.relative_to(root).as_posix())
        for name in filenames:
            path = current / name
            _regular_file_info(path, "目录树文件")
            files.add(path.relative_to(root).as_posix())
    return dirs, files


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    _regular_file_info(path, label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, f"无法读取 {label}") from exc
    if len(raw) > MAX_METADATA_BYTES:
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, f"{label} 过大")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, f"{label} 不是安全 JSON 对象") from exc
    if not isinstance(value, Mapping):
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, f"{label} 必须是对象")
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复 JSON 字段")
        result[key] = value
    return result


def _require_exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须是对象")
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label} 字段不匹配")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    _assert_managed_file_or_missing(path, "原子写入目标")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _assert_managed_file_or_missing(temporary, "原子临时文件")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "原子临时文件已存在") from exc
    try:
        _write_all_and_sync(descriptor, payload)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    except OSError as exc:
        raise RunnerFailure(FailureCode.RUNNER_PROTOCOL, "状态无法原子落地") from exc


def _write_all_and_sync(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("无法写入完整内容")
        offset += written
    os.fsync(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunnerFailure(FailureCode.UPLOAD_INTEGRITY, "无法读取文件 hash") from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} 必须是正整数")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负整数")
    return value


def _plain_marker(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200 or any(ord(c) < 33 for c in value):
        raise ValueError(f"{label} 不合法")
    return value


def controlled_environment() -> dict[str, str]:
    """不让 SSH 客户端注入的环境变量影响 runner 或管理员 launcher。"""

    home = pwd.getpwuid(os.getuid()).pw_dir
    return {
        "HOME": home,
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
