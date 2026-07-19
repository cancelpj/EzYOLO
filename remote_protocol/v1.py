"""ezyolo-remote/v1 的纯 Python wire contract。

桌面端和服务端 runner 必须使用同一版本的本模块。这里故意不处理 SSH、rsync、
PyQt、QSettings、数据库或训练框架；这些职责属于各自的边界模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping


REMOTE_PROTOCOL_VERSION = 1

JOB_ID_PATTERN = re.compile(r"[a-f0-9]{32}\Z")
SHA256_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
MODEL_SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

SUPPORTED_TASK_TYPES = frozenset({"detect", "segment"})


class ProtocolValidationError(ValueError):
    """不满足 ezyolo-remote/v1 wire contract 的输入。"""


class ProtocolStateError(ProtocolValidationError):
    """违反远程任务状态机的状态变化。"""


class JobStatus(str, Enum):
    VALIDATING = "VALIDATING"
    SNAPSHOTTING = "SNAPSHOTTING"
    UPLOADING = "UPLOADING"
    VERIFYING_UPLOAD = "VERIFYING_UPLOAD"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    REMOTE_SUCCEEDED_PENDING_COLLECTION = "REMOTE_SUCCEEDED_PENDING_COLLECTION"
    COLLECTING = "COLLECTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    ATTACHING = "ATTACHING"


class FailureCode(str, Enum):
    PRECHECK_FAILED = "PRECHECK_FAILED"
    CLIENT_TRANSPORT_UNAVAILABLE = "CLIENT_TRANSPORT_UNAVAILABLE"
    ENVIRONMENT_MISMATCH = "ENVIRONMENT_MISMATCH"
    SERVER_BUSY = "SERVER_BUSY"
    GPU_ADMISSION = "GPU_ADMISSION"
    UPLOAD_INTEGRITY = "UPLOAD_INTEGRITY"
    RUNNER_PROTOCOL = "RUNNER_PROTOCOL"
    TRAINING_FAILED = "TRAINING_FAILED"
    CUDA_OOM = "CUDA_OOM"
    CANCEL_TIMEOUT = "CANCEL_TIMEOUT"
    COLLECTION_FAILED = "COLLECTION_FAILED"
    LOCAL_VERIFY_FAILED = "LOCAL_VERIFY_FAILED"
    REMOTE_CRASHED = "REMOTE_CRASHED"
    MAX_RUNTIME = "MAX_RUNTIME"
    JOB_DISK_LIMIT = "JOB_DISK_LIMIT"
    LOW_DISK_SPACE = "LOW_DISK_SPACE"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)

# 由 runner 声明、可以作为远程事实返回给桌面端的状态。服务器不会把尚未经过
# 本机结果校验和原子落地的任务声明为 SUCCEEDED。
RUNNER_OBSERVABLE_STATUSES = frozenset(
    {
        JobStatus.VERIFYING_UPLOAD,
        JobStatus.STARTING,
        JobStatus.RUNNING,
        JobStatus.CANCEL_REQUESTED,
        JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)

CAPABILITIES_FIELDS = (
    "protocol_version",
    "canonical_remote_root",
    "supported_tasks",
    "model_symbols",
    "max_epochs",
    "max_runtime_seconds",
    "max_payload_bytes",
    "max_result_bytes",
)
JOB_SPEC_FIELDS = (
    "protocol_version",
    "job_id",
    "task_type",
    "model_symbol",
    "epochs",
    "batch",
    "imgsz",
    "class_names",
    "snapshot_hash",
)
MANIFEST_ENTRY_FIELDS = ("path", "size", "sha256")
MANIFEST_FIELDS = (
    "protocol_version",
    "job_id",
    "task_type",
    "class_names",
    "layout",
    "entries",
    "total_bytes",
    "snapshot_hash",
)
STATUS_FIELDS = (
    "protocol_version",
    "job_id",
    "status",
    "failure_code",
    "message",
    "epoch",
)
RESULT_RECEIPT_FIELDS = (
    "protocol_version",
    "job_id",
    "result_manifest_hash",
    "result_count",
    "result_bytes",
)
RESULT_MANIFEST_FIELDS = (
    "protocol_version",
    "job_id",
    "entries",
    "total_bytes",
)
RUNNER_FAILURE_FIELDS = (
    "protocol_version",
    "ok",
    "failure_code",
    "message",
)


def validate_job_id(value: object) -> str:
    """验证由客户端生成、可安全用于固定 runner action 的 job id。"""
    if not isinstance(value, str) or not JOB_ID_PATTERN.fullmatch(value):
        raise ProtocolValidationError("job id 必须是 32 位小写十六进制字符串")
    return value


def validate_sha256(value: object, *, field_name: str = "sha256") -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ProtocolValidationError(f"{field_name} 必须是 64 位小写 SHA256")
    return value


def is_terminal_status(value: JobStatus | str) -> bool:
    return _coerce_status(value) in TERMINAL_STATUSES


def begin_attach() -> JobStatus:
    """为已有持久化任务启动重新连接；下一步必须读取 runner 的事实状态。"""
    return JobStatus.ATTACHING


def mark_unknown(current: JobStatus | str) -> JobStatus:
    """网络/客户端中断后标记未知；终态不能被本机猜测性改写。"""
    status = _coerce_status(current)
    if is_terminal_status(status):
        raise ProtocolStateError("终态任务不能被改成 UNKNOWN")
    return JobStatus.UNKNOWN


def advance_local_status(current: JobStatus | str, target: JobStatus | str) -> JobStatus:
    """执行仅由桌面端掌握的前置阶段变化。

    本函数刻意拒绝从 UNKNOWN/ATTACHING 作出猜测性恢复，也拒绝直接标记
    SUCCEEDED。前者必须调用 resolve_runner_status()，后者必须调用
    complete_collection()。
    """
    previous = _coerce_status(current)
    next_status = _coerce_status(target)

    if previous in TERMINAL_STATUSES:
        if previous == next_status:
            return previous
        raise ProtocolStateError("终态任务不能改变状态")
    if previous in {JobStatus.UNKNOWN, JobStatus.ATTACHING}:
        raise ProtocolStateError("UNKNOWN/ATTACHING 只能由 runner 事实状态解决")
    if next_status == JobStatus.UNKNOWN:
        return mark_unknown(previous)
    if next_status in {JobStatus.ATTACHING, JobStatus.SUCCEEDED}:
        raise ProtocolStateError("该状态必须通过专用状态入口设置")
    if next_status not in _LOCAL_TRANSITIONS[previous]:
        raise ProtocolStateError(f"不允许本机状态变化：{previous.value} -> {next_status.value}")
    return next_status


def resolve_runner_status(
    current: JobStatus | str,
    runner_status: JobStatus | str,
) -> JobStatus:
    """用 runner 返回的事实状态更新本地状态。

    此函数是 UNKNOWN 与 ATTACHING 的唯一恢复入口。它也允许客户端 UI 落后时由
    runner 的更靠后事实状态推进任务，但不会接受 runner 报告的本机成功终态。
    """
    previous = _coerce_status(current)
    observed = _coerce_status(runner_status)
    if observed not in RUNNER_OBSERVABLE_STATUSES:
        raise ProtocolStateError("runner 不能声明本机专属或未知状态")
    if previous in TERMINAL_STATUSES:
        if previous == observed:
            return previous
        raise ProtocolStateError("终态任务不能被 runner 改写")
    return observed


def complete_collection(current: JobStatus | str, *, promoted: bool) -> JobStatus:
    """仅在结果已本机校验并原子落地后标记训练成功。"""
    previous = _coerce_status(current)
    if previous != JobStatus.COLLECTING:
        raise ProtocolStateError("只有 COLLECTING 阶段可以完成结果收集")
    if promoted is not True:
        raise ProtocolStateError("结果未完成本机原子落地，不能标记 SUCCEEDED")
    return JobStatus.SUCCEEDED


def _coerce_status(value: JobStatus | str) -> JobStatus:
    if isinstance(value, JobStatus):
        return value
    try:
        return JobStatus(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError("未知远程任务状态") from exc


_LOCAL_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.VALIDATING: frozenset(
        {JobStatus.SNAPSHOTTING, JobStatus.FAILED}
    ),
    JobStatus.SNAPSHOTTING: frozenset(
        {JobStatus.UPLOADING, JobStatus.FAILED}
    ),
    JobStatus.UPLOADING: frozenset(
        {JobStatus.VERIFYING_UPLOAD, JobStatus.FAILED}
    ),
    JobStatus.VERIFYING_UPLOAD: frozenset(
        {JobStatus.STARTING, JobStatus.CANCEL_REQUESTED, JobStatus.FAILED}
    ),
    JobStatus.STARTING: frozenset(
        {
            JobStatus.RUNNING,
            JobStatus.CANCEL_REQUESTED,
            JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
            JobStatus.FAILED,
        }
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.CANCEL_REQUESTED,
            JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
            JobStatus.FAILED,
        }
    ),
    JobStatus.CANCEL_REQUESTED: frozenset(
        {JobStatus.CANCELLED, JobStatus.FAILED}
    ),
    JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION: frozenset(
        {JobStatus.COLLECTING, JobStatus.FAILED}
    ),
    JobStatus.COLLECTING: frozenset({JobStatus.FAILED}),
}


@dataclass(frozen=True)
class ServerCapabilities:
    protocol_version: int
    canonical_remote_root: str
    supported_tasks: tuple[str, ...]
    model_symbols: tuple[str, ...]
    max_epochs: int
    max_runtime_seconds: int
    max_payload_bytes: int
    max_result_bytes: int

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        _validate_remote_root(self.canonical_remote_root)
        _validate_nonempty_strings(self.supported_tasks, "supported_tasks", allowed=SUPPORTED_TASK_TYPES)
        _validate_nonempty_strings(self.model_symbols, "model_symbols", validator=_validate_model_symbol)
        _validate_positive_int(self.max_epochs, "max_epochs")
        _validate_positive_int(self.max_runtime_seconds, "max_runtime_seconds")
        _validate_positive_int(self.max_payload_bytes, "max_payload_bytes")
        _validate_positive_int(self.max_result_bytes, "max_result_bytes")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "canonical_remote_root": self.canonical_remote_root,
            "supported_tasks": list(self.supported_tasks),
            "model_symbols": list(self.model_symbols),
            "max_epochs": self.max_epochs,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_payload_bytes": self.max_payload_bytes,
            "max_result_bytes": self.max_result_bytes,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ServerCapabilities":
        data = _require_exact_fields(payload, CAPABILITIES_FIELDS, "capabilities")
        return cls(
            protocol_version=data["protocol_version"],
            canonical_remote_root=data["canonical_remote_root"],
            supported_tasks=_as_string_tuple(data["supported_tasks"], "supported_tasks"),
            model_symbols=_as_string_tuple(data["model_symbols"], "model_symbols"),
            max_epochs=data["max_epochs"],
            max_runtime_seconds=data["max_runtime_seconds"],
            max_payload_bytes=data["max_payload_bytes"],
            max_result_bytes=data["max_result_bytes"],
        )


@dataclass(frozen=True)
class JobSpec:
    protocol_version: int
    job_id: str
    task_type: str
    model_symbol: str
    epochs: int
    batch: int
    imgsz: int
    class_names: tuple[str, ...]
    snapshot_hash: str

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        validate_job_id(self.job_id)
        _validate_task_type(self.task_type)
        _validate_model_symbol(self.model_symbol)
        _validate_positive_int(self.epochs, "epochs")
        _validate_positive_int(self.batch, "batch")
        _validate_positive_int(self.imgsz, "imgsz")
        _validate_nonempty_strings(self.class_names, "class_names")
        validate_sha256(self.snapshot_hash, field_name="snapshot_hash")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "task_type": self.task_type,
            "model_symbol": self.model_symbol,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "class_names": list(self.class_names),
            "snapshot_hash": self.snapshot_hash,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "JobSpec":
        data = _require_exact_fields(payload, JOB_SPEC_FIELDS, "job spec")
        return cls(
            protocol_version=data["protocol_version"],
            job_id=data["job_id"],
            task_type=data["task_type"],
            model_symbol=data["model_symbol"],
            epochs=data["epochs"],
            batch=data["batch"],
            imgsz=data["imgsz"],
            class_names=_as_string_tuple(data["class_names"], "class_names"),
            snapshot_hash=data["snapshot_hash"],
        )


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_payload_path(self.path)
        _validate_nonnegative_int(self.size, "size")
        validate_sha256(self.sha256)

    def to_wire(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ManifestEntry":
        data = _require_exact_fields(payload, MANIFEST_ENTRY_FIELDS, "manifest entry")
        return cls(path=data["path"], size=data["size"], sha256=data["sha256"])


@dataclass(frozen=True)
class DatasetManifest:
    protocol_version: int
    job_id: str
    task_type: str
    class_names: tuple[str, ...]
    layout: tuple[tuple[str, str], ...]
    entries: tuple[ManifestEntry, ...]
    total_bytes: int
    snapshot_hash: str

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        validate_job_id(self.job_id)
        _validate_task_type(self.task_type)
        _validate_nonempty_strings(self.class_names, "class_names")
        _validate_layout(self.layout)
        if not self.entries:
            raise ProtocolValidationError("manifest 必须至少包含一个文件")
        if len({entry.path for entry in self.entries}) != len(self.entries):
            raise ProtocolValidationError("manifest 不能包含重复路径")
        _validate_nonnegative_int(self.total_bytes, "total_bytes")
        if self.total_bytes != sum(entry.size for entry in self.entries):
            raise ProtocolValidationError("manifest total_bytes 与文件大小不一致")
        validate_sha256(self.snapshot_hash, field_name="snapshot_hash")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "task_type": self.task_type,
            "class_names": list(self.class_names),
            "layout": {key: value for key, value in self.layout},
            "entries": [entry.to_wire() for entry in self.entries],
            "total_bytes": self.total_bytes,
            "snapshot_hash": self.snapshot_hash,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        data = _require_exact_fields(payload, MANIFEST_FIELDS, "manifest")
        raw_entries = data["entries"]
        if not isinstance(raw_entries, list):
            raise ProtocolValidationError("entries 必须是列表")
        raw_layout = data["layout"]
        if not isinstance(raw_layout, Mapping):
            raise ProtocolValidationError("layout 必须是对象")
        return cls(
            protocol_version=data["protocol_version"],
            job_id=data["job_id"],
            task_type=data["task_type"],
            class_names=_as_string_tuple(data["class_names"], "class_names"),
            layout=tuple(raw_layout.items()),
            entries=tuple(ManifestEntry.from_wire(entry) for entry in raw_entries),
            total_bytes=data["total_bytes"],
            snapshot_hash=data["snapshot_hash"],
        )


@dataclass(frozen=True)
class RemoteStatus:
    protocol_version: int
    job_id: str
    status: JobStatus
    failure_code: FailureCode | None = None
    message: str | None = None
    epoch: int | None = None

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        validate_job_id(self.job_id)
        status = _coerce_status(self.status)
        object.__setattr__(self, "status", status)
        if self.failure_code is not None:
            try:
                code = FailureCode(self.failure_code)
            except (TypeError, ValueError) as exc:
                raise ProtocolValidationError("未知 failure_code") from exc
            object.__setattr__(self, "failure_code", code)
        if status == JobStatus.FAILED and self.failure_code is None:
            raise ProtocolValidationError("FAILED 状态必须带 failure_code")
        if status != JobStatus.FAILED and self.failure_code is not None:
            raise ProtocolValidationError("只有 FAILED 状态可以带 failure_code")
        if self.message is not None:
            _validate_message(self.message)
        if self.epoch is not None:
            _validate_nonnegative_int(self.epoch, "epoch")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "status": self.status.value,
            "failure_code": self.failure_code.value if self.failure_code else None,
            "message": self.message,
            "epoch": self.epoch,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "RemoteStatus":
        data = _require_exact_fields(payload, STATUS_FIELDS, "status")
        return cls(
            protocol_version=data["protocol_version"],
            job_id=data["job_id"],
            status=data["status"],
            failure_code=data["failure_code"],
            message=data["message"],
            epoch=data["epoch"],
        )


@dataclass(frozen=True)
class ResultReceipt:
    protocol_version: int
    job_id: str
    result_manifest_hash: str
    result_count: int
    result_bytes: int

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        validate_job_id(self.job_id)
        validate_sha256(self.result_manifest_hash, field_name="result_manifest_hash")
        _validate_nonnegative_int(self.result_count, "result_count")
        _validate_nonnegative_int(self.result_bytes, "result_bytes")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "result_manifest_hash": self.result_manifest_hash,
            "result_count": self.result_count,
            "result_bytes": self.result_bytes,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ResultReceipt":
        data = _require_exact_fields(payload, RESULT_RECEIPT_FIELDS, "result receipt")
        return cls(
            protocol_version=data["protocol_version"],
            job_id=data["job_id"],
            result_manifest_hash=data["result_manifest_hash"],
            result_count=data["result_count"],
            result_bytes=data["result_bytes"],
        )


@dataclass(frozen=True)
class RunnerFailureEnvelope:
    """runner CLI 失败时唯一允许返回的受控 wire envelope。

    失败 envelope 不携带 traceback、命令行、路径或认证材料。桌面端只使用其中
    的枚举错误码推进本机任务状态，界面不会直接回显 ``message``。
    """

    protocol_version: int
    ok: bool
    failure_code: FailureCode
    message: str

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        if self.ok is not False:
            raise ProtocolValidationError("runner failure envelope 的 ok 必须为 false")
        try:
            code = FailureCode(self.failure_code)
        except (TypeError, ValueError) as exc:
            raise ProtocolValidationError("未知 runner failure_code") from exc
        object.__setattr__(self, "failure_code", code)
        _validate_message(self.message)
        if not self.message.strip():
            raise ProtocolValidationError("runner failure message 不能为空")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "ok": False,
            "failure_code": self.failure_code.value,
            "message": self.message,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "RunnerFailureEnvelope":
        data = _require_exact_fields(payload, RUNNER_FAILURE_FIELDS, "runner failure")
        return cls(
            protocol_version=data["protocol_version"],
            ok=data["ok"],
            failure_code=data["failure_code"],
            message=data["message"],
        )


@dataclass(frozen=True)
class ResultManifest:
    """服务器回传结果的受控文件清单。

    ``manifest.json`` 本身不在 ``entries`` 里，避免把它自己的哈希递归进清单。
    ``ResultReceipt.result_manifest_hash`` 是该控制文件原始 UTF-8 字节的 SHA-256，
    而 ``result_count`` / ``result_bytes`` 只统计 entries 中的实际结果文件。
    """

    protocol_version: int
    job_id: str
    entries: tuple[ManifestEntry, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        _validate_protocol_version(self.protocol_version)
        validate_job_id(self.job_id)
        if not self.entries:
            raise ProtocolValidationError("结果 manifest 必须至少包含一个文件")
        if len({entry.path for entry in self.entries}) != len(self.entries):
            raise ProtocolValidationError("结果 manifest 不能包含重复路径")
        _validate_nonnegative_int(self.total_bytes, "result total_bytes")
        if self.total_bytes != sum(entry.size for entry in self.entries):
            raise ProtocolValidationError("结果 manifest total_bytes 与文件大小不一致")

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "entries": [entry.to_wire() for entry in self.entries],
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ResultManifest":
        data = _require_exact_fields(payload, RESULT_MANIFEST_FIELDS, "result manifest")
        raw_entries = data["entries"]
        if not isinstance(raw_entries, list):
            raise ProtocolValidationError("结果 entries 必须是列表")
        return cls(
            protocol_version=data["protocol_version"],
            job_id=data["job_id"],
            entries=tuple(ManifestEntry.from_wire(entry) for entry in raw_entries),
            total_bytes=data["total_bytes"],
        )


def _require_exact_fields(
    payload: Mapping[str, Any], expected: tuple[str, ...], label: str
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProtocolValidationError(f"{label} 必须是对象")
    actual = set(payload)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        detail = []
        if missing:
            detail.append("缺少 " + ", ".join(missing))
        if extra:
            detail.append("包含未知字段 " + ", ".join(extra))
        raise ProtocolValidationError(f"{label} 字段不匹配：{'；'.join(detail)}")
    return payload


def _validate_protocol_version(value: object) -> None:
    if type(value) is not int or value != REMOTE_PROTOCOL_VERSION:
        raise ProtocolValidationError("协议版本不匹配")


def _validate_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ProtocolValidationError(f"{field_name} 必须是正整数")


def _validate_nonnegative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ProtocolValidationError(f"{field_name} 必须是非负整数")


def _validate_task_type(value: object) -> None:
    if value not in SUPPORTED_TASK_TYPES:
        raise ProtocolValidationError("task_type 仅支持 detect 或 segment")


def _validate_model_symbol(value: object) -> None:
    if not isinstance(value, str) or not MODEL_SYMBOL_PATTERN.fullmatch(value):
        raise ProtocolValidationError("model_symbol 格式不合法")


def _validate_remote_root(value: object) -> None:
    if not isinstance(value, str) or not value.startswith("/") or value == "/":
        raise ProtocolValidationError("canonical_remote_root 必须是非根绝对路径")
    if value == "/root" or value.startswith("/root/"):
        raise ProtocolValidationError("canonical_remote_root 不能位于 /root")
    if any(ord(char) < 32 for char in value) or "//" in value:
        raise ProtocolValidationError("canonical_remote_root 不规范")
    parts = value.split("/")[1:]
    if any(
        part in {"", ".", ".."}
        or part.startswith("-")
        or not re.fullmatch(r"[A-Za-z0-9._-]+", part)
        for part in parts
    ):
        raise ProtocolValidationError("canonical_remote_root 不规范")


def _validate_relative_payload_path(value: object) -> None:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ProtocolValidationError("manifest path 必须是非空 POSIX 相对路径")
    if any(ord(char) < 32 for char in value):
        raise ProtocolValidationError("manifest path 含有控制字符")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.startswith("-") for part in parts):
        raise ProtocolValidationError("manifest path 含有不安全路径段")


def _validate_nonempty_strings(
    values: object,
    field_name: str,
    *,
    allowed: frozenset[str] | None = None,
    validator: Any | None = None,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ProtocolValidationError(f"{field_name} 必须是非空字符串元组")
    if len(set(values)) != len(values):
        raise ProtocolValidationError(f"{field_name} 不能包含重复项")
    for value in values:
        if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
            raise ProtocolValidationError(f"{field_name} 包含非法字符串")
        if allowed is not None and value not in allowed:
            raise ProtocolValidationError(f"{field_name} 包含不支持的值")
        if validator is not None:
            validator(value)


def _as_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProtocolValidationError(f"{field_name} 必须是列表")
    return tuple(value)


def _validate_layout(value: object) -> None:
    if not isinstance(value, tuple) or not value:
        raise ProtocolValidationError("layout 必须是非空路径映射")
    keys: set[str] = set()
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ProtocolValidationError("layout 结构不合法")
        key, path = item
        if not isinstance(key, str) or key not in {"train", "val", "test"}:
            raise ProtocolValidationError("layout 仅允许 train、val 和 test")
        if key in keys:
            raise ProtocolValidationError("layout 不能重复字段")
        keys.add(key)
        _validate_relative_payload_path(path)
        if not path.startswith("images/"):
            raise ProtocolValidationError("layout 路径必须位于 images/ 下")
    if not {"train", "val"}.issubset(keys):
        raise ProtocolValidationError("layout 必须包含 train 和 val")


def _validate_message(value: object) -> None:
    if not isinstance(value, str) or len(value) > 500 or any(ord(char) < 32 and char not in "\\n\\t" for char in value):
        raise ProtocolValidationError("status message 不合法")
