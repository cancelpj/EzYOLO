"""远程训练 job record 与回传结果的本机 staging/promote 边界。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Protocol, Sequence

from remote_protocol.v1 import (
    REMOTE_PROTOCOL_VERSION,
    FailureCode,
    JobStatus,
    ProtocolStateError,
    ProtocolValidationError,
    RemoteStatus,
    ResultReceipt,
    advance_local_status,
    complete_collection,
    mark_unknown as protocol_mark_unknown,
    resolve_runner_status,
    validate_job_id,
    validate_sha256,
)


REMOTE_TRAINING_JOBS_KEY = "remote_training_jobs_v1"
_POSIX_ROOT_SEGMENT = re.compile(r"[A-Za-z0-9._-]+\Z")


class RemoteTrainingJobError(ValueError):
    """本机 job record 不符合远程训练持久化约束。"""


class ResultPromotionError(RemoteTrainingJobError):
    """回传结果不能安全地从 staging 原子落地。"""


class SettingsLike(Protocol):
    def value(self, key: str, default_value: Any = ...) -> Any: ...

    def setValue(self, key: str, value: Any) -> None: ...

    def sync(self) -> None: ...


@dataclass(frozen=True)
class RemoteTrainingJobRecord:
    """可恢复的最小本机任务记录；其中不保存认证材料或远程命令。"""

    protocol_version: int
    job_id: str
    project_id: int
    profile_id: str
    canonical_remote_root: str
    snapshot_hash: str
    created_at: str
    updated_at: str
    last_status: JobStatus
    failure_code: FailureCode | None = None
    result_receipt: ResultReceipt | None = None
    local_result_dir: str | None = None

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not int or self.protocol_version != REMOTE_PROTOCOL_VERSION:
            raise RemoteTrainingJobError("job record 协议版本不匹配")
        validate_job_id(self.job_id)
        if type(self.project_id) is not int or self.project_id <= 0:
            raise RemoteTrainingJobError("project_id 必须是正整数")
        validate_job_id(self.profile_id)
        _validate_canonical_remote_root(self.canonical_remote_root)
        validate_sha256(self.snapshot_hash, field_name="snapshot_hash")
        _validate_timestamp(self.created_at, "created_at")
        _validate_timestamp(self.updated_at, "updated_at")
        try:
            status = JobStatus(self.last_status)
        except (TypeError, ValueError) as exc:
            raise RemoteTrainingJobError("last_status 不合法") from exc
        object.__setattr__(self, "last_status", status)
        if self.failure_code is not None:
            try:
                failure_code = FailureCode(self.failure_code)
            except (TypeError, ValueError) as exc:
                raise RemoteTrainingJobError("failure_code 不合法") from exc
            object.__setattr__(self, "failure_code", failure_code)
        if status == JobStatus.FAILED and self.failure_code is None:
            raise RemoteTrainingJobError("FAILED job record 必须带 failure_code")
        if status != JobStatus.FAILED and self.failure_code is not None:
            raise RemoteTrainingJobError("非 FAILED job record 不能带 failure_code")
        if self.result_receipt is not None and self.result_receipt.job_id != self.job_id:
            raise RemoteTrainingJobError("result_receipt 与 job id 不一致")
        if self.local_result_dir is not None:
            if not isinstance(self.local_result_dir, str) or not self.local_result_dir:
                raise RemoteTrainingJobError("local_result_dir 不合法")

    @classmethod
    def new(
        cls,
        *,
        job_id: str,
        project_id: int,
        profile_id: str,
        canonical_remote_root: str,
        snapshot_hash: str,
        now: datetime | None = None,
    ) -> "RemoteTrainingJobRecord":
        timestamp = _timestamp(now)
        return cls(
            protocol_version=REMOTE_PROTOCOL_VERSION,
            job_id=job_id,
            project_id=project_id,
            profile_id=profile_id,
            canonical_remote_root=canonical_remote_root,
            snapshot_hash=snapshot_hash,
            created_at=timestamp,
            updated_at=timestamp,
            last_status=JobStatus.VALIDATING,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "project_id": self.project_id,
            "profile_id": self.profile_id,
            "canonical_remote_root": self.canonical_remote_root,
            "snapshot_hash": self.snapshot_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_status": self.last_status.value,
            "failure_code": self.failure_code.value if self.failure_code else None,
            "result_receipt": (
                self.result_receipt.to_wire() if self.result_receipt else None
            ),
            "local_result_dir": self.local_result_dir,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RemoteTrainingJobRecord":
        expected = {
            "protocol_version",
            "job_id",
            "project_id",
            "profile_id",
            "canonical_remote_root",
            "snapshot_hash",
            "created_at",
            "updated_at",
            "last_status",
            "failure_code",
            "result_receipt",
            "local_result_dir",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise RemoteTrainingJobError("job record 字段不匹配")
        receipt_payload = payload["result_receipt"]
        if receipt_payload is not None and not isinstance(receipt_payload, Mapping):
            raise RemoteTrainingJobError("result_receipt 必须是对象或空")
        try:
            receipt = (
                ResultReceipt.from_wire(receipt_payload)
                if receipt_payload is not None
                else None
            )
        except ProtocolValidationError as exc:
            raise RemoteTrainingJobError("result_receipt 不合法") from exc
        return cls(
            protocol_version=payload["protocol_version"],
            job_id=payload["job_id"],
            project_id=payload["project_id"],
            profile_id=payload["profile_id"],
            canonical_remote_root=payload["canonical_remote_root"],
            snapshot_hash=payload["snapshot_hash"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            last_status=payload["last_status"],
            failure_code=payload["failure_code"],
            result_receipt=receipt,
            local_result_dir=payload["local_result_dir"],
        )

    def with_runner_status(
        self,
        remote_status: RemoteStatus,
        *,
        now: datetime | None = None,
    ) -> "RemoteTrainingJobRecord":
        if remote_status.job_id != self.job_id:
            raise RemoteTrainingJobError("runner status 与 job id 不一致")
        try:
            next_status = resolve_runner_status(self.last_status, remote_status.status)
        except ProtocolStateError as exc:
            raise RemoteTrainingJobError("runner status 不允许改变本机 job") from exc
        return replace(
            self,
            last_status=next_status,
            failure_code=remote_status.failure_code,
            updated_at=_timestamp(now),
        )

    def advance_local(
        self,
        target: JobStatus,
        *,
        now: datetime | None = None,
    ) -> "RemoteTrainingJobRecord":
        """推进非终态的本机步骤；FAILED 必须改用 fail() 并带明确原因。"""
        if target == JobStatus.FAILED:
            raise RemoteTrainingJobError("FAILED 必须带明确 failure_code")
        return _replace_local_status(self, target, now=now)

    def fail(
        self,
        failure_code: FailureCode,
        *,
        now: datetime | None = None,
    ) -> "RemoteTrainingJobRecord":
        """把可失败的本机阶段原子更新为 FAILED + 可解释原因。"""
        try:
            failed = advance_local_status(self.last_status, JobStatus.FAILED)
            code = FailureCode(failure_code)
        except (ProtocolStateError, TypeError, ValueError) as exc:
            raise RemoteTrainingJobError("job record 不允许该失败状态变化") from exc
        return replace(
            self,
            last_status=failed,
            failure_code=code,
            updated_at=_timestamp(now),
        )

    def mark_unknown(self, *, now: datetime | None = None) -> "RemoteTrainingJobRecord":
        """网络中断后持久化 UNKNOWN；下次只能由 runner 事实状态恢复。"""
        try:
            unknown = protocol_mark_unknown(self.last_status)
        except ProtocolStateError as exc:
            raise RemoteTrainingJobError("终态任务不能被标记为 UNKNOWN") from exc
        return replace(
            self,
            last_status=unknown,
            failure_code=None,
            updated_at=_timestamp(now),
        )

    def begin_collection(self, *, now: datetime | None = None) -> "RemoteTrainingJobRecord":
        return _replace_local_status(
            self,
            JobStatus.COLLECTING,
            now=now,
        )

    def fail_collection(self, *, now: datetime | None = None) -> "RemoteTrainingJobRecord":
        try:
            failed = advance_local_status(self.last_status, JobStatus.FAILED)
        except ProtocolStateError as exc:
            raise RemoteTrainingJobError("job record 不允许该本机状态变化") from exc
        return replace(
            self,
            last_status=failed,
            failure_code=FailureCode.COLLECTION_FAILED,
            updated_at=_timestamp(now),
        )

    def finish_collection(
        self,
        *,
        receipt: ResultReceipt,
        local_result_dir: Path | str,
        now: datetime | None = None,
    ) -> "RemoteTrainingJobRecord":
        if receipt.job_id != self.job_id:
            raise RemoteTrainingJobError("result receipt 与 job id 不一致")
        try:
            succeeded = complete_collection(self.last_status, promoted=True)
        except ProtocolStateError as exc:
            raise RemoteTrainingJobError("结果尚不能标记成功") from exc
        return replace(
            self,
            last_status=succeeded,
            failure_code=None,
            result_receipt=receipt,
            local_result_dir=str(Path(local_result_dir)),
            updated_at=_timestamp(now),
        )


class RemoteTrainingJobStore:
    """只管理版本化 QSettings job record；不执行网络或文件传输。"""

    def __init__(self, settings: SettingsLike) -> None:
        self._settings = settings

    def list(self) -> list[RemoteTrainingJobRecord]:
        raw = self._settings.value(REMOTE_TRAINING_JOBS_KEY, "[]")
        if raw in (None, ""):
            return []
        if not isinstance(raw, str):
            raise RemoteTrainingJobError("远程任务存储格式不正确")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RemoteTrainingJobError("远程任务存储不是有效 JSON") from exc
        if not isinstance(payload, list):
            raise RemoteTrainingJobError("远程任务存储必须是列表")
        try:
            records = [RemoteTrainingJobRecord.from_dict(item) for item in payload]
        except (TypeError, RemoteTrainingJobError) as exc:
            raise RemoteTrainingJobError("远程任务存储含有无效记录") from exc
        _ensure_unique_job_ids(records)
        return records

    def get(self, job_id: str) -> RemoteTrainingJobRecord | None:
        validate_job_id(job_id)
        return next((record for record in self.list() if record.job_id == job_id), None)

    def create(self, record: RemoteTrainingJobRecord) -> None:
        records = self.list()
        if any(item.job_id == record.job_id for item in records):
            raise RemoteTrainingJobError("job id 已存在，拒绝覆盖")
        self._save([*records, record])

    def replace(self, record: RemoteTrainingJobRecord) -> None:
        records = self.list()
        replaced = False
        result = []
        for item in records:
            if item.job_id == record.job_id:
                result.append(record)
                replaced = True
            else:
                result.append(item)
        if not replaced:
            raise RemoteTrainingJobError("找不到要更新的 job record")
        self._save(result)

    def _save(self, records: Sequence[RemoteTrainingJobRecord]) -> None:
        _ensure_unique_job_ids(records)
        serialized = json.dumps(
            [record.to_dict() for record in records],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._settings.setValue(REMOTE_TRAINING_JOBS_KEY, serialized)
        self._settings.sync()


class ResultStager:
    """把已经在本机 staging 的回传结果原子落地到 runs/train。"""

    def __init__(
        self,
        staging_parent: Path | str,
        runs_train_root: Path | str,
        *,
        stat_function=os.stat,
    ) -> None:
        self.staging_parent = Path(staging_parent)
        self.runs_train_root = Path(runs_train_root)
        self._stat = stat_function

    def create_staging(self, job_id: str) -> Path:
        validate_job_id(job_id)
        self.staging_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging_dir = self.staging_parent / job_id
        try:
            staging_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ResultPromotionError("该 job 的结果 staging 已存在，拒绝覆盖") from exc
        return staging_dir

    def final_dir_for(self, project_id: int, job_id: str) -> Path:
        if type(project_id) is not int or project_id <= 0:
            raise ResultPromotionError("project_id 必须是正整数")
        validate_job_id(job_id)
        return self.runs_train_root / f"exp_{project_id}_remote_{job_id[:8]}"

    def promote(self, staging_dir: Path | str, *, project_id: int, job_id: str) -> Path:
        """在同一文件系统上 rename；已有最终目录时明确失败且绝不覆盖。"""
        validate_job_id(job_id)
        staging = Path(staging_dir)
        final_dir = self.final_dir_for(project_id, job_id)
        if staging.is_symlink() or not staging.is_dir():
            raise ResultPromotionError("结果 staging 不存在或不是普通目录")
        _require_direct_child(staging, self.staging_parent)

        self.runs_train_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if final_dir.exists() or final_dir.is_symlink():
            raise ResultPromotionError("最终结果目录已存在，拒绝覆盖")
        if self._stat(staging).st_dev != self._stat(self.runs_train_root).st_dev:
            raise ResultPromotionError("staging 与最终目录不在同一文件系统，拒绝非原子移动")

        reservation = final_dir.with_name(final_dir.name + ".promote-lock")
        try:
            descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ResultPromotionError("最终结果目录正在被另一个落地操作占用") from exc
        else:
            os.close(descriptor)

        try:
            if final_dir.exists() or final_dir.is_symlink():
                raise ResultPromotionError("最终结果目录已存在，拒绝覆盖")
            try:
                os.rename(staging, final_dir)
            except OSError as exc:
                raise ResultPromotionError("结果无法原子落地") from exc
        finally:
            try:
                reservation.unlink()
            except FileNotFoundError:
                pass
        return final_dir


def _replace_local_status(
    record: RemoteTrainingJobRecord,
    target: JobStatus,
    *,
    now: datetime | None,
) -> RemoteTrainingJobRecord:
    try:
        next_status = advance_local_status(record.last_status, target)
    except ProtocolStateError as exc:
        raise RemoteTrainingJobError("job record 不允许该本机状态变化") from exc
    return replace(
        record,
        last_status=next_status,
        failure_code=None,
        updated_at=_timestamp(now),
    )


def _ensure_unique_job_ids(records: Sequence[RemoteTrainingJobRecord]) -> None:
    ids = [record.job_id for record in records]
    if len(ids) != len(set(ids)):
        raise RemoteTrainingJobError("远程任务存储不能包含重复 job id")


def _validate_canonical_remote_root(value: object) -> None:
    if not isinstance(value, str) or value == "/" or not value.startswith("/"):
        raise RemoteTrainingJobError("canonical_remote_root 必须是非根绝对路径")
    if value == "/root" or value.startswith("/root/"):
        raise RemoteTrainingJobError("canonical_remote_root 不能位于 /root")
    if any(ord(char) < 32 for char in value) or "//" in value:
        raise RemoteTrainingJobError("canonical_remote_root 不规范")
    parts = value.split("/")[1:]
    if any(
        part in {"", ".", ".."}
        or part.startswith("-")
        or not _POSIX_ROOT_SEGMENT.fullmatch(part)
        for part in parts
    ):
        raise RemoteTrainingJobError("canonical_remote_root 不规范")


def _validate_timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise RemoteTrainingJobError(f"{field_name} 必须是 UTC ISO 时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RemoteTrainingJobError(f"{field_name} 必须是 UTC ISO 时间") from exc
    if parsed.tzinfo is None:
        raise RemoteTrainingJobError(f"{field_name} 必须带时区")


def _timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise RemoteTrainingJobError("时间必须带时区")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_direct_child(candidate: Path, parent: Path) -> None:
    try:
        candidate.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ResultPromotionError("staging 不在受控 staging 目录内") from exc
    if candidate.parent.resolve(strict=True) != parent.resolve(strict=True):
        raise ResultPromotionError("staging 必须是受控目录的直接子目录")
