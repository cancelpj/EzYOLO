"""远程训练的 Qt 后台状态机。

界面只订阅信号，网络、快照、结果 staging 与状态持久化都在本线程中完成。该线程
不会显示确认框；训练页必须在创建它之前向用户确认服务器、目录与预计上传数据。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
import uuid

from PyQt6.QtCore import QThread, pyqtSignal

from core.remote_training.jobs import (
    RemoteTrainingJobError,
    RemoteTrainingJobRecord,
    ResultPromotionError,
    ResultStager,
)
from core.remote_training.launch import RemoteLaunchPlan
from core.remote_training.snapshot import (
    DatasetSnapshotBuilder,
    SnapshotError,
    SnapshotSource,
    verify_remote_payload,
    write_job_spec,
)
from core.remote_training.transport import (
    RemoteTrainingBackend,
    RemoteRunnerReportedFailure,
    RemoteTransportError,
)
from core.remote_training.profiles import RemoteTrainingProfile
from remote_protocol.v1 import (
    FailureCode,
    JobSpec,
    JobStatus,
    ProtocolValidationError,
    ResultReceipt,
)


class JobStoreLike(Protocol):
    def create(self, record: RemoteTrainingJobRecord) -> None: ...

    def replace(self, record: RemoteTrainingJobRecord) -> None: ...


ResultVerifier = Callable[[Path, ResultReceipt], None]
SleepFunction = Callable[[float], None]
JobIdFactory = Callable[[], str]
MonotonicClock = Callable[[], float]


@dataclass(frozen=True)
class RemoteTrainingRequest:
    launch_plan: RemoteLaunchPlan
    class_names: tuple[str, ...]
    layout: Mapping[str, str]
    sources: Sequence[SnapshotSource]


class RemoteTrainingThread(QThread):
    """按 v1 状态机执行一次已确认的远程训练，不阻塞 Qt 主线程。"""

    state_changed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    job_updated = pyqtSignal(object)
    training_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        *,
        request: RemoteTrainingRequest,
        backend: RemoteTrainingBackend,
        job_store: JobStoreLike,
        snapshot_builder: DatasetSnapshotBuilder,
        result_stager: ResultStager,
        result_verifier: ResultVerifier,
        poll_interval_seconds: float = 2.0,
        cancel_timeout_seconds: float = 60.0,
        sleep: SleepFunction = time.sleep,
        clock: MonotonicClock = time.monotonic,
        job_id_factory: JobIdFactory = lambda: uuid.uuid4().hex,
    ) -> None:
        super().__init__()
        if poll_interval_seconds < 0:
            raise ValueError("poll interval 不能小于零")
        if (
            isinstance(cancel_timeout_seconds, bool)
            or not isinstance(cancel_timeout_seconds, (int, float))
            or cancel_timeout_seconds <= 0
        ):
            raise ValueError("取消确认超时必须是正数")
        self._request = request
        self._backend = backend
        self._job_store = job_store
        self._snapshot_builder = snapshot_builder
        self._result_stager = result_stager
        self._result_verifier = result_verifier
        self._poll_interval_seconds = poll_interval_seconds
        self._cancel_timeout_seconds = float(cancel_timeout_seconds)
        self._sleep = sleep
        self._clock = clock
        self._job_id_factory = job_id_factory
        self._cancel_requested = threading.Event()
        self._cancel_sent = False
        self._cancel_started_at: float | None = None
        self._record: RemoteTrainingJobRecord | None = None
        self._capabilities = None

    def request_cancel(self) -> None:
        """UI 线程可安全调用：只记录请求，不在主线程执行网络 I/O。"""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            self._run_workflow()
        except RemoteRunnerReportedFailure as exc:
            self._handle_failure(
                exc.code,
                "服务器拒绝了本次远程训练请求，任务未被标记为完成或已停止",
            )
        except RemoteTransportError:
            self._handle_transport_interruption()
        except _ThreadFailure as exc:
            self._handle_failure(exc.code, exc.message)
        except Exception:
            self._handle_failure(
                FailureCode.LOCAL_VERIFY_FAILED,
                "远程训练在本机校验阶段失败，结果没有被标记为完成",
            )

    def _run_workflow(self) -> None:
        plan = self._request.launch_plan
        self._set_state(JobStatus.VALIDATING, "正在核验已保存服务器与训练能力…")
        capabilities = self._backend.preflight(plan.profile)
        self._validate_preflight(capabilities)
        self._capabilities = capabilities

        job_id = self._job_id_factory()
        self._set_state(JobStatus.SNAPSHOTTING, "正在创建只读数据快照…")
        snapshot = self._snapshot_builder.build(
            job_id=job_id,
            task_type=plan.task_type,
            class_names=self._request.class_names,
            layout=self._request.layout,
            sources=self._request.sources,
        )
        spec = self._job_spec(job_id, snapshot.manifest.snapshot_hash)
        write_job_spec(snapshot, spec)
        verify_remote_payload(snapshot.root)
        if snapshot.manifest.total_bytes > capabilities.max_payload_bytes:
            raise _ThreadFailure(
                FailureCode.UPLOAD_INTEGRITY,
                "数据快照超过服务器允许的上传大小，已阻止上传",
            )

        record = RemoteTrainingJobRecord.new(
            job_id=job_id,
            project_id=plan.project_id,
            profile_id=plan.profile.id,
            canonical_remote_root=plan.profile.remote_root,
            snapshot_hash=snapshot.manifest.snapshot_hash,
        ).advance_local(JobStatus.SNAPSHOTTING)
        self._job_store.create(record)
        self._record = record
        self.job_updated.emit(record)

        self._advance(JobStatus.UPLOADING, "正在安全上传数据快照…")
        self._backend.upload(plan.profile, job_id, snapshot.root)
        self._advance(JobStatus.VERIFYING_UPLOAD, "服务器正在核验上传的数据…")
        self._accept_remote_status(
            self._backend.verify_upload(plan.profile, job_id)
        )
        self._maybe_cancel_or_start()
        self._poll_until_terminal()

    def _validate_preflight(self, capabilities) -> None:
        plan = self._request.launch_plan
        if plan.task_type not in capabilities.supported_tasks:
            raise _ThreadFailure(
                FailureCode.PRECHECK_FAILED,
                "这台服务器不支持当前训练任务类型，已阻止上传",
            )
        if plan.model_symbol not in capabilities.model_symbols:
            raise _ThreadFailure(
                FailureCode.PRECHECK_FAILED,
                "这台服务器没有预置当前模型，已阻止上传",
            )
        epochs = _positive_int(plan.runtime_config, "epochs")
        if epochs > capabilities.max_epochs:
            raise _ThreadFailure(
                FailureCode.PRECHECK_FAILED,
                "训练轮数超过服务器策略上限，已阻止上传",
            )

    def _job_spec(self, job_id: str, snapshot_hash: str) -> JobSpec:
        plan = self._request.launch_plan
        try:
            return JobSpec(
                protocol_version=1,
                job_id=job_id,
                task_type=plan.task_type,
                model_symbol=plan.model_symbol,
                epochs=_positive_int(plan.runtime_config, "epochs"),
                batch=_positive_int(plan.runtime_config, "batch_size"),
                imgsz=_positive_int(plan.runtime_config, "img_size"),
                class_names=self._request.class_names,
                snapshot_hash=snapshot_hash,
            )
        except ProtocolValidationError as exc:
            raise _ThreadFailure(
                FailureCode.LOCAL_VERIFY_FAILED,
                "远程训练任务参数无效，已阻止上传",
            ) from exc

    def _maybe_cancel_or_start(self) -> None:
        if self._cancel_requested.is_set():
            self._cancel()
            return
        self._advance(JobStatus.STARTING, "服务器正在启动训练任务…")
        self._accept_remote_status(
            self._backend.start(self._request.launch_plan.profile, self._require_record().job_id)
        )

    def _poll_until_terminal(self) -> None:
        while True:
            record = self._require_record()
            if record.last_status == JobStatus.FAILED:
                self._finish(False, "服务器已报告训练失败")
                return
            if record.last_status == JobStatus.CANCELLED:
                self._finish(False, "服务器已确认训练任务停止")
                return
            if record.last_status == JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION:
                self._collect_results()
                return
            if self._cancel_requested.is_set():
                self._raise_if_cancel_confirmation_timed_out()
                # runner 才能在进程组真正退出后把状态确认成 CANCELLED。请求只发
                # 一次，之后只轮询；若 runner 始终不确认，会由上面的超时安全退出。
                if not self._cancel_sent:
                    self._cancel()
                    continue
            if self._poll_interval_seconds:
                self._sleep(self._poll_interval_seconds)
            self._accept_remote_status(
                self._backend.poll(
                    self._request.launch_plan.profile,
                    record.job_id,
                )
            )

    def _cancel(self) -> None:
        record = self._require_record()
        if not self._cancel_sent:
            self._cancel_sent = True
            self._cancel_started_at = self._clock()
        if record.last_status not in {JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED}:
            self._advance(JobStatus.CANCEL_REQUESTED, "已请求服务器停止当前任务，正在等待确认…")
        if self._require_record().last_status == JobStatus.CANCELLED:
            return
        self._accept_remote_status(
            self._backend.cancel(
                self._request.launch_plan.profile,
                self._require_record().job_id,
            )
        )

    def _raise_if_cancel_confirmation_timed_out(self) -> None:
        if self._cancel_started_at is None:
            return
        elapsed = self._clock() - self._cancel_started_at
        if elapsed >= self._cancel_timeout_seconds:
            raise _ThreadFailure(
                FailureCode.CANCEL_TIMEOUT,
                "服务器未在限定时间内确认停止；远程任务不能视为已停止，请稍后重新连接核验",
            )

    def _collect_results(self) -> None:
        self._advance(JobStatus.COLLECTING, "正在回传并校验训练结果…")
        record = self._require_record()
        receipt = self._backend.collect_manifest(
            self._request.launch_plan.profile,
            record.job_id,
        )
        if receipt.job_id != record.job_id:
            raise _ThreadFailure(
                FailureCode.COLLECTION_FAILED,
                "服务器结果回执与当前任务不一致，已阻止下载",
            )
        capabilities = self._capabilities
        if capabilities is None or receipt.result_bytes > capabilities.max_result_bytes:
            raise _ThreadFailure(
                FailureCode.COLLECTION_FAILED,
                "服务器声明的结果大小超过已核验上限，已阻止下载",
            )
        staging = self._result_stager.create_staging(record.job_id)
        try:
            self._backend.download_results(
                self._request.launch_plan.profile,
                record.job_id,
                staging,
            )
            self._result_verifier(staging, receipt)
            final_dir = self._result_stager.promote(
                staging,
                project_id=record.project_id,
                job_id=record.job_id,
            )
        except (RemoteTransportError, ResultPromotionError, OSError, ValueError) as exc:
            if isinstance(exc, RemoteTransportError):
                raise
            raise _ThreadFailure(
                FailureCode.LOCAL_VERIFY_FAILED,
                "回传结果未通过本机校验，已阻止落地",
            ) from exc
        completed = record.finish_collection(
            receipt=receipt,
            local_result_dir=final_dir,
        )
        self._persist(completed)
        self._set_state(JobStatus.SUCCEEDED, "远程训练完成，结果已校验并保存到本机")
        self._finish(True, "远程训练完成，结果已安全保存到本机")

    def _accept_remote_status(self, status) -> None:
        record = self._require_record()
        try:
            updated = record.with_runner_status(status)
        except RemoteTrainingJobError as exc:
            raise _ThreadFailure(
                FailureCode.RUNNER_PROTOCOL,
                "服务器返回的任务状态无法安全确认",
            ) from exc
        self._persist(updated)
        self._set_state(updated.last_status, _status_text(updated.last_status))

    def _advance(self, status: JobStatus, message: str) -> None:
        record = self._require_record()
        try:
            updated = record.advance_local(status)
        except RemoteTrainingJobError as exc:
            raise _ThreadFailure(
                FailureCode.LOCAL_VERIFY_FAILED,
                "远程训练本机状态无法安全推进",
            ) from exc
        self._persist(updated)
        self._set_state(status, message)

    def _persist(self, record: RemoteTrainingJobRecord) -> None:
        self._job_store.replace(record)
        self._record = record
        self.job_updated.emit(record)

    def _require_record(self) -> RemoteTrainingJobRecord:
        if self._record is None:
            raise _ThreadFailure(
                FailureCode.LOCAL_VERIFY_FAILED,
                "远程训练任务记录尚未创建",
            )
        return self._record

    def _handle_transport_interruption(self) -> None:
        if self._record is not None and self._record.last_status not in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            try:
                self._persist(self._record.mark_unknown())
                self._set_state(
                    JobStatus.UNKNOWN,
                    "连接中断；远程任务状态未知，未将它标记为完成或已停止",
                )
            except RemoteTrainingJobError:
                pass
        self._finish(False, "远程连接或传输中断；任务状态未知，未标记为完成或已停止")

    def _handle_failure(self, code: FailureCode, message: str) -> None:
        if self._record is not None and self._record.last_status not in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            try:
                self._persist(self._record.fail(code))
                self._set_state(JobStatus.FAILED, message)
            except RemoteTrainingJobError:
                pass
        self._finish(False, message)

    def _set_state(self, status: JobStatus, message: str) -> None:
        self.state_changed.emit(status.value)
        self.log_message.emit(message)

    def _finish(self, succeeded: bool, message: str) -> None:
        self.training_finished.emit(succeeded, message)


class RemoteConnectionTestThread(QThread):
    """用户主动点击后的只读预检线程，不上传、不建 job、不开始训练。"""

    preflight_finished = pyqtSignal(bool, str, object)

    def __init__(
        self,
        *,
        profile: RemoteTrainingProfile,
        backend: RemoteTrainingBackend,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._backend = backend

    def run(self) -> None:
        try:
            capabilities = self._backend.preflight(self._profile)
        except RemoteTransportError:
            self.preflight_finished.emit(
                False,
                "连接预检未通过。请检查网络、系统 OpenSSH/ssh-agent、服务器主机公钥和管理员配置。",
                None,
            )
            return
        except Exception:
            self.preflight_finished.emit(
                False,
                "连接预检未能安全完成；没有上传数据或创建远程任务。",
                None,
            )
            return
        tasks = "、".join(capabilities.supported_tasks)
        self.preflight_finished.emit(
            True,
            f"“{self._profile.name}”连接正常：支持 {tasks}；可用模型 "
            f"{len(capabilities.model_symbols)} 个。未上传数据或创建任务。",
            capabilities,
        )


@dataclass(frozen=True)
class _ThreadFailure(Exception):
    code: FailureCode
    message: str


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise _ThreadFailure(
            FailureCode.LOCAL_VERIFY_FAILED,
            "远程训练参数无效，已阻止上传",
        )
    return value


def _status_text(status: JobStatus) -> str:
    messages = {
        JobStatus.VERIFYING_UPLOAD: "服务器正在核验上传的数据…",
        JobStatus.STARTING: "服务器正在启动训练任务…",
        JobStatus.RUNNING: "服务器正在训练…",
        JobStatus.CANCEL_REQUESTED: "服务器正在确认停止请求…",
        JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION: "服务器训练完成，正在准备回传结果…",
        JobStatus.FAILED: "服务器已报告训练失败",
        JobStatus.CANCELLED: "服务器已确认任务停止",
    }
    return messages.get(status, "远程训练状态已更新")
