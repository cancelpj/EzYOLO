"""EzYOLO 远程训练共享协议。

这个包只包含客户端与服务器 runner 共用的纯 Python wire contract。它不依赖
桌面界面、QSettings、数据库、Ultralytics 或网络传输实现。
"""

from .v1 import (
    REMOTE_PROTOCOL_VERSION,
    DatasetManifest,
    FailureCode,
    JobSpec,
    JobStatus,
    ManifestEntry,
    ProtocolStateError,
    ProtocolValidationError,
    RunnerFailureEnvelope,
    ResultManifest,
    ResultReceipt,
    ServerCapabilities,
    advance_local_status,
    begin_attach,
    complete_collection,
    is_terminal_status,
    mark_unknown,
    resolve_runner_status,
    validate_job_id,
)

__all__ = [
    "REMOTE_PROTOCOL_VERSION",
    "DatasetManifest",
    "FailureCode",
    "JobSpec",
    "JobStatus",
    "ManifestEntry",
    "ProtocolStateError",
    "ProtocolValidationError",
    "RunnerFailureEnvelope",
    "ResultManifest",
    "ResultReceipt",
    "ServerCapabilities",
    "advance_local_status",
    "begin_attach",
    "complete_collection",
    "is_terminal_status",
    "mark_unknown",
    "resolve_runner_status",
    "validate_job_id",
]
