"""服务器本地 ``server.json`` 的严格加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pwd
import stat
from typing import Any, Mapping

from remote_protocol.v1 import REMOTE_PROTOCOL_VERSION, ServerCapabilities


DEFAULT_CONFIG_PATH = (
    Path(pwd.getpwuid(os.getuid()).pw_dir) / ".config" / "ezyolo-remote" / "server.json"
)
MAX_CONFIG_BYTES = 1024 * 1024

_CONFIG_FIELDS = frozenset(
    {
        "protocol_version",
        "canonical_remote_root",
        "model_allowlist",
        "runtime",
        "max_epochs",
        "max_runtime_seconds",
        "max_payload_bytes",
        "max_jobs_bytes",
        "max_results_bytes",
        "min_free_disk_bytes",
        "single_task",
    }
)
_RUNTIME_FIELDS = frozenset({"launcher"})


class ConfigError(ValueError):
    """服务器管理员配置不安全、不完整或与协议不兼容。"""


@dataclass(frozen=True)
class RuntimeConfig:
    """管理员预置的训练 launcher；runner 只向它传递固定参数。"""

    launcher: Path


@dataclass(frozen=True)
class ServerConfig:
    protocol_version: int
    canonical_remote_root: Path
    model_allowlist: dict[str, Path]
    runtime: RuntimeConfig
    max_epochs: int
    max_runtime_seconds: int
    max_payload_bytes: int
    max_jobs_bytes: int
    max_results_bytes: int
    min_free_disk_bytes: int
    single_task: bool

    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            protocol_version=self.protocol_version,
            canonical_remote_root=str(self.canonical_remote_root),
            supported_tasks=("detect", "segment"),
            model_symbols=tuple(sorted(self.model_allowlist)),
            max_epochs=self.max_epochs,
            max_runtime_seconds=self.max_runtime_seconds,
            max_payload_bytes=self.max_payload_bytes,
            max_result_bytes=self.max_results_bytes,
        )


def load_server_config(path: Path | None = None) -> ServerConfig:
    """只读取当前普通账号拥有的固定本地配置文件。

    CLI 不暴露配置路径参数。可选 ``path`` 仅供离线单元测试和嵌入式调用使用，
    不能由 ``remote_runner.cli`` 的命令行传入。
    """

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise ConfigError("runner 必须由普通 Linux 账号运行，不能以 root 运行")
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    _validate_private_regular_file(config_path, "server.json")
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError("无法读取 server.json") from exc
    if len(raw) > MAX_CONFIG_BYTES:
        raise ConfigError("server.json 过大")
    return server_config_from_mapping(_decode_json_object(raw, "server.json"))


def server_config_from_mapping(payload: Mapping[str, Any]) -> ServerConfig:
    """从已解析的 JSON 构建配置，供纯离线测试复用。"""

    data = _require_exact_fields(payload, _CONFIG_FIELDS, "server.json")
    if data["protocol_version"] != REMOTE_PROTOCOL_VERSION:
        raise ConfigError("runner 与 remote_protocol 协议版本不匹配")

    root = _validate_existing_directory(data["canonical_remote_root"], "canonical_remote_root")
    allowlist = _validate_allowlist(data["model_allowlist"])
    runtime = _validate_runtime(data["runtime"])
    max_epochs = _positive_int(data["max_epochs"], "max_epochs")
    max_runtime_seconds = _positive_int(data["max_runtime_seconds"], "max_runtime_seconds")
    max_payload_bytes = _positive_int(data["max_payload_bytes"], "max_payload_bytes")
    max_jobs_bytes = _positive_int(data["max_jobs_bytes"], "max_jobs_bytes")
    max_results_bytes = _positive_int(data["max_results_bytes"], "max_results_bytes")
    min_free_disk_bytes = _nonnegative_int(
        data["min_free_disk_bytes"], "min_free_disk_bytes"
    )
    if type(data["single_task"]) is not bool:
        raise ConfigError("single_task 必须是布尔值")

    return ServerConfig(
        protocol_version=data["protocol_version"],
        canonical_remote_root=root,
        model_allowlist=allowlist,
        runtime=runtime,
        max_epochs=max_epochs,
        max_runtime_seconds=max_runtime_seconds,
        max_payload_bytes=max_payload_bytes,
        max_jobs_bytes=max_jobs_bytes,
        max_results_bytes=max_results_bytes,
        min_free_disk_bytes=min_free_disk_bytes,
        single_task=data["single_task"],
    )


def _validate_allowlist(value: object) -> dict[str, Path]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError("model_allowlist 必须是非空对象")
    allowlist: dict[str, Path] = {}
    for symbol, raw_path in value.items():
        if not isinstance(symbol, str) or not symbol or symbol in allowlist:
            raise ConfigError("model_allowlist 的模型符号不合法")
        # 复用协议对象的模型符号校验，避免两套符号规则漂移。
        try:
            ServerCapabilities(
                protocol_version=REMOTE_PROTOCOL_VERSION,
                canonical_remote_root="/srv/ezyolo/validation",
                supported_tasks=("detect",),
                model_symbols=(symbol,),
                max_epochs=1,
                max_runtime_seconds=1,
                max_payload_bytes=1,
                max_result_bytes=1,
            )
        except ValueError as exc:
            raise ConfigError("model_allowlist 的模型符号不合法") from exc
        allowlist[symbol] = _validate_existing_regular_file(
            raw_path, f"model_allowlist[{symbol!r}]", executable=False
        )
    return allowlist


def _validate_runtime(value: object) -> RuntimeConfig:
    data = _require_exact_fields(value, _RUNTIME_FIELDS, "runtime")
    return RuntimeConfig(
        launcher=_validate_existing_regular_file(
            data["launcher"], "runtime.launcher", executable=True
        )
    )


def _validate_private_regular_file(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ConfigError(f"{label} 必须使用绝对路径")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ConfigError(f"{label} 不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ConfigError(f"{label} 必须是普通文件，不能是软链接")
    if info.st_uid != os.getuid():
        raise ConfigError(f"{label} 必须属于当前普通账号")
    if info.st_mode & 0o022:
        raise ConfigError(f"{label} 不能允许组或其他账号写入")


def _validate_existing_directory(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or value in {"/", "/root"}:
        raise ConfigError(f"{label} 必须是非 root 的绝对目录")
    if value != os.path.normpath(value):
        raise ConfigError(f"{label} 必须是 canonical path")
    raw = Path(value)
    if ".." in raw.parts or "." in raw.parts or any(not part for part in raw.parts[1:]):
        raise ConfigError(f"{label} 不规范")
    if str(raw).startswith("/root/"):
        raise ConfigError(f"{label} 不能位于 /root")
    try:
        info = raw.lstat()
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"{label} 必须是已存在目录") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ConfigError(f"{label} 必须是普通目录，不能是软链接")
    if resolved != raw:
        raise ConfigError(f"{label} 必须是 canonical path")
    if info.st_uid != os.getuid():
        raise ConfigError(f"{label} 必须属于当前普通账号")
    if info.st_mode & 0o022:
        raise ConfigError(f"{label} 不能允许组或其他账号写入")
    return raw


def _validate_existing_regular_file(
    value: object, label: str, *, executable: bool
) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ConfigError(f"{label} 必须是绝对普通文件")
    if value != os.path.normpath(value):
        raise ConfigError(f"{label} 必须是 canonical path")
    raw = Path(value)
    if ".." in raw.parts or str(raw) == "/root" or str(raw).startswith("/root/"):
        raise ConfigError(f"{label} 路径不安全")
    try:
        info = raw.lstat()
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"{label} 指向的文件不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or resolved != raw:
        raise ConfigError(f"{label} 必须是普通文件，不能是软链接")
    if info.st_mode & 0o022:
        raise ConfigError(f"{label} 不能允许组或其他账号写入")
    if executable and not (info.st_mode & stat.S_IXUSR):
        raise ConfigError(f"{label} 必须对当前账号可执行")
    return raw


def _decode_json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(f"{label} 不是安全 JSON 对象") from exc
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} 必须是对象")
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 包含重复字段")
        result[key] = value
    return result


def _require_exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} 必须是对象")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append("缺少 " + ", ".join(missing))
        if extra:
            detail.append("包含不允许字段 " + ", ".join(extra))
        raise ConfigError(f"{label} 字段不匹配：{'；'.join(detail)}")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{label} 必须是正整数")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{label} 必须是非负整数")
    return value
