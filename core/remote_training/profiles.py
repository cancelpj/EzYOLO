"""远程训练服务器档案的数据模型与输入校验。

本模块不执行 SSH、路径访问或任何远程命令。它只负责把外部数据校验成
一个不可变的 ``RemoteTrainingProfile``。
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import re
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class RemoteTrainingProfileError(ValueError):
    """档案数据不符合 v1 schema 或安全约束时抛出的异常。"""


_ALLOWED_FIELDS = frozenset(
    {
        "id",
        "name",
        "host",
        "port",
        "username",
        "remote_root",
        "host_public_key",
    }
)
_ID_RE = re.compile(r"[a-f0-9]{32}")
_HOST_LABEL_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?"
)
_REMOTE_ROOT_RE = re.compile(r"/[A-Za-z0-9._/-]+")
_KEY_LINE_RE = re.compile(
    r"(?P<algorithm>ssh-ed25519|ecdsa-sha2-nistp256|ecdsa-sha2-nistp384|"
    r"ecdsa-sha2-nistp521|rsa-sha2-256|rsa-sha2-512|ssh-rsa) "
    r"(?P<payload>[A-Za-z0-9+/]+={0,2})"
    r"(?: (?P<comment>\S(?:.*\S)?))?"
)


def _invalid(field: str) -> RemoteTrainingProfileError:
    """返回不包含用户输入值的稳定错误。"""

    return RemoteTrainingProfileError(f"invalid remote training profile field: {field}")


def _validate_text(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise _invalid(field)
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise _invalid(field)
    return value


def _validate_id(value: Any) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise _invalid("id")
    return value


def _validate_name(value: Any) -> str:
    value = _validate_text("name", value)
    if not value.strip():
        raise _invalid("name")
    return value


def _validate_host(value: Any) -> str:
    value = _validate_text("host", value)
    if not value or value.startswith("-") or not value.isascii():
        raise _invalid("host")
    if any(char.isspace() for char in value):
        raise _invalid("host")

    if ":" in value:
        try:
            ipaddress.IPv6Address(value)
        except ValueError as exc:
            raise _invalid("host") from exc
        return value

    if "." in value and all(char.isdigit() or char == "." for char in value):
        try:
            ipaddress.IPv4Address(value)
        except ValueError as exc:
            raise _invalid("host") from exc
        return value

    labels = value.split(".")
    if any(_HOST_LABEL_RE.fullmatch(label) is None for label in labels):
        raise _invalid("host")
    return value


def _validate_port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("port")
    if not 1 <= value <= 65535:
        raise _invalid("port")
    return value


def _validate_username(value: Any) -> str:
    value = _validate_text("username", value)
    if value == "root" or re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", value) is None:
        raise _invalid("username")
    return value


def _validate_remote_root(value: Any) -> str:
    value = _validate_text("remote_root", value)
    if _REMOTE_ROOT_RE.fullmatch(value) is None or value == "/":
        raise _invalid("remote_root")
    if value == "/root" or value.startswith("/root/"):
        raise _invalid("remote_root")

    segments = value.split("/")
    if segments[0] != "" or any(not segment for segment in segments[1:]):
        raise _invalid("remote_root")
    if any(segment in {".", ".."} for segment in segments[1:]):
        raise _invalid("remote_root")
    if any(segment.startswith("-") for segment in segments[1:]):
        raise _invalid("remote_root")
    return value


def _read_ssh_string(payload: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(payload):
        raise ValueError
    length = int.from_bytes(payload[offset : offset + 4], "big")
    start = offset + 4
    end = start + length
    if end > len(payload):
        raise ValueError
    return payload[start:end], end


def _read_ascii_ssh_string(payload: bytes, offset: int) -> tuple[str, int]:
    value, offset = _read_ssh_string(payload, offset)
    try:
        return value.decode("ascii"), offset
    except UnicodeDecodeError as exc:
        raise ValueError from exc


def _validate_key_blob(algorithm: str, payload: bytes) -> None:
    try:
        blob_algorithm, offset = _read_ascii_ssh_string(payload, 0)
        if algorithm.startswith("rsa-sha2-"):
            if blob_algorithm != "ssh-rsa":
                raise ValueError
            exponent, offset = _read_ssh_string(payload, offset)
            modulus, offset = _read_ssh_string(payload, offset)
            if not exponent or not modulus:
                raise ValueError
            if int.from_bytes(exponent, "big") <= 0 or int.from_bytes(modulus, "big") <= 0:
                raise ValueError
        elif algorithm == "ssh-rsa":
            if blob_algorithm != "ssh-rsa":
                raise ValueError
            exponent, offset = _read_ssh_string(payload, offset)
            modulus, offset = _read_ssh_string(payload, offset)
            if not exponent or not modulus:
                raise ValueError
            if int.from_bytes(exponent, "big") <= 0 or int.from_bytes(modulus, "big") <= 0:
                raise ValueError
        elif algorithm == "ssh-ed25519":
            if blob_algorithm != algorithm:
                raise ValueError
            key, offset = _read_ssh_string(payload, offset)
            if len(key) != 32:
                raise ValueError
        else:
            if blob_algorithm != algorithm:
                raise ValueError
            curve, offset = _read_ascii_ssh_string(payload, offset)
            point, offset = _read_ssh_string(payload, offset)
            expected_curve = algorithm.removeprefix("ecdsa-sha2-")
            expected_point_length = {
                "nistp256": 65,
                "nistp384": 97,
                "nistp521": 133,
            }[expected_curve]
            if curve != expected_curve or len(point) != expected_point_length or point[:1] != b"\x04":
                raise ValueError
        if offset != len(payload):
            raise ValueError
    except (KeyError, ValueError, OverflowError):
        raise _invalid("host_public_key") from None


def _validate_host_public_key(value: Any) -> str:
    value = _validate_text("host_public_key", value)
    if not value or value.startswith("-"):
        raise _invalid("host_public_key")

    match = _KEY_LINE_RE.fullmatch(value)
    if match is None:
        raise _invalid("host_public_key")
    algorithm = match.group("algorithm")
    encoded = match.group("payload")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid("host_public_key") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != encoded:
        raise _invalid("host_public_key")
    _validate_key_blob(algorithm, decoded)
    return value


@dataclass(frozen=True, init=False)
class RemoteTrainingProfile:
    """经过 v1 安全校验的不可变远程训练服务器档案。

    新档案不要求调用者提供 ID，模型会生成 UUID4 hex。加载已保存档案时，
    ``from_dict`` 会校验并保留其中的合法 ID。
    """

    id: str
    name: str
    host: str
    port: int
    username: str
    remote_root: str
    host_public_key: str

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        username: str,
        remote_root: str,
        host_public_key: str,
        *,
        id: str | None = None,
    ) -> None:
        profile_id = uuid.uuid4().hex if id is None else _validate_id(id)
        if _ID_RE.fullmatch(profile_id) is None:
            raise _invalid("id")
        object.__setattr__(self, "id", profile_id)
        object.__setattr__(self, "name", _validate_name(name))
        object.__setattr__(self, "host", _validate_host(host))
        object.__setattr__(self, "port", _validate_port(port))
        object.__setattr__(self, "username", _validate_username(username))
        object.__setattr__(self, "remote_root", _validate_remote_root(remote_root))
        object.__setattr__(self, "host_public_key", _validate_host_public_key(host_public_key))

    def to_dict(self) -> dict[str, Any]:
        """返回只包含冻结 v1 七字段的新字典。"""

        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "remote_root": self.remote_root,
            "host_public_key": self.host_public_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RemoteTrainingProfile":
        """严格校验并从字典恢复一个档案。"""

        if not isinstance(value, Mapping):
            raise RemoteTrainingProfileError("remote training profile must be an object")
        unknown = set(value) - _ALLOWED_FIELDS
        missing = _ALLOWED_FIELDS - set(value)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise RemoteTrainingProfileError(f"unknown remote training profile fields: {names}")
        if missing:
            names = ", ".join(sorted(missing))
            raise RemoteTrainingProfileError(f"missing remote training profile fields: {names}")
        return cls(
            name=value["name"],
            host=value["host"],
            port=value["port"],
            username=value["username"],
            remote_root=value["remote_root"],
            host_public_key=value["host_public_key"],
            id=value["id"],
        )


def validate_profile_dict(value: Mapping[str, Any]) -> RemoteTrainingProfile:
    """显式验证入口，等价于 ``RemoteTrainingProfile.from_dict``。"""

    return RemoteTrainingProfile.from_dict(value)


__all__ = [
    "RemoteTrainingProfile",
    "RemoteTrainingProfileError",
    "validate_profile_dict",
]
