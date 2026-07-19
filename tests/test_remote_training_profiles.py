"""RemoteTrainingProfile 的直接可运行安全校验测试。"""

from __future__ import annotations

import base64
import re
import struct

import _bootstrap  # noqa: F401  将仓库根目录加入 import path

from core.remote_training.profiles import (
    RemoteTrainingProfile,
    RemoteTrainingProfileError,
    validate_profile_dict,
)
from _bootstrap import run_module_tests


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _key_line(algorithm: str, blob: bytes, comment: str | None = None) -> str:
    encoded = base64.b64encode(blob).decode("ascii")
    return f"{algorithm} {encoded}" + (f" {comment}" if comment else "")


def _ed25519_key() -> str:
    return _key_line("ssh-ed25519", _ssh_string(b"ssh-ed25519") + _ssh_string(b"e" * 32))


def _ecdsa_key(algorithm: str) -> str:
    curve = algorithm.removeprefix("ecdsa-sha2-")
    point_length = {"nistp256": 65, "nistp384": 97, "nistp521": 133}[curve]
    point = b"\x04" + b"p" * (point_length - 1)
    blob = _ssh_string(algorithm.encode()) + _ssh_string(curve.encode()) + _ssh_string(point)
    return _key_line(algorithm, blob)


def _rsa_key(algorithm: str) -> str:
    blob = _ssh_string(b"ssh-rsa") + _ssh_string(b"\x01\x00\x01") + _ssh_string(b"n" * 256)
    return _key_line(algorithm, blob)


def _valid_values() -> dict[str, object]:
    return {
        "name": "训练服务器",
        "host": "train-gpu-01",
        "port": 22,
        "username": "trainer_1",
        "remote_root": "/srv/ezyolo/project_1",
        "host_public_key": _ed25519_key(),
    }


def _profile(**overrides: object) -> RemoteTrainingProfile:
    values = _valid_values()
    values.update(overrides)
    return RemoteTrainingProfile(**values)  # type: ignore[arg-type]


def _assert_rejected(**overrides: object) -> None:
    try:
        _profile(**overrides)
    except RemoteTrainingProfileError:
        return
    raise AssertionError(f"expected rejection for {next(iter(overrides))}")


def test_valid_profile_generates_uuid4_hex_and_is_immutable():
    profile = _profile()
    assert re.fullmatch(r"[a-f0-9]{32}", profile.id)
    assert profile.to_dict()["id"] == profile.id
    try:
        profile.name = "changed"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("profile must be immutable")


def test_constructor_accepts_only_valid_existing_id_and_from_dict_round_trips():
    source = _profile().to_dict()
    restored = RemoteTrainingProfile.from_dict(source)
    assert restored == RemoteTrainingProfile.from_dict(restored.to_dict())
    assert restored.id == source["id"]
    assert validate_profile_dict(source) == restored
    for bad_id in ("", "a" * 31, "g" * 32, "A" * 32, 123):
        _assert_rejected(id=bad_id)
    missing = dict(source)
    del missing["id"]
    try:
        RemoteTrainingProfile.from_dict(missing)
    except RemoteTrainingProfileError:
        pass
    else:
        raise AssertionError("missing id must be rejected when loading")


def test_name_rejects_blank_and_control_characters():
    for name in ("", "   ", "\t", "name\n", "name\x00"):
        _assert_rejected(name=name)
    assert _profile(name="display name")


def test_host_accepts_alias_hostname_ipv4_and_ipv6_but_rejects_shellish_values():
    for host in ("gpu_alias", "train-gpu.example", "192.0.2.10", "2001:db8::10", "::1"):
        assert _profile(host=host).host == host
    for host in (
        "",
        " -host",
        "-host",
        "host name",
        "user@host",
        "host'quote",
        'host"quote',
        "host`cmd`",
        "host$HOME",
        "host;rm",
        "host\n",
        "2001:db8::bad::1",
        "999.999.1.1",
        "/tmp/host",
        "[::1]",
    ):
        _assert_rejected(host=host)


def test_port_requires_real_int_in_range():
    for port in (1, 22, 65535):
        assert _profile(port=port).port == port
    for port in (True, False, 0, -1, 65536, "22", 22.0):
        _assert_rejected(port=port)


def test_username_rejects_root_and_invalid_forms():
    assert _profile(username="rootless").username == "rootless"
    for username in ("root", "Root", "", "1trainer", "trainer!", "trainer name", "a" * 33):
        _assert_rejected(username=username)


def test_remote_root_uses_posix_rules_and_rejects_noncanonical_segments():
    for root in ("/srv/ezyolo", "/opt/train.v1", "/a_b/c-1"):
        assert _profile(remote_root=root).remote_root == root
    for root in (
        "/",
        "",
        "relative/path",
        "~/ezyolo",
        "/srv//ezyolo",
        "/srv/",
        "/srv/../other",
        "/srv/./other",
        "/srv/-project",
        "/srv/project name",
        "/srv/project\n",
        "/srv\\project",
        "/root",
        "/root/ezyolo",
    ):
        _assert_rejected(remote_root=root)


def test_host_public_key_accepts_supported_complete_key_formats():
    assert _profile(host_public_key=_ed25519_key()).host_public_key
    assert _profile(host_public_key=_key_line("ssh-ed25519", _ssh_string(b"ssh-ed25519") + _ssh_string(b"e" * 32), "host key comment"))
    for algorithm in (
        "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521",
        "ssh-rsa",
        "rsa-sha2-256",
        "rsa-sha2-512",
    ):
        key = _ecdsa_key(algorithm) if algorithm.startswith("ecdsa-") else _rsa_key(algorithm)
        assert _profile(host_public_key=key).host_public_key == key


def test_host_public_key_rejects_fingerprint_incomplete_base64_and_private_material():
    valid = _ed25519_key()
    algorithm, _ = valid.split(" ", 1)
    for key in (
        "SHA256:abc123",
        f"{algorithm} not-base64!",
        f"{algorithm} AAAA",
        f"{algorithm} {base64.b64encode(_ssh_string(b'wrong') + _ssh_string(b'x')).decode()}",
        f"{algorithm} {base64.b64encode(_ssh_string(b'ssh-ed25519') + _ssh_string(b'x' * 31)).decode()}",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "ssh-ed25519\nAAAA",
        "ssh-ed25519 AAAA\x00comment",
    ):
        _assert_rejected(host_public_key=key)


def test_unknown_fields_and_secret_fields_are_rejected_without_echoing_secret_value():
    source = _profile().to_dict()
    secret = "DO-NOT-ECHO-THIS-SECRET"
    for field in ("password", "private_key", "private_key_content", "passphrase", "token", "api_key", "other_future_field"):
        candidate = dict(source)
        candidate[field] = secret
        try:
            RemoteTrainingProfile.from_dict(candidate)
        except RemoteTrainingProfileError as exc:
            assert secret not in str(exc)
        else:
            raise AssertionError(f"unknown field must be rejected: {field}")


def test_to_dict_contains_exactly_the_frozen_schema():
    assert set(_profile().to_dict()) == {
        "id",
        "name",
        "host",
        "port",
        "username",
        "remote_root",
        "host_public_key",
    }


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
