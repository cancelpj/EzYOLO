"""RemoteTrainingProfileStore 的直接可运行 QSettings 隔离测试。"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401  必须先隔离 QSettings 和数据库

from PyQt6.QtCore import QSettings

from _bootstrap import run_module_tests
from core.remote_training.profile_store import (
    REMOTE_TRAINING_PROFILES_KEY,
    RemoteTrainingProfileStore,
    RemoteTrainingProfileStoreError,
)
from core.remote_training.profiles import RemoteTrainingProfile, RemoteTrainingProfileError
from test_remote_training_profiles import _ed25519_key


def _settings() -> QSettings:
    settings = QSettings("EzYOLO", "RemoteTrainingProfilesTest")
    settings.clear()
    settings.sync()
    return settings


def _profile(index: int = 1) -> RemoteTrainingProfile:
    return RemoteTrainingProfile(
        name=f"训练服务器 {index}",
        host=f"gpu-{index}",
        port=22,
        username="trainer",
        remote_root=f"/srv/ezyolo/project_{index}",
        host_public_key=_ed25519_key(),
    )


def _stored_dicts(*profiles: RemoteTrainingProfile) -> list[dict[str, object]]:
    return [profile.to_dict() for profile in profiles]


def test_empty_store_reads_as_empty_list_and_fixed_key_is_used():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    assert store.read() == []
    assert store.list() == []
    assert REMOTE_TRAINING_PROFILES_KEY == "remote_training_profiles_v1"


def test_save_and_read_round_trip_through_json_qsettings_value():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    profiles = [_profile(1), _profile(2)]
    store.save(profiles)
    raw = settings.value(REMOTE_TRAINING_PROFILES_KEY, None)
    assert isinstance(raw, str)
    assert json.loads(raw) == _stored_dicts(*profiles)
    assert store.read() == profiles
    assert store.list() == profiles


def test_save_rejects_duplicate_ids_before_overwriting_existing_value():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    existing = _profile()
    store.save([existing])
    raw_before = settings.value(REMOTE_TRAINING_PROFILES_KEY, None)
    duplicate = RemoteTrainingProfile.from_dict(
        {
            **_profile(2).to_dict(),
            "id": existing.id,
        }
    )
    try:
        store.save([existing, duplicate])
    except RemoteTrainingProfileStoreError:
        pass
    else:
        raise AssertionError("duplicate ids must be rejected")
    assert settings.value(REMOTE_TRAINING_PROFILES_KEY, None) == raw_before


def test_malformed_json_and_non_list_values_fail_closed():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    for value in ("not-json", "{}", json.dumps({"password": "SECRET"}), 123, True):
        settings.setValue(REMOTE_TRAINING_PROFILES_KEY, value)
        settings.sync()
        try:
            store.read()
        except RemoteTrainingProfileStoreError as exc:
            assert "SECRET" not in str(exc)
        else:
            raise AssertionError("malformed store value must be rejected")


def test_malformed_list_item_unknown_secret_and_duplicate_stored_id_fail_closed():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    first = _profile(1)
    second = _profile(2)
    cases = [
        ["not an object"],
        [{"password": "SECRET"}],
        _stored_dicts(first, second, RemoteTrainingProfile.from_dict({**first.to_dict(), "id": first.id})),
        [{**first.to_dict(), "private_key_content": "SECRET"}],
    ]
    for value in cases:
        settings.setValue(REMOTE_TRAINING_PROFILES_KEY, json.dumps(value))
        settings.sync()
        try:
            store.read()
        except RemoteTrainingProfileStoreError as exc:
            assert "SECRET" not in str(exc)
        else:
            raise AssertionError("malformed list item must be rejected")


def test_delete_removes_only_matching_profile_and_returns_boolean():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    first = _profile(1)
    second = _profile(2)
    store.save([first, second])
    assert store.delete(first.id) is True
    assert store.read() == [second]
    raw_after_delete = settings.value(REMOTE_TRAINING_PROFILES_KEY, None)
    assert store.delete(first.id) is False
    assert settings.value(REMOTE_TRAINING_PROFILES_KEY, None) == raw_after_delete
    assert store.delete(second.id) is True
    assert store.read() == []


def test_delete_rejects_malformed_id():
    store = RemoteTrainingProfileStore(_settings())
    for profile_id in ("", "not-an-id", "A" * 32):
        try:
            store.delete(profile_id)
        except RemoteTrainingProfileError:
            pass
        else:
            raise AssertionError("delete must validate profile id")


def test_save_rejects_non_profile_item_without_writing():
    settings = _settings()
    store = RemoteTrainingProfileStore(settings)
    store.save([_profile()])
    raw_before = settings.value(REMOTE_TRAINING_PROFILES_KEY, None)
    try:
        store.save([_profile(), {"id": "not-a-profile"}])  # type: ignore[list-item]
    except RemoteTrainingProfileStoreError:
        pass
    else:
        raise AssertionError("non-profile items must be rejected")
    assert settings.value(REMOTE_TRAINING_PROFILES_KEY, None) == raw_before


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
