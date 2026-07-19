"""远程训练服务器档案的 QSettings 存储适配层。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from .profiles import RemoteTrainingProfile, RemoteTrainingProfileError, _validate_id


REMOTE_TRAINING_PROFILES_KEY = "remote_training_profiles_v1"


class RemoteTrainingProfileStoreError(RemoteTrainingProfileError):
    """QSettings 中的档案数据无法安全恢复时抛出的异常。"""


class RemoteTrainingProfileStore:
    """只负责固定 QSettings key 的档案读写，不执行任何远程操作。

    ``read``/``list`` 对未设置的 key 返回空列表；对非 JSON 列表、坏列表项、
    未知字段、非法档案或重复 ID 一律 fail closed，抛出
    ``RemoteTrainingProfileStoreError``，不会静默跳过潜在秘密字段。
    """

    def __init__(self, settings: Any | None = None) -> None:
        if settings is None:
            from PyQt6.QtCore import QSettings

            settings = QSettings("EzYOLO", "Settings")
        self._settings = settings

    def read(self) -> list[RemoteTrainingProfile]:
        """读取并严格验证全部已保存档案。"""

        raw = self._settings.value(REMOTE_TRAINING_PROFILES_KEY, None)
        if raw is None:
            return []

        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RemoteTrainingProfileStoreError(
                    "stored remote training profiles are not valid JSON"
                ) from exc
        elif isinstance(raw, list):
            decoded = raw
        else:
            raise RemoteTrainingProfileStoreError(
                "stored remote training profiles must be a list"
            )

        if not isinstance(decoded, list):
            raise RemoteTrainingProfileStoreError(
                "stored remote training profiles must be a list"
            )

        profiles: list[RemoteTrainingProfile] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(decoded):
            if not isinstance(item, Mapping):
                raise RemoteTrainingProfileStoreError(
                    f"stored remote training profile entry {index} is not an object"
                )
            try:
                profile = RemoteTrainingProfile.from_dict(item)
            except RemoteTrainingProfileError as exc:
                raise RemoteTrainingProfileStoreError(
                    f"stored remote training profile entry {index} is invalid: {exc}"
                ) from exc
            if profile.id in seen_ids:
                raise RemoteTrainingProfileStoreError(
                    f"stored remote training profiles contain duplicate id at entry {index}"
                )
            seen_ids.add(profile.id)
            profiles.append(profile)
        return profiles

    def list(self) -> list[RemoteTrainingProfile]:
        """``read`` 的命名别名，便于调用方按列表语义使用。"""

        return self.read()

    def save(self, profiles: Iterable[RemoteTrainingProfile]) -> None:
        """只在全部输入合法且 ID 唯一时写入一次 JSON 值。"""

        entries = list(profiles)
        seen_ids: set[str] = set()
        serialized: list[dict[str, Any]] = []
        for index, profile in enumerate(entries):
            if not isinstance(profile, RemoteTrainingProfile):
                raise RemoteTrainingProfileStoreError(
                    f"remote training profile entry {index} is not a profile"
                )
            if profile.id in seen_ids:
                raise RemoteTrainingProfileStoreError(
                    f"remote training profiles contain duplicate id at entry {index}"
                )
            seen_ids.add(profile.id)
            serialized.append(profile.to_dict())

        payload = json.dumps(
            serialized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._settings.setValue(REMOTE_TRAINING_PROFILES_KEY, payload)
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()

    def delete(self, profile_id: str) -> bool:
        """删除匹配 ID 的一个档案；没有匹配项时不写入并返回 False。"""

        profile_id = _validate_id(profile_id)
        profiles = self.read()
        remaining = [profile for profile in profiles if profile.id != profile_id]
        if len(remaining) == len(profiles):
            return False
        self.save(remaining)
        return True


__all__ = [
    "REMOTE_TRAINING_PROFILES_KEY",
    "RemoteTrainingProfileStore",
    "RemoteTrainingProfileStoreError",
]
