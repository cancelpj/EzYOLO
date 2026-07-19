"""训练位置解析：把已经校验的训练表单分为本机或远程启动计划。

这里不启动线程、不会创建数据快照，更不会执行 SSH、rsync 或训练框架。真正的
远程 executor 只能在后续后台层显式注册后使用 RemoteLaunchPlan。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .profile_store import RemoteTrainingProfileStoreError
from .profiles import RemoteTrainingProfile


REMOTE_TARGET_PREFIX = "remote:"
REMOTE_SUPPORTED_TASKS = frozenset({"detect", "segment"})


class TrainingLaunchError(ValueError):
    """训练位置或已校验训练配置不能形成启动计划。"""


class RemoteTargetValidationError(TrainingLaunchError):
    """远程 target/profile/config 无法安全解析。"""


class RemoteExecutionUnavailable(TrainingLaunchError):
    """代码尚未注册远程执行器，不能把远程选择悄悄降级为本机训练。"""


class ProfileStoreLike(Protocol):
    def list(self) -> list[RemoteTrainingProfile]: ...


@dataclass(frozen=True)
class LocalLaunchPlan:
    project_id: int
    task_type: str
    runtime_config: Mapping[str, Any]


@dataclass(frozen=True)
class RemoteLaunchPlan:
    project_id: int
    task_type: str
    model_symbol: str
    profile: RemoteTrainingProfile
    runtime_config: Mapping[str, Any]


class TrainingLaunchController:
    """唯一的本机/远程分叉点；永不以隐式本机回退掩盖远程失败。"""

    def __init__(
        self,
        profile_store: ProfileStoreLike,
        *,
        remote_executor_registered: bool = False,
    ) -> None:
        self._profile_store = profile_store
        self._remote_executor_registered = remote_executor_registered

    def resolve(
        self,
        *,
        project_id: int,
        runtime_config: Mapping[str, Any],
        target: str,
    ) -> LocalLaunchPlan | RemoteLaunchPlan:
        """生成纯数据计划；不创建线程、不调用网络、也不写入本机数据集。"""
        _validate_project_id(project_id)
        config = _normalise_runtime_config(runtime_config)
        task_type = _required_task(config)

        if target == "local":
            return LocalLaunchPlan(
                project_id=project_id,
                task_type=task_type,
                runtime_config=MappingProxyType(config),
            )
        if not isinstance(target, str) or not target.startswith(REMOTE_TARGET_PREFIX):
            raise RemoteTargetValidationError("训练位置无效，请重新选择本机或已保存服务器")

        profile_id = target.removeprefix(REMOTE_TARGET_PREFIX)
        if not _is_profile_id(profile_id):
            raise RemoteTargetValidationError("远程服务器档案标识无效，请到设置页重新选择")
        if task_type not in REMOTE_SUPPORTED_TASKS:
            raise RemoteTargetValidationError(
                "远程训练首版只支持目标检测和实例分割，请改用本机训练"
            )

        profile = self._find_profile(profile_id)
        remote_config = dict(config)
        remote_config.pop("device", None)
        return RemoteLaunchPlan(
            project_id=project_id,
            task_type=task_type,
            model_symbol=_model_symbol(config),
            profile=profile,
            runtime_config=MappingProxyType(remote_config),
        )

    def resolve_for_execution(
        self,
        *,
        project_id: int,
        runtime_config: Mapping[str, Any],
        target: str,
    ) -> LocalLaunchPlan | RemoteLaunchPlan:
        """在真正启动之前再确认远程执行器已注册。"""
        plan = self.resolve(
            project_id=project_id,
            runtime_config=runtime_config,
            target=target,
        )
        if isinstance(plan, RemoteLaunchPlan) and not self._remote_executor_registered:
            raise RemoteExecutionUnavailable(
                "远程训练后台尚未准备好；已阻止启动，不会自动改成本机训练"
            )
        return plan

    def _find_profile(self, profile_id: str) -> RemoteTrainingProfile:
        try:
            profiles = self._profile_store.list()
        except RemoteTrainingProfileStoreError as exc:
            raise RemoteTargetValidationError(
                "远程服务器档案无法安全读取，请到设置页检查后再试"
            ) from exc
        except Exception as exc:
            raise RemoteTargetValidationError(
                "远程服务器档案无法读取，请到设置页检查后再试"
            ) from exc
        for profile in profiles:
            if profile.id == profile_id:
                return profile
        raise RemoteTargetValidationError(
            "选择的远程服务器已不存在或已删除；请明确改选本机或其他服务器"
        )


def _validate_project_id(project_id: object) -> None:
    if type(project_id) is not int or project_id <= 0:
        raise TrainingLaunchError("项目无效，不能启动训练")


def _normalise_runtime_config(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime_config, Mapping):
        raise TrainingLaunchError("训练配置无效")
    return dict(runtime_config)


def _required_task(config: Mapping[str, Any]) -> str:
    task = config.get("task")
    if not isinstance(task, str) or not task:
        raise TrainingLaunchError("训练配置缺少任务类型")
    return task


def _model_symbol(config: Mapping[str, Any]) -> str:
    explicit = config.get("model_symbol")
    if isinstance(explicit, str) and explicit:
        return explicit
    prefix = config.get("model_prefix")
    size = config.get("model_size")
    if not isinstance(prefix, str) or not prefix or not isinstance(size, str) or not size:
        raise RemoteTargetValidationError("远程训练配置缺少可映射的模型符号名")
    symbol = prefix + size
    if not all(char.isascii() and (char.isalnum() or char in "._-") for char in symbol):
        raise RemoteTargetValidationError("远程训练模型符号名无效")
    return symbol


def _is_profile_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(char in "0123456789abcdef" for char in value)
    )
