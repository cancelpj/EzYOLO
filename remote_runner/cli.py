"""固定 SSH action 的最小 CLI。

不接受 runner path、remote root、GPU、环境变量或任意 shell 命令；server.json 仅从
服务器普通账号自己的固定位置加载。
"""

from __future__ import annotations

import json
import sys
from typing import Any, Sequence

from remote_protocol.v1 import (
    REMOTE_PROTOCOL_VERSION,
    FailureCode,
    RunnerFailureEnvelope,
    validate_job_id,
)

from .config import ConfigError, load_server_config
from .jobs import Runner, RunnerFailure


_ACTIONS_WITH_JOB_ID = frozenset(
    {"verify-upload", "start", "status", "cancel", "collect-manifest"}
)
_ALL_ACTIONS = _ACTIONS_WITH_JOB_ID | {"preflight"}


def parse_action(argv: Sequence[str]) -> tuple[str, str | None]:
    args = list(argv)
    if not args or args[0] not in _ALL_ACTIONS:
        raise ValueError("仅允许 preflight、verify-upload、start、status、cancel、collect-manifest")
    action = args[0]
    if action == "preflight":
        if len(args) != 1:
            raise ValueError("preflight 不接受 job id 或其他参数")
        return action, None
    if len(args) != 2:
        raise ValueError(f"{action} 只接受一个 32 位小写十六进制 job id")
    return action, validate_job_id(args[1])


def dispatch(action: str, job_id: str | None, runner: Runner) -> dict[str, Any]:
    if action == "preflight":
        return runner.preflight().to_wire()
    assert job_id is not None
    if action == "verify-upload":
        return runner.verify_upload(job_id).to_wire()
    if action == "start":
        return runner.start(job_id).to_wire()
    if action == "status":
        return runner.status(job_id).to_wire()
    if action == "cancel":
        status, _cancelled = runner.cancel(job_id)
        if status is None:
            raise RunnerFailure(
                FailureCode.RUNNER_PROTOCOL,
                "cancel 未返回可验证的远程状态",
            )
        return status.to_wire()
    if action == "collect-manifest":
        _manifest, receipt = runner.collect_manifest(job_id)
        return receipt.to_wire()
    raise AssertionError("parse_action 已保证 action 是固定集合")


def _failure_payload(code: FailureCode, message: str) -> dict[str, Any]:
    """将可公开的 runner 失败固定成共享协议 envelope。"""
    return RunnerFailureEnvelope(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        ok=False,
        failure_code=code,
        message=message,
    ).to_wire()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        action, job_id = parse_action(sys.argv[1:] if argv is None else argv)
        runner = Runner(load_server_config())
        payload = dispatch(action, job_id, runner)
    except RunnerFailure as exc:
        payload = _failure_payload(exc.code, exc.message)
    except ConfigError:
        payload = _failure_payload(
            FailureCode.RUNNER_PROTOCOL,
            "服务器 runner 配置未通过安全校验",
        )
    except (ValueError, TypeError):
        payload = _failure_payload(
            FailureCode.RUNNER_PROTOCOL,
            "远程训练请求不符合 runner 协议",
        )
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
