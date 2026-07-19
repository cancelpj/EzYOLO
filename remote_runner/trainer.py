"""由 ``start`` 派生的受控训练 supervisor。

该模块不导入 Ultralytics；管理员在 server.json 中预置 launcher。supervisor 自身
以独立进程组运行，launcher 子进程继承该组，因此 cancel/watchdog 只能对已核验
的 job PGID 止损。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess
import sys
import time
from typing import Callable, Sequence

from remote_protocol.v1 import REMOTE_PROTOCOL_VERSION, FailureCode, JobStatus, RemoteStatus, validate_job_id

from .config import load_server_config
from .jobs import ProcessIdentity, Runner, RunnerFailure, controlled_environment


@dataclass
class Watchdog:
    """只报告/止损当前 job；时间、文件系统和信号均可在离线测试中 fake。"""

    runner: Runner
    job_id: str
    clock: Callable[[], float] = time.monotonic
    _started: float = field(init=False)

    def __post_init__(self) -> None:
        self.job_id = validate_job_id(self.job_id)
        self._started = self.clock()

    def check(self) -> FailureCode | None:
        failure = self.runner.watchdog_failure(self.job_id, self.clock() - self._started)
        if failure is None:
            return None
        self.runner.stop_verified_job(
            self.job_id,
            failure,
            f"训练 watchdog 触发 {failure.value}",
        )
        return failure


def launcher_command(runner: Runner, job_id: str) -> Sequence[str]:
    """构造唯一固定的管理员 launcher 命令，不读取客户端命令或环境变量。"""

    spec, _manifest = runner._load_verified_job(job_id)
    return (
        str(runner.config.runtime.launcher),
        "--job-id",
        job_id,
        "--task",
        spec.task_type,
        "--model",
        str(runner.config.model_allowlist[spec.model_symbol]),
        "--data",
        str(runner.paths.job_dir(job_id) / "data.yaml"),
        "--epochs",
        str(spec.epochs),
        "--batch",
        str(spec.batch),
        "--imgsz",
        str(spec.imgsz),
        "--results-dir",
        str(runner.paths.result_dir(job_id)),
    )


def supervise_job(
    job_id: str,
    *,
    runner: Runner | None = None,
    spawner: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """运行一个已有 RUNNING job，直到 launcher 退出或 watchdog 止损。"""

    job_id = validate_job_id(job_id)
    runner = runner or Runner(load_server_config())
    # 父进程必须先 Popen supervisor，才可取得其 PID/PGID 并把 single-task 锁原子
    # 落地。短暂等待这一事实可避免子进程抢在父进程写 RUNNING 之前退出。
    own_identity: ProcessIdentity | None = None
    for _attempt in range(50):
        status = runner.read_status(job_id)
        candidate = runner.inspector.identity_for(job_id, os.getpid())
        locked_identity = runner.lock.read()
        if status.status == JobStatus.RUNNING and candidate is not None and locked_identity == candidate:
            own_identity = candidate
            break
        if status.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            return 2
        sleeper(0.1)
    if own_identity is None:
        return 2

    try:
        process = spawner(
            launcher_command(runner, job_id),
            cwd=str(runner.paths.job_dir(job_id)),
            close_fds=True,
            env=controlled_environment(),
        )
    except OSError as exc:
        runner._write_failed(job_id, FailureCode.TRAINING_FAILED, f"训练 launcher 无法启动：{exc}")
        runner.lock.release_if_matches(own_identity)
        return 1

    watchdog = Watchdog(runner, job_id, clock=clock)
    while process.poll() is None:
        if watchdog.check() is not None:
            # 正常 Linux 路径中 killpg 会终止本 supervisor；返回仅用于 fake 测试。
            return 1
        sleeper(1.0)

    if process.returncode != 0:
        runner._write_failed(
            job_id,
            FailureCode.TRAINING_FAILED,
            f"训练 launcher 以退出码 {process.returncode} 结束",
        )
        runner.lock.release_if_matches(own_identity)
        return 1

    try:
        runner.declare_results(job_id)
        runner.write_status(
            RemoteStatus(
                REMOTE_PROTOCOL_VERSION,
                job_id,
                JobStatus.REMOTE_SUCCEEDED_PENDING_COLLECTION,
            )
        )
    except RunnerFailure as exc:
        runner._write_failed(job_id, exc.code, exc.message)
        runner.lock.release_if_matches(own_identity)
        return 1
    runner.lock.release_if_matches(own_identity)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    try:
        return supervise_job(args[0])
    except (RunnerFailure, ValueError):
        return 2


if __name__ == "__main__":  # pragma: no cover - 由 start 派生，不作为测试入口
    raise SystemExit(main())
