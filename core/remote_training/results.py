"""远程训练结果回传后的本机完整性校验。

runner 通过 rsync 放入 staging 的内容一律视为不可信。这里仅接受一个受控
``manifest.json`` 与其明确列出的普通文件；不解析权重、pickle 或任意训练产物，
只核验路径、大小和 SHA-256。校验成功后才允许 ``ResultStager`` 原子落地。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from remote_protocol.v1 import (
    ManifestEntry,
    ProtocolValidationError,
    ResultManifest,
    ResultReceipt,
)


RESULT_MANIFEST_FILENAME = "manifest.json"
MAX_RESULT_MANIFEST_BYTES = 4 * 1024 * 1024
RESULT_HASH_CHUNK_BYTES = 1024 * 1024


class ResultBundleVerificationError(ValueError):
    """回传结果不符合 v1 清单约定，不能落地到 runs。"""


def verify_result_bundle(root: Path | str, receipt: ResultReceipt) -> ResultManifest:
    """严格核验下载到本机 staging 的结果 bundle。

    ``ResultReceipt`` 的哈希覆盖控制文件的原始字节；文件数量、总大小、每个
    条目的 SHA-256 则覆盖控制文件之外的实际结果。任何额外、缺失、符号链接或
    内容变化都会拒绝，不会促成最终结果目录。
    """
    if not isinstance(receipt, ResultReceipt):
        raise ResultBundleVerificationError("服务器结果回执格式无效")
    bundle_root = _require_plain_directory(root)
    manifest_path = bundle_root / RESULT_MANIFEST_FILENAME
    manifest_bytes = _read_regular_file(
        manifest_path,
        maximum_bytes=MAX_RESULT_MANIFEST_BYTES,
        label="结果 manifest",
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != receipt.result_manifest_hash:
        raise ResultBundleVerificationError("结果 manifest 哈希与服务器回执不一致")
    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
        manifest = ResultManifest.from_wire(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError, ProtocolValidationError) as exc:
        raise ResultBundleVerificationError("结果 manifest 格式无效") from exc

    if manifest.job_id != receipt.job_id:
        raise ResultBundleVerificationError("结果 manifest 与服务器回执的 job id 不一致")
    if len(manifest.entries) != receipt.result_count:
        raise ResultBundleVerificationError("结果文件数量与服务器回执不一致")
    if manifest.total_bytes != receipt.result_bytes:
        raise ResultBundleVerificationError("结果文件总大小与服务器回执不一致")

    expected = {entry.path: entry for entry in manifest.entries}
    actual, actual_directories = _list_regular_tree(bundle_root)
    expected_paths = set(expected) | {RESULT_MANIFEST_FILENAME}
    expected_directories = _expected_parent_directories(manifest.entries)
    if set(actual) != expected_paths or actual_directories != expected_directories:
        missing = sorted(expected_paths - set(actual))
        extra = sorted(set(actual) - expected_paths)
        missing_directories = sorted(expected_directories - actual_directories)
        extra_directories = sorted(actual_directories - expected_directories)
        detail = []
        if missing:
            detail.append("缺少 " + ", ".join(missing))
        if extra:
            detail.append("包含未声明文件 " + ", ".join(extra))
        if missing_directories:
            detail.append("缺少目录 " + ", ".join(missing_directories))
        if extra_directories:
            detail.append("包含未声明目录 " + ", ".join(extra_directories))
        raise ResultBundleVerificationError("结果目录与 manifest 不一致：" + "；".join(detail))

    for entry in manifest.entries:
        _verify_entry(bundle_root, entry)
    return manifest


def _require_plain_directory(value: Path | str) -> Path:
    root = Path(value)
    try:
        status = root.lstat()
    except OSError as exc:
        raise ResultBundleVerificationError("结果 staging 目录不存在") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ResultBundleVerificationError("结果 staging 必须是普通目录")
    try:
        return root.resolve(strict=True)
    except OSError as exc:
        raise ResultBundleVerificationError("结果 staging 无法解析") from exc


def _list_regular_tree(root: Path) -> tuple[dict[str, Path], set[str]]:
    """列出完整普通树，同时拒绝 symlink、特殊文件和未声明空目录。"""
    result: dict[str, Path] = {}
    directories: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError as exc:
            raise ResultBundleVerificationError("结果目录无法安全读取") from exc
        for child in children:
            try:
                status = child.lstat()
            except OSError as exc:
                raise ResultBundleVerificationError("结果目录在校验中发生变化") from exc
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(status.st_mode):
                raise ResultBundleVerificationError(f"结果目录包含符号链接：{relative}")
            if stat.S_ISDIR(status.st_mode):
                directories.add(relative)
                stack.append(child)
                continue
            if not stat.S_ISREG(status.st_mode):
                raise ResultBundleVerificationError(f"结果目录包含非普通文件：{relative}")
            if relative in result:
                raise ResultBundleVerificationError("结果目录包含重复路径")
            result[relative] = child
    return result, directories


def _expected_parent_directories(entries: tuple[ManifestEntry, ...]) -> set[str]:
    """从受控文件条目推导允许出现的全部祖先目录。"""
    expected: set[str] = set()
    for entry in entries:
        parent = PurePosixPath(entry.path).parent
        while parent.as_posix() != ".":
            expected.add(parent.as_posix())
            parent = parent.parent
    return expected


def _verify_entry(root: Path, entry: ManifestEntry) -> None:
    path = _safe_entry_path(root, entry.path)
    _verify_regular_file_hash(path, entry)


def _safe_entry_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    path = root.joinpath(*parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ResultBundleVerificationError(f"结果文件缺失：{relative}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResultBundleVerificationError(f"结果文件越出 staging：{relative}") from exc
    return path


def _read_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """用前后 stat 比较检测校验期间的本地文件替换。"""
    if maximum_bytes < 0:
        raise ResultBundleVerificationError(f"{label} 大小无效")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ResultBundleVerificationError(f"{label} 不存在") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ResultBundleVerificationError(f"{label} 必须是普通文件")
    if before.st_size > maximum_bytes:
        raise ResultBundleVerificationError(f"{label} 超过声明大小")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResultBundleVerificationError(f"无法安全读取{label}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ResultBundleVerificationError(f"{label} 在校验中发生变化")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            content = handle.read(maximum_bytes + 1)
        if len(content) > maximum_bytes:
            raise ResultBundleVerificationError(f"{label} 超过声明大小")
    finally:
        if descriptor != -1:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ResultBundleVerificationError(f"{label} 在校验中消失") from exc
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
    ):
        raise ResultBundleVerificationError(f"{label} 在校验中发生变化")
    return content


def _verify_regular_file_hash(path: Path, entry: ManifestEntry) -> None:
    """流式哈希大型训练产物，不把模型权重完整读入内存。"""
    label = f"结果文件 {entry.path}"
    try:
        before = path.lstat()
    except OSError as exc:
        raise ResultBundleVerificationError(f"{label} 不存在") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ResultBundleVerificationError(f"{label} 必须是普通文件")
    if before.st_size != entry.size:
        raise ResultBundleVerificationError(f"{label} 大小不匹配")

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResultBundleVerificationError(f"无法安全读取{label}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ResultBundleVerificationError(f"{label} 在校验中发生变化")
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = handle.read(RESULT_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > entry.size:
                    raise ResultBundleVerificationError(f"{label} 大小不匹配")
                digest.update(chunk)
        if total != entry.size or digest.hexdigest() != entry.sha256:
            raise ResultBundleVerificationError(f"{label} 大小或哈希不匹配")
    finally:
        if descriptor != -1:
            os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ResultBundleVerificationError(f"{label} 在校验中消失") from exc
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
    ):
        raise ResultBundleVerificationError(f"{label} 在校验中发生变化")
