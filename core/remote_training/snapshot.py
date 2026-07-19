"""远程训练的数据集只读快照。

本模块接收已经选好的源文件，把它们复制到一个新的临时快照目录，并生成受控
manifest。它绝不调用本机 TrainingThread.prepare_data_yaml()，也不修改源数据。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import shutil
from typing import Iterable, Mapping, Sequence

from remote_protocol.v1 import (
    REMOTE_PROTOCOL_VERSION,
    DatasetManifest,
    JobSpec,
    ManifestEntry,
    ProtocolValidationError,
    validate_job_id,
)


IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
LABEL_SUFFIXES = frozenset({".txt"})
FORBIDDEN_PAYLOAD_SUFFIXES = frozenset(
    {".bat", ".command", ".pkl", ".pt", ".pth", ".py", ".sh"}
)
MANIFEST_FILENAME = "manifest.json"
JOB_SPEC_FILENAME = "job-spec.json"


class SnapshotError(ValueError):
    """数据源、快照目录或 manifest 不满足远程训练快照边界。"""


class SnapshotIntegrityError(SnapshotError):
    """已落地的快照与其 manifest 不一致。"""


@dataclass(frozen=True)
class SnapshotSource:
    """一个经过调用方选择的源文件及它在远程 payload 中的相对位置。"""

    source_path: Path
    payload_path: str


@dataclass(frozen=True)
class GeneratedSnapshotSource:
    """由本机受控逻辑生成的标签文本；绝不来自客户端任意脚本或 YAML。"""

    payload_path: str
    content: bytes


@dataclass(frozen=True)
class SnapshotEstimate:
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class DatasetSnapshot:
    root: Path
    manifest_path: Path
    manifest: DatasetManifest


class DatasetSnapshotBuilder:
    """在受控 staging 目录创建完整、可校验的只读数据快照。"""

    def __init__(
        self,
        snapshot_parent: Path | str,
        *,
        allowed_roots: Iterable[Path | str],
    ) -> None:
        self.snapshot_parent = Path(snapshot_parent)
        roots = tuple(_resolve_allowed_root(Path(root)) for root in allowed_roots)
        if not roots:
            raise SnapshotError("至少需要一个允许的数据根目录")
        self._allowed_roots = roots

    def estimate(self, sources: Sequence[SnapshotSource]) -> SnapshotEstimate:
        """只检查并统计源文件；不会创建快照目录或写入任何源文件。"""
        checked = self._checked_sources(sources)
        return SnapshotEstimate(
            file_count=len(checked),
            total_bytes=sum(_checked_source_size(item) for item in checked),
        )

    def build(
        self,
        *,
        job_id: str,
        task_type: str,
        class_names: Sequence[str],
        layout: Mapping[str, str],
        sources: Sequence[SnapshotSource],
    ) -> DatasetSnapshot:
        """复制受控数据到新的 staging 快照，并生成 manifest.json。

        每个 job id 只能创建一次；已有目录会直接失败，绝不覆盖旧快照。调用方把
        staging parent 放在应用数据目录，随后再由传输层上传该快照。
        """
        validate_job_id(job_id)
        checked = self._checked_sources(sources)
        layout_pairs = _normalise_layout(layout)
        _require_images_for_layout(checked, layout_pairs)

        self.snapshot_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        snapshot_root = self.snapshot_parent / job_id
        try:
            snapshot_root.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SnapshotError("该 job 已有快照目录，拒绝覆盖") from exc

        try:
            entries = []
            for item in checked:
                destination = snapshot_root / item.payload_path
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if item.generated_content is not None:
                    size, digest = _write_generated_source(
                        destination,
                        item.generated_content,
                    )
                else:
                    if item.source_path is None or item.source_stat is None:
                        raise SnapshotError("普通数据源缺少文件信息")
                    size, digest = _copy_checked_source(
                        item.source_path,
                        item.source_stat,
                        destination,
                    )
                entries.append(
                    ManifestEntry(
                        path=item.payload_path,
                        size=size,
                        sha256=digest,
                    )
                )

            entries.sort(key=lambda entry: entry.path)
            total_bytes = sum(entry.size for entry in entries)
            snapshot_hash = compute_snapshot_hash(
                job_id=job_id,
                task_type=task_type,
                class_names=tuple(class_names),
                layout=layout_pairs,
                entries=tuple(entries),
                total_bytes=total_bytes,
            )
            manifest = DatasetManifest(
                protocol_version=REMOTE_PROTOCOL_VERSION,
                job_id=job_id,
                task_type=task_type,
                class_names=tuple(class_names),
                layout=layout_pairs,
                entries=tuple(entries),
                total_bytes=total_bytes,
                snapshot_hash=snapshot_hash,
            )
            manifest_path = snapshot_root / MANIFEST_FILENAME
            _write_manifest(manifest_path, manifest)
            return DatasetSnapshot(
                root=snapshot_root,
                manifest_path=manifest_path,
                manifest=manifest,
            )
        except Exception:
            # 只清理本次刚刚创建、位于受控 staging parent 下的目录。源数据不会被
            # 触碰，且不会删除任何已存在 job 快照。
            shutil.rmtree(snapshot_root, ignore_errors=True)
            raise

    def verify(self, snapshot: DatasetSnapshot | Path | str) -> DatasetManifest:
        """验证 manifest、所有数据文件、哈希、大小和额外文件。"""
        root = snapshot.root if isinstance(snapshot, DatasetSnapshot) else Path(snapshot)
        return verify_snapshot(root)

    def _checked_sources(self, sources: Sequence[SnapshotSource]) -> list["_CheckedSource"]:
        if not sources:
            raise SnapshotError("快照至少需要一个数据文件")

        checked = []
        seen_paths: set[str] = set()
        for source in sources:
            if isinstance(source, SnapshotSource):
                payload_path = _validate_payload_path(source.payload_path)
                source_path = Path(source.source_path)
                source_stat = _check_source_file(source_path, self._allowed_roots)
                generated_content = None
            elif isinstance(source, GeneratedSnapshotSource):
                payload_path = _validate_generated_payload_path(source.payload_path)
                if not isinstance(source.content, bytes):
                    raise SnapshotError("生成的标签内容必须是 bytes")
                source_path = None
                source_stat = None
                generated_content = source.content
            else:
                raise SnapshotError("sources 必须由受控数据源组成")
            if payload_path in seen_paths:
                raise SnapshotError("同一个 payload 路径只能出现一次")
            seen_paths.add(payload_path)
            checked.append(
                _CheckedSource(
                    source_path=source_path,
                    payload_path=payload_path,
                    source_stat=source_stat,
                    generated_content=generated_content,
                )
            )
        return checked


@dataclass(frozen=True)
class _CheckedSource:
    source_path: Path | None
    payload_path: str
    source_stat: os.stat_result | None
    generated_content: bytes | None


def compute_snapshot_hash(
    *,
    job_id: str,
    task_type: str,
    class_names: tuple[str, ...],
    layout: tuple[tuple[str, str], ...],
    entries: tuple[ManifestEntry, ...],
    total_bytes: int,
) -> str:
    """为不包含 snapshot_hash 本身的规范 manifest 内容计算 SHA256。"""
    body = {
        "protocol_version": REMOTE_PROTOCOL_VERSION,
        "job_id": job_id,
        "task_type": task_type,
        "class_names": list(class_names),
        "layout": {key: value for key, value in sorted(layout)},
        "entries": [entry.to_wire() for entry in sorted(entries, key=lambda item: item.path)],
        "total_bytes": total_bytes,
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_snapshot(snapshot_root: Path | str) -> DatasetManifest:
    """验证一个已生成快照，拒绝缺文件、篡改、软链接和 manifest 外文件。"""
    root = Path(snapshot_root)
    if root.is_symlink() or not root.is_dir():
        raise SnapshotIntegrityError("快照根目录不存在或不是普通目录")
    manifest = _read_manifest(root)
    _verify_snapshot_contents(root, manifest, {MANIFEST_FILENAME})
    return manifest


def write_job_spec(snapshot: DatasetSnapshot | Path | str, job_spec: JobSpec) -> Path:
    """为已经校验的数据快照增加唯一、受控的 job-spec.json。"""
    root = snapshot.root if isinstance(snapshot, DatasetSnapshot) else Path(snapshot)
    manifest = verify_snapshot(root)
    if (
        job_spec.job_id != manifest.job_id
        or job_spec.task_type != manifest.task_type
        or job_spec.class_names != manifest.class_names
        or job_spec.snapshot_hash != manifest.snapshot_hash
    ):
        raise SnapshotError("job spec 与数据快照不一致")
    path = root / JOB_SPEC_FILENAME
    if path.exists() or path.is_symlink():
        raise SnapshotError("job spec 已存在，拒绝覆盖")
    _write_json_private(path, job_spec.to_wire(), "job spec")
    return path


def verify_remote_payload(snapshot_root: Path | str) -> tuple[DatasetManifest, JobSpec]:
    """验证准备上传给 runner 的完整 payload（数据 manifest + job spec）。"""
    root = Path(snapshot_root)
    if root.is_symlink() or not root.is_dir():
        raise SnapshotIntegrityError("快照根目录不存在或不是普通目录")
    manifest = _read_manifest(root)
    job_spec_path = root / JOB_SPEC_FILENAME
    if job_spec_path.is_symlink() or not job_spec_path.is_file():
        raise SnapshotIntegrityError("快照缺少 job-spec.json")
    try:
        job_spec = JobSpec.from_wire(
            json.loads(job_spec_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ProtocolValidationError) as exc:
        raise SnapshotIntegrityError("快照 job spec 无法验证") from exc
    if (
        job_spec.job_id != manifest.job_id
        or job_spec.task_type != manifest.task_type
        or job_spec.class_names != manifest.class_names
        or job_spec.snapshot_hash != manifest.snapshot_hash
    ):
        raise SnapshotIntegrityError("job spec 与数据快照不一致")
    _verify_snapshot_contents(
        root,
        manifest,
        {MANIFEST_FILENAME, JOB_SPEC_FILENAME},
    )
    return manifest, job_spec


def _resolve_allowed_root(root: Path) -> Path:
    try:
        stat_result = root.lstat()
    except OSError as exc:
        raise SnapshotError("允许的数据根目录不存在") from exc
    if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISDIR(stat_result.st_mode):
        raise SnapshotError("允许的数据根目录必须是普通目录")
    return root.resolve(strict=True)


def _check_source_file(path: Path, allowed_roots: tuple[Path, ...]) -> os.stat_result:
    try:
        source_stat = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SnapshotError("数据源文件不存在") from exc
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise SnapshotError("数据源必须是普通文件，不能是符号链接")
    if not any(_is_within(resolved, root) for root in allowed_roots):
        raise SnapshotError("数据源不在允许的数据根目录内")
    return source_stat


def _copy_checked_source(
    source_path: Path,
    source_stat: os.stat_result,
    destination: Path,
) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise SnapshotError("无法安全读取数据源文件") from exc

    try:
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != source_stat.st_dev
            or opened_stat.st_ino != source_stat.st_ino
            or opened_stat.st_size != source_stat.st_size
        ):
            raise SnapshotError("数据源在快照期间发生变化")

        digest = hashlib.sha256()
        copied = 0
        with os.fdopen(descriptor, "rb", closefd=True) as source_file:
            descriptor = -1
            with destination.open("xb") as destination_file:
                while chunk := source_file.read(1024 * 1024):
                    destination_file.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
        if copied != source_stat.st_size:
            raise SnapshotError("数据源在快照期间发生变化")
        return copied, digest.hexdigest()
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _write_generated_source(destination: Path, content: bytes) -> tuple[int, str]:
    """将受控生成的标签写入新 staging 文件，不触碰任何用户源数据。"""
    digest = hashlib.sha256(content).hexdigest()
    try:
        with destination.open("xb") as destination_file:
            destination_file.write(content)
    except OSError as exc:
        raise SnapshotError("无法写入生成的标签文件") from exc
    return len(content), digest


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: DatasetManifest) -> None:
    _write_json_private(path, manifest.to_wire(), "快照 manifest")


def _write_json_private(path: Path, value: Mapping[str, object], label: str) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        with path.open("x", encoding="utf-8") as file_handle:
            file_handle.write(serialized)
            file_handle.write("\n")
        os.chmod(path, 0o600)
    except OSError as exc:
        raise SnapshotError(f"无法写入{label}") from exc


def _validate_payload_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SnapshotError("payload 路径必须是非空 POSIX 相对路径")
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute() or pure_path.as_posix() != value:
        raise SnapshotError("payload 路径不规范")
    parts = pure_path.parts
    if (
        len(parts) < 3
        or any(part in {"", ".", ".."} or part.startswith(".") or part.startswith("-") for part in parts)
        or any(ord(char) < 32 for char in value)
    ):
        raise SnapshotError("payload 路径含有不安全路径段")

    suffix = pure_path.suffix.lower()
    if suffix in FORBIDDEN_PAYLOAD_SUFFIXES:
        raise SnapshotError("payload 禁止包含可执行文件或模型文件")
    if parts[0] == "images" and suffix in IMAGE_SUFFIXES:
        return pure_path.as_posix()
    if parts[0] == "labels" and suffix in LABEL_SUFFIXES:
        return pure_path.as_posix()
    raise SnapshotError("payload 仅允许 images 下的图片或 labels 下的 txt")


def _validate_generated_payload_path(value: object) -> str:
    path = _validate_payload_path(value)
    if not path.startswith("labels/"):
        raise SnapshotError("只有 labels 下的 txt 可以由本机受控逻辑生成")
    return path


def _normalise_layout(layout: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(layout, Mapping):
        raise SnapshotError("layout 必须是 train/val/test 路径映射")
    keys = set(layout)
    if not {"train", "val"}.issubset(keys) or not keys.issubset({"train", "val", "test"}):
        raise SnapshotError("layout 必须包含 train 与 val，且只允许 train/val/test")
    pairs = []
    for split in ("train", "val", "test"):
        if split not in layout:
            continue
        path = layout[split]
        if not isinstance(path, str):
            raise SnapshotError("layout 路径必须是字符串")
        _validate_payload_path(path + "/placeholder.jpg")
        pairs.append((split, path))
    return tuple(pairs)


def _require_images_for_layout(
    sources: Sequence[_CheckedSource],
    layout: tuple[tuple[str, str], ...],
) -> None:
    image_paths = [item.payload_path for item in sources if item.payload_path.startswith("images/")]
    for split, image_root in layout:
        prefix = image_root.rstrip("/") + "/"
        if not any(path.startswith(prefix) for path in image_paths):
            raise SnapshotError(f"layout 的 {split} 没有图片文件")


def _list_snapshot_files(root: Path) -> set[str]:
    found: set[str] = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        for directory_name in list(directory_names):
            directory = current_path / directory_name
            if directory.is_symlink():
                raise SnapshotIntegrityError("快照不能包含符号链接目录")
        for file_name in file_names:
            file_path = current_path / file_name
            if file_path.is_symlink() or not file_path.is_file():
                raise SnapshotIntegrityError("快照不能包含符号链接或特殊文件")
            found.add(file_path.relative_to(root).as_posix())
    return found


def _read_manifest(root: Path) -> DatasetManifest:
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SnapshotIntegrityError("快照缺少 manifest.json")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return DatasetManifest.from_wire(raw_manifest)
    except (OSError, json.JSONDecodeError, ProtocolValidationError) as exc:
        raise SnapshotIntegrityError("快照 manifest 无法验证") from exc


def _verify_snapshot_contents(
    root: Path,
    manifest: DatasetManifest,
    metadata_paths: set[str],
) -> None:
    expected_hash = compute_snapshot_hash(
        job_id=manifest.job_id,
        task_type=manifest.task_type,
        class_names=manifest.class_names,
        layout=manifest.layout,
        entries=manifest.entries,
        total_bytes=manifest.total_bytes,
    )
    if manifest.snapshot_hash != expected_hash:
        raise SnapshotIntegrityError("snapshot_hash 不匹配")

    expected_paths = {*metadata_paths, *(entry.path for entry in manifest.entries)}
    actual_paths = _list_snapshot_files(root)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        detail = []
        if missing:
            detail.append("缺少 " + ", ".join(missing))
        if extra:
            detail.append("包含额外文件 " + ", ".join(extra))
        raise SnapshotIntegrityError("快照文件清单不匹配：" + "；".join(detail))

    for entry in manifest.entries:
        file_path = root / entry.path
        try:
            file_stat = file_path.lstat()
        except OSError as exc:
            raise SnapshotIntegrityError("manifest 文件无法读取") from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise SnapshotIntegrityError("manifest 文件不是普通文件")
        if file_stat.st_size != entry.size:
            raise SnapshotIntegrityError("manifest 文件大小不匹配")
        if _hash_file(file_path) != entry.sha256:
            raise SnapshotIntegrityError("manifest 文件哈希不匹配")


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _checked_source_size(source: _CheckedSource) -> int:
    if source.generated_content is not None:
        return len(source.generated_content)
    if source.source_stat is None:
        raise SnapshotError("普通数据源缺少文件信息")
    return source.source_stat.st_size
