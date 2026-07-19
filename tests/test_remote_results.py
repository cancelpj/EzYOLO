# -*- coding: utf-8 -*-
"""远程训练结果 bundle 的本机校验测试。"""

import _bootstrap  # noqa: F401

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import core.remote_training.results as results_module  # noqa: E402
from core.remote_training.results import (  # noqa: E402
    RESULT_MANIFEST_FILENAME,
    RESULT_HASH_CHUNK_BYTES,
    ResultBundleVerificationError,
    verify_result_bundle,
)
from remote_protocol.v1 import (  # noqa: E402
    REMOTE_PROTOCOL_VERSION,
    ManifestEntry,
    ResultManifest,
    ResultReceipt,
)


JOB_ID = "e" * 32


def _rejected(callable_object, *args, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except ResultBundleVerificationError:
        return
    raise AssertionError("预期不安全结果被拒绝")


def _write_bundle(
    root: Path,
    *,
    extra: bool = False,
    corrupt_hash: bool = False,
    artifact_bytes: bytes = b"not a serialized object; verifier must only hash it",
):
    artifact = root / "weights" / "best.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(artifact_bytes)
    content = artifact.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    entry = ManifestEntry(
        path="weights/best.pt",
        size=len(content),
        sha256=("0" * 64 if corrupt_hash else digest),
    )
    manifest = ResultManifest(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        entries=(entry,),
        total_bytes=len(content),
    )
    manifest_bytes = (
        json.dumps(manifest.to_wire(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    (root / RESULT_MANIFEST_FILENAME).write_bytes(manifest_bytes)
    if extra:
        (root / "unexpected.txt").write_text("no", encoding="utf-8")
    return ResultReceipt(
        protocol_version=REMOTE_PROTOCOL_VERSION,
        job_id=JOB_ID,
        result_manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        result_count=1,
        result_bytes=len(content),
    )


def test_valid_result_bundle_is_verified_without_deserializing_weight():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root)
        manifest = verify_result_bundle(root, receipt)
        assert manifest.job_id == JOB_ID
        assert manifest.entries[0].path == "weights/best.pt"


def test_hash_extra_file_and_wrong_receipt_are_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root, corrupt_hash=True)
        _rejected(verify_result_bundle, root, receipt)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root, extra=True)
        _rejected(verify_result_bundle, root, receipt)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root)
        bad_receipt = ResultReceipt(
            protocol_version=REMOTE_PROTOCOL_VERSION,
            job_id=JOB_ID,
            result_manifest_hash=receipt.result_manifest_hash,
            result_count=2,
            result_bytes=receipt.result_bytes,
        )
        _rejected(verify_result_bundle, root, bad_receipt)


def test_symlink_and_missing_file_are_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root)
        (root / "weights" / "best.pt").unlink()
        _rejected(verify_result_bundle, root, receipt)


def test_extra_empty_directory_is_rejected_too():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root)
        (root / "unexpected-empty-directory").mkdir()
        _rejected(verify_result_bundle, root, receipt)


def test_large_weight_is_hashed_in_bounded_chunks_not_read_all_at_once():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact_bytes = b"x" * (RESULT_HASH_CHUNK_BYTES * 2 + 17)
        receipt = _write_bundle(root, artifact_bytes=artifact_bytes)
        read_sizes = []
        real_fdopen = results_module.os.fdopen

        class _GuardedFile:
            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                self._handle.__enter__()
                return self

            def __exit__(self, *args):
                return self._handle.__exit__(*args)

            def read(self, size=-1):
                read_sizes.append(size)
                return self._handle.read(size)

        def guarded_fdopen(*args, **kwargs):
            return _GuardedFile(real_fdopen(*args, **kwargs))

        with patch.object(results_module.os, "fdopen", side_effect=guarded_fdopen):
            verify_result_bundle(root, receipt)

        assert len(artifact_bytes) + 1 not in read_sizes
        artifact_reads = [
            size for size in read_sizes
            if size != results_module.MAX_RESULT_MANIFEST_BYTES + 1
        ]
        assert artifact_reads
        assert max(artifact_reads) <= RESULT_HASH_CHUNK_BYTES

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = _write_bundle(root)
        link = root / "weights" / "link.pt"
        try:
            link.symlink_to(root / "weights" / "best.pt")
        except OSError:
            return
        _rejected(verify_result_bundle, root, receipt)


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
