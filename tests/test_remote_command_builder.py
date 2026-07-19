# -*- coding: utf-8 -*-
"""SSH/rsync 精确 argv 与 host-key pin 的离线测试。"""

import _bootstrap  # noqa: F401  保持测试路径与现有测试一致

import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile

from core.remote_training.profiles import RemoteTrainingProfile  # noqa: E402
from core.remote_training.transport import (  # noqa: E402
    ClientTransportTools,
    HostTrustStore,
    RemoteCommandBuilder,
    RemoteTransportError,
    host_key_alias,
)
from test_remote_training_profiles import _ed25519_key  # noqa: E402


PROFILE_ID = "5" * 32
JOB_ID = "6" * 32


def assert_rejected(callable_object, *args, **kwargs):
    try:
        callable_object(*args, **kwargs)
    except RemoteTransportError:
        return
    raise AssertionError("预期输入被拒绝")


def make_profile(host="train-lab"):
    return RemoteTrainingProfile(
        name="实验室 A100",
        host=host,
        port=2202,
        username="trainer",
        remote_root="/srv/ezyolo/trainer",
        host_public_key=_ed25519_key(),
        id=PROFILE_ID,
    )


def make_builder(root):
    tools = ClientTransportTools(
        ssh="/usr/bin/ssh",
        rsync="/usr/bin/rsync",
        null_device="/dev/null",
    )
    return RemoteCommandBuilder(tools, HostTrustStore(root / "known-hosts"))


def option_values(argv):
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "-o"]


def test_plain_ssh_argv_has_all_required_security_options_and_fixed_runner_command():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-command-builder-"))
    builder = make_builder(root)
    argv = builder.ssh_argv(make_profile(), "status", JOB_ID)
    options = option_values(argv)
    expected = {
        "BatchMode=yes",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "NumberOfPasswordPrompts=0",
        "ForwardAgent=no",
        "ClearAllForwardings=yes",
        "StrictHostKeyChecking=yes",
        "GlobalKnownHostsFile=/dev/null",
        f"HostKeyAlias={host_key_alias(make_profile())}",
        "ConnectTimeout=10",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=3",
    }
    assert expected.issubset(set(options))
    assert argv[1:3] == ["-F", "/dev/null"]
    assert {
        "ControlMaster=no",
        "ControlPath=none",
        "ControlPersist=no",
        "ProxyCommand=none",
        "ProxyJump=none",
        "PermitLocalCommand=no",
        "LocalCommand=none",
    }.issubset(set(options))
    assert any(option.startswith("UserKnownHostsFile=") for option in options)
    assert "StrictHostKeyChecking=no" not in options
    assert "ForwardAgent=yes" not in options
    assert "--" in argv
    assert argv[-4:] == ["trainer@train-lab", "ezyolo-remote-runner", "status", JOB_ID]


def test_known_hosts_is_private_unique_and_rewritten_from_profile_pin():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-command-builder-"))
    profile = make_profile()
    store = HostTrustStore(root / "known-hosts")
    path = store.prepare(profile)
    expected_line = f"{host_key_alias(profile)} {profile.host_public_key}\n"
    assert path.read_text(encoding="utf-8") == expected_line
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    path.write_text("old value\n", encoding="utf-8")
    assert store.prepare(profile).read_text(encoding="utf-8") == expected_line
    assert list(path.parent.glob("*.known_hosts")) == [path]


def test_rsync_upload_download_share_the_same_security_options_and_no_delete():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-command-builder-"))
    snapshot = root / "snapshot"
    staging = root / "staging"
    snapshot.mkdir()
    staging.mkdir()
    builder = make_builder(root)
    profile = make_profile()
    upload = builder.rsync_upload_argv(
        profile,
        job_id=JOB_ID,
        snapshot_root=snapshot,
    )
    download = builder.rsync_download_results_argv(
        profile,
        job_id=JOB_ID,
        staging_dir=staging,
    )
    for argv in (upload, download):
        assert "--protect-args" in argv
        assert "--safe-links" in argv
        assert "--delete" not in argv
        remote_shell = argv[argv.index("-e") + 1]
        shell_args = shlex.split(remote_shell)
        options = option_values(shell_args)
        assert shell_args[1:3] == ["-F", "/dev/null"]
        assert "StrictHostKeyChecking=yes" in options
        assert "ForwardAgent=no" in options
        assert "GlobalKnownHostsFile=/dev/null" in options
        assert "ControlMaster=no" in options
        assert "ControlPath=none" in options
        assert "ProxyCommand=none" in options
        assert not any("accept-new" in value for value in shell_args)
    assert upload[-1].endswith(f":/srv/ezyolo/trainer/incoming/{JOB_ID}")
    assert download[-2].endswith(f":/srv/ezyolo/trainer/results/{JOB_ID}/")


def test_runner_action_and_job_id_are_fixed_and_reject_injection():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-command-builder-"))
    builder = make_builder(root)
    profile = make_profile()
    for action, job_id in (
        ("shell", JOB_ID),
        ("start", None),
        ("preflight", JOB_ID),
        ("status", JOB_ID[:-1] + ";"),
    ):
        assert_rejected(builder.ssh_argv, profile, action, job_id)


def test_rsync_ipv6_destination_is_bracketed_and_local_symlink_is_rejected():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-command-builder-"))
    snapshot = root / "snapshot"
    snapshot.mkdir()
    builder = make_builder(root)
    argv = builder.rsync_upload_argv(
        make_profile(host="2001:db8::10"),
        job_id=JOB_ID,
        snapshot_root=snapshot,
    )
    assert f"trainer@[2001:db8::10]:/srv/ezyolo/trainer/incoming/{JOB_ID}" in argv

    link = root / "snapshot-link"
    try:
        os.symlink(snapshot, link)
    except (NotImplementedError, OSError):
        print("SKIP symlink unsupported")
        return
    assert_rejected(
        builder.rsync_upload_argv,
        make_profile(),
        job_id=JOB_ID,
        snapshot_root=link,
    )


def test_windows_rsync_shell_uses_createprocess_quoting_and_never_requires_wsl():
    root = Path(tempfile.mkdtemp(prefix="ezyolo-command-builder-"))
    snapshot = root / "snapshot"
    snapshot.mkdir()
    tools = ClientTransportTools(
        ssh=r"C:\Program Files\OpenSSH\ssh.exe",
        rsync=r"C:\Tools\rsync.exe",
        null_device="NUL",
    )
    builder = RemoteCommandBuilder(tools, HostTrustStore(root / "known-hosts"))
    argv = builder.rsync_upload_argv(
        make_profile(),
        job_id=JOB_ID,
        snapshot_root=snapshot,
    )
    remote_shell = argv[argv.index("-e") + 1]
    assert '"C:\\Program Files\\OpenSSH\\ssh.exe"' in remote_shell
    assert "-F NUL" in remote_shell
    assert "GlobalKnownHostsFile=NUL" in remote_shell
    assert "wsl" not in remote_shell.lower()
    assert argv[-2].endswith("/")


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
