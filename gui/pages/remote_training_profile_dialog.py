# -*- coding: utf-8 -*-
"""远程训练服务器档案的最小编辑对话框。

这里只收公开连接信息和服务器主机公钥 pin。认证仍由系统 OpenSSH / ssh-agent
完成，故意没有密码、私钥、口令或自定义命令输入框。
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
)

from core.remote_training.profiles import (
    RemoteTrainingProfile,
    RemoteTrainingProfileError,
)


class RemoteTrainingProfileDialog(QDialog):
    """创建或编辑一个严格受限的远程训练服务器档案。"""

    def __init__(
        self,
        profile: Optional[RemoteTrainingProfile] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._existing_profile = profile
        self._saved_profile: Optional[RemoteTrainingProfile] = None
        self.setWindowTitle("添加远程服务器" if profile is None else "编辑远程服务器")
        self.setMinimumWidth(520)
        self._init_ui()
        if profile is not None:
            self._populate(profile)

    @property
    def saved_profile(self) -> Optional[RemoteTrainingProfile]:
        return self._saved_profile

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        explanation = QLabel(
            "填写服务器管理员提供的公开连接信息。EzYOLO 不保存密码、私钥或口令；"
            "登录认证仍由本机系统 OpenSSH 和 ssh-agent 负责。"
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("caption")
        layout.addWidget(explanation)

        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.name_input = QLineEdit()
        self.name_input.setObjectName("remote_profile_name_input")
        self.name_input.setPlaceholderText("例如：实验室 A100")
        form.addRow("名称:", self.name_input)

        self.host_input = QLineEdit()
        self.host_input.setObjectName("remote_profile_host_input")
        self.host_input.setPlaceholderText("主机名或 IP 地址")
        form.addRow("服务器:", self.host_input)

        self.port_input = QSpinBox()
        self.port_input.setObjectName("remote_profile_port_input")
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(22)
        form.addRow("端口:", self.port_input)

        self.username_input = QLineEdit()
        self.username_input.setObjectName("remote_profile_username_input")
        self.username_input.setPlaceholderText("普通账号（不能使用 root）")
        form.addRow("登录账号:", self.username_input)

        self.remote_root_input = QLineEdit()
        self.remote_root_input.setObjectName("remote_profile_root_input")
        self.remote_root_input.setPlaceholderText("例如：/srv/ezyolo/trainer")
        self.remote_root_input.setToolTip("服务器上专供此账号远程训练 runner 使用的目录。")
        form.addRow("服务器目录:", self.remote_root_input)

        self.host_key_input = QPlainTextEdit()
        self.host_key_input.setObjectName("remote_profile_host_key_input")
        self.host_key_input.setFixedHeight(82)
        self.host_key_input.setPlaceholderText("ssh-ed25519 AAAA…（服务器主机公钥）")
        self.host_key_input.setToolTip(
            "这是服务器主机公钥，不是你的私钥。它用于固定校验连接到的服务器身份。"
        )
        form.addRow("主机公钥:", self.host_key_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save
        )
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self._save)
        layout.addWidget(buttons)

    def _populate(self, profile: RemoteTrainingProfile) -> None:
        self.name_input.setText(profile.name)
        self.host_input.setText(profile.host)
        self.port_input.setValue(profile.port)
        self.username_input.setText(profile.username)
        self.remote_root_input.setText(profile.remote_root)
        self.host_key_input.setPlainText(profile.host_public_key)

    def _save(self) -> None:
        try:
            self._saved_profile = RemoteTrainingProfile(
                name=self.name_input.text().strip(),
                host=self.host_input.text().strip(),
                port=self.port_input.value(),
                username=self.username_input.text().strip(),
                remote_root=self.remote_root_input.text().strip(),
                host_public_key=" ".join(self.host_key_input.toPlainText().split()),
                id=self._existing_profile.id if self._existing_profile is not None else None,
            )
        except RemoteTrainingProfileError as exc:
            QMessageBox.warning(
                self,
                "服务器信息不完整",
                "请检查服务器、账号、目录和完整的服务器主机公钥。\n\n"
                f"具体原因：{exc}",
            )
            return
        self.accept()


__all__ = ["RemoteTrainingProfileDialog"]
