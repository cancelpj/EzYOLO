#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EzYOLO - 本地YOLO全流程训练软件
主程序入口
"""

import sys
import os
from pathlib import Path

# 关闭 Ultralytics 的“依赖自动安装”：
# 本项目用 onnxruntime-directml 提供 DML 加速，而 Ultralytics 会把“缺少 onnxruntime 发行包”误判为
# 需要安装 CPU 版 onnxruntime（二者争抢同一个顶层包名）。自动安装会用 CPU 版覆盖 directml，
# 导致 DmlExecutionProvider 失效。因此禁用其自动安装，导出 ONNX 所需的 onnx / onnxslim 由
# requirements.txt 显式安装。setdefault 保证若用户主动开启则尊重其设置。
os.environ.setdefault("YOLO_AUTOINSTALL", "0")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, qInstallMessageHandler, QtMsgType
from PyQt6.QtGui import QIcon

from gui.main_window import MainWindow
from gui.styles import get_primary_font_family


def qt_message_handler(msg_type, context, message):
    """自定义Qt消息处理器，过滤QFont警告"""
    # 过滤掉QFont::setPointSize的警告
    msg_str = str(message).strip()
    if "QFont::setPointSize" in msg_str and "Point size <= 0" in msg_str:
        return  # 忽略这个警告
    
    # 其他消息正常输出到 stderr（Qt 的默认行为）
    if msg_type in (
        QtMsgType.QtWarningMsg,
        QtMsgType.QtCriticalMsg,
        QtMsgType.QtFatalMsg,
    ):
        print(msg_str, file=sys.stderr)


def main():
    """主函数"""
    # 安装自定义消息处理器，屏蔽QFont警告
    qInstallMessageHandler(qt_message_handler)
    
    # 启用高DPI支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # 获取应用根目录
    app_root = Path(__file__).parent
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("EzYOLO")
    app.setApplicationVersion("1.0.0")

    # 按当前系统真实存在的字体设置界面字体：
    # macOS/Linux 上不会再去请求 Windows 才有的 "Microsoft YaHei"，字体缺失警告随之消失
    font_family = get_primary_font_family()
    if font_family:
        app_font = app.font()
        app_font.setFamily(font_family)
        app.setFont(app_font)

    # 设置应用图标（使用相对路径）
    icon_path = app_root / "icon.png"
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
    
    # 创建主窗口
    window = MainWindow()
    
    # 为主窗口设置图标
    if 'app_icon' in locals():
        window.setWindowIcon(app_icon)
    
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
