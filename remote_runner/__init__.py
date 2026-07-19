"""EzYOLO 远程训练服务器 runner。

本包只依赖 Python 标准库和 ``remote_protocol.v1``。它由服务器管理员在普通
Linux 账号下手工部署；桌面端只能调用固定 action，不能把路径、命令或配置传入
runner。
"""

from remote_protocol.v1 import REMOTE_PROTOCOL_VERSION

__all__ = ["REMOTE_PROTOCOL_VERSION"]
