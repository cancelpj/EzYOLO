# EzYOLO 远程训练实施计划（SSH + rsync）

**前置设计：** [远程训练安全设计说明](../specs/2026-07-13-remote-training-design.md)
**状态：** 已完成代码地图与独立安全审查；待按本计划实施
**实施目标：** 在不改变既有本机训练的前提下，为 EzYOLO 增加可测试、可恢复、可多人隔离的 SSH 远程训练能力。

## 0. 当前事实、硬边界与完成定义

### 当前事实

- 仓库：`/Users/huyi/dev/EzYOLO`
- 当前分支：`codex/ezyolo-optimization-20260713`
- 设计提交：`ec2c809 新增远程训练安全设计说明`
- 原有未提交内容：`config/sam_config.json` 和 `.claude/`；二者均不属于本任务写集。
- 当前 `.venv` 有 PyQt、PyTorch 和 Ultralytics，但缺少 `pytest`。这不阻塞既有测试：测试文件内置直接运行入口；已实际直接运行 `test_navigation.py`、`test_ui_layout.py`、`test_ui_smoke.py`、`test_workflow.py`，全部通过。为避免无关依赖面，首版不因远程训练新增 pytest 或 pytest-qt。
- 当前桌面端没有 SSH、rsync、远程 profile、fingerprint 或 server runner 的现有实现。

### 绝对不触碰

- `config/sam_config.json`；
- `.claude/`；
- 本机 `TrainingThread` 的实现主体（尤其 `run()`、`run_real_training()`、`prepare_data_yaml()`）；
- qh-server 的 `/root/hy-work`、其他工程、系统 Python、系统服务和已有 GPU 进程；
- 密码、私钥、passphrase、token、任意 Shell 命令、任意 runner 路径、GPU ID。

### 本计划完成的定义

1. 用户可以在设置页保存非 root、非敏感的 SSH 服务器档案；
2. 训练页可以选择“本机训练”或一个已保存档案；
3. 远程 target 仅接受 detect/segment，且不传递本机 device 选择；
4. 客户端和 runner 都实现并测试协议、路径、manifest、payload allowlist、原子状态与取消语义；
5. 远程完成只在本机结果验证并原子落地后成立；
6. 所有离线测试与 macOS UI smoke 通过；
7. 真实服务器只在用户另行授权、普通账号与 runner 环境已经准备好后做最小数据 smoke；
8. 不 push、不创建/修改 PR，除非用户之后另行授权。

## 1. 交付结构、单一 writer 与提交边界

| 单元 | 单一 writer 写集 | 主要验收 | 建议独立提交 |
| --- | --- | --- | --- |
| A. 测试基线 | 无源码写集；沿用既有直接测试入口 | 相关导航、布局、冒烟和 workflow 基线真实通过 | 无提交 |
| B. 服务器档案 | `core/remote_training/profiles.py`、`profile_store.py`、对应测试 | 拒绝 root/秘密/危险字段；QSettings 隔离 | `新增远程训练服务器档案` |
| C. 启动计划 | `launch.py` 与对应测试；不暴露远程 UI | 本机不回归；远程只支持 detect/segment，且没有 executor 时无法启动 | `新增远程训练启动计划` |
| D. 共享协议与客户端安全层 | `remote_protocol/`、`snapshot.py`、`jobs.py`、对应测试 | 独立 wire contract、只读快照、manifest、状态机和本机原子落地 | `新增远程训练安全协议` |
| E. 服务端 runner | `remote_runner/` 与 runner 测试 | 协议握手、allowlist、锁、状态、取消全部离线可测 | `新增远程训练服务端 runner` |
| F. SSH/rsync 与 Qt 接入 | `transport.py`、`gui/remote_training_thread.py`、训练页远程绑定 | fake transport 跑通全流程；UI 不阻塞 | `接入远程训练后台流程` |
| G. 回归与现场前置 | 测试补充、UI smoke 证据、管理员清单 | 无服务器验证完整；服务器 write 前明确停住 | `补充远程训练回归验证` |

同一单元内的文件和逻辑只由一个 writer 修改；其他 MCP 助手只能调查或复核，不并发写同一个单元。每个单元完成后必须：运行聚焦测试、做本机 UI smoke、检查 diff、检查保护文件仍未暂存、独立提交。

## 2. 单元 A：确认既有直接测试基线（无源码改动）

仓库现有测试文件以 `if __name__ == "__main__"` 方式调用 `_bootstrap.run_module_tests(...)`，不依赖 pytest。后续新增测试必须沿用同一直接运行方式，避免为本次功能额外引入测试框架或生产依赖。

基线命令：

```bash
cd /Users/huyi/dev/EzYOLO
for test_file in \
  tests/test_navigation.py \
  tests/test_ui_layout.py \
  tests/test_ui_smoke.py \
  tests/test_workflow.py
do
  PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
    ./.venv/bin/python "$test_file" || exit 1
done
```

本计划写入前已经实际运行该基线，四组均通过。测试会使用既有 bootstrap 创建的临时 INI、临时数据库和 `/tmp` 项目数据；不执行 SSH、rsync、GPU、真实训练或模型下载。该单元没有源代码提交。

## 3. 单元 B：非敏感远程服务器档案

### 新增模块

建立目录：

```text
core/remote_training/
  __init__.py
  profiles.py
  profile_store.py
```

`profiles.py` 只包含纯 Python 数据与校验：

- `RemoteTrainingProfile`；
- `ProfileValidationError`；
- `new_profile_id()`：UUID4 hex；
- `normalize_profile()`、`validate_profile()`；
- `serialize_profiles()`、`deserialize_profiles()`；
- allowlist 序列化与未知字段拒绝。

profile 允许字段固定为：

```text
id, name, host, port, username, remote_root, host_public_key
```

校验必须拒绝：

- `username == root`；
- 密码、私钥、passphrase、token、API key、未知字段；
- 以 `-` 开头的 host/path 值；
- 空白、控制字符、引号、反引号、`$`、`;`、`..`、`~`；
- 非绝对 remote root、非法端口、非完整 OpenSSH 主机公钥格式。

`profile_store.py` 仅适配 `QSettings`，key 固定为：

```text
remote_training_profiles_v1
```

它不能访问 SSH、`~/.ssh`、网络或任何项目配置文件。JSON 损坏时返回可显示的错误和空安全列表，不崩溃、不猜测目标。

### 测试优先

先新增：

- `tests/test_remote_training_profiles.py`；
- `tests/test_remote_training_profile_store.py`。

两个文件的首个项目导入必须是：

```python
import _bootstrap
```

测试矩阵至少覆盖合法普通账号、root 拒绝、秘密字段拒绝、恶意 host/path、完整主机公钥格式、损坏 JSON、稳定 ID、QSettings 重建、不会覆盖 `training_templates` 与 `pretrained_path`。

### 设置页接入

只在以下文件做最小 UI 胶水：

- `gui/pages/settings_page.py`：把“远程训练服务器”放入“常用设置”之后、“AI 与自动标注”之前；新增/编辑/删除调用 store；`reset_settings()` 不删除 profiles；
- `gui/main_window.py`：仅新增一个 profile-changed 信号连接，为后续训练页刷新服务。

档案编辑对话框没有密码框、私钥框或任意命令输入框。它要求管理员带外提供的完整主机公钥；这不是秘密，界面可派生并展示 SHA256 指纹供用户人工核对。“测试连接”“重置信任”只在单元 F 的 backend 已具备后新增；单元 B 不显示无法真正工作的按钮。

### 验收

```bash
for test_file in \
  tests/test_remote_training_profiles.py \
  tests/test_remote_training_profile_store.py \
  tests/test_navigation.py tests/test_ui_layout.py tests/test_ui_smoke.py
do
  PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
    ./.venv/bin/python "$test_file" || exit 1
done
```

完成后检查最小与笔记本窗口宽度，确认长服务器名称不截断、中文说明无重叠、没有横向滚动。

## 4. 单元 C：训练位置与启动计划分叉

### 新增模块

新增 `core/remote_training/launch.py`：

- `LocalLaunchPlan`；
- `RemoteLaunchPlan`；
- `TrainingLaunchController.resolve()`。

controller 只根据“已校验训练表单 + target + profile store”返回计划，不导入 Ultralytics、不启动线程、不运行 SSH/rsync、也不创建快照。

规则：

- `local` 返回本机计划，保留当前任务类型和本机 `device`；
- `remote:<profile-id>` 仅接受 `detect` / `segment`；
- 远程计划不包含当前本机 `device` 字段；
- profile 不存在、已删除、损坏或不满足安全校验时阻止启动；
- 不允许因错误悄悄改选别的服务器；界面层只允许显示“回退本机训练”的显式结果。

### 暂不暴露远程 UI

此单元**不修改** `gui/pages/train_page.py`、`gui/main_window.py` 或设置页的导航连接，也不在训练页面显示可选的远程 target。这样避免在 SSH/rsync backend 与 `RemoteTrainingThread` 尚未存在时，让用户点击一个会崩溃、回退到本机训练或意外消耗本机 GPU 的“远程训练”选项。

`RemoteLaunchPlan` 是纯数据计划；它只能被后续单元 F 的 executor 消费。若内部调用点在 executor 未注册时意外请求远程启动，controller 必须抛出带用户可读信息的 `RemoteExecutionUnavailable`，绝不能回退为本机训练。

严禁修改 `TrainingThread.run()`、`run_real_training()` 或 `prepare_data_yaml()`。

### 测试

新增 `tests/test_training_launch_controller.py`，并直接运行它：

```bash
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
  ./.venv/bin/python tests/test_training_launch_controller.py
```

覆盖：

- 本机计划保持 device 与既有任务支持；
- 远程 detect/segment 允许；classify/pose/world 明确拒绝；
- 远程计划没有 `device`；
- profile 不存在、删除或损坏时返回 typed validation error；
- executor 未注册时远程启动明确拒绝，不回退到本机；
- 本机计划不导入 SSH、rsync 或 Ultralytics。

## 5. 单元 D：共享协议、只读快照与本机 job record

### 新增模块

在仓库顶层新增纯 Python、无 GUI/无 QSettings/无服务器副作用的共享包，并在客户端目录新增：

```text
remote_protocol/
  __init__.py
  v1.py
core/remote_training/
snapshot.py
jobs.py
```

`remote_protocol/v1.py` 是 client 与 runner 唯一共享的 wire contract：

- `REMOTE_PROTOCOL_VERSION = 1`；
- capabilities、job spec、manifest、status、failure code、result receipt；
- `VALIDATING`、`SNAPSHOTTING`、`UPLOADING`、`VERIFYING_UPLOAD`、`STARTING`、`RUNNING`、`CANCEL_REQUESTED`、`REMOTE_SUCCEEDED_PENDING_COLLECTION`、`COLLECTING`、`SUCCEEDED`、`FAILED`、`CANCELLED`、`UNKNOWN`、`ATTACHING`；
- 状态转移函数：`UNKNOWN` 只能由 runner 返回的事实状态解决；`SUCCEEDED` 只能由本机验证+原子落地触发。

`remote_protocol/` 不得 import `core.*`、`gui.*`、PyQt、数据库或 Ultralytics。`remote_runner/` 只能 import `remote_protocol/`，不能 import desktop 的 `core/` 或 `gui/`。后续手工部署 runner 时，将 `remote_runner/` 与这一小型纯协议包一起作为同一版本的源码工件部署；不拖入桌面应用包。新增 contract test 断言 client/runner 使用同一版本、同一字段名和同一状态集合。

`snapshot.py` 定义 `DatasetSnapshotBuilder`：

- 用只读路径遍历建立 source selection 和可选的大小估算；
- 不调用 `TrainingThread.prepare_data_yaml()`；
- 不移动、删除、重命名、覆盖原始数据；
- 不跟随逃出允许数据根的符号链接；
- 只允许图片、标签与受控元数据；
- 生成相对路径、大小和 SHA256 的完整 manifest；
- 只生成让 runner 构造 data YAML 所需的受控输入（任务类型、类别表、相对布局），不上传客户端 data YAML；
- 拒绝 `.pt/.pth/.pkl/.py/.sh` 等非数据 payload。

`jobs.py` 负责版本化的本地 `remote_training_jobs_v1` record 和 staging/promote：

- record 保存 job id、profile id、规范 root、snapshot hash、协议版本、时间、最后状态、结果摘要、本机结果目录；
- staging 位于应用数据目录，最终结果目录固定为 `runs/train/exp_<project-id>_remote_<job-id-prefix>`；
- promote 前检查 staging 与最终目录在同一文件系统；
- 最终目录已存在时失败，绝不覆盖；
- `best.pt` 等回传二进制不自动反序列化。

### 测试

新增：

- `tests/test_remote_protocol_contract.py`；
- `tests/test_remote_snapshot.py`；
- `tests/test_remote_state_machine.py`；
- `tests/test_remote_jobs.py`。

这些测试只用临时目录和 fake 文件，覆盖：共享 contract、软链接逃逸、manifest 缺失/哈希不符/额外文件、无写入源数据、runner 构造 data YAML 所需输入、原子落地、重复目录、`UNKNOWN`、`REMOTE_CRASHED`、收集失败和取消确认。

## 6. 单元 E：同仓服务端 runner 源码（不自动部署）

### 源码落点

服务端 runner 源码明确放在同一个 EzYOLO 仓库的顶层新目录：

```text
remote_runner/
  __init__.py
  cli.py
  config.py
  jobs.py
  trainer.py
  README.md
  server.example.json
 tests/test_remote_runner_protocol.py
```

这是**源码和测试位置**，不是 qh-server 的部署位置。runner 只可 import 同一版本的 `remote_protocol/`，不得 import `core/`、`gui/` 或桌面应用配置。部署必须由服务器管理员在普通账号下明确执行；桌面应用不能自动复制、安装或启动该目录。

### runner 责任

- 固定 CLI action：`preflight`、`verify-upload`、`start`、`status`、`cancel`、`collect-manifest`；
- 加载普通账号自己的 `server.json`，而不是客户端传来的 shell 参数；
- 与客户端交换协议版本，版本不匹配硬失败；
- 校验 canonical remote root、任务类型、模型符号名、模型 allowlist、batch/imgsz 上限；
- 以 server config 的设备策略决定 GPU/CPU；
- 从已验证的 manifest、任务类型、类别表和相对数据布局**自行构造** job 内 data YAML；绝不加载客户端原样 YAML，拒绝 `download`、未知键、绝对路径、`..`、服务器外路径和动态入口；
- 将模型符号名映射到 server config 中预先存在的绝对本地权重路径；缺失即失败，绝不把符号名交给会下载模型的 Ultralytics API；
- 在 remote root 内执行 incoming → jobs 原子移动；
- 用 `mkdir` 或 `flock` 实现 server-side 单任务锁；
- 状态 JSON “临时文件 + rename”原子写；
- 锁记录 boot id、PID、进程组 ID 和启动标识；仅在 boot id 改变、PID 已死或启动标识不一致时回收 stale lock；无法证明已死则不回收；
- 训练进程创建独立进程组；取消前复核 boot id 与启动标识，只影响本 job 的已核验进程组；未知/已结束任务的取消是 no-op；禁止 `pkill -f`、GPU reset 和外部 PID 操作；
- 拒绝 job payload 中的权重、脚本、客户端 YAML、越界路径、manifest 外文件、哈希不匹配文件和任何额外文件；
- CUDA OOM、环境不匹配、runner 协议错误和进程异常转为清晰 failure code；
- 只产生 allowlist 中的结果 manifest。

runner 使用安全 YAML dumper 生成 data YAML，并校验类别名/路径成分不含控制字符或越界语义。训练期间以固定间隔检查最大运行时长、incoming/jobs/结果目录字节数和最小可用磁盘空间；超过硬上限时只按已经核验的当前 job 进程组取消规则停止自己，写入 `MAX_RUNTIME`、`JOB_DISK_LIMIT` 或 `LOW_DISK_SPACE`，释放锁但不自动清理文件或其他 job。

### 可测试 seam

runner 的实际 Ultralytics 调用必须封装为可注入 trainer callable。其余锁、状态、payload 验证、路径处理和取消逻辑可在无 GPU、无 SSH 条件下运行测试。

`remote_runner/server.example.json` 仅是无秘密模板，展示协议版本、root、allowlist 及其绝对权重路径、虚拟环境路径、最大 epoch、最大运行时长、incoming/jobs/结果字节上限、单任务规则和资源上限；不包含账户密码、token、私钥或真实主机信息。超出容量时拒绝新任务；首版不自动清理旧任务。

`remote_runner/README.md` 必须给管理员一张逐项清单：创建普通账号、创建目录/venv、部署同版本 runner + `remote_protocol`、生成 server config、准备 allowlist 权重、运行本地 `preflight` 自检，以及将**完整 OpenSSH 主机公钥**带外交给桌面端用户。README 不能包含自动部署命令、root 登录命令或真实服务器信息。

## 7. 单元 F：SSH/rsync backend 与 Qt 后台线程

### 新增模块

在 `core/remote_training/` 新增 `transport.py`，在 `gui/` 新增 `remote_training_thread.py`。

`transport.py` 提供：

- `RemoteTrainingBackend` 抽象；
- `RemoteCommandBuilder`；
- `SshRsyncBackend`；
- 可注入的 subprocess/fake transport seam；
- `ClientTransportResolver`：通过受控 PATH 查找 `ssh` / `rsync`，不自动安装、下载、调用 WSL 或选择替代工具；
- host trust helper（应用私有 `known_hosts`，权限 0600）。

所有 SSH 调用使用 argv list，固定含：

```text
BatchMode=yes
PasswordAuthentication=no
KbdInteractiveAuthentication=no
NumberOfPasswordPrompts=0
ForwardAgent=no
ClearAllForwardings=yes
StrictHostKeyChecking=yes
UserKnownHostsFile=<app-private path>
GlobalKnownHostsFile=<os.devnull>
HostKeyAlias=ezyolo-<profile-id>
```

认证明确使用用户已经在本机 OpenSSH config / SSH agent 中准备好的凭据；没有密码框、密钥路径扫描或交互式回退。`ForwardAgent=no` 只禁止 agent 转发给远端，不阻止本机 ssh 使用本地 agent。

host trust helper 必须从 profile 的**完整主机公钥**生成 profile 专属、权限 0600 的 private known_hosts `HostKeyAlias + key` 唯一条目；每次 ssh 或 rsync 调用前均从 profile 重写该文件，绝不追加历史 key。`GlobalKnownHostsFile` 使用 Python `os.devnull` 的平台空设备（macOS/Linux 为 `/dev/null`，Windows 为 `NUL`），防止系统全局条目覆盖该 pin。不使用 `accept-new`、TOFU 或 `ssh-keyscan` 自动信任。任何未知/不匹配 key 都是不可绕过的失败；“重置信任”只接受管理员带外提供的新完整公钥。普通 ssh 与 rsync 的 `-e` 远程 shell 参数必须共用同一安全 argv builder，避免 rsync 静默回退到默认 known_hosts 或 agent 行为。

`ClientTransportResolver` 在 macOS/Linux/Windows 上只接受 PATH 中可发现的 `ssh` 与 `rsync` 可执行程序。Windows 缺少 OpenSSH 或 rsync 时返回 `CLIENT_TRANSPORT_UNAVAILABLE`，显示安装前置条件并禁用远程启动；不自动安装、下载或调用 WSL。本机训练不受影响。

rsync 双向使用 `--protect-args`、`--safe-links`，不使用 `--delete`。远程 shell 字符串只包含固定 runner action 和 32 位 hex job id；profile 目录、模型、训练参数均不得进入远程命令行。所有需远端理解的参数通过上传的受控 JSON job spec 传递。

`RemoteTrainingThread` 依次执行：

```text
preflight → 用户确认 → snapshot → upload → verify-upload → start → poll
→ collect manifest/results → local verify → atomic promote → completed signal
```

网络错误进入 `UNKNOWN` 并持久化 job record。停止按钮发送取消请求后进入 `CANCEL_REQUESTED`，仅在 runner 确认该 job 进程组退出后进入 `CANCELLED`。窗口关闭不伪造停止；用户下次打开应用可执行 `ATTACHING` / “重新连接核验”。

### 页面绑定

远程训练 target 直到本单元 backend 与 `RemoteTrainingThread` 都存在后才第一次出现在 UI。只在此时修改：

- `gui/pages/train_page.py`：在训练模板之后、YOLO 版本之前显示“训练位置”；连接 profile refresh、远程计划和新线程；远程 target 时本机 device 下拉禁用并说明“由服务器策略决定设备”；
- `gui/pages/settings_page.py`：添加真正可工作的“测试连接”和“重置信任”入口；
- `gui/main_window.py`：连接 profile-changed signal 到训练页刷新。

成功信号沿用现有结果刷新方式，但只在结果目录已经原子落地后发出。UI 不直接拼 SSH、rsync、manifest 或 JSON。远程 selector 无法执行时显示 typed error，绝不落回本机训练。

### 测试

新增：

- `tests/test_remote_command_builder.py`；
- `tests/test_remote_transport.py`；
- `tests/test_remote_training_thread.py`。
- `tests/test_remote_training_boundaries.py`。

重点使用 `; $(...)`、反引号、换行、Unicode、`..`、绝对路径、前导 `-` 等恶意输入验证精确 argv；golden test 分别断言 plain ssh argv 与 rsync `-e` 参数中都有同一套安全选项，并且不能有 `StrictHostKeyChecking=no`、`accept-new`、`ForwardAgent=yes` 或客户端注入的 `ProxyCommand`。resolver 测试模拟 macOS/Linux/Windows PATH，验证缺工具时为 `CLIENT_TRANSPORT_UNAVAILABLE`、`os.devnull` 在 Windows 为 `NUL`，且不触发自动安装或 WSL。fake backend 验证全流程不触发真实 SSH/rsync、状态不会丢失、成功只在 promote 后出现；额外增加“回传权重截断或哈希错误”测试，断言 UI 失败且不 promote。boundary test 用 `inspect.getsource()` 机械断言旧 `TrainingThread` 和 `prepare_data_yaml()` 不含 remote backend、SSH、rsync、runner 或 remote snapshot 逻辑。

## 8. 单元 G：完整回归、UI smoke 与服务器前置停点

### 本机回归

新增/扩展测试后运行：

```bash
cd /Users/huyi/dev/EzYOLO
for test_file in \
  tests/test_remote_training_profiles.py \
  tests/test_remote_training_profile_store.py \
  tests/test_training_launch_controller.py \
  tests/test_remote_protocol_contract.py \
  tests/test_remote_snapshot.py \
  tests/test_remote_state_machine.py \
  tests/test_remote_jobs.py \
  tests/test_remote_command_builder.py \
  tests/test_remote_transport.py \
  tests/test_remote_training_thread.py \
  tests/test_remote_training_boundaries.py \
  remote_runner/tests/test_remote_runner_protocol.py \
  tests/test_navigation.py tests/test_ui_layout.py \
  tests/test_ui_smoke.py tests/test_workflow.py
do
  PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
    ./.venv/bin/python "$test_file" || exit 1
done
```

随后启动真实 macOS 应用，至少检查：应用能打开；设置页 profile 编辑、完整主机公钥格式错误提示和长中文/长 profile 名不裁切；训练页 target 切换、下拉箭头、主题切换、窗口缩放、删除回退；本机训练入口的行为保持不变；没有可执行 backend 时远程选项不可点击或给明确不可用提示，绝不落回本机训练。

### 服务器前置条件清单（此计划不执行）

在任何真实 SSH 上传、runner 部署、服务器目录创建或训练启动之前，必须由服务器管理员明确完成并记录：

1. 创建普通 Linux 训练账号，不能是 root；
2. 为该账号建立独立 remote root、虚拟环境、runner 固定入口和 server config；
3. 预装已批准的 Python/PyTorch/CUDA/Ultralytics 与模型 allowlist；
4. 确认 SSH 公钥认证可用，agent 不会被转发；
5. 确认服务器磁盘空间、GPU 策略、单任务规则和管理员维护责任；
6. 用带外方式提供完整 OpenSSH 主机公钥，供桌面端以 `HostKeyAlias` 写入私有 known_hosts；
7. 明确允许一次最小数据 smoke，且不下载大模型、不运行长训练、不影响其他 GPU 用户。

这是一条**硬停点**：没有用户对服务器写入的当次明确授权，就不进行 SSH 写操作、rsync 上传、runner 部署或训练。即使客户端代码已经完成，也只报告“本机实现与 fake 验证完成，现场服务器尚未验证”。

## 9. 全局验收、防回归与报告格式

### 防回归

- 不新增字体裁切、布局重叠、横向滚动或状态丢失；
- 本机 TrainingThread 和本机 detect/classify/pose/world 支持范围不变化；
- 远程分支不会触发本机 destructive `prepare_data_yaml()`；
- remote run 命名兼容现有 workflow 的 `best.pt` 识别，不无故改 `gui/workflow.py`；
- 没有秘密进入 QSettings、job record、日志、截图或 Git diff；
- 服务器端没有来自客户端的任意命令、路径越界、额外 payload、无锁启动或错误进程杀伤。

### 每个单元的汇报必须分开写

| 状态 | 必须说明 |
| --- | --- |
| 已修改 | 文件和用户可见行为。 |
| 已测试 | 实际命令、通过/失败数、是否 fake 或真实。 |
| 已提交 | commit hash 和内容。 |
| 已推送 | 是否实际 push；本计划默认不 push。 |
| 已部署 | 是否有服务器 write；本计划默认没有。 |
| 已现场验证 | macOS、Windows/DPI、qh-server 各自独立说明。 |
| 未完成/阻塞 | 普通服务器账号、runner 环境、完整主机公钥、授权或现场条件。 |

## 10. 实施前检查清单

开始写第一个实现单元前，主执行者必须重新确认：

- [ ] 当前分支、远端、ahead/behind、已有 PR、工作树；
- [ ] `config/sam_config.json` 和 `.claude/` 仍未被暂存；
- [ ] 既有直接测试基线仍可运行，新增测试也具有直接运行入口；
- [ ] 不存在另一个 writer 在同一逻辑写集；
- [ ] 服务器尚未写入，除非取得单独明确授权；
- [ ] 每个单元的测试、UI smoke、commit 名称和独立复核时点已确定。
