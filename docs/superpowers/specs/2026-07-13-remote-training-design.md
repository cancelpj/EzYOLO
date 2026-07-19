# EzYOLO 远程训练设计说明（SSH + rsync）

**状态：** 已冻结方案，待实现前审核
**适用范围：** EzYOLO 本地桌面端、首个兼容目标为 qh-server，后续可添加其他用户自己的 SSH 服务器
**首版协议：** `ezyolo-remote/v1`

## 1. 目标与边界

EzYOLO 目前只在本机训练。本设计增加一个可选的“远程训练”目标：用户在 EzYOLO 中选择已保存的服务器档案，应用将当前项目的数据快照安全地上传到服务器，在服务器 GPU 上训练，并把经过校验的结果同步回本机。

首版要同时满足四件事：

1. **不改变本机训练。** 选择“本机训练”时，继续走现有 `TrainingThread` 路径。
2. **让不同用户可以使用自己的服务器账号。** 每个档案是用户自己的连接信息，不共享密码、私钥或任务目录。
3. **不把远程训练变成远程执行入口。** 用户不能通过主机名、目录、模型名或训练参数把任意 Shell 命令送到服务器。
4. **训练过程可恢复、结果可信。** 网络断开时不虚报成功或停止；上传和回传都必须校验；本机只在完整结果落地后才显示“训练完成”。

本设计只覆盖数据集的**目标检测（detect）**和**实例分割（segment）**训练。分类、姿态、多人排队、多服务器并行、自动重试、远程垃圾清理、暂停/恢复不属于首版。

## 2. 已确认的产品决定

### 2.1 使用普通服务器账号，拒绝 root

EzYOLO 的远程训练档案**不得使用 `root` 账号**。首版也不得把远程工作目录放在 `/root` 下。

每个可用服务器账号必须是普通 Linux 用户，拥有自己的工作目录，例如：

```text
/home/ezyolo/ezyolo-remote
```

或由服务器管理员创建、并明确归该普通账号所有的目录：

```text
/srv/ezyolo/alice
```

这条规则是多人使用的基础：不同账号的任务、数据和 SSH 权限天然隔离。对 qh-server 而言，现有管理员 root 连接只用于维护；在管理员创建普通训练账号、该账号自己的 runner 和虚拟环境之前，EzYOLO 不会把 qh-server 视为可训练目标。

### 2.2 认证由系统 OpenSSH 负责，EzYOLO 不保存秘密

用户可以在 EzYOLO 中填写服务器定位信息，但认证只使用其电脑中已经可用的系统 OpenSSH 配置或 SSH agent。首版不提供密码输入、私钥粘贴、私钥口令输入、Token 输入或密钥文件扫描。

EzYOLO 绝不保存以下内容到 `QSettings`、项目文件、日志、错误消息、Git 或远程任务包：

- SSH 密码；
- 私钥正文、私钥口令或私钥副本；
- API Token、访问令牌或其他可认证秘密；
- 任意可执行 Shell 命令；
- GPU 编号或任意服务器端运行命令。

OpenSSH Host 别名可以作为 `host` 使用。别名中的跳板、`ProxyCommand`、`IdentityFile` 等行为由用户自己的 OpenSSH 配置处理；EzYOLO 不读取、解析、复制或修改 `~/.ssh/config`。

### 2.3 服务端不由应用自动安装或改造

EzYOLO 的“测试连接/预检”只验证已存在的环境。它不会自动：

- 创建 Linux 用户；
- 安装系统包、CUDA、PyTorch、Ultralytics 或 rsync；
- 下载模型权重；
- 修改 `/root/hy-work`、其他项目目录、系统 Python 或系统服务；
- 重启服务器、清空任务目录或删除远程数据。

服务器管理员需要在普通账号范围内独立准备 runner、Python 虚拟环境、可用模型和服务器策略文件。此准备工作是一次单独、可审计的管理员操作，而不是桌面应用的隐式副作用。

## 3. 用户界面与使用流程

### 3.1 设置页：远程训练服务器

在“常用设置”之后、“AI 与自动标注”之前新增“远程训练服务器”分组。它管理全局服务器档案，训练页不重复实现档案编辑。

每个档案只保存下列非敏感字段：

```json
{
  "id": "a58a9d0491c84da8800a3946d3b7a301",
  "name": "实验室 A100",
  "host": "train-lab",
  "port": 22,
  "username": "ezyolo",
  "remote_root": "/home/ezyolo/ezyolo-remote",
  "host_public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI..."
}
```

字段规则：

| 字段 | 规则 |
| --- | --- |
| `id` | 由应用生成 32 位 UUID4 hex；不可由显示名、项目名或数据集名派生。 |
| `name` | 人类可读的服务器名称；用于界面显示，不参与路径或远程命令。 |
| `host` | OpenSSH alias、主机名或 IP；只允许 ASCII 主机字符，不允许空白、`@`、引号、反引号、`$`、`;` 或以 `-` 开头。 |
| `port` | 1–65535 的整数。 |
| `username` | 普通 Linux 用户名；必须匹配 `[a-z_][a-z0-9_-]{0,31}`，并明确拒绝 `root`。 |
| `remote_root` | 绝对、规范的 POSIX 路径；只允许 `[A-Za-z0-9._/-]`，不允许空白、`~`、`..`、重复语义段、控制字符或以 `-` 开头的路径段。 |
| `host_public_key` | 管理员带外提供的完整 OpenSSH 主机公钥（如 `ssh-ed25519 AAAA...`）；非秘密，但首次真实连接前必须存在。界面可从它派生 SHA256 指纹供人核对，不能只保存指纹。 |

档案保存在单独、版本化的 `QSettings` key：

```text
remote_training_profiles_v1
```

`QSettings` 不是秘密仓库，所以 profile schema 和序列化函数必须拒绝 `password`、`private_key`、`private_key_content`、`passphrase`、`token`、`api_key` 及其他未允许字段。恢复默认设置不得删除远程服务器档案，避免用户误失去非敏感连接配置。

设置页可以提供一个次级操作“测试连接”，它只执行 SSH 预检，不上传数据、不创建目录、不安装软件。服务器管理员必须在带外渠道给出完整主机公钥；EzYOLO 用 profile id 生成固定 `HostKeyAlias`，将 `HostKeyAlias + host_public_key` 写入权限为 `0600` 的应用私有 `known_hosts`，再以 `StrictHostKeyChecking=yes` 连接。这样即使用户填写的是 OpenSSH alias 或经跳板连接，实际服务器主机公钥仍固定由 profile 的公钥 pin 校验。

首版不做 TOFU：不使用 `accept-new`，不调用 `ssh-keyscan` 自动信任，也不能只凭 SHA256 指纹生成 `known_hosts` 条目。主机键不匹配一律阻止连接；“重置信任”要求用户粘贴管理员带外提供的新完整主机公钥，不能自动更新。

### 3.2 训练页：只选择训练位置

在“训练模板”之后、“YOLO 版本”之前新增一行：

```text
训练位置： [本机训练 / 实验室 A100 / 其他已保存服务器]
```

训练页只负责选择目标；档案的新增、编辑和删除仍在设置页完成。档案被删除或失效时，训练页安全回退到“本机训练”，并给出可理解的提示，不会静默改选另一台服务器。

用户点击“开始训练”后：

1. 本机目标继续构造现有 `TrainingThread`；
2. 远程目标先做只读预检；
3. 预检通过后显示简短确认信息：服务器名称、普通账号、远程目录、任务类型、模型符号名、预计上传的数据量；
4. 用户确认后才创建本次不可变数据快照并上传；
5. UI 持续显示状态、日志摘要、当前 epoch（可用时）和“重新连接核验”入口；
6. 训练成功的定义是“远程训练结束、结果已回传、本机校验通过并原子落地”，而不是“服务器说训练结束”。

页面不显示原始 SSH 命令、密码输入框或 GPU ID 选择器。GPU 设备选择由服务器管理员的 runner 策略控制。

## 4. 模块边界

远程训练不能塞进现有本机 `TrainingThread`。本机线程会在本机导入 Ultralytics、准备本机数据、读取本机 `pretrained/` 并假设结果在本机 `runs/`；把 SSH 字段加到它的 config 中会产生“界面声称远程、实际仍在本机执行”的错误实现。

新增的模块边界如下：

| 模块 | 责任 | 不负责 |
| --- | --- | --- |
| `remote_training_profiles` | profile schema、校验、序列化和拒绝秘密字段 | 连接服务器、执行训练 |
| `RemoteTrainingProfileStore` | 通过唯一 QSettings key 读写 profile | 直接操作页面控件 |
| `TrainingLaunchController` | 将“本机/远程目标 + 已校验训练配置”解析为启动计划 | 直接调用 ssh、rsync 或 YOLO |
| `remote_protocol/v1` | client 与 runner 共享的版本、job spec、manifest、状态和 failure code wire contract | PyQt、QSettings、数据库、Ultralytics、SSH 或服务器配置 |
| `RemoteTrainingBackend` | 统一远程协议接口 | 管理 PyQt 页面布局 |
| `SshRsyncBackend` | v1 的 OpenSSH、rsync、runner 交互 | 复用本机 `TrainingThread` |
| `RemoteTrainingThread` | 在后台执行远程状态机并通过信号更新 UI | 在主线程阻塞网络 I/O |
| `DatasetSnapshotBuilder` | 从当前项目构造只读、可校验的数据快照 | 修改原始数据集或调用本机 destructive dataset prepare |
| `RemoteCommandBuilder` | 用固定规则构造精确 argv，并拒绝危险输入 | 执行任意用户命令 |
| server runner | 服务端预检、锁、数据验证、训练、状态和结果清单 | 接受任意 Python、Shell、权重或 YAML 下载指令 |

`RemoteTrainingBackend` 的最小接口：

```text
preflight(profile) -> ServerCapabilities
snapshot(project, plan) -> DatasetSnapshot
upload(snapshot, profile, job) -> UploadReceipt
verify_upload(profile, job) -> VerifiedUpload
start(profile, job) -> RemoteJobRecord
poll(profile, job) -> RemoteStatus
cancel(profile, job) -> CancelRequest
collect(profile, job) -> CollectedResults
attach(profile, persisted_job) -> RemoteStatus
```

`TrainingLaunchController` 是唯一的本机/远程分叉点：

```text
已校验表单配置
      ↓
TrainingLaunchController.resolve(...)
      ├─ LocalLaunchPlan  → 现有 TrainingThread
      └─ RemoteLaunchPlan → RemoteTrainingThread + SshRsyncBackend
```

`remote_protocol/v1` 必须是独立的纯 Python 包：client 和 runner 都可以使用它，但它不得 import `core.*`、`gui.*`、PyQt、QSettings、数据库或 Ultralytics。手动部署 runner 时，runner 与同版本的 `remote_protocol` 一起部署；runner 不得通过 import 桌面应用来取得协议定义。共享 contract 的版本、字段名和状态集合必须有自动测试锁定。

## 5. SSH、rsync 与远程命令安全规则

### 5.1 固定的 SSH 行为

所有 SSH 调用都由应用构造固定 argv，至少带有：

```text
BatchMode=yes
PasswordAuthentication=no
KbdInteractiveAuthentication=no
NumberOfPasswordPrompts=0
ForwardAgent=no
ClearAllForwardings=yes
StrictHostKeyChecking=yes
UserKnownHostsFile=<应用私有 known_hosts>
GlobalKnownHostsFile=<os.devnull>
HostKeyAlias=ezyolo-<profile-id>
ConnectTimeout=<固定短超时>
ServerAliveInterval=<固定值>
ServerAliveCountMax=<固定值>
```

应用私有 `known_hosts` 位于系统应用配置目录中 profile 专属的路径，创建权限为 `0600`。每次 SSH 或 rsync 调用前，应用都从该 profile 的 `host_public_key` 重写唯一的 `HostKeyAlias + key` 条目；不追加历史条目，也不写用户 `~/.ssh/known_hosts`。`GlobalKnownHostsFile` 使用 Python `os.devnull` 的平台空设备（macOS/Linux 为 `/dev/null`，Windows 为 `NUL`），防止系统全局记录覆盖 profile pin。认证使用用户已经在本机 OpenSSH config / SSH agent 中准备好的凭据；`ForwardAgent=no` 只禁止把 agent 转发给服务器，并不阻止本机 ssh 使用 agent。连接失败时不得弹出隐藏密码提示或在后台无限等待。

客户端预检还必须用受控 command resolver 查找系统 `ssh` 与 `rsync`。macOS/Linux 与 Windows 都只能使用 PATH 中明确可执行的系统工具；Windows 需要用户或管理员预先安装 OpenSSH 与 rsync 并让它们可被发现。EzYOLO 不自动安装、下载、调用 WSL 或选择来源不明的替代工具。任何一个工具缺失时，远程 target 显示 `CLIENT_TRANSPORT_UNAVAILABLE` 并保持本机训练可用。

本机 `subprocess` 使用参数列表且永不使用 `shell=True`，但这还不够：`ssh host command` 和 `rsync host:path` 最终仍会经过远程端解析。因此下列规则同时强制执行：

1. job id 只能是应用生成的 32 位 hex；
2. 所有 profile 字段先按第 3.1 节规范校验；
3. remote root 由服务端预检返回其规范真实路径，客户端要求它与 profile 精确一致；
4. 每个派生远程路径都必须通过“在 remote root 下”的前缀/路径段断言；
5. SSH 远程命令只允许固定 runner 程序和校验后的 job id；任务类型、模型、epoch、batch、图像大小等只写入上传的 JSON job spec，不能进入远程命令行；
6. 不接受用户填写 runner 命令、解释器路径、附加命令、环境变量或 GPU ID；
7. 所有支持 `--` 的工具使用该分隔符，所有以 `-` 开头的用户值一律拒绝。

runner 使用固定、由服务器管理员部署的入口，例如普通账号 `$HOME/.local/bin/ezyolo-remote-runner`。SSH 传递的是固定 `exec` 调用和 hex job id；runner 自己读取普通账号专属的服务器配置，不从客户端命令行接收 `remote_root` 或其它可执行参数。

### 5.2 rsync 规则

rsync 是大数据传输通道，负责续传而不是命令执行。上传和回传都必须：

- 使用 `--protect-args`（`-s`）；
- 使用 `--safe-links`；
- 禁止 `--delete`；
- 使用固定、只含程序常量和已校验数值端口的远程 shell 参数；
- 只同步 manifest 列出的相对路径；
- 拒绝 `..`、绝对路径、软链接逃逸、控制字符、空路径和未在 allowlist 中的额外文件；
- 在传输前检查文件数、总字节数和本机/远程可用磁盘空间；
- 回传到本机临时 staging 目录，验证后才在同一文件系统上 `rename` 到最终结果目录；
- 绝不覆盖已经存在的本机运行目录。

manifest 校验必须同时发现三类问题：缺文件、哈希/大小不匹配、**额外文件**。仅检查“上传的文件都在”不足以防止 `.incoming` 或结果目录被污染。

## 6. 数据快照与允许载荷

### 6.1 快照必须只读

远程路径不复用本机训练中可能重建数据集目录的流程。`DatasetSnapshotBuilder` 从当前项目创建一个只读快照：

- 不移动、删除、重命名或原地重写用户图片、标签和项目数据；
- 不跟随逃出项目数据根目录的符号链接；
- 只接受图像、标签及所需的相对元数据；
- 排除缓存、临时文件、隐藏垃圾和非预期文件类型；
- 为每个文件记录相对路径、大小和 SHA256；
- 将总文件数、总字节数、任务类型、类别表和快照哈希写入本地持久化 job record。

### 6.2 远程 job payload 只能是数据

远程 runner 把客户端上传内容视为不可信数据，不能执行或反序列化其中的任意文件。客户端 payload 中禁止出现：

```text
.pt  .pth  .pkl  .py  .sh  .bat  .command  可执行文件  自定义下载脚本
```

客户端不上传可直接执行的 `data.yaml`。它只上传经校验的 manifest、任务类型、类别表和相对数据布局；runner 在 manifest 验证通过后，**自行构造**最终的 `data.yaml`。runner 用安全 YAML dumper 输出，并校验类别名和路径成分不含控制字符或越界语义；不得消费客户端原样 YAML，并且构造后的 YAML 不得包含：

- `download:`；
- 绝对路径；
- `..` 路径段；
- 服务器外路径；
- 任意脚本、URL 下载或动态 Python 入口。

训练模型以**符号模型名**而不是本机权重路径提交。例如客户端可表达“YOLOv10 nano”，runner 再把它映射到服务器管理员预先登记的、存在且为普通文件的**绝对本地权重路径**。runner 只能把该绝对路径交给 Ultralytics；不得把符号名或客户端模型路径直接传给会触发解析/下载的 API。模型不存在即失败，不能自动下载模型，也不加载 payload 中的 checkpoint。

服务端 runner 配置维护 allowlist：可用任务类型、模型符号名、Ultralytics 版本、可用设备策略、batch/imgsz 上限、任务目录上限。客户端和 runner 都校验一次；服务端校验是最终权威。

## 7. 服务端 runner 与协议

### 7.1 管理员部署约定

每个普通服务器账号拥有自己的 runner 目录、虚拟环境、任务目录和配置。示例结构：

```text
$HOME/.local/bin/ezyolo-remote-runner
$HOME/.config/ezyolo-remote/server.json
<remote_root>/
  incoming/
  jobs/
  locks/
  logs/
```

`server.json` 由服务器管理员维护，至少包括：协议版本、规范 remote root、允许的模型资产及其绝对路径、虚拟环境解释器、设备/资源上限、最大 epoch、最大运行时长、incoming/jobs 最大字节数、结果最大字节数和单任务策略。它不由桌面应用写入。超过资源上限时 runner 拒绝新任务；运行中超过最大时长、任务目录字节上限或最小可用磁盘空间时，runner 仅终止已核验的本 job 进程组并写入明确 failure code。首版不自动清理旧任务，管理员清理必须另行执行。

桌面端启动前调用固定 runner 的 `preflight`。返回至少包含：

- `protocol_version` 与 runner 版本；
- 规范 remote root；
- Python、PyTorch、CUDA、Ultralytics 兼容信息；
- 模型资产 allowlist；
- 磁盘空间、GPU 可用性和服务器忙闲状态；
- server policy 中允许的任务、模型、batch/imgsz 上限。

客户端和 runner 的 `protocol_version` 不一致时直接拒绝。环境、模型或设备不满足时也直接拒绝，不能自动安装、升级、下载或降级。

### 7.2 原子上传与原子任务创建

每个 job 的目录完全由 runner 管理：

```text
<remote_root>/incoming/<job-id>
<remote_root>/jobs/<job-id>
```

流程：

1. 客户端生成 job id，构造快照和 manifest；
2. rsync 到 `<remote_root>/incoming/<job-id>`；
3. runner 重新验证 manifest、路径和 payload allowlist；
4. 验证通过后，通过同一文件系统内原子 rename 把 incoming job 移到 jobs；
5. runner 创建不可变 job spec、初始状态和日志；
6. runner 才能开始训练。

没有验证通过的 incoming 目录永远不能训练。首版不提供远程清理动作，更不能执行 `rm -rf`；垃圾清理将来必须独立设计，先 dry-run 再执行。

### 7.3 服务器端锁、GPU 与停止

“同一服务器一次一个 EzYOLO 任务”必须在服务器端用原子机制实施，不能只由桌面端预先检查。runner 用 `mkdir` 锁目录或 `flock` 获得锁，锁中记录 job id、PID、进程组 ID、boot id 与进程启动标识。

锁只在下列条件下回收：记录的 boot id 已改变，或同一 boot id 下 PID 已不存在，或 PID 的启动标识与记录不一致。不能仅凭时间戳删除锁；无法证明进程已死时保留锁并要求管理员核验。

训练进程必须单独创建自己的进程组。取消规则：

1. 客户端发出取消请求后，状态为 `CANCEL_REQUESTED`；
2. runner 先核对 boot id、PID、进程组 ID 和进程启动标识，再只向该 job 的进程组发送停止信号；
3. 超时后可按固定策略向**同一已核验进程组**升级信号；
4. job 已结束、状态未知或进程身份无法核验时，取消是安全 no-op，不能扩大为按名称杀进程；
5. 只有进程组已退出、状态文件已原子更新后，才显示 `CANCELLED`；
6. 严禁 `pkill -f`、按名字匹配杀进程、GPU reset，或杀死不属于当前 job 的 PID。

GPU 准入检查只是保护，不是抢占承诺。服务器端根据自己的 device policy 与当下可用显存决定是否可启动；忙或资源不足时首版直接拒绝，不排队。runner 强制 batch/imgsz 上限或其它自我内存上限；CUDA OOM 是带原因码的终态失败，不会自动重试，也不会影响其他 GPU 进程。

runner 在训练期间以固定间隔检查 wall-clock 时长、任务目录字节数、结果目录字节数和可用磁盘空间。超过 `server.json` 规定的硬上限时，runner 按已核验的当前 job 取消规则停止自己的进程组，记录 `MAX_RUNTIME`、`JOB_DISK_LIMIT` 或 `LOW_DISK_SPACE`，并释放锁；它不删除文件、不清理其他 job，也不触碰其他进程。

### 7.4 原子状态与结果

runner 的状态 JSON 通过“写临时文件后 rename”原子更新，避免客户端读到半截 JSON。状态是服务器事实的唯一来源。

首版状态：

| 状态 | 含义 |
| --- | --- |
| `VALIDATING` | 本机 profile、训练表单与预检正在校验。 |
| `SNAPSHOTTING` | 正在构造本机只读快照与 manifest。 |
| `UPLOADING` | 正在上传到 remote incoming。 |
| `VERIFYING_UPLOAD` | runner 正在校验上传内容。 |
| `STARTING` | runner 已获得锁，正在启动当前 job。 |
| `RUNNING` | runner 确认进程组存在，训练正在运行。 |
| `CANCEL_REQUESTED` | 已请求停止，尚未确认进程组退出。 |
| `REMOTE_SUCCEEDED_PENDING_COLLECTION` | 服务器训练完成，但本机还没有完成结果收集和校验。 |
| `COLLECTING` | 正在回传、校验和本机 staging。 |
| `SUCCEEDED` | 结果已回传、本机验证通过并原子落地。 |
| `FAILED` | 明确失败；必须带失败原因码。 |
| `CANCELLED` | 已确认当前 job 的进程组退出。 |
| `UNKNOWN` | 网络或客户端中断；只能重新读取远程状态来解决，不能凭猜测改成成功/失败/停止。 |
| `ATTACHING` | 用户正在用已持久化 job record 重新连接核验。 |

若状态文件声称运行中、但 runner 发现该 job 进程组已经不存在，则状态必须转为 `FAILED`，原因码为 `REMOTE_CRASHED`。常见失败原因码还包括：`PRECHECK_FAILED`、`ENVIRONMENT_MISMATCH`、`SERVER_BUSY`、`GPU_ADMISSION`、`UPLOAD_INTEGRITY`、`RUNNER_PROTOCOL`、`TRAINING_FAILED`、`CUDA_OOM`、`CANCEL_TIMEOUT`、`COLLECTION_FAILED`、`LOCAL_VERIFY_FAILED`。

## 8. 结果收集、本机工作流与恢复

runner 只暴露声明过的结果清单，例如 metrics JSON/CSV、训练日志摘要、曲线图片、`best.pt`、`last.pt`。结果回传仍执行 manifest、相对路径、软链接、大小、总文件数和磁盘空间验证。

本机先同步到：

```text
<app-data>/remote-staging/<job-id>
```

验证通过后，使用同一文件系统内 rename 原子提升到一个从未存在过的本机运行目录：

```text
runs/train/exp_<project-id>_remote_<job-id-prefix>
```

不得覆盖已有运行目录。若回传或本机验证失败，状态是 `FAILED` 并保留可诊断的 staging 信息；不得将项目标成“已完成训练”。

远程 `.pt` 结果是经过用户已信任服务器产生的二进制产物，但桌面端仍不得在收集后自动反序列化或自动执行它。结果页可读取已校验的指标和图片；模型测试或显式加载权重属于后续用户动作。若将来允许不受信任的第三方服务器，必须另行设计签名产物或安全权重格式，不能把 v1 的“信任已确认服务器”边界扩展为通用下载信任。

本机持久化的 `remote_training_jobs_v1` record 至少包含 job id、profile id、规范 remote root、快照哈希、协议版本、创建/更新时间、最后已知状态、结果 manifest 摘要和本机落地目录。应用重启后，用户可用“重新连接核验”进入 `ATTACHING`，向 runner 读取事实状态；它不会凭最后一次 UI 文案推测训练仍在进行。

## 9. 测试与验收

### 9.1 单元测试

新增可离线运行的测试分块：

| 测试文件/范围 | 必须覆盖 |
| --- | --- |
| `test_remote_training_profiles.py` | profile 校验、稳定 ID、拒绝 root/秘密字段、完整主机公钥格式、危险 host/path、损坏 JSON 安全降级。 |
| `test_remote_training_profile_store.py` | 独立 QSettings key、不会碰训练模板或 SAM 配置、删除 profile 后安全回退、测试用临时 QSettings。 |
| `test_training_launch_controller.py` | local/remote plan 分叉、失效 profile、无 SSH 副作用、现有本机路径不变。 |
| `test_remote_command_builder.py` | 精确 argv；对 `; $(...)`、反引号、换行、Unicode、`..`、绝对路径、软链接、前导 `-` 等输入明确拒绝。 |
| `test_remote_training_boundaries.py` | 机械断言旧 `TrainingThread` 与 `prepare_data_yaml()` 中没有远程 backend、SSH、rsync、runner 或 remote snapshot 逻辑。 |
| `test_remote_snapshot.py` | 只读快照、不越过数据根目录、完整 manifest、禁止载荷类型；runner 重建 data YAML 的输入契约。 |
| `test_remote_state_machine.py` | UNKNOWN 只能由远程事实解决、取消确认、远程崩溃、收集失败、成功定义。 |
| `test_remote_runner_protocol.py` | 协议不匹配、共享 wire contract、allowlist、runner 重建 data YAML、原子锁、状态原子写、进程组身份核验、路径/manifest 校验；不需要 GPU。 |
| UI focused tests | 设置页 profile 保存/展示、训练页目标下拉、删除回退、360px 左栏无横向滚动、中文文字不裁切。 |

所有纯函数和 fake transport 测试禁止真实连接 SSH、上传数据、运行 rsync 或启动训练。runner 的非训练逻辑应可在无 GPU 环境下被单独测试，真实 YOLO 调用只有一个可注入 seam。

### 9.2 UI 与现场 smoke

实现完成后，至少验证：

1. macOS 真机：新增普通账号 profile、带外主机公钥 pin、无密码提示、启动本机训练仍正常；
2. 自动 UI：不同窗口宽度/DPI 下设置卡片、训练位置下拉、长服务器名和错误提示无文字裁切或横向滚动；
3. fake SSH/rsync：路径和参数严格按预期构造，无法命令注入；
4. 经授权的真实服务器 preflight：普通账号、runner、协议、环境、模型 allowlist、空间和 GPU 准入均可读取；
5. 经授权的最小数据 smoke：不下载大模型、不调用付费 API、不进行长训练；验证上传、状态读取、取消、回传与 attach；
6. 回归：已有训练、结果分析、模型测试、设置与 UI layout 测试全部通过。

Windows 与不同 DPI 的自动化测试通过不等于完成 Windows 服务器真机验证；报告必须把“自动测试已过”和“真实 Windows SSH/rsync 环境已验证”分开陈述。

## 10. 明确不做的事

首版不会：

- 自动使用 root、自动创建服务器账号或自动部署 runner；
- 在 EzYOLO 保存密码、私钥、passphrase、token；
- 自动下载权重、自动安装依赖或启动长训练；
- 上传任意 Python/Shell/checkpoint 或执行 payload 中的 YAML 下载指令；
- 允许用户输入任意远程命令、环境变量、runner 路径或 GPU ID；
- 多用户共享 root 目录、抢占 GPU、终止其他人的进程；
- 实现多任务排队、并行 fan-out、暂停/恢复、自动重试、远程清理；
- 修改 `config/sam_config.json`、`.claude/`、`/root/hy-work` 或其他既有服务器工程目录。

## 11. 实施顺序

1. 先实现 profile schema、QSettings store 与设置页/训练页目标选择，不产生 SSH 副作用；
2. 实现启动计划、命令构造、snapshot 与 fake transport/state-machine 测试；
3. 以独立、可测试的 server runner 实现协议、allowlist、锁、状态和结果 manifest；
4. 经管理员明确授权，在普通账号下部署 runner 和虚拟环境；
5. 用预检和最小数据 smoke 接入 qh-server；
6. 完成完整回归、跨模型复核与用户可见 UI smoke 后，再提交远程训练代码。

每一步都保持单一 writer：同一模块或逻辑写集不并发修改。每个可交付单元独立测试、截图/日志留证、Git commit；现有本机训练仍必须通过回归。

## 12. 审核结论

本说明已经把以下安全边界写成实现前提：非 root 多用户隔离、系统 SSH agent 认证、带外主机公钥 pin、无秘密存储、无任意远程命令、数据-only payload、runner 重建 data YAML、服务器模型 allowlist、服务器端原子锁与进程身份核验、路径/软链接/空间保护、版本握手、不可猜测的远程状态和原子结果落地。

只有在这些边界全部以代码和测试实现后，EzYOLO 才能把远程训练标为可用；在此之前，qh-server 的 root 管理连接和任何没有普通账号/runner 的服务器都只能显示为“尚未满足远程训练前置条件”。
