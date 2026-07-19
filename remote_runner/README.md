# EzYOLO Remote Runner v1

`remote_runner` 是由**服务器管理员手工部署**的普通 Linux 账号服务端源码。它与桌面端只共享
`remote_protocol` 的纯 Python wire contract；runner 不导入 `core`、GUI、PyQt、QSettings、
数据库或 Ultralytics，也不负责 SSH、部署、下载模型、安装依赖或分配 GPU。

## 管理员部署边界

1. 使用一个普通 Linux 账号及其自己的目录和 venv。不要用 root 登录或把运行目录放在 `/root`。
2. 将**同一版本**的 `remote_runner/` 和 `remote_protocol/` 部署到该账号的 venv 可导入位置。
   本模块不会自动创建 venv、安装包、下载权重或改写服务器配置。
3. 管理员预先创建 canonical remote root **及其三个普通子目录**，并确保均属于该普通账号、
   不是软链接、不可由组或其他账号写入（建议目录权限为 `0700`）。桌面端的「测试连接」只读检查这些目录，缺任何一个
   都会拒绝继续，绝不会替管理员创建：

   ```text
   incoming/<job-id>/
   jobs/<job-id>/
   results/<job-id>/
   ```

4. 从 [`server.example.json`](server.example.json) 复制出该账号自己的
   `~/.config/ezyolo-remote/server.json`。配置文件必须是该账号拥有、不可由组或其他账号写入的
   普通文件；示例没有任何密码、token、主机地址或真实服务器资料。
5. 管理员在 `model_allowlist` 中预置每个模型符号对应的**已存在、不可由组或其他账号写入的绝对普通权重文件**。runner 不会
   下载缺失模型，也不接受桌面端传入的权重路径。
6. `runtime.launcher` 也是管理员预置的、不可由组或其他账号写入的绝对可执行普通文件。它接收 runner 固定给出的
   `--job-id`、`--task`、`--model`、`--data`、`--epochs`、`--batch`、`--imgsz` 和
   `--results-dir` 参数；不从客户端接收任意命令。launcher 可以在管理员已经准备好的 venv 中
   使用训练框架，但该依赖不进入 runner。

在让桌面端使用前，管理员本机执行轻量检查：

```bash
python -m remote_runner.cli preflight
```

该 action 只检查协议、普通账号配置、canonical root 与 `incoming/jobs/results` 预置目录、预置
launcher/模型、Python、Linux 进程身份信息和磁盘下限；它不安装依赖、不创建目录或用户、不运行训练。

桌面端用户应通过带外、可信渠道拿到**完整 OpenSSH host public key** 后再建立连接信任；不要从
runner 输出、截图或临时聊天内容中取得或替换 host key。

## 固定 CLI protocol

唯一允许的 CLI action 为：

```text
preflight
verify-upload <32-lowercase-hex-job-id>
start <32-lowercase-hex-job-id>
status <32-lowercase-hex-job-id>
cancel <32-lowercase-hex-job-id>
collect-manifest <32-lowercase-hex-job-id>
```

没有 runner path、remote root、GPU id、环境变量、配置路径、shell 片段或任意额外命令参数。所有
输出都是标准库 JSON：成功时为共享协议对象；失败时为共享的
`RunnerFailureEnvelope`（固定 `protocol_version`、`ok=false`、`failure_code` 和短消息）。
runner 不会输出 traceback、任意命令行、认证材料或配置路径。

## 上传 wire

客户端只可将一个未验证 job 放到 `incoming/<job-id>/`。其文件集合必须**恰好**为：

```text
manifest.json                 # DatasetManifest.to_wire()
job-spec.json                 # JobSpec.to_wire()
<manifest.entries 中逐项声明的相对普通文件>
```

`verify-upload` 用 `remote_protocol.v1` 严格解析两个 JSON 文件，并校验协议版本、job id、task type、
class names、snapshot hash、文件大小与 SHA-256。它拒绝额外或缺失文件、软链接、绝对/`..` 路径、
客户端 `data.yaml`，以及 `.pt`、`.pth`、`.pkl`、`.py`、`.sh`、`.bat`、`.command` payload。
验证成功后才会将同一文件系统上的 `incoming/<job-id>` 原子 rename 为 `jobs/<job-id>`，绝不覆盖已有
job。

runner 只根据协议中受控的 `layout` 与 `class_names` 生成自己的 `jobs/<job-id>/data.yaml`。该 YAML
没有 `download` 字段，不读取客户端 YAML，也不写入客户端提供的绝对或上跳路径。

## 执行、状态与取消

- `single_task=true` 时以锁记录 job id、PID、PGID、boot id 与 Linux process start marker。锁只有在
  boot id 改变、PID 已死亡或 start marker 不匹配时才会回收；无法证明死亡时保持拒绝，不猜测。
- `start` 在容量检查后启动独立进程组的 supervisor。supervisor 监测最大运行时长、job/result 容量和
  最低磁盘空间；命中时仅在本 job 身份仍被核验时向该 PGID 发送信号，并写入 `MAX_RUNTIME`、
  `JOB_DISK_LIMIT` 或 `LOW_DISK_SPACE`。
- `status` 发现 `RUNNING` job 的锁、进程身份或进程组已经消失时，写入
  `FAILED + REMOTE_CRASHED`。
- `cancel` 只在 boot id、PID、PGID 和 start marker 全部一致时处理该 job 的进程组。未知 job、已完成
  job 或身份不匹配都是安全 no-op；它不用 `pkill -f`、进程名匹配、GPU reset 或外部 PID。

## 受控结果 bundle

成功 launcher 只可在 `results/<job-id>/` 产出 runner 声明的固定结果：`metrics.json`、`results.csv`、
`weights/best.pt`、`weights/last.pt`。runner 从普通文件计算 path/size/SHA-256，写入受控
`results/<job-id>/manifest.json`；不反序列化、导入或执行 `.pt`。

`collect-manifest` 只在 runner 已声明远程成功后直接返回 `ResultReceipt`：

```json
{
  "protocol_version": 1,
  "job_id": "…",
  "result_manifest_hash": "…",
  "result_count": 0,
  "result_bytes": 0
}
```

`result_manifest_hash` 是受控 `manifest.json` 的 SHA-256。桌面端仍要在本地完成自己的下载、
校验与原子落地后，才可以把训练标为本机成功。
