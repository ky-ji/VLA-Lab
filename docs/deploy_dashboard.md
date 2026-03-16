# Deploy Dashboard

`/deploy` 已重构为一个面向 realworld deploy 的极简控制台，不再依赖远端 `deploy-agent`，而是由 `VLA-Lab` 后端直接通过本机 `ssh` 执行两台目标机器上的固定命令。

## 设计目标

- 固定 2 台机器：`server` 和 `client`
- 固定 4 个命令：
  - `start_model_server`
  - `start_robot_server`
  - `set_joint_preset`
  - `start_inference_client`
- 固定 2 个输入：
  - `model_server_config_path`
  - `inference_client_config_path`
- 页面加载时自动做 SSH 健康检查
- 点击按钮后由后端线程异步执行命令，并展示最近 job、远端 PID、stdout/stderr 摘要

## 配置方式

deploy 只读取一个 JSON 配置文件，优先级如下：

1. `vlalab view --deploy-config /abs/path/dashboard.json`
2. `vlalab serve --deploy-config /abs/path/dashboard.json`
3. 环境变量 `VLALAB_DEPLOY_CONFIG`
4. 默认路径 `configs/deploy/dashboard.json`

后端通过本机 `ssh <host_alias> ...` 连接目标机器，推荐提前在 `~/.ssh/config` 中配置好免交互 host alias。

最小配置结构如下：

```json
{
  "targets": {
    "server": {
      "label": "Model Server",
      "ssh_host": "groot-gpu",
      "workdir": "/path/to/realworld_deploy/server",
      "shell": "bash -lc"
    },
    "client": {
      "label": "Robot Client",
      "ssh_host": "groot-robot",
      "workdir": "/path/to/realworld_deploy/robot_inference",
      "shell": "bash -lc"
    }
  },
  "inputs": {
    "model_server_config_path": {
      "label": "Model Server Config Path",
      "type": "path",
      "required": true,
      "target": "server"
    },
    "inference_client_config_path": {
      "label": "Inference Client Config Path",
      "type": "path",
      "required": true,
      "target": "client"
    }
  },
  "commands": {
    "start_model_server": {
      "label": "启动模型服务",
      "target": "server",
      "background": true,
      "template": "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate groot && python inference_server_groot.py --config {model_server_config_path}",
      "stdout_log": "/tmp/vlalab/start_model_server.stdout.log",
      "stderr_log": "/tmp/vlalab/start_model_server.stderr.log"
    },
    "start_robot_server": {
      "label": "启动机器人服务",
      "target": "client",
      "background": true,
      "template": "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate robot && python control/start_server.py",
      "stdout_log": "/tmp/vlalab/start_robot_server.stdout.log",
      "stderr_log": "/tmp/vlalab/start_robot_server.stderr.log"
    },
    "set_joint_preset": {
      "label": "设置关节位姿",
      "target": "client",
      "background": false,
      "template": "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate robot && python control/set_joint_positions.py"
    },
    "start_inference_client": {
      "label": "启动推理客户端",
      "target": "client",
      "background": true,
      "template": "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate robot && python inference_client.py --config {inference_client_config_path}",
      "stdout_log": "/tmp/vlalab/start_inference_client.stdout.log",
      "stderr_log": "/tmp/vlalab/start_inference_client.stderr.log"
    }
  }
}
```

## 配置约束

- `targets` 必须同时包含 `server` 和 `client`
- `inputs` 必须同时包含 `model_server_config_path`、`inference_client_config_path`
- `commands` 必须同时包含 4 个固定命令 ID
- `start_model_server` 模板必须包含 `{model_server_config_path}`
- `start_inference_client` 模板必须包含 `{inference_client_config_path}`
- 所有后台命令必须同时提供 `stdout_log` 和 `stderr_log`

也就是说，按钮和命令 ID 是固定的，但每条命令的具体 shell 内容、conda 环境、工作目录、日志路径都可以在 JSON 里改。

## 执行模型

- 健康检查：`ssh <host_alias> "echo __vlalab_ok__"`
- 后台任务：远端 `nohup ... & echo $!`，记录 PID
- 前台任务：同步执行并回写 stdout/stderr
- 后端会把 job 状态持久化到配置文件旁边的 `.deploy_state/`
- 服务重启后会重新加载最近 job，并通过 `kill -0 <pid>` 继续判断后台进程是否存活

## 启动方式

```bash
vlalab view --deploy-config configs/deploy/dashboard.json
```

或只起 API：

```bash
vlalab serve --no-frontend --deploy-config configs/deploy/dashboard.json
```

默认 API 为 `http://127.0.0.1:8000`，默认前端为 `http://127.0.0.1:3000/deploy`。

## 页面内容

新的 dashboard 固定只展示四块内容：

- 两台机器的 SSH 连接状态
- 2 个输入参数
- 4 个固定命令按钮
- 最近 jobs 与日志摘要

页面不再包含：

- deploy-agent 状态
- workflow 编排卡片
- action catalog
- 任意命令编辑
- 文件上传
- stop/restart/kill 按钮

## 建议使用方式

1. 先确认 dashboard 主机能无密码 SSH 到 `server` 和 `client`
2. 按实际环境修改 `configs/deploy/dashboard.json`
3. 把两条 `config_path` 的值填写为目标机器上的远端路径
4. 通过 `/deploy` 页面直接启动 realworld deploy 流程

如果后续命令入口有变化，优先修改 JSON 配置中的 `template`，而不是再改前端按钮或后端接口。
