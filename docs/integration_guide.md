# VLA-Lab 集成指南

本指南介绍如何将 VLA-Lab 集成到 RealWorld-DP 和 Isaac-GR00T 项目中。

## 目录

- [安装](#安装)
- [RealWorld-DP 集成](#realworld-dp-集成)
- [Isaac-GR00T 集成](#isaac-groot-集成)
- [配置选项](#配置选项)
- [查看日志](#查看日志)
- [常见问题](#常见问题)

## 安装

在推理服务器所在的环境中安装 VLA-Lab：

```bash
# 进入 VLA-Lab 目录
cd /path/to/VLA-Lab

# 安装（推荐可编辑模式）
pip install -e .

# 或者完整安装（包含 zarr 支持）
pip install -e ".[full]"
```

## RealWorld-DP 集成

### 1. 在 server_config.py 中启用 VLA-Lab

```python
# 在 RealWorld-DP/realworld_deploy/server/server_config.py 末尾添加

# VLA-Lab 配置
ENABLE_VLALAB = True  # 设为 True 启用 VLA-Lab 日志
VLALAB_RUN_DIR = None  # None = 自动生成，或指定路径如 "/path/to/runs/my_run"
```

### 2. 启动推理服务器

```bash
cd RealWorld-DP/realworld_deploy/server
python inference_server.py
```

启动后，如果 VLA-Lab 已启用，会看到：
```
[VLA-Lab] 日志目录: /path/to/runs/dp_20240115_103000_checkpoint
```

### 3. 日志输出

VLA-Lab 日志会保存在 `server/runs/` 目录下：

```
runs/dp_20240115_103000_checkpoint/
├── meta.json           # 运行元数据
├── steps.jsonl         # 步骤记录（每行一个 JSON）
└── artifacts/
    └── images/         # 图像文件
        ├── step_000000_default.jpg
        ├── step_000001_default.jpg
        └── ...
```

## Isaac-GR00T 集成

### 1. 在 server_config_groot.py 中启用 VLA-Lab

```python
# 在 Isaac-GR00T/realworld_deploy/server/server_config_groot.py 末尾添加

# VLA-Lab 配置
ENABLE_VLALAB = True  # 设为 True 启用 VLA-Lab 日志
VLALAB_RUN_DIR = None  # None = 自动生成
```

### 2. 启动推理服务器

```bash
cd Isaac-GR00T/realworld_deploy/server
python inference_server_groot.py
```

### 3. 多相机支持

GR00T 支持多相机输入，VLA-Lab 会自动保存所有相机的图像：

```
runs/groot_20240115_103000_checkpoint/
├── meta.json
├── steps.jsonl
└── artifacts/
    └── images/
        ├── step_000000_camera_0.jpg  # ego_view
        ├── step_000000_camera_1.jpg  # front_view
        ├── step_000001_camera_0.jpg
        └── ...
```

## 配置选项

### RunLogger 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `run_dir` | str | 必填 | 日志保存目录 |
| `model_name` | str | "unknown" | 模型名称 |
| `model_path` | str | None | 模型路径 |
| `model_type` | str | None | 模型类型 |
| `task_name` | str | "unknown" | 任务名称 |
| `task_prompt` | str | None | 语言指令 |
| `robot_name` | str | "unknown" | 机器人名称 |
| `inference_freq` | float | None | 推理频率 (Hz) |
| `image_quality` | int | 85 | JPEG 压缩质量 |

### log_step 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `step_idx` | int | 步骤索引 |
| `state` | List[float] | 状态向量 |
| `pose` | List[float] | 位姿 [x,y,z,qx,qy,qz,qw] |
| `gripper` | float | 夹爪状态 |
| `action` | List/List[List] | 动作（单个或 chunk） |
| `images` | Dict[str, ndarray/str] | {相机名: 图像数组或 base64} |
| `timing` | Dict | 时延信息 |
| `prompt` | str | 语言指令 |

## 查看日志

### 方式 1: 使用 CLI

```bash
# 启动可视化界面
vlalab view

# 指定端口
vlalab view --port 8502

# 查看运行信息
vlalab info /path/to/run_dir
```

### 方式 2: 直接运行 Streamlit

```bash
cd VLA-Lab
streamlit run src/vlalab/apps/streamlit/app.py
```

### 方式 3: 转换旧日志

```bash
# 自动检测格式
vlalab convert /path/to/inference_log_xxx.json -o /path/to/output_run

# 指定格式
vlalab convert /path/to/log.json -f dp
vlalab convert /path/to/log.json -f groot
```

## 常见问题

### Q: 如何禁用 VLA-Lab？

在配置文件中设置 `ENABLE_VLALAB = False`，或者不安装 VLA-Lab 包。

### Q: 图像文件太大怎么办？

调整 `image_quality` 参数（默认 85），降低到 60-70 可以显著减小文件大小。

### Q: 如何只记录部分步骤？

在调用 `log_step` 之前添加条件判断：

```python
if step_idx % 10 == 0:  # 每 10 步记录一次
    logger.log_step(...)
```

### Q: 传统日志和 VLA-Lab 日志有什么区别？

| 特性 | 传统日志 | VLA-Lab 日志 |
|------|---------|-------------|
| 格式 | 单个 JSON 文件 | JSONL + 图像文件 |
| 图像存储 | Base64 嵌入 | 独立 JPEG 文件 |
| 文件大小 | 很大（含 base64） | 较小（引用） |
| 可视化 | 需要解码 | 直接查看 |
| 多相机 | 不支持 | 支持 |

### Q: 如何自定义相机名称？

```python
images_dict = {
    "ego_view": ego_image,
    "front_view": front_image,
    "wrist_cam": wrist_image,
}
logger.log_step(..., images=images_dict)
```
