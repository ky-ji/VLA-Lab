# VLA-Lab

**A toolbox for tracking and visualizing the real-world deployment process of VLA models.**

VLA-Lab 提供统一的日志采集接口和可视化工具，帮助研究人员快速 debug VLA 模型在真实世界部署时的问题。

## Features

- **📊 统一日志格式**: 标准化的 Run 目录结构，支持 JSONL + 图像 artifact
- **🔬 推理回放**: 逐步回放推理过程，支持多相机、3D 轨迹、动作可视化
- **📈 时延分析**: 深度分析传输延迟、推理延迟、总回路时间
- **🗂️ 数据集浏览**: 浏览 Zarr 格式的训练/评估数据集
- **🔌 多框架支持**: 支持 Diffusion Policy 和 NVIDIA GR00T

## Installation

```bash
# 基础安装
pip install -e .

# 完整安装（含 zarr 数据集支持）
pip install -e ".[full]"

# 开发安装
pip install -e ".[dev]"
```

## Quick Start

### 1. 启动可视化界面

```bash
# 方式 1: 使用 CLI
vlalab view

# 方式 2: 直接运行 Streamlit
streamlit run src/vlalab/apps/streamlit/app.py
```

### 2. 在推理服务器中接入 VLA-Lab

```python
from vlalab import RunLogger

# 初始化 logger
logger = RunLogger(
    run_dir="runs/my_experiment",
    model_name="diffusion_policy",
    model_path="/path/to/checkpoint",
    task_name="pick_and_place",
    robot_name="franka",
)

# 在推理循环中记录每一步
logger.log_step(
    step_idx=0,
    state=[0.5, 0.2, 0.3, 0, 0, 0, 1, 1.0],  # pose + gripper
    action=[[0.51, 0.21, 0.31, 0, 0, 0, 1, 1.0]],  # action chunk
    images={"front": image_rgb},  # 多相机支持
    timing={
        "client_send": t1,
        "server_recv": t2,
        "infer_start": t3,
        "infer_end": t4,
    },
)

# 结束时关闭
logger.close()
```

### 3. 转换旧版日志

```bash
# 自动检测格式并转换
vlalab convert /path/to/inference_log_xxx.json -o /path/to/output_run

# 指定格式
vlalab convert /path/to/log.json -f dp -o /path/to/output
vlalab convert /path/to/log.json -f groot -o /path/to/output
```

## Run Directory Structure

VLA-Lab 使用标准化的 Run 目录结构：

```
run_dir/
├── meta.json           # 元数据（模型、任务、机器人、相机配置等）
├── steps.jsonl         # 步骤记录（每行一个 JSON）
└── artifacts/
    └── images/         # 图像文件
        ├── step_000000_front.jpg
        ├── step_000000_ego.jpg
        └── ...
```

### meta.json 示例

```json
{
    "run_name": "experiment_001",
    "start_time": "2024-01-15T10:30:00",
    "model_name": "diffusion_policy",
    "model_path": "/path/to/checkpoint",
    "task_name": "pick_and_place",
    "robot_name": "franka",
    "cameras": [
        {"name": "front", "resolution": [640, 480]},
        {"name": "ego", "resolution": [320, 240]}
    ],
    "inference_freq": 10.0,
    "total_steps": 150
}
```

### steps.jsonl 示例

```json
{"step_idx": 0, "obs": {"state": [0.5, 0.2, ...], "images": [{"path": "artifacts/images/step_000000_front.jpg", "camera_name": "front"}]}, "action": {"values": [[0.51, 0.21, ...]]}, "timing": {"inference_latency_ms": 45.2, "total_latency_ms": 78.5}}
```

## Supported Frameworks

### Diffusion Policy

接入方式：在 `inference_server.py` 中初始化 `RunLogger`

```python
# 在 DPInferenceServerSSH.__init__ 中
from vlalab import RunLogger

self.logger = RunLogger(
    run_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    model_name="diffusion_policy",
    model_path=str(checkpoint_path),
    model_type="diffusion_policy",
)
```

### Isaac-GR00T

接入方式：在 `inference_server_groot.py` 中初始化 `RunLogger`

```python
# 在 GrootInferenceServer.__init__ 中
from vlalab import RunLogger

self.logger = RunLogger(
    run_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    model_name="groot",
    model_path=str(model_path),
    model_type="groot",
    task_prompt=task_prompt,
)
```

## CLI Commands

```bash
# 启动可视化界面
vlalab view [--port 8501] [--run-dir /path/to/run]

# 转换旧版日志
vlalab convert <input_path> [-o output_dir] [-f dp|groot|auto]

# 初始化新的 run 目录
vlalab init-run <run_dir> [-m model] [-t task] [-r robot]

# 查看 run 信息
vlalab info <run_dir>
```

## API Reference

### RunLogger

```python
class RunLogger:
    def __init__(
        self,
        run_dir: str,
        model_name: str = "unknown",
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        task_name: str = "unknown",
        task_prompt: Optional[str] = None,
        robot_name: str = "unknown",
        cameras: Optional[List[Dict]] = None,
        inference_freq: Optional[float] = None,
        image_quality: int = 85,
    ): ...
    
    def log_step(
        self,
        step_idx: int,
        state: Optional[List[float]] = None,
        action: Optional[Union[List, List[List]]] = None,
        images: Optional[Dict[str, np.ndarray]] = None,
        timing: Optional[Dict] = None,
        prompt: Optional[str] = None,
    ): ...
    
    def close(self): ...
```

### Schema Classes

- `StepRecord`: 单步记录
- `ObsData`: 观测数据（状态 + 图像引用）
- `ActionData`: 动作数据（支持 chunk）
- `TimingData`: 时延数据
- `RunMeta`: 运行元数据

## Development

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black src/
ruff check src/
```

## License

MIT License

