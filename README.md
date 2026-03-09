<div align="center">
  
# 🦾 VLA-Lab

### The Missing Toolkit for Vision-Language-Action Model Deployment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.1-orange.svg)](https://pypi.org/project/vlalab/)

**Log · Replay · Analyze · Evaluate** — All-in-one toolkit for real-world VLA deployment

[🚀 Quick Start](#-quick-start) · [📸 Screenshots](#-screenshots) · [🎯 Features](#-features) · [🔧 Installation](#-installation)

</div>

---

## 🎯 Why VLA-Lab?

Deploying VLA models to real robots is **hard**. You face:

- 🕵️ **Black-box inference** — Can't see what the model "sees" or why it fails
- ⏱️ **Hidden latencies** — Transport delays, inference bottlenecks, control loop timing issues
- 📊 **Fragmented logging** — Every framework logs differently, making cross-model comparison painful
- 🔄 **Tedious debugging** — Replaying failures requires manual log parsing and visualization

**VLA-Lab solves this.** A unified logging format + interactive visualization dashboard covering the full workflow from data collection to open-loop evaluation.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔬 Inference Replay
Step-by-step playback of policy inference: multi-camera views, 3D end-effector trajectories, full-dimensional state/action curves — pinpoint deployment failures instantly.

### 📊 Dataset Viewer
Interactively browse Zarr-format training data: frame-by-frame analysis, global statistics overview, and workspace distribution heatmaps.

</td>
<td width="50%">

### 📈 Latency Analysis
Decompose transport delay, GPU inference latency, and end-to-end loop time. Time-series plots + statistical distributions + multi-run comparison to identify bottlenecks fast.

### 🎯 Open-Loop Eval
Compare predicted actions against ground truth: MSE / MAE metric summaries, temporal alignment, error heatmaps, and 3D trajectory overlay.

</td>
</tr>
</table>

### 🔧 Supported Frameworks

| Framework | Status |
|:---|:---:|
| **Diffusion Policy** | ✅ Supported |
| **Isaac-GR00T** | ✅ Supported |
| **Pi 0.5** | ✅ Supported |
| **DreamZero** | ✅ Supported |
| **VITA** | ✅ Supported |

> VLA-Lab uses a unified logging protocol — adapting a new framework takes only a few lines of glue code.

---

## 📸 Screenshots

<table>
<tr>
<td colspan="2" align="center">

#### 🚀 Get Started — Feature Overview
<img src="assets/intro.png" width="100%"/>

</td>
</tr>
<tr>
<td align="center" width="50%">

#### 🔬 Inference Replay
<img src="assets/replay.png" width="100%"/>
<sub>Multi-camera views · 3D end-effector trajectory · Full state/action curves</sub>

</td>
<td align="center" width="50%">

#### 📈 Latency Analysis
<img src="assets/delay_analysis.png" width="100%"/>
<sub>Transport & inference latency time-series · Statistical distributions · Multi-run comparison</sub>

</td>
</tr>
<tr>
<td align="center" width="50%">

#### 📊 Dataset Frame-by-Frame Analysis
<img src="assets/dataset_view.png" width="100%"/>
<sub>Camera views · Robot state · Timeline scrubbing</sub>

</td>
<td align="center" width="50%">

#### 📊 Dataset Global Overview
<img src="assets/dataset_global.png" width="100%"/>
<sub>Episode statistics · Action distributions · Image grid</sub>

</td>
</tr>
<tr>
<td align="center" width="50%">

#### 📊 Workspace Distribution
<img src="assets/dataset_space.png" width="100%"/>
<sub>3D workspace sampling density visualization</sub>

</td>
<td align="center" width="50%">

#### 🎯 Open-Loop Evaluation
<img src="assets/openloop_eval.png" width="100%"/>
<sub>MSE/MAE summary · Temporal comparison · Error heatmap · 3D trajectories</sub>

</td>
</tr>
</table>

---

## 🔧 Installation

```bash
pip install vlalab
```

Install with full dependencies (including Zarr dataset support):

```bash
pip install vlalab[full]
```

Or install from source:

```bash
git clone https://github.com/ky-ji/VLA-Lab.git
cd VLA-Lab
pip install -e .
```

---

## 🚀 Quick Start

### Minimal Example (3 Lines!)

```python
import vlalab

# Initialize a run
run = vlalab.init(project="pick_and_place", config={"model": "diffusion_policy"})

# Log during inference
vlalab.log({"state": obs["state"], "action": action, "images": {"front": obs["image"]}})
```

### Full Example

```python
import vlalab

# Initialize with detailed config
run = vlalab.init(
    project="pick_and_place",
    config={
        "model": "diffusion_policy",
        "action_horizon": 8,
        "inference_freq": 10,
    },
)

# Access config anywhere
print(f"Action horizon: {run.config.action_horizon}")

# Inference loop
for step in range(100):
    obs = get_observation()
    
    t_start = time.time()
    action = model.predict(obs)
    latency = (time.time() - t_start) * 1000
    
    # Log everything in one call
    vlalab.log({
        "state": obs["state"],
        "action": action,
        "images": {"front": obs["front_cam"], "wrist": obs["wrist_cam"]},
        "inference_latency_ms": latency,
    })

    robot.execute(action)

# Auto-finishes on exit, or call manually
vlalab.finish()
```

### Launch Visualization

```bash
# One command to view all your runs
vlalab view
```

---

## 📖 Documentation

### Core Concepts

**Run** — A single deployment session (one experiment, one episode, one evaluation)

**Step** — A single inference timestep with observations, actions, and timing

**Artifacts** — Images, point clouds, and other media saved alongside logs

### API Reference

<details>
<summary><b>vlalab.init() — Initialize a run</b></summary>

```python
run = vlalab.init(
    project: str = "default",     # Project name (creates subdirectory)
    name: str = None,             # Run name (auto-generated if None)
    config: dict = None,          # Config accessible via run.config.key
    dir: str = "./vlalab_runs",   # Base directory (or $VLALAB_DIR)
    tags: list = None,            # Optional tags
    notes: str = None,            # Optional notes
)
```

</details>

<details>
<summary><b>vlalab.log() — Log a step</b></summary>

```python
vlalab.log({
    # Robot state
    "state": [...],                    # Full state vector
    "pose": [x, y, z, qx, qy, qz, qw], # Position + quaternion
    "gripper": 0.5,                    # Gripper opening (0-1)
    
    # Actions
    "action": [...],                   # Single action or action chunk
    
    # Images (multi-camera support)
    "images": {
        "front": np.ndarray,           # HWC numpy array
        "wrist": np.ndarray,
    },
    
    # Timing (any *_ms field auto-captured)
    "inference_latency_ms": 32.1,
    "transport_latency_ms": 5.2,
    "custom_metric_ms": 10.0,
})
```

</details>

<details>
<summary><b>RunLogger — Advanced API</b></summary>

For fine-grained control over logging:

```python
from vlalab import RunLogger

logger = RunLogger(
    run_dir="runs/experiment_001",
    model_name="diffusion_policy",
    model_path="/path/to/checkpoint.pt",
    task_name="pick_and_place",
    robot_name="franka",
    cameras=[
        {"name": "front", "resolution": [640, 480]},
        {"name": "wrist", "resolution": [320, 240]},
    ],
    inference_freq=10.0,
)

logger.log_step(
    step_idx=0,
    state=[0.5, 0.2, 0.3, 0, 0, 0, 1, 1.0],
    action=[[0.51, 0.21, 0.31, 0, 0, 0, 1, 1.0]],
    images={"front": image_rgb},
    timing={
        "client_send": t1,
        "server_recv": t2,
        "infer_start": t3,
        "infer_end": t4,
    },
)

logger.close()
```

</details>

### CLI Commands

```bash
# Launch visualization dashboard
vlalab view [--port 8501]

# Convert legacy logs (auto-detects format)
vlalab convert /path/to/old_log.json -o /path/to/output

# Inspect a run
vlalab info /path/to/run_dir
```

---

## 📁 Run Directory Structure

```
vlalab_runs/
└── pick_and_place/                 # Project
    └── run_20240115_103000/        # Run
        ├── meta.json               # Metadata (model, task, robot, cameras)
        ├── steps.jsonl             # Step records (one JSON per line)
        └── artifacts/
            └── images/             # Saved images
                ├── step_000000_front.jpg
                ├── step_000000_wrist.jpg
                └── ...
```

---

## 🗺️ Roadmap

- [x] Core logging API & unified run format
- [x] Streamlit visualization suite (5 pages)
- [x] Diffusion Policy adapter
- [x] Isaac-GR00T adapter
- [x] Pi 0.5 adapter
- [x] DreamZero adapter
- [x] VITA adapter
- [x] Open-loop evaluation pipeline
- [ ] Cloud sync & team collaboration
- [ ] Real-time streaming dashboard
- [ ] Automatic failure detection
- [ ] Integration with robot simulators

---

## 🤝 Contributing

We welcome contributions! 

```bash
git clone https://github.com/ky-ji/VLA-Lab.git
cd VLA-Lab
pip install -e ".[dev]"
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  
**⭐ Star us on GitHub if VLA-Lab helps your research!**

*Built with ❤️ for the robotics community*

</div>
