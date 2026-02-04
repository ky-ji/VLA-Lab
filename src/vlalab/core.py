"""
VLA-Lab Core API

SwanLab-style simple API for VLA deployment logging.

Usage:
    import vlalab

    # Initialize a run
    run = vlalab.init(
        project="pick_and_place",
        config={
            "model": "diffusion_policy",
            "action_horizon": 8,
        },
    )

    # Log a step
    vlalab.log({
        "state": [0.5, 0.2, 0.3, 0, 0, 0, 1, 1.0],
        "action": [0.51, 0.21, 0.31, 0, 0, 0, 1, 1.0],
        "images": {"front": image_array},
        "inference_latency_ms": 32.1,
    })

    # Finish
    vlalab.finish()
"""

import os
import atexit
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
import numpy as np


def _find_project_root(start_path: Optional[Path] = None, max_depth: int = 10) -> Optional[Path]:
    """
    自动检测项目根目录（效仿 SwanLab 的做法）
    
    从调用 vlalab.init() 的位置向上查找项目根目录，查找标志文件：
    - .git/ (Git 仓库)
    - setup.py (Python 包)
    - pyproject.toml (现代 Python 项目)
    - README.md (项目文档)
    
    Args:
        start_path: 起始路径（如果为 None，则从调用者文件位置开始）
        max_depth: 最大向上查找深度
    
    Returns:
        项目根目录路径，如果未找到则返回 None
    """
    if start_path is None:
        # 获取调用者的文件路径
        frame = inspect.currentframe()
        try:
            # 向上查找调用栈，找到第一个不在 vlalab 包内的调用者
            while frame:
                frame = frame.f_back
                if frame is None:
                    break
                filename = frame.f_code.co_filename
                if 'vlalab' not in filename and filename != '<stdin>':
                    start_path = Path(filename).parent.resolve()
                    break
        finally:
            del frame
    
    if start_path is None:
        # 如果无法获取调用者路径，使用当前工作目录
        start_path = Path.cwd()
    
    # 确保是绝对路径
    current = Path(start_path).resolve()
    
    # 项目根目录的标志文件
    markers = ['.git', 'setup.py', 'pyproject.toml', 'README.md']
    
    depth = 0
    while depth < max_depth:
        # 检查当前目录是否包含项目标志
        for marker in markers:
            marker_path = current / marker
            if marker_path.exists():
                return current
        
        # 向上查找
        parent = current.parent
        if parent == current:  # 到达文件系统根目录
            break
        current = parent
        depth += 1
    
    # 如果没找到，返回调用者文件所在目录的父目录（通常是项目根目录）
    return start_path.parent if start_path else None


class Config:
    """
    Configuration object that allows attribute-style access.
    
    Similar to SwanLab's run.config
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._data = config_dict or {}
        # Set attributes for direct access
        for key, value in self._data.items():
            setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def update(self, d: Dict[str, Any]):
        self._data.update(d)
        for key, value in d.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)
    
    def __repr__(self) -> str:
        return f"Config({self._data})"


class Run:
    """
    A single run/experiment session.
    
    This is returned by vlalab.init() and provides:
    - config: Access to configuration via run.config.key
    - log(): Log a step
    - log_image(): Log an image
    - finish(): Finish the run
    """
    
    def __init__(
        self,
        project: str = "default",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",  # "online", "offline", "disabled"
    ):
        """
        Initialize a run.
        
        Args:
            project: Project name (creates subdirectory)
            name: Run name (auto-generated if None)
            config: Configuration dict (accessible via run.config.key)
            dir: Base directory for runs (default: ./vlalab_runs)
            tags: Optional tags for the run
            notes: Optional notes about the run
            mode: "online" (future cloud sync), "offline" (local only), "disabled" (no logging)
        """
        from vlalab.logging import RunLogger
        
        self.project = project
        self.mode = mode
        self.tags = tags or []
        self.notes = notes or ""
        
        # Generate run name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"run_{timestamp}"
        self.name = name
        
        # Setup config object for attribute access
        self.config = Config(config)
        
        # Store step counter
        self._step_idx = 0
        self._disabled = (mode == "disabled")
        
        if self._disabled:
            self._logger = None
            print(f"[vlalab] Run disabled, no logging will occur")
            return
        
        # Setup run directory
        # 优先级：1. 显式指定的 dir 参数  2. VLALAB_DIR 环境变量  3. 自动检测项目根目录
        if dir is not None:
            base_dir = dir
        elif "VLALAB_DIR" in os.environ:
            base_dir = os.environ.get("VLALAB_DIR")
        else:
            # 自动检测项目根目录（效仿 SwanLab 的做法）
            project_root = _find_project_root()
            if project_root:
                # 在项目根目录下创建 vlalab_runs/ 目录
                base_dir = project_root / "vlalab_runs"
            else:
                # 如果无法检测，使用当前工作目录
                base_dir = Path.cwd() / "vlalab_runs"
        
        self.run_dir = Path(base_dir) / project / name
        
        # Extract model/task info from config if available
        model_name = self.config.get("model", self.config.get("model_name", "unknown"))
        task_name = self.config.get("task", self.config.get("task_name", project))
        robot_name = self.config.get("robot", self.config.get("robot_name", "unknown"))
        
        # Initialize the underlying RunLogger
        self._logger = RunLogger(
            run_dir=self.run_dir,
            model_name=model_name,
            model_type=self.config.get("model_type"),
            model_path=self.config.get("model_path"),
            task_name=task_name,
            task_prompt=self.config.get("task_prompt"),
            robot_name=robot_name,
            robot_type=self.config.get("robot_type"),
            inference_freq=self.config.get("inference_freq"),
            action_dim=self.config.get("action_dim"),
            action_horizon=self.config.get("action_horizon"),
            server_config=self.config.get("server_config", {}),
            client_config=self.config.get("client_config", {}),
        )
        
        # Store full config in metadata
        self._logger.meta.extra = {"config": self.config.to_dict()}
        self._logger._save_meta()
        
        print(f"[vlalab] 🚀 Run initialized: {self.run_dir}")
        print(f"[vlalab] 📊 Project: {project} | Name: {name}")
    
    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log a step with flexible data format.
        
        Args:
            data: Dict containing any of:
                - state: Robot state vector
                - pose: Robot pose [x, y, z, qx, qy, qz, qw]
                - gripper: Gripper state (0-1)
                - action: Action values (single or chunk)
                - images: Dict of camera_name -> image_array/base64
                - *_latency_ms: Any timing field ending in _latency_ms
                - Any other timing fields (client_send, server_recv, etc.)
            step: Step index (auto-incremented if None)
            commit: Whether to write immediately (True) or batch
        
        Examples:
            # Simple logging
            run.log({"state": [0.5, 0.2], "action": [0.1, 0.2]})
            
            # With images
            run.log({
                "state": [0.5, 0.2],
                "action": [0.1, 0.2],
                "images": {"front": img_array},
            })
            
            # With timing
            run.log({
                "state": [0.5, 0.2],
                "inference_latency_ms": 32.1,
                "transport_latency_ms": 5.2,
            })
        """
        if self._disabled or self._logger is None:
            return
        
        # Determine step index
        if step is not None:
            current_step = step
        else:
            current_step = self._step_idx
            self._step_idx += 1
        
        # Extract known fields
        state = data.get("state")
        pose = data.get("pose")
        gripper = data.get("gripper")
        action = data.get("action")
        images = data.get("images")
        prompt = data.get("prompt")
        
        # Extract timing fields
        timing = {}
        timing_keys = [
            "client_send", "server_recv", "infer_start", "infer_end", "send_timestamp",
            "transport_latency_ms", "inference_latency_ms", "total_latency_ms",
            "message_interval_ms", "preprocess_ms", "postprocess_ms",
        ]
        for key in timing_keys:
            if key in data:
                timing[key] = data[key]
        
        # Also capture any custom *_latency_ms or *_ms fields
        for key, value in data.items():
            if (key.endswith("_latency_ms") or key.endswith("_ms")) and key not in timing:
                timing[key] = value
        
        # Extract any extra tags
        tags = {}
        known_keys = {"state", "pose", "gripper", "action", "images", "prompt"} | set(timing_keys)
        for key, value in data.items():
            if key not in known_keys and not key.endswith("_ms"):
                # Store as tag if it's a simple value
                if isinstance(value, (int, float, str, bool)):
                    tags[key] = value
        
        # Log the step
        self._logger.log_step(
            step_idx=current_step,
            state=state,
            pose=pose,
            gripper=gripper,
            action=action,
            images=images,
            timing=timing if timing else None,
            prompt=prompt,
            tags=tags if tags else None,
        )
    
    def log_image(
        self,
        camera_name: str,
        image: Union[np.ndarray, str],
        step: Optional[int] = None,
    ):
        """
        Log a single image.
        
        Args:
            camera_name: Name of the camera
            image: Image array (H, W, C) or base64 string
            step: Step index (uses current step if None)
        """
        if self._disabled or self._logger is None:
            return
        
        current_step = step if step is not None else self._step_idx
        self.log({"images": {camera_name: image}}, step=current_step, commit=True)
    
    def finish(self):
        """Finish the run and save all data."""
        if self._logger is not None:
            self._logger.close()
            print(f"[vlalab] ✅ Run finished: {self._logger.step_count} steps logged")
            print(f"[vlalab] 📁 Data saved to: {self.run_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


# Global state
_current_run: Optional[Run] = None


def init(
    project: str = "default",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    dir: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    mode: str = "offline",
) -> Run:
    """
    Initialize a new VLA-Lab run.
    
    This is the main entry point for VLA-Lab, similar to swanlab.init().
    
    Args:
        project: Project name (creates subdirectory)
        name: Run name (auto-generated with timestamp if None)
        config: Configuration dict, accessible via run.config.key
        dir: Base directory for runs (default: ./vlalab_runs or $VLALAB_DIR)
        tags: Optional tags for the run
        notes: Optional notes about the run
        mode: "offline" (local only), "disabled" (no logging)
    
    Returns:
        Run object with config attribute and log() method
    
    Example:
        import vlalab
        
        run = vlalab.init(
            project="pick_and_place",
            config={
                "model": "diffusion_policy",
                "action_horizon": 8,
                "inference_freq": 10,
            },
        )
        
        print(f"Action horizon: {run.config.action_horizon}")
        
        # Log steps
        vlalab.log({"state": [...], "action": [...]})
        
        # Finish
        vlalab.finish()
    """
    global _current_run
    
    # Close previous run if exists
    if _current_run is not None:
        _current_run.finish()
    
    # Create new run
    _current_run = Run(
        project=project,
        name=name,
        config=config,
        dir=dir,
        tags=tags,
        notes=notes,
        mode=mode,
    )
    
    # Register atexit handler to auto-finish
    atexit.register(_auto_finish)
    
    return _current_run


def log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
):
    """
    Log a step to the current run.
    
    Must call vlalab.init() first.
    
    Args:
        data: Dict containing state, action, images, timing, etc.
        step: Step index (auto-incremented if None)
        commit: Whether to write immediately
    
    Example:
        vlalab.log({
            "state": [0.5, 0.2, 0.3, 0, 0, 0, 1, 1.0],
            "action": [0.51, 0.21, 0.31, 0, 0, 0, 1, 1.0],
            "images": {"front": image_array},
            "inference_latency_ms": 32.1,
        })
    """
    global _current_run
    
    if _current_run is None:
        raise RuntimeError("vlalab.init() must be called before vlalab.log()")
    
    _current_run.log(data, step=step, commit=commit)


def log_image(
    camera_name: str,
    image: Union[np.ndarray, str],
    step: Optional[int] = None,
):
    """
    Log a single image to the current run.
    
    Args:
        camera_name: Name of the camera
        image: Image array (H, W, C) or base64 string
        step: Step index (uses current step if None)
    """
    global _current_run
    
    if _current_run is None:
        raise RuntimeError("vlalab.init() must be called before vlalab.log_image()")
    
    _current_run.log_image(camera_name, image, step=step)


def finish():
    """
    Finish the current run.
    
    This is called automatically on exit, but can be called manually.
    """
    global _current_run
    
    if _current_run is not None:
        _current_run.finish()
        _current_run = None


def _auto_finish():
    """Auto-finish handler for atexit."""
    global _current_run
    if _current_run is not None:
        try:
            _current_run.finish()
        except Exception:
            pass
        _current_run = None


def get_run() -> Optional[Run]:
    """Get the current active run, or None if no run is active."""
    return _current_run


# ============================================================================
# Run Discovery - 和 init() 共享同一套目录配置
# ============================================================================

def get_runs_dir(dir: Optional[str] = None) -> Path:
    """
    Get the runs directory.
    
    For visualization tools, this detects the project root from the current
    working directory (where the user runs the command).
    
    Args:
        dir: Override directory (default: auto-detect project root or $VLALAB_DIR or ./vlalab_runs)
    
    Returns:
        Path to the runs directory
    """
    if dir is not None:
        base_dir = dir
    elif "VLALAB_DIR" in os.environ:
        base_dir = os.environ.get("VLALAB_DIR")
    else:
        # 对于可视化/查询工具，从当前工作目录检测项目根目录
        # （用户运行 streamlit 或其他可视化命令的目录）
        project_root = _find_project_root(start_path=Path.cwd())
        if project_root:
            base_dir = project_root / "vlalab_runs"
        else:
            base_dir = Path.cwd() / "vlalab_runs"
    
    return Path(base_dir)


def list_projects(dir: Optional[str] = None) -> List[str]:
    """
    List all projects in the runs directory.
    
    Args:
        dir: Override directory (default: $VLALAB_DIR or ./vlalab_runs)
    
    Returns:
        List of project names
    """
    runs_dir = get_runs_dir(dir)
    if not runs_dir.exists():
        return []
    
    projects = []
    for item in runs_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            projects.append(item.name)
    
    return sorted(projects)


def list_runs(
    project: Optional[str] = None,
    dir: Optional[str] = None,
) -> List[Path]:
    """
    List all runs, optionally filtered by project.
    
    Uses the same directory as init() to ensure consistency.
    
    Args:
        project: Filter by project name (None for all projects)
        dir: Override directory (default: $VLALAB_DIR or ./vlalab_runs)
    
    Returns:
        List of run paths, sorted by modification time (newest first)
    
    Example:
        import vlalab
        
        # List all runs
        runs = vlalab.list_runs()
        
        # List runs in a specific project
        runs = vlalab.list_runs(project="pick_and_place")
        
        for run_path in runs:
            print(run_path.name)
    """
    runs_dir = get_runs_dir(dir)
    if not runs_dir.exists():
        return []
    
    runs = []
    
    def is_run_dir(path: Path) -> bool:
        return (path / "meta.json").exists() or (path / "steps.jsonl").exists()
    
    if project:
        # List runs in a specific project
        project_dir = runs_dir / project
        if project_dir.exists() and project_dir.is_dir():
            for item in project_dir.iterdir():
                if item.is_dir() and is_run_dir(item):
                    runs.append(item)
    else:
        # List all runs across all projects
        for project_item in runs_dir.iterdir():
            if project_item.is_dir() and not project_item.name.startswith("."):
                for run_item in project_item.iterdir():
                    if run_item.is_dir() and is_run_dir(run_item):
                        runs.append(run_item)
    
    # Sort by modification time (newest first)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
