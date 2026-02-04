"""
VLA-Lab Run Loader

Load and access run data.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import numpy as np

from vlalab.schema.run import RunMeta
from vlalab.schema.step import StepRecord
from vlalab.logging.jsonl_writer import JsonlReader


def load_run_info(run_dir: Path) -> Dict[str, Any]:
    """
    Load basic run information.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with run info
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    steps_path = run_dir / "steps.jsonl"
    
    info = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
    }
    
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            info.update({
                "model_name": meta.get("model_name", "unknown"),
                "task_name": meta.get("task_name", "unknown"),
                "robot_name": meta.get("robot_name", "unknown"),
                "start_time": meta.get("start_time", "unknown"),
                "total_steps": meta.get("total_steps", 0),
            })
    
    if steps_path.exists():
        reader = JsonlReader(steps_path)
        info["actual_steps"] = reader.count()
    
    return info


class RunLoader:
    """
    Load and access run data.
    
    Usage:
        loader = RunLoader("runs/my_run")
        
        # Access metadata
        print(loader.meta.model_name)
        
        # Iterate over steps
        for step in loader.iter_steps():
            print(step.step_idx, step.action.values)
        
        # Access specific step
        step = loader.get_step(10)
        
        # Load image for a step
        image = loader.load_image(step, camera_name="front")
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize run loader.
        
        Args:
            run_dir: Path to run directory
        """
        self.run_dir = Path(run_dir)
        
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
        
        # Load metadata
        meta_path = self.run_dir / "meta.json"
        if meta_path.exists():
            self.meta = RunMeta.load(str(meta_path))
        else:
            # Create minimal metadata
            self.meta = RunMeta(
                run_name=self.run_dir.name,
                start_time="unknown",
            )
        
        # Initialize step reader
        self._steps_path = self.run_dir / "steps.jsonl"
        self._steps_cache: Optional[List[StepRecord]] = None
    
    @property
    def step_count(self) -> int:
        """Get the number of steps."""
        if self._steps_path.exists():
            return JsonlReader(self._steps_path).count()
        return 0
    
    def iter_steps(self) -> Iterator[StepRecord]:
        """Iterate over all steps."""
        if not self._steps_path.exists():
            return
        
        reader = JsonlReader(self._steps_path)
        for data in reader:
            yield StepRecord.from_dict(data)
    
    def get_steps(self) -> List[StepRecord]:
        """Get all steps as a list (cached)."""
        if self._steps_cache is None:
            self._steps_cache = list(self.iter_steps())
        return self._steps_cache
    
    def get_step(self, step_idx: int) -> Optional[StepRecord]:
        """
        Get a specific step by index.
        
        Note: This loads steps sequentially, so it's O(n).
        For random access, use get_steps() first.
        """
        for step in self.iter_steps():
            if step.step_idx == step_idx:
                return step
        return None
    
    def load_image(
        self,
        step: StepRecord,
        camera_name: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Load image for a step.
        
        Args:
            step: StepRecord
            camera_name: Camera name (None = first camera)
            
        Returns:
            Image array (RGB) or None if not found
        """
        import cv2
        
        if not step.obs.images:
            return None
        
        # Find image ref
        image_ref = None
        for ref in step.obs.images:
            if camera_name is None or ref.camera_name == camera_name:
                image_ref = ref
                break
        
        if image_ref is None:
            return None
        
        # Load image
        image_path = self.run_dir / image_ref.path
        if not image_path.exists():
            return None
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def get_all_states(self) -> np.ndarray:
        """Get all states as a numpy array."""
        steps = self.get_steps()
        if not steps:
            return np.array([])
        
        states = [step.obs.state for step in steps if step.obs.state]
        if not states:
            return np.array([])
        
        return np.array(states)
    
    def get_all_actions(self) -> List[np.ndarray]:
        """Get all actions as a list of arrays."""
        steps = self.get_steps()
        actions = []
        for step in steps:
            if step.action.values:
                actions.append(np.array(step.action.values))
        return actions
    
    def get_timing_series(self) -> Dict[str, np.ndarray]:
        """Get timing data as arrays."""
        steps = self.get_steps()
        
        timing_keys = [
            "transport_latency_ms",
            "inference_latency_ms",
            "total_latency_ms",
            "message_interval_ms",
        ]
        
        result = {key: [] for key in timing_keys}
        
        for step in steps:
            for key in timing_keys:
                value = getattr(step.timing, key, None)
                result[key].append(value if value is not None else np.nan)
        
        return {key: np.array(values) for key, values in result.items()}
