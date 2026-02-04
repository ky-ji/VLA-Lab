"""
VLA-Lab Run Logger

Unified logging interface for VLA real-world deployment.
"""

import os
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import numpy as np

from vlalab.schema.step import StepRecord, ObsData, ActionData, TimingData, ImageRef
from vlalab.schema.run import RunMeta, CameraConfig
from vlalab.logging.jsonl_writer import JsonlWriter


class RunLogger:
    """
    Unified logger for VLA deployment runs.
    
    Creates a run directory with:
    - meta.json: Run metadata
    - steps.jsonl: Step records (one per line)
    - artifacts/images/: Image files
    
    Usage:
        logger = RunLogger(
            run_dir="runs/my_run",
            model_name="diffusion_policy",
            task_name="pick_and_place",
        )
        
        # Log a step
        logger.log_step(
            step_idx=0,
            state=[0.5, 0.2, 0.3, 0, 0, 0, 1, 1.0],
            action=[[0.51, 0.21, 0.31, 0, 0, 0, 1, 1.0]],
            images={"front": image_array},
            timing={...},
        )
        
        # Close when done
        logger.close()
    """
    
    def __init__(
        self,
        run_dir: Union[str, Path],
        model_name: str = "unknown",
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        task_name: str = "unknown",
        task_prompt: Optional[str] = None,
        robot_name: str = "unknown",
        robot_type: Optional[str] = None,
        cameras: Optional[List[Dict[str, Any]]] = None,
        inference_freq: Optional[float] = None,
        action_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        server_config: Optional[Dict[str, Any]] = None,
        client_config: Optional[Dict[str, Any]] = None,
        auto_create: bool = True,
        image_quality: int = 85,
    ):
        """
        Initialize the run logger.
        
        Args:
            run_dir: Directory to store run data
            model_name: Name of the model
            model_path: Path to the model checkpoint
            model_type: Type of model (diffusion_policy, groot, etc.)
            task_name: Name of the task
            task_prompt: Language prompt for the task
            robot_name: Name of the robot
            robot_type: Type of robot (franka, ur5, etc.)
            cameras: List of camera configurations
            inference_freq: Inference frequency in Hz
            action_dim: Action dimension
            action_horizon: Action horizon (chunk size)
            server_config: Server configuration dict
            client_config: Client configuration dict
            auto_create: Whether to create the run directory
            image_quality: JPEG quality for saved images (1-100)
        """
        self.run_dir = Path(run_dir)
        self.image_quality = image_quality
        self._step_count = 0
        self._closed = False
        
        if auto_create:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "artifacts" / "images").mkdir(parents=True, exist_ok=True)
        
        # Create run name from directory name
        run_name = self.run_dir.name
        start_time = datetime.now().isoformat()
        
        # Build camera configs
        camera_configs = []
        if cameras:
            for cam in cameras:
                if isinstance(cam, dict):
                    camera_configs.append(CameraConfig(**cam))
                else:
                    camera_configs.append(cam)
        
        # Create metadata
        self.meta = RunMeta(
            run_name=run_name,
            start_time=start_time,
            model_name=model_name,
            model_path=model_path,
            model_type=model_type,
            task_name=task_name,
            task_prompt=task_prompt,
            robot_name=robot_name,
            robot_type=robot_type,
            cameras=camera_configs,
            inference_freq=inference_freq,
            action_dim=action_dim,
            action_horizon=action_horizon,
            server_config=server_config or {},
            client_config=client_config or {},
        )
        
        # Save initial metadata
        self._save_meta()
        
        # Initialize JSONL writer
        self._jsonl_writer = JsonlWriter(self.run_dir / "steps.jsonl")
    
    def _save_meta(self):
        """Save metadata to meta.json."""
        self.meta.save(str(self.run_dir / "meta.json"))
    
    def _save_image(
        self,
        image: np.ndarray,
        step_idx: int,
        camera_name: str = "default",
    ) -> ImageRef:
        """
        Save an image to the artifacts directory.
        
        Args:
            image: Image array (H, W, C) in RGB or BGR format
            step_idx: Step index
            camera_name: Camera name
            
        Returns:
            ImageRef with path information
        """
        # Import cv2 here to avoid hard dependency
        import cv2
        
        # Create filename
        filename = f"step_{step_idx:06d}_{camera_name}.jpg"
        rel_path = f"artifacts/images/{filename}"
        abs_path = self.run_dir / rel_path
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert RGB to BGR for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Save image
        cv2.imwrite(str(abs_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        return ImageRef(
            path=rel_path,
            camera_name=camera_name,
            shape=list(image.shape),
            encoding="jpeg",
        )
    
    def _decode_base64_image(self, b64_str: str) -> Optional[np.ndarray]:
        """Decode a base64-encoded image."""
        try:
            import cv2
            img_data = base64.b64decode(b64_str)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    
    def log_step(
        self,
        step_idx: int,
        state: Optional[List[float]] = None,
        pose: Optional[List[float]] = None,
        gripper: Optional[float] = None,
        action: Optional[Union[List[float], List[List[float]]]] = None,
        images: Optional[Dict[str, Union[np.ndarray, str]]] = None,
        timing: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a single step.
        
        Args:
            step_idx: Step index
            state: Low-dim state vector
            pose: Robot pose [x, y, z, qx, qy, qz, qw] or similar
            gripper: Gripper state (0-1)
            action: Action values (single or chunk)
            images: Dict of camera_name -> image_array or base64_string
            timing: Timing information dict
            prompt: Language prompt
            tags: Additional tags/metadata
        """
        if self._closed:
            raise RuntimeError("Logger is closed")
        
        # Process images
        image_refs = []
        if images:
            for camera_name, image_data in images.items():
                # Handle base64 string
                if isinstance(image_data, str):
                    image_array = self._decode_base64_image(image_data)
                    if image_array is None:
                        continue
                else:
                    image_array = image_data
                
                # Save image and get reference
                ref = self._save_image(image_array, step_idx, camera_name)
                image_refs.append(ref)
        
        # Build observation
        obs = ObsData(
            state=state or [],
            images=image_refs,
            pose=pose,
            gripper=gripper,
        )
        
        # Build action
        action_data = ActionData()
        if action is not None:
            # Normalize to list of lists (chunk format)
            if action and not isinstance(action[0], (list, tuple)):
                action = [action]  # Single action -> chunk of 1
            action_data.values = [list(a) for a in action]
            if action:
                action_data.action_dim = len(action[0])
                action_data.chunk_size = len(action)
        
        # Build timing
        timing_data = TimingData()
        if timing:
            timing_data = TimingData.from_dict(timing)
            timing_data.compute_latencies()
        
        # Create step record
        record = StepRecord(
            step_idx=step_idx,
            obs=obs,
            action=action_data,
            timing=timing_data,
            prompt=prompt,
            tags=tags or {},
        )
        
        # Write to JSONL
        self._jsonl_writer.write(record)
        self._step_count += 1
    
    def log_step_raw(
        self,
        step_idx: int,
        obs_dict: Dict[str, Any],
        action_dict: Dict[str, Any],
        timing_dict: Dict[str, Any],
        **kwargs,
    ):
        """
        Log a step from raw dictionaries (for adapters).
        
        Args:
            step_idx: Step index
            obs_dict: Observation dictionary
            action_dict: Action dictionary
            timing_dict: Timing dictionary
            **kwargs: Additional fields
        """
        record = StepRecord(
            step_idx=step_idx,
            obs=ObsData.from_dict(obs_dict),
            action=ActionData.from_dict(action_dict),
            timing=TimingData.from_dict(timing_dict),
            **kwargs,
        )
        self._jsonl_writer.write(record)
        self._step_count += 1
    
    def update_meta(self, **kwargs):
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self.meta, key):
                setattr(self.meta, key, value)
        self._save_meta()
    
    def close(self):
        """Close the logger and finalize metadata."""
        if self._closed:
            return
        
        # Update final metadata
        self.meta.end_time = datetime.now().isoformat()
        self.meta.total_steps = self._step_count
        
        # Calculate duration
        start = datetime.fromisoformat(self.meta.start_time)
        end = datetime.fromisoformat(self.meta.end_time)
        self.meta.total_duration_s = (end - start).total_seconds()
        
        self._save_meta()
        self._jsonl_writer.close()
        self._closed = True
    
    @property
    def step_count(self) -> int:
        """Get the number of logged steps."""
        return self._step_count
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
