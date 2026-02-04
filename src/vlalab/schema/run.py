"""
VLA-Lab Run Schema

Defines the metadata structure for a deployment run.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class CameraConfig:
    """Camera configuration."""
    name: str
    resolution: Optional[List[int]] = None  # [width, height]
    fps: Optional[float] = None
    camera_type: Optional[str] = None  # "realsense", "usb", etc.
    serial_number: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name}
        if self.resolution is not None:
            d["resolution"] = self.resolution
        if self.fps is not None:
            d["fps"] = self.fps
        if self.camera_type is not None:
            d["camera_type"] = self.camera_type
        if self.serial_number is not None:
            d["serial_number"] = self.serial_number
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraConfig":
        return cls(**data)


@dataclass
class RunMeta:
    """Metadata for a deployment run."""
    # Required fields
    run_name: str
    start_time: str  # ISO format timestamp
    
    # Model info
    model_name: str = "unknown"
    model_path: Optional[str] = None
    model_type: Optional[str] = None  # "diffusion_policy", "groot", etc.
    
    # Task info
    task_name: str = "unknown"
    task_prompt: Optional[str] = None
    
    # Robot info
    robot_name: str = "unknown"
    robot_type: Optional[str] = None  # "franka", "ur5", etc.
    
    # Camera info
    cameras: List[CameraConfig] = field(default_factory=list)
    
    # Inference config
    inference_freq: Optional[float] = None  # Hz
    action_dim: Optional[int] = None
    action_horizon: Optional[int] = None
    
    # Deployment info
    server_config: Dict[str, Any] = field(default_factory=dict)
    client_config: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics (updated during/after run)
    end_time: Optional[str] = None
    total_steps: int = 0
    total_duration_s: Optional[float] = None
    
    # Version info
    vlalab_version: str = "0.1.0"
    framework_version: Optional[str] = None
    
    # Extra fields
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_name": self.run_name,
            "start_time": self.start_time,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "robot_name": self.robot_name,
            "cameras": [cam.to_dict() for cam in self.cameras],
            "total_steps": self.total_steps,
            "vlalab_version": self.vlalab_version,
        }
        
        # Add optional fields if set
        optional_fields = [
            "model_path", "model_type", "task_prompt", "robot_type",
            "inference_freq", "action_dim", "action_horizon",
            "end_time", "total_duration_s", "framework_version",
        ]
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                d[field_name] = value
        
        if self.server_config:
            d["server_config"] = self.server_config
        if self.client_config:
            d["client_config"] = self.client_config
        if self.extra:
            d["extra"] = self.extra
        
        return d
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str):
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMeta":
        cameras = [
            CameraConfig.from_dict(cam) if isinstance(cam, dict) else cam
            for cam in data.get("cameras", [])
        ]
        
        return cls(
            run_name=data["run_name"],
            start_time=data["start_time"],
            model_name=data.get("model_name", "unknown"),
            model_path=data.get("model_path"),
            model_type=data.get("model_type"),
            task_name=data.get("task_name", "unknown"),
            task_prompt=data.get("task_prompt"),
            robot_name=data.get("robot_name", "unknown"),
            robot_type=data.get("robot_type"),
            cameras=cameras,
            inference_freq=data.get("inference_freq"),
            action_dim=data.get("action_dim"),
            action_horizon=data.get("action_horizon"),
            server_config=data.get("server_config", {}),
            client_config=data.get("client_config", {}),
            end_time=data.get("end_time"),
            total_steps=data.get("total_steps", 0),
            total_duration_s=data.get("total_duration_s"),
            vlalab_version=data.get("vlalab_version", "0.1.0"),
            framework_version=data.get("framework_version"),
            extra=data.get("extra", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "RunMeta":
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: str) -> "RunMeta":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())
