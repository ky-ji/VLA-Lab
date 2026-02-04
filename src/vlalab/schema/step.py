"""
VLA-Lab Step Schema

Defines the data structure for a single inference step.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json


@dataclass
class ImageRef:
    """Reference to an image artifact file."""
    path: str  # Relative path from run_dir
    camera_name: str = "default"
    shape: Optional[List[int]] = None  # [H, W, C]
    encoding: str = "jpeg"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRef":
        return cls(**data)


@dataclass
class ObsData:
    """Observation data for a step."""
    state: List[float] = field(default_factory=list)  # Low-dim state (pose, gripper, etc.)
    images: List[ImageRef] = field(default_factory=list)  # Image references
    
    # Optional detailed state breakdown
    pose: Optional[List[float]] = None  # [x, y, z, qx, qy, qz, qw] or [x, y, z, qw, qx, qy, qz]
    gripper: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "state": self.state,
            "images": [img.to_dict() for img in self.images],
        }
        if self.pose is not None:
            d["pose"] = self.pose
        if self.gripper is not None:
            d["gripper"] = self.gripper
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObsData":
        images = [ImageRef.from_dict(img) for img in data.get("images", [])]
        return cls(
            state=data.get("state", []),
            images=images,
            pose=data.get("pose"),
            gripper=data.get("gripper"),
        )


@dataclass
class ActionData:
    """Action data for a step."""
    values: List[List[float]] = field(default_factory=list)  # Action chunk: [[a1], [a2], ...]
    
    # Optional metadata
    action_dim: Optional[int] = None
    chunk_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"values": self.values}
        if self.action_dim is not None:
            d["action_dim"] = self.action_dim
        if self.chunk_size is not None:
            d["chunk_size"] = self.chunk_size
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionData":
        return cls(
            values=data.get("values", []),
            action_dim=data.get("action_dim"),
            chunk_size=data.get("chunk_size"),
        )


@dataclass
class TimingData:
    """Timing data for a step (all times in milliseconds or Unix timestamps)."""
    # Timestamps (Unix time, float)
    client_send: Optional[float] = None
    server_recv: Optional[float] = None
    infer_start: Optional[float] = None
    infer_end: Optional[float] = None
    send_timestamp: Optional[float] = None
    
    # Computed latencies (milliseconds)
    transport_latency_ms: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    message_interval_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in asdict(self).items():
            if v is not None:
                d[k] = v
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimingData":
        return cls(
            client_send=data.get("client_send"),
            server_recv=data.get("server_recv"),
            infer_start=data.get("infer_start"),
            infer_end=data.get("infer_end"),
            send_timestamp=data.get("send_timestamp"),
            transport_latency_ms=data.get("transport_latency_ms"),
            inference_latency_ms=data.get("inference_latency_ms"),
            total_latency_ms=data.get("total_latency_ms"),
            message_interval_ms=data.get("message_interval_ms"),
        )
    
    def compute_latencies(self):
        """Compute latency values from timestamps."""
        if self.server_recv is not None and self.client_send is not None:
            self.transport_latency_ms = (self.server_recv - self.client_send) * 1000
        
        if self.infer_end is not None and self.infer_start is not None:
            self.inference_latency_ms = (self.infer_end - self.infer_start) * 1000
        
        if self.send_timestamp is not None and self.client_send is not None:
            self.total_latency_ms = (self.send_timestamp - self.client_send) * 1000


@dataclass
class StepRecord:
    """A single step record in the inference log."""
    step_idx: int
    obs: ObsData = field(default_factory=ObsData)
    action: ActionData = field(default_factory=ActionData)
    timing: TimingData = field(default_factory=TimingData)
    
    # Optional fields
    tags: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None  # For language-conditioned models
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "step_idx": self.step_idx,
            "obs": self.obs.to_dict(),
            "action": self.action.to_dict(),
            "timing": self.timing.to_dict(),
        }
        if self.tags:
            d["tags"] = self.tags
        if self.prompt is not None:
            d["prompt"] = self.prompt
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepRecord":
        return cls(
            step_idx=data["step_idx"],
            obs=ObsData.from_dict(data.get("obs", {})),
            action=ActionData.from_dict(data.get("action", {})),
            timing=TimingData.from_dict(data.get("timing", {})),
            tags=data.get("tags", {}),
            prompt=data.get("prompt"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "StepRecord":
        return cls.from_dict(json.loads(json_str))
