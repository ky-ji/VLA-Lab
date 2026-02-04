"""
VLA-Lab Adapter for Isaac-GR00T

Converts between GR00T log format and VLA-Lab unified format.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from vlalab.schema.step import StepRecord, ObsData, ActionData, TimingData, ImageRef


class GR00TAdapter:
    """
    Adapter for GR00T inference logs.
    
    GR00T log format (inference_log_groot_*.json):
    {
        "meta": {
            "model_path": "...",
            "model_type": "groot",
            "start_time": "..."
        },
        "steps": [
            {
                "step": 0,
                "timing": {
                    "client_send": float,
                    "server_recv": float,
                    "infer_start": float,
                    "infer_end": float,
                    "inference_latency_ms": float,
                },
                "input": {
                    "state8": [float, ...],  # [x, y, z, qx, qy, qz, qw, gripper]
                    "prompt": "..."
                },
                "action": {
                    "action8": [[float, ...], ...]  # Chunk of 8D actions
                }
            },
            ...
        ]
    }
    """
    
    @staticmethod
    def convert_step(
        step_data: Dict[str, Any],
        run_dir: Optional[str] = None,
        save_images: bool = False,
    ) -> StepRecord:
        """
        Convert a GR00T step to VLA-Lab StepRecord.
        
        Args:
            step_data: GR00T step dictionary
            run_dir: Run directory (for saving images)
            save_images: Whether to save images to disk
            
        Returns:
            StepRecord
        """
        step_idx = step_data.get("step", 0)
        
        # Convert timing
        timing_raw = step_data.get("timing", {})
        timing = TimingData(
            client_send=timing_raw.get("client_send"),
            server_recv=timing_raw.get("server_recv"),
            infer_start=timing_raw.get("infer_start"),
            infer_end=timing_raw.get("infer_end"),
            inference_latency_ms=timing_raw.get("inference_latency_ms"),
        )
        
        # Compute additional latencies if possible
        if timing.server_recv and timing.client_send:
            timing.transport_latency_ms = (timing.server_recv - timing.client_send) * 1000
        
        # Convert observation
        input_data = step_data.get("input", {})
        state8 = input_data.get("state8", [])
        prompt = input_data.get("prompt")
        
        # Parse state8: [x, y, z, qx, qy, qz, qw, gripper]
        pose = state8[:7] if len(state8) >= 7 else None
        gripper = state8[7] if len(state8) >= 8 else None
        
        obs = ObsData(
            state=state8,
            images=[],  # GR00T logs typically don't include images
            pose=pose,
            gripper=gripper,
        )
        
        # Convert action
        action_raw = step_data.get("action", {})
        action8 = action_raw.get("action8", [])
        
        action = ActionData(
            values=action8,
            action_dim=len(action8[0]) if action8 else 8,
            chunk_size=len(action8) if action8 else None,
        )
        
        return StepRecord(
            step_idx=step_idx,
            obs=obs,
            action=action,
            timing=timing,
            prompt=prompt,
        )
    
    @staticmethod
    def convert_meta(meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert GR00T metadata to VLA-Lab format.
        
        Args:
            meta_data: GR00T meta dictionary
            
        Returns:
            VLA-Lab meta dictionary
        """
        return {
            "model_name": "groot",
            "model_path": meta_data.get("model_path"),
            "model_type": "groot",
            "start_time": meta_data.get("start_time"),
        }
    
    @staticmethod
    def state8_to_pose_gripper(state8: List[float]) -> tuple:
        """
        Parse state8 into pose and gripper.
        
        Args:
            state8: [x, y, z, qx, qy, qz, qw, gripper]
            
        Returns:
            (pose_7d, gripper)
        """
        if len(state8) >= 8:
            return state8[:7], state8[7]
        elif len(state8) >= 7:
            return state8[:7], None
        else:
            return state8, None
