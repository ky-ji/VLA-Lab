"""
VLA-Lab Adapter for Diffusion Policy (RealWorld-DP)

Converts between DP log format and VLA-Lab unified format.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from vlalab.schema.step import StepRecord, ObsData, ActionData, TimingData, ImageRef


class DPAdapter:
    """
    Adapter for Diffusion Policy inference logs.
    
    DP log format (inference_log_*.json):
    {
        "meta": {
            "checkpoint": "...",
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
                    "send_timestamp": float,
                    "transport_latency_ms": float,
                    "inference_latency_ms": float,
                    "total_latency_ms": float,
                    "message_interval_ms": float,
                },
                "input": {
                    "state": [float, ...],
                    "image_base64": "..."
                },
                "action": {
                    "values": [[float, ...], ...]
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
        Convert a DP step to VLA-Lab StepRecord.
        
        Args:
            step_data: DP step dictionary
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
            send_timestamp=timing_raw.get("send_timestamp"),
            transport_latency_ms=timing_raw.get("transport_latency_ms"),
            inference_latency_ms=timing_raw.get("inference_latency_ms"),
            total_latency_ms=timing_raw.get("total_latency_ms"),
            message_interval_ms=timing_raw.get("message_interval_ms"),
        )
        
        # Convert observation
        input_data = step_data.get("input", {})
        state = input_data.get("state", [])
        
        # Handle image
        image_refs = []
        image_b64 = input_data.get("image_base64")
        if image_b64 and save_images and run_dir:
            # Save image to disk
            from vlalab.logging.run_logger import RunLogger
            import base64
            import cv2
            
            try:
                img_data = base64.b64decode(image_b64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Save image
                from pathlib import Path
                run_path = Path(run_dir)
                images_dir = run_path / "artifacts" / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"step_{step_idx:06d}_default.jpg"
                image_path = images_dir / filename
                cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                image_refs.append(ImageRef(
                    path=f"artifacts/images/{filename}",
                    camera_name="default",
                    shape=list(image_rgb.shape),
                    encoding="jpeg",
                ))
            except Exception:
                pass
        
        obs = ObsData(
            state=state,
            images=image_refs,
        )
        
        # Convert action
        action_raw = step_data.get("action", {})
        action_values = action_raw.get("values", [])
        action = ActionData(
            values=action_values,
            action_dim=len(action_values[0]) if action_values else None,
            chunk_size=len(action_values) if action_values else None,
        )
        
        return StepRecord(
            step_idx=step_idx,
            obs=obs,
            action=action,
            timing=timing,
        )
    
    @staticmethod
    def convert_meta(meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DP metadata to VLA-Lab format.
        
        Args:
            meta_data: DP meta dictionary
            
        Returns:
            VLA-Lab meta dictionary
        """
        return {
            "model_name": "diffusion_policy",
            "model_path": meta_data.get("checkpoint"),
            "model_type": "diffusion_policy",
            "start_time": meta_data.get("start_time"),
        }
    
    @staticmethod
    def get_latency_ms(timing_dict: Dict[str, Any], key_base: str) -> float:
        """
        Get latency value in ms (compatible with old/new format).
        
        Args:
            timing_dict: Timing dictionary
            key_base: Base key name (e.g., 'transport_latency')
            
        Returns:
            Latency in milliseconds
        """
        # Try new format (already in ms)
        new_key = f"{key_base}_ms"
        if new_key in timing_dict and timing_dict[new_key] is not None:
            return timing_dict[new_key]
        
        # Try old format (in seconds)
        if key_base in timing_dict and timing_dict[key_base] is not None:
            return timing_dict[key_base] * 1000
        
        return 0.0
