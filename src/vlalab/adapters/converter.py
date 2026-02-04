"""
VLA-Lab Legacy Log Converter

Convert old log formats to VLA-Lab run format.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from vlalab.logging.run_logger import RunLogger
from vlalab.adapters.dp_adapter import DPAdapter
from vlalab.adapters.groot_adapter import GR00TAdapter


def detect_log_format(log_path: Path) -> str:
    """
    Detect the format of a log file.
    
    Args:
        log_path: Path to log file
        
    Returns:
        Format string: "dp", "groot", or "unknown"
    """
    with open(log_path, "r") as f:
        data = json.load(f)
    
    meta = data.get("meta", {})
    
    # Check for GR00T markers
    if meta.get("model_type") == "groot":
        return "groot"
    
    # Check for DP markers
    if "checkpoint" in meta:
        return "dp"
    
    # Check step format
    steps = data.get("steps", [])
    if steps:
        first_step = steps[0]
        if "input" in first_step:
            input_data = first_step["input"]
            if "state8" in input_data:
                return "groot"
            if "state" in input_data or "image_base64" in input_data:
                return "dp"
    
    return "unknown"


def convert_legacy_log(
    input_path: Path,
    output_dir: Path,
    input_format: str = "auto",
) -> Dict[str, Any]:
    """
    Convert a legacy log file to VLA-Lab run format.
    
    Args:
        input_path: Path to input log file
        output_dir: Path to output run directory
        input_format: Input format ("dp", "groot", or "auto")
        
    Returns:
        Statistics dictionary
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Load input log
    with open(input_path, "r") as f:
        log_data = json.load(f)
    
    # Detect format if auto
    if input_format == "auto":
        input_format = detect_log_format(input_path)
        if input_format == "unknown":
            raise ValueError(f"Could not detect log format for {input_path}")
    
    # Select adapter
    if input_format == "dp":
        adapter = DPAdapter
    elif input_format == "groot":
        adapter = GR00TAdapter
    else:
        raise ValueError(f"Unknown format: {input_format}")
    
    # Extract metadata
    meta_data = log_data.get("meta", {})
    converted_meta = adapter.convert_meta(meta_data)
    
    # Create run logger
    logger = RunLogger(
        run_dir=output_dir,
        model_name=converted_meta.get("model_name", "unknown"),
        model_path=converted_meta.get("model_path"),
        model_type=converted_meta.get("model_type"),
        task_name="converted",
    )
    
    # Convert steps
    steps = log_data.get("steps", [])
    image_count = 0
    
    for step_data in steps:
        step_record = adapter.convert_step(
            step_data,
            run_dir=str(output_dir),
            save_images=True,
        )
        
        # Count images
        image_count += len(step_record.obs.images)
        
        # Log step
        logger.log_step_raw(
            step_idx=step_record.step_idx,
            obs_dict=step_record.obs.to_dict(),
            action_dict=step_record.action.to_dict(),
            timing_dict=step_record.timing.to_dict(),
            prompt=step_record.prompt,
            tags=step_record.tags,
        )
    
    # Close logger
    logger.close()
    
    return {
        "steps": len(steps),
        "images": image_count,
        "format": input_format,
        "output_dir": str(output_dir),
    }


def convert_dp_log(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Convenience function for DP logs."""
    return convert_legacy_log(input_path, output_dir, "dp")


def convert_groot_log(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Convenience function for GR00T logs."""
    return convert_legacy_log(input_path, output_dir, "groot")
