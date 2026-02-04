"""
VLA-Lab: A toolbox for tracking and visualizing the real-world deployment process of VLA models.

Simple API (SwanLab-style):
    import vlalab

    # Initialize a run
    run = vlalab.init(
        project="pick_and_place",
        config={
            "model": "diffusion_policy",
            "action_horizon": 8,
        },
    )

    # Access config
    print(f"Action horizon: {run.config.action_horizon}")

    # Log steps
    for step in range(100):
        vlalab.log({
            "state": obs["state"],
            "action": action,
            "images": {"front": obs["image"]},
            "inference_latency_ms": latency,
        })

    # Finish (auto-called on exit)
    vlalab.finish()

Advanced API:
    from vlalab import RunLogger
    
    logger = RunLogger(run_dir="runs/my_run", model_name="diffusion_policy")
    logger.log_step(step_idx=0, state=[...], action=[...])
    logger.close()
"""

__version__ = "0.1.0"

# Simple API (SwanLab-style)
from vlalab.core import (
    init,
    log,
    log_image,
    finish,
    get_run,
    Run,
    Config,
    # Run discovery (uses same dir as init)
    get_runs_dir,
    list_projects,
    list_runs,
)

# Advanced API
from vlalab.logging import RunLogger
from vlalab.schema import StepRecord, RunMeta, ImageRef

__all__ = [
    # Version
    "__version__",
    # Simple API
    "init",
    "log",
    "log_image",
    "finish",
    "get_run",
    "Run",
    "Config",
    # Run discovery
    "get_runs_dir",
    "list_projects",
    "list_runs",
    # Advanced API
    "RunLogger",
    "StepRecord",
    "RunMeta",
    "ImageRef",
]
