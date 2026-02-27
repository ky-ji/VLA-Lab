"""
VLA-Lab: The missing toolkit for VLA model deployment.

Debug, visualize, and analyze your VLA deployments in the real world.

Quick Start:
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

__version__ = "0.1.1"

# Simple API
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

# Evaluation API — lazy import to avoid pulling in heavy deps (torch, matplotlib)
# at module load time. Actual imports happen on first attribute access.
_EVAL_LAZY = {
    "EvalPolicy": "vlalab.eval.policy_interface",
    "ModalityConfig": "vlalab.eval.policy_interface",
    "OpenLoopEvaluator": "vlalab.eval.open_loop_eval",
}


def __getattr__(name: str):
    if name in _EVAL_LAZY:
        import importlib
        mod = importlib.import_module(_EVAL_LAZY[name])
        obj = getattr(mod, name)
        # Cache on module so subsequent access is fast
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'vlalab' has no attribute {name}")


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
    # Evaluation API (lazy)
    "EvalPolicy",
    "ModalityConfig",
    "OpenLoopEvaluator",
]
