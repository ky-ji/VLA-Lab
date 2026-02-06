"""
VLA-Lab Evaluation Module

Provides open-loop evaluation tools for VLA models.
"""

from vlalab.eval.policy_interface import EvalPolicy, ModalityConfig
from vlalab.eval.open_loop_eval import OpenLoopEvaluator, evaluate_trajectory, DatasetLoader

__all__ = [
    "EvalPolicy",
    "ModalityConfig",
    "OpenLoopEvaluator",
    "evaluate_trajectory",
    "DatasetLoader",
]
