"""
VLA-Lab Policy Adapters

Adapters that wrap specific VLA model implementations to conform to
the unified EvalPolicy interface.
"""

from vlalab.eval.adapters.groot_adapter import GR00TAdapter
from vlalab.eval.adapters.dp_adapter import DiffusionPolicyAdapter

__all__ = [
    "GR00TAdapter",
    "DiffusionPolicyAdapter",
]
