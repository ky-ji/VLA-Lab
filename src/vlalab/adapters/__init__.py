"""
VLA-Lab Adapters Module

Provides adapters for different VLA frameworks (Diffusion Policy, GR00T, etc.)
"""

from vlalab.adapters.dp_adapter import DPAdapter
from vlalab.adapters.groot_adapter import GR00TAdapter

__all__ = ["DPAdapter", "GR00TAdapter"]
