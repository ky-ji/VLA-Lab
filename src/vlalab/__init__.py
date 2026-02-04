"""
VLA-Lab: A toolbox for tracking and visualizing the real-world deployment process of VLA models.

This package provides:
- Unified logging interface for VLA deployment (RunLogger)
- Standardized data schema for observations, actions, and timing
- Streamlit-based visualization tools
- CLI utilities for data conversion and viewing
"""

__version__ = "0.1.0"

from vlalab.logging import RunLogger
from vlalab.schema import StepRecord, RunMeta, ImageRef

__all__ = [
    "__version__",
    "RunLogger",
    "StepRecord",
    "RunMeta",
    "ImageRef",
]
