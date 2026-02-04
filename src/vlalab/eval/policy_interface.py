"""
VLA-Lab Unified Policy Interface

Defines a standard interface for VLA policies to enable model-agnostic evaluation.
Each model (GR00T, Diffusion Policy, OpenVLA, etc.) implements an adapter that
conforms to this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ModalityConfig:
    """
    Configuration describing the modalities a policy expects/produces.
    
    This enables the evaluator to understand what data to extract from
    datasets and how to format inputs/outputs.
    """
    # State modality keys (e.g., ["joint_position", "gripper_position"])
    state_keys: List[str] = field(default_factory=list)
    
    # Action modality keys (e.g., ["arm_action", "gripper_action"])
    action_keys: List[str] = field(default_factory=list)
    
    # Image/video modality keys (e.g., ["ego_view", "front_view"])
    image_keys: List[str] = field(default_factory=list)
    
    # Language modality keys (e.g., ["annotation.human.action.task_description"])
    language_keys: List[str] = field(default_factory=list)
    
    # Action horizon (number of future actions predicted)
    action_horizon: int = 16
    
    # Action dimension (total dim after concatenating all action keys)
    action_dim: Optional[int] = None
    
    # State dimension (total dim after concatenating all state keys)
    state_dim: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_keys": self.state_keys,
            "action_keys": self.action_keys,
            "image_keys": self.image_keys,
            "language_keys": self.language_keys,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModalityConfig":
        return cls(
            state_keys=d.get("state_keys", []),
            action_keys=d.get("action_keys", []),
            image_keys=d.get("image_keys", []),
            language_keys=d.get("language_keys", []),
            action_horizon=d.get("action_horizon", 16),
            action_dim=d.get("action_dim"),
            state_dim=d.get("state_dim"),
        )


class EvalPolicy(ABC):
    """
    Abstract base class for VLA policies in evaluation mode.
    
    Each model implementation (GR00T, DP, OpenVLA) provides an adapter
    that wraps the actual policy and implements this interface.
    
    The interface is designed to be minimal and model-agnostic:
    - get_action(): Takes a standardized observation dict, returns action array
    - get_modality_config(): Returns what modalities the policy expects
    - reset(): Resets any internal state (for stateful policies)
    """
    
    @abstractmethod
    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get action from the policy given an observation.
        
        Args:
            obs: Standardized observation dictionary with keys:
                - "state": Dict[str, np.ndarray] - state vectors by key
                - "images": Dict[str, np.ndarray] - images by camera name (H, W, C)
            task_description: Optional language instruction
            
        Returns:
            Action array of shape (action_horizon, action_dim)
            The action_dim is the concatenation of all action modality keys.
        """
        pass
    
    @abstractmethod
    def get_modality_config(self) -> ModalityConfig:
        """
        Get the modality configuration for this policy.
        
        Returns:
            ModalityConfig describing expected inputs and outputs
        """
        pass
    
    def reset(self) -> None:
        """
        Reset any internal state.
        
        Override this for stateful policies (e.g., those with history buffers).
        """
        pass
    
    @property
    def action_horizon(self) -> int:
        """Convenience property for action horizon."""
        return self.get_modality_config().action_horizon


class DummyPolicy(EvalPolicy):
    """
    A dummy policy for testing that returns random actions.
    """
    
    def __init__(
        self,
        action_dim: int = 8,
        action_horizon: int = 16,
        state_keys: Optional[List[str]] = None,
        action_keys: Optional[List[str]] = None,
        image_keys: Optional[List[str]] = None,
    ):
        self._config = ModalityConfig(
            state_keys=state_keys or ["joint_position"],
            action_keys=action_keys or ["action"],
            image_keys=image_keys or ["front"],
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        return np.random.randn(self._config.action_horizon, self._config.action_dim)
    
    def get_modality_config(self) -> ModalityConfig:
        return self._config
