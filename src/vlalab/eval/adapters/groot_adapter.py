"""
GR00T Policy Adapter

Wraps NVIDIA GR00T policy to conform to the unified EvalPolicy interface.
This adapter handles the conversion between VLA-Lab's standardized observation
format and GR00T's specific input/output formats.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from vlalab.eval.policy_interface import EvalPolicy, ModalityConfig


def parse_observation_gr00t(
    obs: Dict[str, Any],
    modality_configs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert standardized observation to GR00T's expected format.
    
    GR00T expects observations in the format:
    {
        "video": {camera_name: array (1, T, H, W, C)},
        "state": {state_key: array (1, T, D)},
        "language": {lang_key: [[text]]}
    }
    
    Args:
        obs: Standardized observation dict with:
            - "state": Dict[str, np.ndarray] - state vectors
            - "images": Dict[str, np.ndarray] - images (H, W, C)
            - "task_description": Optional[str] - language instruction
        modality_configs: GR00T modality configuration
    
    Returns:
        GR00T-formatted observation dict
    """
    new_obs = {
        "video": {},
        "state": {},
        "language": {},
    }
    
    # Process state modalities
    state_config = modality_configs.get("state", {})
    state_keys = getattr(state_config, "modality_keys", []) if hasattr(state_config, "modality_keys") else state_config.get("modality_keys", [])
    
    for key in state_keys:
        if "state" in obs and key in obs["state"]:
            arr = obs["state"][key]
            # Add batch dimension: (D,) -> (1, 1, D) or (T, D) -> (1, T, D)
            if arr.ndim == 1:
                arr = arr[None, None, :]
            elif arr.ndim == 2:
                arr = arr[None, :]
            new_obs["state"][key] = arr
    
    # Process video/image modalities
    video_config = modality_configs.get("video", {})
    video_keys = getattr(video_config, "modality_keys", []) if hasattr(video_config, "modality_keys") else video_config.get("modality_keys", [])
    
    for key in video_keys:
        if "images" in obs and key in obs["images"]:
            img = obs["images"][key]
            # Add batch and time dimensions: (H, W, C) -> (1, 1, H, W, C)
            if img.ndim == 3:
                img = img[None, None, :]
            elif img.ndim == 4:
                img = img[None, :]
            new_obs["video"][key] = img
    
    # Process language modalities
    lang_config = modality_configs.get("language", {})
    lang_keys = getattr(lang_config, "modality_keys", []) if hasattr(lang_config, "modality_keys") else lang_config.get("modality_keys", [])
    
    task_desc = obs.get("task_description", "")
    for key in lang_keys:
        new_obs["language"][key] = [[task_desc]]
    
    return new_obs


def parse_action_gr00t(
    action: Dict[str, Any],
    action_keys: List[str],
) -> np.ndarray:
    """
    Convert GR00T action output to standardized array format.
    
    GR00T outputs actions in the format:
    {"arm_action": array (T, D1), "gripper_action": array (T, D2), ...}
    
    This function concatenates all action keys into a single array.
    
    Args:
        action: GR00T action dict
        action_keys: List of action keys to concatenate
    
    Returns:
        Action array of shape (action_horizon, total_action_dim)
    """
    action_parts = []
    for key in action_keys:
        full_key = f"action.{key}" if not key.startswith("action.") else key
        if full_key in action:
            arr = action[full_key]
            # Ensure 2D: (T, D)
            arr = np.atleast_1d(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            action_parts.append(arr)
    
    if not action_parts:
        raise ValueError(f"No action keys found. Available: {list(action.keys())}")
    
    return np.concatenate(action_parts, axis=-1)


class GR00TAdapter(EvalPolicy):
    """
    Adapter for NVIDIA GR00T policy.
    
    Usage:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.policy.server_client import PolicyClient
        
        # Option 1: Wrap local policy
        gr00t_policy = Gr00tPolicy(embodiment_tag=..., model_path=...)
        adapter = GR00TAdapter(gr00t_policy)
        
        # Option 2: Wrap remote policy client
        client = PolicyClient(host="localhost", port=5555)
        adapter = GR00TAdapter(client)
        
        # Use with evaluator
        action = adapter.get_action(obs, task_description="pick up the cube")
    """
    
    def __init__(
        self,
        policy: Any,
        embodiment_tag: Optional[str] = None,
    ):
        """
        Initialize the GR00T adapter.
        
        Args:
            policy: GR00T policy instance (Gr00tPolicy or PolicyClient)
            embodiment_tag: Optional embodiment tag (auto-detected if not provided)
        """
        self.policy = policy
        self.embodiment_tag = embodiment_tag
        
        # Get modality config from policy
        self._raw_modality_config = self._get_raw_modality_config()
        self._modality_config = self._build_modality_config()
    
    def _get_raw_modality_config(self) -> Dict[str, Any]:
        """Get raw modality config from the underlying policy."""
        if hasattr(self.policy, "get_modality_config"):
            config = self.policy.get_modality_config()
            # If it's already a dict of ModalityConfig objects, convert
            if isinstance(config, dict):
                return config
        
        if hasattr(self.policy, "modality_configs"):
            return self.policy.modality_configs
        
        # Fallback: return empty config
        return {}
    
    def _build_modality_config(self) -> ModalityConfig:
        """Build VLA-Lab ModalityConfig from GR00T config."""
        raw = self._raw_modality_config
        
        # Extract keys from each modality
        state_keys = []
        action_keys = []
        image_keys = []
        language_keys = []
        action_horizon = 16
        
        if "state" in raw:
            state_cfg = raw["state"]
            state_keys = getattr(state_cfg, "modality_keys", []) if hasattr(state_cfg, "modality_keys") else state_cfg.get("modality_keys", [])
        
        if "action" in raw:
            action_cfg = raw["action"]
            action_keys = getattr(action_cfg, "modality_keys", []) if hasattr(action_cfg, "modality_keys") else action_cfg.get("modality_keys", [])
            # Get action horizon from delta_indices
            delta_indices = getattr(action_cfg, "delta_indices", None) if hasattr(action_cfg, "delta_indices") else action_cfg.get("delta_indices")
            if delta_indices is not None:
                action_horizon = len(delta_indices)
        
        if "video" in raw:
            video_cfg = raw["video"]
            image_keys = getattr(video_cfg, "modality_keys", []) if hasattr(video_cfg, "modality_keys") else video_cfg.get("modality_keys", [])
        
        if "language" in raw:
            lang_cfg = raw["language"]
            language_keys = getattr(lang_cfg, "modality_keys", []) if hasattr(lang_cfg, "modality_keys") else lang_cfg.get("modality_keys", [])
        
        return ModalityConfig(
            state_keys=state_keys,
            action_keys=action_keys,
            image_keys=image_keys,
            language_keys=language_keys,
            action_horizon=action_horizon,
        )
    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get action from GR00T policy.
        
        Args:
            obs: Standardized observation dict
            task_description: Language instruction
        
        Returns:
            Action array of shape (action_horizon, action_dim)
        """
        # Add task description to obs
        obs_with_lang = dict(obs)
        if task_description:
            obs_with_lang["task_description"] = task_description
        
        # Convert to GR00T format
        gr00t_obs = parse_observation_gr00t(obs_with_lang, self._raw_modality_config)
        
        # Get action from policy
        action_dict, info = self.policy.get_action(gr00t_obs)
        
        # Convert action to array
        action_array = parse_action_gr00t(
            action_dict,
            self._modality_config.action_keys,
        )
        
        return action_array
    
    def get_modality_config(self) -> ModalityConfig:
        """Get modality configuration."""
        return self._modality_config
    
    def reset(self) -> None:
        """Reset the policy."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
