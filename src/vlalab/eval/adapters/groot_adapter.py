"""
GR00T Policy Adapter

Wraps NVIDIA GR00T policy to conform to the unified EvalPolicy interface.
This adapter handles the conversion between VLA-Lab's standardized observation
format and GR00T's specific input/output formats.

Key design: Key mappings between VLA-Lab's expected keys and the actual
policy output keys are auto-discovered at first inference, not hardcoded.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from vlalab.eval.policy_interface import EvalPolicy, ModalityConfig


logger = logging.getLogger(__name__)


def _get_modality_keys(config, key: str) -> List[str]:
    """Extract modality_keys from a config object (supports both attr and dict)."""
    cfg = config.get(key, {}) if isinstance(config, dict) else getattr(config, key, {})
    if hasattr(cfg, "modality_keys"):
        return cfg.modality_keys
    if isinstance(cfg, dict):
        return cfg.get("modality_keys", [])
    return []


def resolve_key_mapping(
    expected_keys: List[str],
    available_keys: List[str],
) -> Dict[str, str]:
    """
    Auto-discover the mapping from expected keys to actual keys in a dict.

    Resolution strategy (first match wins):
      1. Exact match: "robot_eef_pose" in available
      2. Bare-name match: strip any dotted prefix from both sides
         e.g. expected "robot_eef_pose" matches available "action.robot_eef_pose"
              or expected "action.gripper" matches available "gripper"

    Args:
        expected_keys: Keys the adapter expects (from modality config)
        available_keys: Keys actually present in the policy output

    Returns:
        Dict mapping expected_key -> available_key
    """
    mapping: Dict[str, str] = {}
    avail_set = set(available_keys)

    # Build a bare-name -> full-name index for available keys
    bare_to_avail: Dict[str, str] = {}
    for ak in available_keys:
        bare = ak.rsplit(".", 1)[-1]  # "action.gripper" -> "gripper"
        bare_to_avail[bare] = ak

    for key in expected_keys:
        # Strategy 1: exact match
        if key in avail_set:
            mapping[key] = key
            continue

        # Strategy 2: compare bare names
        bare_expected = key.rsplit(".", 1)[-1]
        if bare_expected in bare_to_avail:
            mapping[key] = bare_to_avail[bare_expected]
            continue

        logger.warning(
            f"Could not resolve key '{key}' in available keys {available_keys}"
        )

    return mapping


def parse_action_output(
    action_dict: Dict[str, Any],
    key_mapping: Dict[str, str],
) -> np.ndarray:
    """
    Convert a policy's action dict output to a single concatenated array.

    Uses a pre-resolved key mapping so this function is model-agnostic —
    it does not assume any naming convention (e.g. "action." prefix).

    Args:
        action_dict: Raw action dict from policy.get_action()
        key_mapping: Mapping {expected_key -> actual_key_in_dict}

    Returns:
        Action array of shape (action_horizon, total_action_dim)
    """
    action_parts = []
    for expected_key, actual_key in key_mapping.items():
        if actual_key not in action_dict:
            raise KeyError(
                f"Resolved key '{actual_key}' (for '{expected_key}') "
                f"not found in action dict. Available: {list(action_dict.keys())}"
            )
        arr = np.asarray(action_dict[actual_key])
        # Unbatch if needed: (B, T, D) -> (T, D)
        if arr.ndim == 3:
            arr = arr[0]
        # Ensure at least 2D: scalar -> (1, 1), (T,) -> (T, 1)
        arr = np.atleast_1d(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        action_parts.append(arr)

    if not action_parts:
        raise ValueError(
            f"No action keys resolved. Action dict keys: {list(action_dict.keys())}"
        )

    return np.concatenate(action_parts, axis=-1)


def build_gr00t_observation(
    obs: Dict[str, Any],
    modality_configs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert VLA-Lab standardized observation to GR00T's nested format.

    VLA-Lab obs:
        {"state": {k: arr}, "images": {k: arr}, "task_description": str}
    GR00T obs:
        {"video": {k: (1,T,H,W,C)}, "state": {k: (1,T,D)}, "language": {k: [[str]]}}
    """
    new_obs: Dict[str, dict] = {"video": {}, "state": {}, "language": {}}

    # State
    for key in _get_modality_keys(modality_configs, "state"):
        if "state" in obs and key in obs["state"]:
            arr = obs["state"][key]
            if arr.ndim == 1:
                arr = arr[None, None, :]
            elif arr.ndim == 2:
                arr = arr[None, :]
            new_obs["state"][key] = arr

    # Video / images
    for key in _get_modality_keys(modality_configs, "video"):
        if "images" in obs and key in obs["images"]:
            img = obs["images"][key]
            if img.ndim == 3:
                img = img[None, None, :]
            elif img.ndim == 4:
                img = img[None, :]
            new_obs["video"][key] = img

    # Language
    task_desc = obs.get("task_description", "")
    for key in _get_modality_keys(modality_configs, "language"):
        new_obs["language"][key] = [[task_desc]]

    return new_obs


class GR00TAdapter(EvalPolicy):
    """
    Adapter for NVIDIA GR00T policy.

    Key design: action key mapping is auto-discovered on the first call
    to get_action(), so no naming convention (e.g. "action." prefix)
    is hardcoded.

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
        self.policy = policy
        self.embodiment_tag = embodiment_tag

        # Get modality config from policy
        self._raw_modality_config = self._get_raw_modality_config()
        self._modality_config = self._build_modality_config()

        # Lazy-initialized: resolved once on first get_action call
        self._action_key_mapping: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_modality_config(self) -> Dict[str, Any]:
        """Get raw modality config from the underlying policy."""
        if hasattr(self.policy, "get_modality_config"):
            config = self.policy.get_modality_config()
            if isinstance(config, dict):
                return config

        if hasattr(self.policy, "modality_configs"):
            return self.policy.modality_configs

        return {}

    def _build_modality_config(self) -> ModalityConfig:
        """Build VLA-Lab ModalityConfig from the raw policy config."""
        raw = self._raw_modality_config

        state_keys = _get_modality_keys(raw, "state")
        action_keys = _get_modality_keys(raw, "action")
        image_keys = _get_modality_keys(raw, "video")
        language_keys = _get_modality_keys(raw, "language")

        action_horizon = 16
        if "action" in raw:
            action_cfg = raw["action"]
            delta = (
                getattr(action_cfg, "delta_indices", None)
                if hasattr(action_cfg, "delta_indices")
                else action_cfg.get("delta_indices") if isinstance(action_cfg, dict) else None
            )
            if delta is not None:
                action_horizon = len(delta)

        return ModalityConfig(
            state_keys=state_keys,
            action_keys=action_keys,
            image_keys=image_keys,
            language_keys=language_keys,
            action_horizon=action_horizon,
        )

    def _ensure_action_key_mapping(self, action_dict: Dict[str, Any]) -> None:
        """Resolve and cache the action key mapping on first call."""
        if self._action_key_mapping is not None:
            return

        self._action_key_mapping = resolve_key_mapping(
            expected_keys=self._modality_config.action_keys,
            available_keys=list(action_dict.keys()),
        )
        logger.info(f"Auto-resolved action key mapping: {self._action_key_mapping}")

    # ------------------------------------------------------------------
    # EvalPolicy interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get action from GR00T policy.

        Returns:
            Action array of shape (action_horizon, action_dim)
        """
        obs_with_lang = dict(obs)
        if task_description:
            obs_with_lang["task_description"] = task_description

        # Convert to GR00T format
        gr00t_obs = build_gr00t_observation(obs_with_lang, self._raw_modality_config)

        # Inference
        action_dict, _info = self.policy.get_action(gr00t_obs)

        # Lazy-resolve key mapping on first call
        self._ensure_action_key_mapping(action_dict)

        # Convert to array
        return parse_action_output(action_dict, self._action_key_mapping)

    def get_modality_config(self) -> ModalityConfig:
        return self._modality_config

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
