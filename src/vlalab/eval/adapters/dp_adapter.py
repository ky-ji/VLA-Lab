"""
Diffusion Policy Adapter

Wraps Diffusion Policy implementations to conform to the unified EvalPolicy interface.
This adapter handles the conversion between VLA-Lab's standardized observation
format and Diffusion Policy's specific input/output formats.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np

from vlalab.eval.policy_interface import EvalPolicy, ModalityConfig


def parse_observation_dp(
    obs: Dict[str, Any],
    state_key: str = "state",
    image_key: str = "image",
) -> Dict[str, Any]:
    """
    Convert standardized observation to Diffusion Policy's expected format.
    
    Diffusion Policy typically expects:
    {
        "state": np.ndarray (state_dim,) or (T, state_dim),
        "image": np.ndarray (H, W, C) or (T, H, W, C),
    }
    
    Args:
        obs: Standardized observation dict with:
            - "state": Dict[str, np.ndarray] - state vectors
            - "images": Dict[str, np.ndarray] - images (H, W, C)
        state_key: Key to use for concatenated state
        image_key: Key to use for primary image
    
    Returns:
        DP-formatted observation dict
    """
    dp_obs = {}
    
    # Concatenate all state keys into single state vector
    if "state" in obs:
        state_parts = []
        for key, arr in obs["state"].items():
            arr = np.atleast_1d(arr).astype(np.float32)
            state_parts.append(arr)
        if state_parts:
            dp_obs[state_key] = np.concatenate(state_parts, axis=-1)
    
    # Use first image as primary image
    if "images" in obs:
        for cam_name, img in obs["images"].items():
            dp_obs[image_key] = img
            break  # Use first image
    
    return dp_obs


def parse_action_dp(
    action: Any,
) -> np.ndarray:
    """
    Convert Diffusion Policy action output to standardized array format.
    
    Diffusion Policy outputs action chunks as:
    - np.ndarray of shape (action_horizon, action_dim)
    - or torch.Tensor of same shape
    
    Args:
        action: DP action output
    
    Returns:
        Action array of shape (action_horizon, action_dim)
    """
    # Handle torch tensors
    if hasattr(action, "cpu"):
        action = action.cpu().numpy()
    
    action = np.asarray(action)
    
    # Ensure 2D
    if action.ndim == 1:
        action = action[None, :]
    
    return action


class DiffusionPolicyAdapter(EvalPolicy):
    """
    Adapter for Diffusion Policy.
    
    This adapter is designed to work with various Diffusion Policy implementations.
    You can either:
    1. Pass a policy object with a predict() or get_action() method
    2. Pass a callable inference function directly
    
    Usage:
        # Option 1: Wrap policy object
        from dp_server import DPPolicy
        policy = DPPolicy.load(checkpoint_path)
        adapter = DiffusionPolicyAdapter(policy)
        
        # Option 2: Wrap inference client
        from dp_client import DPClient
        client = DPClient(host="localhost", port=5000)
        adapter = DiffusionPolicyAdapter(
            client,
            inference_fn=lambda p, obs: p.predict(obs["state"], obs["image"])
        )
        
        # Option 3: Wrap callable
        adapter = DiffusionPolicyAdapter(
            inference_fn=my_inference_function,
            action_horizon=8,
            action_dim=7,
        )
    """
    
    def __init__(
        self,
        policy: Any = None,
        inference_fn: Optional[Callable] = None,
        action_horizon: int = 8,
        action_dim: int = 7,
        state_keys: Optional[List[str]] = None,
        image_keys: Optional[List[str]] = None,
    ):
        """
        Initialize the Diffusion Policy adapter.
        
        Args:
            policy: DP policy instance (optional if inference_fn provided)
            inference_fn: Custom inference function (policy, obs) -> action
            action_horizon: Number of future actions predicted
            action_dim: Dimension of action space
            state_keys: List of state modality keys
            image_keys: List of image modality keys
        """
        self.policy = policy
        self._inference_fn = inference_fn
        
        self._modality_config = ModalityConfig(
            state_keys=state_keys or ["robot_state"],
            action_keys=["action"],
            image_keys=image_keys or ["front"],
            language_keys=[],  # DP typically doesn't use language
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
    
    def _default_inference(self, obs: Dict[str, Any]) -> np.ndarray:
        """Default inference logic when no custom inference_fn provided."""
        dp_obs = parse_observation_dp(obs)
        
        # Try different method names
        if hasattr(self.policy, "predict"):
            action = self.policy.predict(dp_obs)
        elif hasattr(self.policy, "get_action"):
            action = self.policy.get_action(dp_obs)
        elif hasattr(self.policy, "__call__"):
            action = self.policy(dp_obs)
        else:
            raise ValueError(
                "Policy must have predict(), get_action(), or __call__() method. "
                "Alternatively, provide inference_fn parameter."
            )
        
        return parse_action_dp(action)
    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get action from Diffusion Policy.
        
        Args:
            obs: Standardized observation dict
            task_description: Ignored (DP doesn't use language)
        
        Returns:
            Action array of shape (action_horizon, action_dim)
        """
        if self._inference_fn is not None:
            if self.policy is not None:
                action = self._inference_fn(self.policy, obs)
            else:
                action = self._inference_fn(obs)
            return parse_action_dp(action)
        
        return self._default_inference(obs)
    
    def get_modality_config(self) -> ModalityConfig:
        """Get modality configuration."""
        return self._modality_config
    
    def reset(self) -> None:
        """Reset the policy."""
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()


class DiffusionPolicyClientAdapter(DiffusionPolicyAdapter):
    """
    Adapter for Diffusion Policy inference client (e.g., ZMQ client).
    
    This is a convenience class for wrapping remote DP inference servers.
    
    Usage:
        adapter = DiffusionPolicyClientAdapter(
            host="localhost",
            port=5000,
            action_horizon=8,
            action_dim=7,
        )
        action = adapter.get_action(obs)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        action_horizon: int = 8,
        action_dim: int = 7,
        state_keys: Optional[List[str]] = None,
        image_keys: Optional[List[str]] = None,
    ):
        """
        Initialize client adapter.
        
        Args:
            host: Server hostname
            port: Server port
            action_horizon: Number of future actions
            action_dim: Action dimension
            state_keys: State modality keys
            image_keys: Image modality keys
        """
        super().__init__(
            policy=None,
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_keys=state_keys,
            image_keys=image_keys,
        )
        
        self.host = host
        self.port = port
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of ZMQ client."""
        if self._client is None:
            try:
                import zmq
                context = zmq.Context()
                self._client = context.socket(zmq.REQ)
                self._client.connect(f"tcp://{self.host}:{self.port}")
            except ImportError:
                raise ImportError("pyzmq required for DiffusionPolicyClientAdapter")
        return self._client
    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get action from remote DP server.
        
        Note: This is a placeholder. Actual implementation depends on
        the specific DP server protocol used.
        """
        raise NotImplementedError(
            "DiffusionPolicyClientAdapter.get_action() requires implementation "
            "specific to your DP server protocol. Override this method or use "
            "DiffusionPolicyAdapter with a custom inference_fn."
        )
