"""
VITA Policy Adapter

Wraps RealWorld-VITA's VitaPolicy to conform to the unified EvalPolicy interface.
Handles loading checkpoint, normalizing observations, and generating actions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vlalab.eval.policy_interface import EvalPolicy, ModalityConfig

logger = logging.getLogger(__name__)


def load_vita_policy(
    config_path: str,
    checkpoint_dir: str,
    device: str = "cuda:0",
):
    """
    Load a trained VITA policy from config + checkpoint.

    Args:
        config_path: Path to the training config YAML (e.g. train_config_*.yaml)
        checkpoint_dir: Path to checkpoint directory (e.g. step_0000020000/)
        device: Torch device

    Returns:
        (policy, cfg) tuple
    """
    import sys
    import torch
    from omegaconf import OmegaConf
    # Ensure flare and gym_av_aloha are importable
    cfg = OmegaConf.load(config_path)

    # Load dataset stats
    from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDatasetMeta
    dataset_meta = AVAlohaDatasetMeta(
        repo_id=cfg.task.dataset_repo_id,
        root=cfg.task.dataset_root,
    )
    stats = dict(dataset_meta.stats)
    stats.update(cfg.task.override_stats)

    # Build policy
    from flare.factory import get_policy_class
    policy_cls = get_policy_class(cfg.policy.name)
    policy = policy_cls(cfg, stats)

    # Load checkpoint weights
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "model.safetensors"
    if model_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(model_path))
        policy.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_dir}")

    # Load EMA weights if available
    training_state_path = checkpoint_dir / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(str(training_state_path), map_location="cpu")
        if "ema" in training_state:
            from diffusers.training_utils import EMAModel
            ema = EMAModel(parameters=policy.parameters(), power=cfg.ema_power)
            ema.load_state_dict(training_state["ema"])
            ema.copy_to(policy.parameters())
            logger.info("Applied EMA weights to policy")

    policy = policy.to(device)
    policy.eval()
    logger.info(f"VITA policy loaded on {device}")
    return policy, cfg


class VitaAdapter(EvalPolicy):
    """
    Adapter for RealWorld-VITA's VitaPolicy.

    Usage:
        policy, cfg = load_vita_policy(config_path, checkpoint_dir)
        adapter = VitaAdapter(policy, cfg, device="cuda:0")

        # Use with OpenLoopEvaluator
        evaluator = OpenLoopEvaluator(policy=adapter, ...)
        results = evaluator.evaluate(traj_ids=[0, 1, 2])
    """

    def __init__(
        self,
        policy,
        cfg,
        device: str = "cuda:0",
    ):
        self.policy = policy
        self.cfg = cfg
        self.device = device

        self._modality_config = ModalityConfig(
            state_keys=[cfg.task.state_key],
            action_keys=[cfg.task.action_key],
            image_keys=list(cfg.task.image_keys),
            language_keys=[],
            action_horizon=cfg.policy.action_horizon,
            action_dim=cfg.task.action_dim,
        )

    def get_action(
        self,
        obs: Dict[str, Any],
        task_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get action from VITA policy.

        Args:
            obs: Standardized observation dict with:
                - "state": {state_key: np.ndarray (state_dim,)}
                - "images": {image_key: np.ndarray (H, W, C) uint8 or float}

        Returns:
            Action array of shape (action_horizon, action_dim)
        """
        import torch

        batch = {}

        # Process state: (state_dim,) -> (1, 1, state_dim)
        state_key = self.cfg.task.state_key
        if "state" in obs and state_key in obs["state"]:
            state = np.atleast_1d(obs["state"][state_key]).astype(np.float32)
            batch[state_key] = torch.from_numpy(state).unsqueeze(0).to(self.device)
        elif "state" in obs:
            # Concatenate all state parts
            parts = [np.atleast_1d(v).astype(np.float32) for v in obs["state"].values()]
            state = np.concatenate(parts, axis=-1)
            batch[state_key] = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Process images: (H, W, C) uint8 -> (1, 1, C, H, W) float [0,1]
        if "images" in obs:
            for img_key in self.cfg.task.image_keys:
                if img_key in obs["images"]:
                    img = obs["images"][img_key]
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    # (H, W, C) -> (C, H, W)
                    if img.ndim == 3 and img.shape[-1] in [1, 3]:
                        img = np.transpose(img, (2, 0, 1))
                    batch[img_key] = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # select_action handles normalize + generate + unnormalize + action queue
        # But for open-loop eval we want the full action chunk, so call generate directly
        with torch.no_grad():
            # Add temporal dim: (1, D) -> (1, 1, D) for obs_horizon=1
            for k in batch:
                if batch[k].ndim == 2:
                    batch[k] = batch[k].unsqueeze(1)

            batch = self.policy.normalize_inputs(batch)
            actions_norm = self.policy.generate_actions(batch)
            # actions_norm: (1, pred_horizon, action_dim)
            actions_norm = actions_norm[:, :self.cfg.policy.action_horizon]
            actions = self.policy.unnormalize_outputs({"action": actions_norm})["action"]

        # (1, action_horizon, action_dim) -> (action_horizon, action_dim)
        actions = actions[0].cpu().numpy()
        return actions

    def get_modality_config(self) -> ModalityConfig:
        return self._modality_config

    def reset(self) -> None:
        self.policy.reset()
