"""
VLA-Lab Open-Loop Evaluation

Model-agnostic open-loop evaluation for VLA policies.
Compares predicted actions against ground-truth actions from a dataset.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json

import numpy as np
import matplotlib.pyplot as plt

from vlalab.eval.policy_interface import EvalPolicy, ModalityConfig


@dataclass
class EvalResult:
    """Result of evaluating a single trajectory."""
    trajectory_id: int
    mse: float
    mae: float
    num_steps: int
    gt_actions: np.ndarray  # (T, action_dim)
    pred_actions: np.ndarray  # (T, action_dim)
    states: Optional[np.ndarray] = None  # (T, state_dim)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "mse": float(self.mse),
            "mae": float(self.mae),
            "num_steps": self.num_steps,
        }


@dataclass
class EvalConfig:
    """Configuration for open-loop evaluation."""
    max_steps: int = 300
    action_horizon: int = 16
    task_description: Optional[str] = None
    save_plot_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "action_horizon": self.action_horizon,
            "task_description": self.task_description,
            "save_plot_path": self.save_plot_path,
        }


class DatasetLoader:
    """
    Abstract interface for loading trajectory data from datasets.
    
    Subclass this to support different dataset formats (Zarr, LeRobot, HDF5, etc.)
    """
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        raise NotImplementedError
    
    def get_trajectory_length(self, traj_id: int) -> int:
        """Return length of a specific trajectory."""
        raise NotImplementedError
    
    def get_step_data(
        self,
        traj_id: int,
        step_idx: int,
        state_keys: List[str],
        image_keys: List[str],
    ) -> Dict[str, Any]:
        """
        Get observation data for a specific step.
        
        Returns:
            Dict with:
            - "state": Dict[str, np.ndarray] - state vectors by key
            - "images": Dict[str, np.ndarray] - images by camera name
        """
        raise NotImplementedError
    
    def get_action(
        self,
        traj_id: int,
        step_idx: int,
        action_keys: List[str],
    ) -> np.ndarray:
        """
        Get ground-truth action for a specific step.
        
        Returns:
            Action array of shape (action_dim,)
        """
        raise NotImplementedError
    
    def get_trajectory_actions(
        self,
        traj_id: int,
        action_keys: List[str],
        max_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get all ground-truth actions for a trajectory.
        
        Returns:
            Action array of shape (T, action_dim)
        """
        raise NotImplementedError


class ZarrDatasetLoader(DatasetLoader):
    """
    Loader for Zarr format datasets (used by Diffusion Policy, etc.)
    
    Expected structure:
    dataset.zarr/
    ├── data/
    │   ├── action (T, action_dim)
    │   ├── state (T, state_dim)  # or robot_state, etc.
    │   └── image_* (T, H, W, C)
    └── meta/
        └── episode_ends (num_episodes,)
    """
    
    def __init__(self, zarr_path: str):
        import zarr
        self.zarr_path = Path(zarr_path)
        self.root = zarr.open(str(zarr_path), mode='r')
        self.data = self.root['data']
        self.meta = self.root['meta']
        self.episode_ends = self.meta['episode_ends'][:]
    
    def __len__(self) -> int:
        return len(self.episode_ends)
    
    def _get_episode_slice(self, traj_id: int) -> Tuple[int, int]:
        start = 0 if traj_id == 0 else int(self.episode_ends[traj_id - 1])
        end = int(self.episode_ends[traj_id])
        return start, end
    
    def get_trajectory_length(self, traj_id: int) -> int:
        start, end = self._get_episode_slice(traj_id)
        return end - start
    
    def get_step_data(
        self,
        traj_id: int,
        step_idx: int,
        state_keys: List[str],
        image_keys: List[str],
    ) -> Dict[str, Any]:
        start, end = self._get_episode_slice(traj_id)
        global_idx = start + step_idx
        
        obs = {"state": {}, "images": {}}
        
        # Load state data
        for key in state_keys:
            if key in self.data:
                obs["state"][key] = self.data[key][global_idx]
        
        # Load image data
        for key in image_keys:
            # Try different naming conventions
            for img_key in [key, f"image_{key}", f"img_{key}"]:
                if img_key in self.data:
                    img = self.data[img_key][global_idx]
                    # Handle (C, H, W) -> (H, W, C)
                    if img.ndim == 3 and img.shape[0] in [1, 3]:
                        img = np.transpose(img, (1, 2, 0))
                    obs["images"][key] = img
                    break
        
        return obs
    
    def get_action(
        self,
        traj_id: int,
        step_idx: int,
        action_keys: List[str],
    ) -> np.ndarray:
        start, _ = self._get_episode_slice(traj_id)
        global_idx = start + step_idx
        
        # For Zarr, action is typically stored as single array
        return self.data['action'][global_idx]
    
    def get_trajectory_actions(
        self,
        traj_id: int,
        action_keys: List[str],
        max_steps: Optional[int] = None,
    ) -> np.ndarray:
        start, end = self._get_episode_slice(traj_id)
        if max_steps:
            end = min(end, start + max_steps)
        return self.data['action'][start:end]


def evaluate_trajectory(
    policy: EvalPolicy,
    dataset: DatasetLoader,
    traj_id: int,
    config: EvalConfig,
) -> EvalResult:
    """
    Evaluate a policy on a single trajectory.
    
    Args:
        policy: Policy adapter implementing EvalPolicy
        dataset: Dataset loader
        traj_id: Trajectory ID to evaluate
        config: Evaluation configuration
    
    Returns:
        EvalResult with metrics and action arrays
    """
    modality = policy.get_modality_config()
    action_horizon = config.action_horizon or modality.action_horizon
    
    # Get trajectory length
    traj_length = dataset.get_trajectory_length(traj_id)
    actual_steps = min(config.max_steps, traj_length)
    
    logging.info(f"Evaluating trajectory {traj_id}: {actual_steps} steps")
    
    # Collect predicted actions
    pred_actions_list = []
    
    for step_idx in range(0, actual_steps, action_horizon):
        logging.debug(f"Inferencing at step {step_idx}")
        
        # Get observation
        obs = dataset.get_step_data(
            traj_id,
            step_idx,
            modality.state_keys,
            modality.image_keys,
        )
        
        # Get action from policy
        action_chunk = policy.get_action(obs, config.task_description)
        
        # Collect actions from chunk
        for j in range(action_horizon):
            if step_idx + j >= actual_steps:
                break
            if j < len(action_chunk):
                pred_actions_list.append(action_chunk[j])
    
    # Get ground truth actions
    gt_actions = dataset.get_trajectory_actions(
        traj_id,
        modality.action_keys,
        max_steps=actual_steps,
    )
    
    pred_actions = np.array(pred_actions_list)[:actual_steps]
    gt_actions = gt_actions[:actual_steps]
    
    # Ensure shapes match
    min_len = min(len(pred_actions), len(gt_actions))
    pred_actions = pred_actions[:min_len]
    gt_actions = gt_actions[:min_len]
    
    # Handle dimension mismatch
    if pred_actions.shape[-1] != gt_actions.shape[-1]:
        # Take minimum dimension (common case: pred has extra dims)
        min_dim = min(pred_actions.shape[-1], gt_actions.shape[-1])
        pred_actions = pred_actions[..., :min_dim]
        gt_actions = gt_actions[..., :min_dim]
        logging.warning(
            f"Dimension mismatch: pred {pred_actions.shape[-1]}, gt {gt_actions.shape[-1]}. "
            f"Using first {min_dim} dims."
        )
    
    # Calculate metrics
    mse = float(np.mean((gt_actions - pred_actions) ** 2))
    mae = float(np.mean(np.abs(gt_actions - pred_actions)))
    
    logging.info(f"Trajectory {traj_id}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    return EvalResult(
        trajectory_id=traj_id,
        mse=mse,
        mae=mae,
        num_steps=min_len,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
    )


def plot_trajectory_results(
    result: EvalResult,
    action_keys: Optional[List[str]] = None,
    action_horizon: int = 16,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot evaluation results comparing GT vs predicted actions.
    
    Args:
        result: EvalResult from evaluate_trajectory
        action_keys: Optional labels for action dimensions
        action_horizon: Action horizon for marking inference points
        save_path: Path to save plot (optional)
        show: Whether to display plot
    
    Returns:
        Matplotlib figure
    """
    gt_actions = result.gt_actions
    pred_actions = result.pred_actions
    
    num_dims = gt_actions.shape[1]
    actual_steps = len(gt_actions)
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=num_dims,
        ncols=1,
        figsize=(10, 3 * num_dims),
        squeeze=False,
    )
    
    fig.suptitle(
        f"Trajectory {result.trajectory_id} | MSE: {result.mse:.4f} | MAE: {result.mae:.4f}",
        fontsize=14,
    )
    
    for i in range(num_dims):
        ax = axes[i, 0]
        
        # Plot GT and predicted
        ax.plot(gt_actions[:, i], label="GT Action", alpha=0.8)
        ax.plot(pred_actions[:, i], label="Pred Action", alpha=0.8, linestyle="--")
        
        # Mark inference points
        for j in range(0, actual_steps, action_horizon):
            ax.axvline(x=j, color='gray', linestyle=':', alpha=0.3)
            if j == 0:
                ax.plot(j, gt_actions[j, i], "ro", markersize=4, label="Inference Point")
            else:
                ax.plot(j, gt_actions[j, i], "ro", markersize=4)
        
        # Labels
        dim_label = action_keys[i] if action_keys and i < len(action_keys) else f"Dim {i}"
        ax.set_title(f"Action {dim_label}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


class OpenLoopEvaluator:
    """
    High-level evaluator for running open-loop evaluation.
    
    Usage:
        from vlalab.eval import OpenLoopEvaluator
        from vlalab.eval.adapters import GR00TAdapter
        
        # Create adapter
        adapter = GR00TAdapter(policy)
        
        # Create evaluator
        evaluator = OpenLoopEvaluator(
            policy=adapter,
            dataset_path="/path/to/dataset.zarr",
            dataset_format="zarr",
        )
        
        # Evaluate
        results = evaluator.evaluate(
            traj_ids=[0, 1, 2],
            max_steps=200,
            save_plots_dir="outputs/",
        )
        
        print(f"Average MSE: {results['avg_mse']:.4f}")
    """
    
    def __init__(
        self,
        policy: EvalPolicy,
        dataset_path: str,
        dataset_format: str = "zarr",
        task_description: Optional[str] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            policy: Policy adapter implementing EvalPolicy
            dataset_path: Path to dataset
            dataset_format: Dataset format ("zarr", "lerobot", etc.)
            task_description: Default task description for language-conditioned models
        """
        self.policy = policy
        self.dataset_path = dataset_path
        self.task_description = task_description
        
        # Load dataset
        if dataset_format == "zarr":
            self.dataset = ZarrDatasetLoader(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
        
        logging.info(f"Loaded dataset with {len(self.dataset)} trajectories")
    
    def evaluate(
        self,
        traj_ids: Optional[List[int]] = None,
        max_steps: int = 300,
        action_horizon: Optional[int] = None,
        save_plots_dir: Optional[str] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on specified trajectories.
        
        Args:
            traj_ids: List of trajectory IDs to evaluate (default: [0])
            max_steps: Maximum steps per trajectory
            action_horizon: Action horizon (default: from policy config)
            save_plots_dir: Directory to save plots (optional)
            task_description: Override default task description
        
        Returns:
            Dict with results:
            - "results": List of EvalResult dicts
            - "avg_mse": Average MSE across trajectories
            - "avg_mae": Average MAE across trajectories
            - "num_trajectories": Number of trajectories evaluated
        """
        if traj_ids is None:
            traj_ids = [0]
        
        modality = self.policy.get_modality_config()
        config = EvalConfig(
            max_steps=max_steps,
            action_horizon=action_horizon or modality.action_horizon,
            task_description=task_description or self.task_description,
            save_plot_path=None,
        )
        
        results = []
        all_mse = []
        all_mae = []
        
        for traj_id in traj_ids:
            if traj_id >= len(self.dataset):
                logging.warning(f"Trajectory {traj_id} out of range, skipping")
                continue
            
            # Reset policy state
            self.policy.reset()
            
            # Evaluate
            result = evaluate_trajectory(
                self.policy,
                self.dataset,
                traj_id,
                config,
            )
            
            results.append(result)
            all_mse.append(result.mse)
            all_mae.append(result.mae)
            
            # Save plot
            if save_plots_dir:
                plot_path = Path(save_plots_dir) / f"traj_{traj_id}.png"
                plot_trajectory_results(
                    result,
                    action_keys=modality.action_keys,
                    action_horizon=config.action_horizon,
                    save_path=str(plot_path),
                )
                plt.close()
        
        # Aggregate results
        output = {
            "results": [r.to_dict() for r in results],
            "num_trajectories": len(results),
        }
        
        if all_mse:
            output["avg_mse"] = float(np.mean(all_mse))
            output["avg_mae"] = float(np.mean(all_mae))
            logging.info(f"Average MSE: {output['avg_mse']:.6f}")
            logging.info(f"Average MAE: {output['avg_mae']:.6f}")
        
        return output
    
    def evaluate_and_save(
        self,
        output_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run evaluation and save results to JSON.
        
        Args:
            output_path: Path to save results JSON
            **kwargs: Arguments passed to evaluate()
        
        Returns:
            Evaluation results dict
        """
        results = self.evaluate(**kwargs)
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
        
        return results
