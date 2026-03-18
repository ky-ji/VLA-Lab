"""
AVAloha Dataset Loader for VLA-Lab Open-Loop Evaluation

Bridges RealWorld-VITA's AVAloha Zarr-based dataset format with VLA-Lab's
DatasetLoader interface, enabling open-loop evaluation of VITA models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vlalab.eval.open_loop_eval import DatasetLoader

logger = logging.getLogger(__name__)


class AVAlohaDatasetLoader(DatasetLoader):
    """
    Loader for AVAloha Zarr format datasets used by RealWorld-VITA.

    Expected structure (Zarr):
        dataset_root/
        ├── data/
        │   ├── action (T, action_dim)
        │   ├── observation.state (T, state_dim)
        │   ├── observation.images.front_view (T, H, W, C) uint8
        │   ├── observation.images.wrist_view (T, H, W, C) uint8
        │   ├── episode_index (T,)
        │   └── timestamp (T,)
        └── config.json

    Args:
        dataset_root: Path to the AVAloha dataset root directory
        episodes: Optional list of episode indices to use (default: all)
    """

    def __init__(
        self,
        dataset_root: str,
        episodes: Optional[List[int]] = None,
    ):
        try:
            from gym_av_aloha.common.replay_buffer import ReplayBuffer
            from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDatasetMeta
        except ImportError:
            raise ImportError(
                "gym_av_aloha is required for AVAlohaDatasetLoader. "
                "Install it following the RealWorld-VITA setup instructions."
            ) from None

        self.root = Path(dataset_root)
        self.meta = AVAlohaDatasetMeta(root=self.root)
        self.replay_buffer = ReplayBuffer.copy_from_path(self.root)

        # Determine episodes
        if episodes is not None:
            self.episodes = episodes
        else:
            self.episodes = list(range(self.meta.num_episodes))

        # Build episode start/end index mapping
        ep_indices = np.array(self.replay_buffer["episode_index"])
        self._episode_slices = {}  # episode_id -> (start, end)
        for ep_id in self.episodes:
            mask = ep_indices == ep_id
            indices = np.where(mask)[0]
            if len(indices) > 0:
                self._episode_slices[ep_id] = (int(indices[0]), int(indices[-1]) + 1)

        logger.info(
            f"AVAlohaDatasetLoader: {len(self.episodes)} episodes, "
            f"image_keys={self.meta.image_keys}, "
            f"features={list(self.meta.features.keys())}"
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def _get_episode_slice(self, traj_id: int):
        """Map traj_id (0-based index into self.episodes) to global (start, end)."""
        ep_id = self.episodes[traj_id]
        return self._episode_slices[ep_id]

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

        # Load state
        for key in state_keys:
            if key in self.replay_buffer.keys():
                obs["state"][key] = np.array(
                    self.replay_buffer[key][global_idx], dtype=np.float32
                )

        # Load images — stored as (H, W, C) uint8 in Zarr
        for key in image_keys:
            if key in self.replay_buffer.keys():
                img = np.array(self.replay_buffer[key][global_idx])
                obs["images"][key] = img

        return obs

    def get_action(
        self,
        traj_id: int,
        step_idx: int,
        action_keys: List[str],
    ) -> np.ndarray:
        start, _ = self._get_episode_slice(traj_id)
        global_idx = start + step_idx
        return np.array(self.replay_buffer["action"][global_idx], dtype=np.float32)

    def get_trajectory_actions(
        self,
        traj_id: int,
        action_keys: List[str],
        max_steps: Optional[int] = None,
    ) -> np.ndarray:
        start, end = self._get_episode_slice(traj_id)
        if max_steps:
            end = min(end, start + max_steps)
        return np.array(self.replay_buffer["action"][start:end], dtype=np.float32)
