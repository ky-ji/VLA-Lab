"""
LeRobot Dataset Loader for VLA-Lab Open-Loop Evaluation

Bridges Isaac-GR00T's LeRobot v2 format datasets with VLA-Lab's
DatasetLoader interface, enabling open-loop evaluation of GR00T models
through VLA-Lab's unified evaluation pipeline.
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from vlalab.eval.open_loop_eval import DatasetLoader

_MAX_EPISODE_CACHE = 32


class LeRobotDatasetLoader(DatasetLoader):
    """
    Loader for LeRobot v2 format datasets (used by Isaac-GR00T).

    This loader wraps Isaac-GR00T's LeRobotEpisodeLoader to provide
    data through VLA-Lab's standardized DatasetLoader interface.

    Expected dataset structure (LeRobot v2 format):
        dataset/
        ├── data/
        │   └── chunk-000/
        │       ├── episode_000000.parquet
        │       └── ...
        ├── meta/
        │   ├── info.json
        │   ├── episodes.jsonl
        │   ├── tasks.jsonl
        │   ├── modality.json
        │   └── stats.json
        └── videos/
            └── chunk-000/
                └── observation.images.*/
                    └── episode_*.mp4

    Args:
        dataset_path: Path to the LeRobot dataset root directory
        modality_configs: GR00T-style modality configuration dict
        embodiment_tag: EmbodimentTag for data extraction
        video_backend: Video decoding backend (default: "torchcodec")
        max_cache_episodes: Max number of episodes to keep in LRU cache
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: Dict[str, Any],
        embodiment_tag: Any,
        video_backend: str = "torchcodec",
        max_cache_episodes: int = _MAX_EPISODE_CACHE,
    ):
        try:
            from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        except ImportError:
            raise ImportError(
                "Isaac-GR00T is required for LeRobotDatasetLoader. "
                "Install it following: https://github.com/NVIDIA/Isaac-GR00T"
            ) from None

        self.dataset_path = dataset_path
        self.modality_configs = modality_configs
        self.embodiment_tag = embodiment_tag
        self._max_cache = max_cache_episodes

        self.loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=None,
        )

        self._episode_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

        # Pre-build observation-only modality config (avoids deepcopy per step)
        self._obs_modality_configs = deepcopy(modality_configs)
        self._obs_modality_configs.pop("action", None)

    def _get_episode(self, traj_id: int) -> pd.DataFrame:
        """Load and cache an episode DataFrame with LRU eviction."""
        if traj_id in self._episode_cache:
            self._episode_cache.move_to_end(traj_id)
            return self._episode_cache[traj_id]
        episode = self.loader[traj_id]
        self._episode_cache[traj_id] = episode
        while len(self._episode_cache) > self._max_cache:
            self._episode_cache.popitem(last=False)
        return episode

    def __len__(self) -> int:
        return len(self.loader)

    def get_trajectory_length(self, traj_id: int) -> int:
        episode = self._get_episode(traj_id)
        return len(episode)

    def get_step_data(
        self,
        traj_id: int,
        step_idx: int,
        state_keys: List[str],
        image_keys: List[str],
    ) -> Dict[str, Any]:
        """
        Get observation data for a specific step using GR00T's extract_step_data.

        Returns:
            Dict with:
            - "state": Dict[str, np.ndarray] - state vectors by key
            - "images": Dict[str, np.ndarray] - images by camera name
        """
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: F811

        episode = self._get_episode(traj_id)

        step_data = extract_step_data(
            episode,
            step_idx,
            self._obs_modality_configs,
            self.embodiment_tag,
            allow_padding=True,
        )

        obs = {"state": {}, "images": {}}

        for key, value in step_data.states.items():
            obs["state"][key] = value

        for key, value in step_data.images.items():
            obs["images"][key] = np.array(value)

        return obs

    def get_action(
        self,
        traj_id: int,
        step_idx: int,
        action_keys: List[str],
    ) -> np.ndarray:
        """
        Get ground-truth action for a specific step.

        Returns:
            Action array of shape (action_dim,) - concatenation of all action keys
        """
        episode = self._get_episode(traj_id)

        action_parts = []
        for key in action_keys:
            col_name = f"action.{key}"
            if col_name in episode.columns:
                arr = np.array(episode[col_name].iloc[step_idx]).astype(np.float32)
                arr = np.atleast_1d(arr)
                action_parts.append(arr)

        return np.concatenate(action_parts, axis=0)

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
        episode = self._get_episode(traj_id)
        end = min(max_steps, len(episode)) if max_steps else len(episode)

        parts = []
        for key in action_keys:
            col_name = f"action.{key}"
            if col_name in episode.columns:
                col_data = episode[col_name].iloc[:end]
                arr = np.stack([np.atleast_1d(np.asarray(v, dtype=np.float32)) for v in col_data])
                parts.append(arr)

        if not parts:
            return np.empty((end, 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    def get_task_description(self, traj_id: int) -> Optional[str]:
        """
        Get the task description / language instruction for a trajectory.

        Returns:
            Task description string, or None if not available
        """
        episode = self._get_episode(traj_id)

        language_config = self.modality_configs.get("language")
        if language_config is not None:
            lang_keys = getattr(language_config, "modality_keys", [])
            for key in lang_keys:
                if key in episode.columns:
                    return str(episode[key].iloc[0])

        return None
