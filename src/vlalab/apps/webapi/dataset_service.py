"""Dataset inspection helpers for the VLA-Lab web API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .media import image_to_data_url, normalize_image
from .service import stat_signature


@lru_cache(maxsize=16)
def _open_zarr_cached(path_str: str, mtime_ns: int, size: int):
    import zarr

    return zarr.open(path_str, mode="r")


def _open_dataset(path: Union[str, Path]):
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return dataset_path, _open_zarr_cached(str(dataset_path), *stat_signature(dataset_path))


def _classify_keys(root) -> Tuple[List[str], List[str]]:
    image_keys = []
    lowdim_keys = []
    for key in root["data"].keys():
        arr = root["data"][key]
        if len(arr.shape) == 4:
            image_keys.append(str(key))
        else:
            lowdim_keys.append(str(key))
    return image_keys, lowdim_keys


def inspect_dataset(path: Union[str, Path]) -> Dict[str, Any]:
    dataset_path, root = _open_dataset(path)
    image_keys, lowdim_keys = _classify_keys(root)
    episode_ends = root["meta"]["episode_ends"][:]
    total_steps = int(episode_ends[-1]) if len(episode_ends) else 0
    action = root["data"]["action"]
    episode_lengths = []
    prev = 0
    for idx, end_idx in enumerate(episode_ends.tolist()):
        episode_lengths.append({"episode_idx": idx, "length": int(end_idx - prev)})
        prev = int(end_idx)

    return {
        "path": str(dataset_path),
        "episode_count": len(episode_ends),
        "total_steps": total_steps,
        "image_keys": image_keys,
        "lowdim_keys": lowdim_keys,
        "action_dim": int(action.shape[1]) if len(action.shape) > 1 else 1,
        "episode_lengths": episode_lengths,
    }


def _episode_slice(episode_ends: np.ndarray, episode_idx: int) -> slice:
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])
    return slice(start_idx, end_idx)


def _safe_series(values: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(values).tolist()]


def load_dataset_episode_view(
    path: Union[str, Path],
    episode_idx: int,
    step_idx: int = 0,
    step_interval: int = 5,
    max_frames: int = 20,
    workspace_ratio: float = 0.1,
) -> Dict[str, Any]:
    dataset_path, root = _open_dataset(path)
    data = root["data"]
    meta = root["meta"]
    episode_ends = meta["episode_ends"][:]

    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(f"Episode index out of range: {episode_idx}")

    episode_slice = _episode_slice(episode_ends, episode_idx)
    actions = np.asarray(data["action"][episode_slice])
    total_steps = len(actions)
    if total_steps == 0:
        raise RuntimeError(f"Episode {episode_idx} is empty.")

    step_idx = max(0, min(int(step_idx), total_steps - 1))
    image_keys, lowdim_keys = _classify_keys(root)

    step_images = {}
    for key in image_keys:
        raw = data[key][episode_slice][step_idx]
        step_images[key] = image_to_data_url(raw)

    lowdim_step = {}
    for key in lowdim_keys:
        value = np.asarray(data[key][episode_slice][step_idx]).tolist()
        lowdim_step[key] = value

    frame_indices = list(range(0, total_steps, max(1, step_interval)))[:max_frames]
    image_grids = {}
    for key in image_keys:
        grid = []
        episode_images = data[key][episode_slice]
        for frame_idx in frame_indices:
            grid.append(
                {
                    "step_idx": int(frame_idx),
                    "data_url": image_to_data_url(episode_images[frame_idx]),
                }
            )
        image_grids[key] = grid

    action_dim = actions.shape[1] if actions.ndim > 1 else 1
    action_labels = [f"dim_{idx}" for idx in range(action_dim)]
    if action_dim >= 3:
        action_labels[:3] = ["x", "y", "z"]
    if action_dim >= 8:
        action_labels[7] = "gripper"

    sample_count = max(1, int(total_steps * max(0.01, workspace_ratio)))
    sample_indices = np.linspace(0, total_steps - 1, num=sample_count, dtype=int)
    workspace_points = actions[sample_indices, : min(3, action_dim)].tolist() if action_dim >= 3 else []

    return {
        "dataset": inspect_dataset(dataset_path),
        "episode_idx": episode_idx,
        "episode_length": total_steps,
        "step_idx": step_idx,
        "step_images": step_images,
        "lowdim_step": lowdim_step,
        "current_action": _safe_series(actions[step_idx]),
        "action_labels": action_labels,
        "actions": actions.tolist(),
        "xyz_series": {
            "x": _safe_series(actions[:, 0]) if action_dim >= 1 else [],
            "y": _safe_series(actions[:, 1]) if action_dim >= 2 else [],
            "z": _safe_series(actions[:, 2]) if action_dim >= 3 else [],
        },
        "gripper_series": _safe_series(actions[:, 7]) if action_dim >= 8 else [],
        "image_grids": image_grids,
        "workspace_points": workspace_points,
    }
