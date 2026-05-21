"""Dataset inspection helpers for the VLA-Lab web API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import cv2
import numpy as np

from .media import image_to_data_url, normalize_image
from .service import stat_signature


DEFAULT_DATASET_ROOTS = [Path("/data3/jikangye/vla-data")]
DEFAULT_DATASET_PATH = Path("/data3/jikangye/vla-data/008_stack_bowls")


@lru_cache(maxsize=16)
def _open_zarr_cached(path_str: str, mtime_ns: int, size: int):
    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for dataset inspection. "
            "Install it with: pip install vlalab[full]"
        ) from None

    return zarr.open(path_str, mode="r")


def _open_zarr_dataset(path: Union[str, Path]):
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return dataset_path, _open_zarr_cached(str(dataset_path), *stat_signature(dataset_path))


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists()


def _dataset_format(path: Path) -> str:
    if _is_lerobot_dataset(path):
        return "lerobot"
    return "zarr"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _chunk_for_episode(info: Dict[str, Any], episode_idx: int) -> int:
    chunk_size = int(info.get("chunks_size") or 1000)
    return int(episode_idx) // max(1, chunk_size)


def _format_lerobot_path(template: str, info: Dict[str, Any], episode_idx: int, **extra: Any) -> str:
    return template.format(
        episode_chunk=_chunk_for_episode(info, episode_idx),
        episode_index=int(episode_idx),
        **extra,
    )


def _lerobot_episode_path(dataset_path: Path, info: Dict[str, Any], episode_idx: int) -> Path:
    template = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return dataset_path / _format_lerobot_path(template, info, episode_idx)


def _lerobot_video_path(dataset_path: Path, info: Dict[str, Any], episode_idx: int, video_key: str) -> Path:
    template = info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return dataset_path / _format_lerobot_path(template, info, episode_idx, video_key=video_key)


def _lerobot_feature_keys(info: Dict[str, Any], dtype: str) -> List[str]:
    return [
        key
        for key, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == dtype
    ]


def _feature_labels(info: Dict[str, Any], key: str, dim: int) -> List[str]:
    names = ((info.get("features") or {}).get(key) or {}).get("names") or []
    return [str(name) for name in names[:dim]] or [f"dim_{idx}" for idx in range(dim)]


def _series(values: np.ndarray, index: int) -> List[float]:
    if values.ndim < 2 or values.shape[1] <= index:
        return []
    return [float(v) for v in values[:, index].tolist()]


def _safe_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _read_lerobot_table(path: Path):
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return pq.read_table(path)


def _table_column_matrix(table: Any, column: str) -> np.ndarray:
    if column not in table.column_names:
        return np.empty((table.num_rows, 0), dtype=float)
    return _safe_matrix(table[column].to_pylist())


def _video_frame_data_url(path: Path, step_idx: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(step_idx)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {step_idx} from {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image_to_data_url(rgb)
    finally:
        capture.release()


def _inspect_lerobot_dataset(dataset_path: Path) -> Dict[str, Any]:
    info = _read_json(dataset_path / "meta" / "info.json")
    episodes = _read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(dataset_path / "meta" / "tasks.jsonl")
    image_keys = _lerobot_feature_keys(info, "video")
    lowdim_keys = [
        key
        for key, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") != "video" and key != "action"
    ]
    action_spec = (info.get("features") or {}).get("action") or {}
    state_spec = (info.get("features") or {}).get("observation.state") or {}

    episode_lengths = [
        {
            "episode_idx": int(row.get("episode_index", idx)),
            "length": int(row.get("length", 0)),
            "tasks": row.get("tasks", []),
        }
        for idx, row in enumerate(episodes)
    ]
    task = ""
    if tasks:
        task = str(tasks[0].get("task") or "")
    elif episode_lengths and episode_lengths[0]["tasks"]:
        task = str(episode_lengths[0]["tasks"][0])

    return {
        "format": "lerobot",
        "path": str(dataset_path),
        "episode_count": int(info.get("total_episodes") or len(episode_lengths)),
        "total_steps": int(info.get("total_frames") or sum(item["length"] for item in episode_lengths)),
        "fps": int(info.get("fps") or 0),
        "task": task,
        "image_keys": image_keys,
        "lowdim_keys": lowdim_keys,
        "action_dim": int((action_spec.get("shape") or [0])[0] or 0),
        "state_dim": int((state_spec.get("shape") or [0])[0] or 0),
        "action_labels": _feature_labels(info, "action", int((action_spec.get("shape") or [0])[0] or 0)),
        "state_labels": _feature_labels(info, "observation.state", int((state_spec.get("shape") or [0])[0] or 0)),
        "episode_lengths": episode_lengths,
    }


def list_dataset_candidates(base_paths: Iterable[Union[str, Path]] | None = None) -> Dict[str, Any]:
    use_builtin_default = base_paths is None
    roots = [Path(path).expanduser() for path in (base_paths or DEFAULT_DATASET_ROOTS)]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for item in sorted(root.iterdir()):
            if not item.is_dir() or not _is_lerobot_dataset(item):
                continue
            try:
                info = _inspect_lerobot_dataset(item.resolve())
            except Exception:
                continue
            candidates.append(
                {
                    "path": info["path"],
                    "name": item.name,
                    "format": info["format"],
                    "episode_count": info["episode_count"],
                    "total_steps": info["total_steps"],
                    "task": info.get("task", ""),
                }
            )

    default_path = (
        str(DEFAULT_DATASET_PATH)
        if use_builtin_default and DEFAULT_DATASET_PATH.exists()
        else (candidates[0]["path"] if candidates else "")
    )
    candidates.sort(key=lambda item: (item["path"] != default_path, item["path"]))
    return {"default_path": default_path, "candidates": candidates}


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
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if _dataset_format(dataset_path) == "lerobot":
        return _inspect_lerobot_dataset(dataset_path)

    dataset_path, root = _open_zarr_dataset(dataset_path)
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
        "format": "zarr",
        "path": str(dataset_path),
        "episode_count": len(episode_ends),
        "total_steps": total_steps,
        "image_keys": image_keys,
        "lowdim_keys": lowdim_keys,
        "action_dim": int(action.shape[1]) if len(action.shape) > 1 else 1,
        "episode_lengths": episode_lengths,
    }


def _load_lerobot_episode_view(
    dataset_path: Path,
    episode_idx: int,
    step_idx: int,
    step_interval: int,
    max_frames: int,
    workspace_ratio: float,
) -> Dict[str, Any]:
    info = _read_json(dataset_path / "meta" / "info.json")
    dataset_info = _inspect_lerobot_dataset(dataset_path)
    episode_lengths = dataset_info["episode_lengths"]
    if episode_idx < 0 or episode_idx >= len(episode_lengths):
        raise IndexError(f"Episode index out of range: {episode_idx}")

    table = _read_lerobot_table(_lerobot_episode_path(dataset_path, info, episode_idx))
    actions = _table_column_matrix(table, "action")
    states = _table_column_matrix(table, "observation.state")
    total_steps = int(table.num_rows)
    if total_steps == 0:
        raise RuntimeError(f"Episode {episode_idx} is empty.")
    step_idx = max(0, min(int(step_idx), total_steps - 1))

    image_keys = dataset_info["image_keys"]
    step_images = {}
    unavailable = []
    for key in image_keys:
        video_path = _lerobot_video_path(dataset_path, info, episode_idx, key)
        try:
            step_images[key] = _video_frame_data_url(video_path, step_idx)
        except Exception as exc:
            unavailable.append({"key": key, "reason": str(exc)})

    frame_indices = list(range(0, total_steps, max(1, int(step_interval))))[: max(1, int(max_frames))]
    image_grids = {}
    for key in image_keys:
        video_path = _lerobot_video_path(dataset_path, info, episode_idx, key)
        grid = []
        for frame_idx in frame_indices:
            try:
                grid.append({"step_idx": int(frame_idx), "data_url": _video_frame_data_url(video_path, frame_idx)})
            except Exception:
                continue
        image_grids[key] = grid

    action_dim = actions.shape[1] if actions.ndim > 1 else 0
    state_dim = states.shape[1] if states.ndim > 1 else 0
    sample_count = max(1, int(total_steps * max(0.01, float(workspace_ratio))))
    sample_indices = np.linspace(0, total_steps - 1, num=sample_count, dtype=int)
    workspace_points = actions[sample_indices, : min(3, action_dim)].tolist() if action_dim >= 3 else []

    lowdim_step = {}
    if state_dim:
        lowdim_step["observation.state"] = [float(v) for v in states[step_idx].tolist()]

    return {
        "format": "lerobot",
        "dataset": dataset_info,
        "episode_idx": int(episode_idx),
        "episode_length": total_steps,
        "step_idx": step_idx,
        "step_images": step_images,
        "image_unavailable": unavailable,
        "lowdim_step": lowdim_step,
        "current_action": [float(v) for v in actions[step_idx].tolist()] if action_dim else [],
        "current_state": [float(v) for v in states[step_idx].tolist()] if state_dim else [],
        "action_labels": dataset_info.get("action_labels", []),
        "state_labels": dataset_info.get("state_labels", []),
        "actions": actions.tolist() if action_dim else [],
        "states": states.tolist() if state_dim else [],
        "xyz_series": {
            "x": _series(actions, 0),
            "y": _series(actions, 1),
            "z": _series(actions, 2),
        },
        "state_xyz_series": {
            "x": _series(states, 0),
            "y": _series(states, 1),
            "z": _series(states, 2),
        },
        "gripper_series": _series(actions, 7),
        "state_gripper_series": _series(states, 7),
        "image_grids": image_grids,
        "workspace_points": workspace_points,
    }


def _episode_slice(episode_ends: np.ndarray, episode_idx: int) -> slice:
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])
    return slice(start_idx, end_idx)


def _safe_series(values: np.ndarray) -> List[float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr.tolist()


def load_dataset_episode_view(
    path: Union[str, Path],
    episode_idx: int,
    step_idx: int = 0,
    step_interval: int = 5,
    max_frames: int = 20,
    workspace_ratio: float = 0.1,
) -> Dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if _dataset_format(dataset_path) == "lerobot":
        return _load_lerobot_episode_view(
            dataset_path,
            episode_idx=episode_idx,
            step_idx=step_idx,
            step_interval=step_interval,
            max_frames=max_frames,
            workspace_ratio=workspace_ratio,
        )

    dataset_path, root = _open_zarr_dataset(dataset_path)
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

    frame_indices = list(range(0, total_steps, max(1, step_interval)))[:max_frames]

    # Batch-load image data per key: one Zarr read for the whole episode slice,
    # then index in-memory for both step_images and image_grids.
    step_images = {}
    image_grids = {}
    for key in image_keys:
        episode_images = data[key][episode_slice]
        step_images[key] = image_to_data_url(episode_images[step_idx])
        image_grids[key] = [
            {"step_idx": int(fi), "data_url": image_to_data_url(episode_images[fi])}
            for fi in frame_indices
        ]

    lowdim_step = {}
    for key in lowdim_keys:
        lowdim_step[key] = np.asarray(data[key][episode_slice.start + step_idx]).tolist()

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
        "format": "zarr",
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
