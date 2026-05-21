"""Eval-result helpers for the VLA-Lab web API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .service import read_json_cached, stat_signature

DEFAULT_OPENLOOP_EVAL_PATH = (
    "/data3/jikangye/workspace/baselines/vla-baselines/Isaac-GR00T/outputs/vlalab_eval"
)

DEFAULT_OPENLOOP_EVAL_CANDIDATES = [
    DEFAULT_OPENLOOP_EVAL_PATH,
    "/data3/jikangye/workspace/baselines/dp-baselines/RealWorld-VITA/ckpts/"
    "stack_bowls_double_view/vita-20260226_140930/eval_openloop/step_0000020000",
    "/data3/jikangye/tmp/main_self_contained_eval",
]

DEFAULT_OPENLOOP_EVAL_SCAN_ROOTS: List[str] = []


def _load_json(path: Path) -> Dict[str, Any]:
    return read_json_cached(str(path), *stat_signature(path))


def _generate_demo_data() -> Dict[str, Any]:
    np.random.seed(42)
    steps, dims = 200, 8
    t = np.linspace(0, 4 * np.pi, steps)
    gt = np.column_stack(
        [
            0.5 * np.sin(t),
            0.3 * np.cos(t),
            0.2 * np.sin(0.5 * t),
            *(0.1 * np.sin(t * (idx - 2)) for idx in range(3, dims)),
        ]
    ) + 0.02 * np.random.randn(steps, dims)
    pred = gt + 0.05 * np.random.randn(steps, dims) + 0.02
    err = gt - pred
    results = {
        "results": [
            {
                "trajectory_id": 0,
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "num_steps": steps,
            }
        ],
        "avg_mse": float(np.mean(err**2)),
        "avg_mae": float(np.mean(np.abs(err))),
        "num_trajectories": 1,
        "action_keys": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        "action_horizon": 16,
    }
    return {
        "source": "demo",
        "results": results,
        "available_traj_ids": [0],
        "gt_actions": gt.tolist(),
        "pred_actions": pred.tolist(),
        "states": [],
    }


def _discover_trajectories(results_dir: Path) -> List[int]:
    traj_ids = set()
    for file_path in results_dir.glob("traj_*_gt.npy"):
        parts = file_path.stem.replace("traj_", "").replace("_gt", "")
        try:
            traj_ids.add(int(parts))
        except ValueError:
            continue
    return sorted(traj_ids)


def _split_env_paths(env_name: str) -> List[Path]:
    raw_value = os.getenv(env_name, "")
    return [Path(item).expanduser() for item in raw_value.split(os.pathsep) if item.strip()]


def _summarize_eval_dir(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_dir():
        return None

    json_files = sorted(path.glob("*.json"))
    gt_ids = set(_discover_trajectories(path))
    pred_ids = set()
    for file_path in path.glob("traj_*_pred.npy"):
        parts = file_path.stem.replace("traj_", "").replace("_pred", "")
        try:
            pred_ids.add(int(parts))
        except ValueError:
            continue

    trajectory_ids = sorted(gt_ids & pred_ids) or sorted(gt_ids | pred_ids)
    png_files = sorted(path.glob("*.png"))
    if not json_files and not trajectory_ids and not png_files:
        return None

    if trajectory_ids:
        eval_format = "vlalab-openloop"
    elif json_files:
        eval_format = "openloop-json"
    else:
        eval_format = "openloop-plots"

    latest_mtime = max(
        [item.stat().st_mtime for item in [*json_files, *png_files] if item.exists()] + [path.stat().st_mtime]
    )
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "format": eval_format,
        "json_count": len(json_files),
        "trajectory_count": len(trajectory_ids),
        "trajectory_ids": trajectory_ids[:20],
        "png_count": len(png_files),
        "updated_at": latest_mtime,
    }


def _iter_eval_dirs(root: Path, max_depth: int = 4) -> Iterable[Path]:
    root = root.expanduser()
    if not root.exists():
        return
    if _summarize_eval_dir(root):
        yield root
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth >= max_depth:
            dirs[:] = []
        if any(name.endswith(".json") or name.endswith(".png") or name.endswith(".npy") for name in files):
            if _summarize_eval_dir(current_path):
                yield current_path


def list_eval_candidates(
    candidate_dirs: Optional[Iterable[str | Path]] = None,
    search_roots: Optional[Iterable[str | Path]] = None,
) -> Dict[str, Any]:
    """Return useful OpenLoop eval folders for the web selector."""
    explicit_dirs = (
        [Path(item).expanduser() for item in candidate_dirs]
        if candidate_dirs is not None
        else _split_env_paths("VLALAB_EVAL_CANDIDATE_DIRS")
    )
    if candidate_dirs is None and search_roots is None and not explicit_dirs:
        explicit_dirs = [Path(item).expanduser() for item in DEFAULT_OPENLOOP_EVAL_CANDIDATES]

    roots = (
        [Path(item).expanduser() for item in search_roots]
        if search_roots is not None
        else _split_env_paths("VLALAB_EVAL_SCAN_ROOTS")
    )
    if candidate_dirs is None and search_roots is None and not roots:
        roots = [Path(item).expanduser() for item in DEFAULT_OPENLOOP_EVAL_SCAN_ROOTS]

    candidates_by_path: Dict[str, Dict[str, Any]] = {}
    default_path = ""
    for path in explicit_dirs:
        summary = _summarize_eval_dir(path)
        if not summary:
            continue
        candidates_by_path[summary["path"]] = summary
        if not default_path:
            default_path = summary["path"]

    for root in roots:
        for eval_dir in _iter_eval_dirs(root):
            summary = _summarize_eval_dir(eval_dir)
            if not summary:
                continue
            candidates_by_path.setdefault(summary["path"], summary)
            if not default_path:
                default_path = summary["path"]
            if len(candidates_by_path) >= 40:
                break

    candidates = list(candidates_by_path.values())
    if not default_path and candidates:
        default_path = max(
            candidates,
            key=lambda item: (item["trajectory_count"], item["json_count"], item["png_count"], item["updated_at"]),
        )["path"]

    candidates.sort(
        key=lambda item: (
            item["path"] != default_path,
            -item["trajectory_count"],
            -item["json_count"],
            item["path"],
        )
    )
    return {"default_path": default_path, "candidates": candidates}


def _load_trajectory_arrays(results_dir: Path, traj_id: int) -> Dict[str, Any]:
    arrays: Dict[str, Any] = {}
    for suffix, key in [("gt", "gt_actions"), ("pred", "pred_actions"), ("states", "states")]:
        array_path = results_dir / f"traj_{traj_id}_{suffix}.npy"
        if array_path.exists():
            arrays[key] = np.load(array_path).tolist()
    return arrays


def load_eval_inline(results_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize inline/uploaded eval JSON for the web viewer."""
    payload = dict(results_payload or {})
    gt_actions = payload.pop("gt_actions", None)
    pred_actions = payload.pop("pred_actions", None)
    states = payload.pop("states", None)
    return {
        "source": "inline",
        "results": payload,
        "available_traj_ids": [],
        "selected_traj_id": None,
        "gt_actions": gt_actions,
        "pred_actions": pred_actions,
        "states": states,
        "static_pngs": [],
    }


def load_eval_view(
    source: str = "dir",
    dir_path: Optional[str] = None,
    results_file: Optional[str] = None,
    traj_id: Optional[int] = None,
) -> Dict[str, Any]:
    if source == "demo":
        return _generate_demo_data()

    if not dir_path:
        raise FileNotFoundError("dir_path is required for source='dir'")

    results_dir = Path(dir_path).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {results_dir}")

    json_files = sorted(results_dir.glob("*.json"))
    selected_json = results_dir / results_file if results_file else (json_files[0] if json_files else None)
    results = _load_json(selected_json) if selected_json and selected_json.exists() else {}
    available_traj_ids = _discover_trajectories(results_dir)
    selected_traj_id = traj_id if traj_id is not None else (available_traj_ids[0] if available_traj_ids else None)
    arrays = _load_trajectory_arrays(results_dir, selected_traj_id) if selected_traj_id is not None else {}
    static_pngs = [
        {
            "name": file_path.name,
            "path": str(file_path),
        }
        for file_path in sorted(results_dir.glob("*.png"))
    ]

    return {
        "source": "dir",
        "dir_path": str(results_dir),
        "json_files": [file_path.name for file_path in json_files],
        "selected_json": selected_json.name if selected_json else None,
        "results": results,
        "available_traj_ids": available_traj_ids,
        "selected_traj_id": selected_traj_id,
        "gt_actions": arrays.get("gt_actions"),
        "pred_actions": arrays.get("pred_actions"),
        "states": arrays.get("states"),
        "static_pngs": static_pngs,
    }
