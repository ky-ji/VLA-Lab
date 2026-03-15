"""Eval-result helpers for the VLA-Lab web API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .service import read_json_cached, stat_signature


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
    arrays = _load_trajectory_arrays(results_dir, traj_id) if traj_id is not None else {}
    static_pngs = [
        {
            "name": file_path.name,
            "path": str(file_path),
        }
        for file_path in sorted(results_dir.glob("traj_*.png"))
    ]

    return {
        "source": "dir",
        "dir_path": str(results_dir),
        "json_files": [file_path.name for file_path in json_files],
        "selected_json": selected_json.name if selected_json else None,
        "results": results,
        "available_traj_ids": available_traj_ids,
        "selected_traj_id": traj_id,
        "gt_actions": arrays.get("gt_actions"),
        "pred_actions": arrays.get("pred_actions"),
        "states": arrays.get("states"),
        "static_pngs": static_pngs,
    }
