import json
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_eval_dir(root: Path) -> Path:
    eval_dir = root / "openloop_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        eval_dir / "results.json",
        {
            "results": [{"trajectory_id": 3, "mse": 0.01, "mae": 0.08, "num_steps": 4}],
            "avg_mse": 0.01,
            "avg_mae": 0.08,
            "num_trajectories": 1,
            "action_keys": ["x", "y", "z", "gripper"],
        },
    )
    np.save(eval_dir / "traj_3_gt.npy", np.arange(16, dtype=np.float64).reshape(4, 4))
    np.save(eval_dir / "traj_3_pred.npy", np.arange(16, dtype=np.float64).reshape(4, 4) + 0.1)
    np.save(eval_dir / "traj_3_states.npy", np.arange(12, dtype=np.float64).reshape(4, 3))
    (eval_dir / "summary.png").write_bytes(b"png")
    return eval_dir


def test_openloop_eval_candidates_and_default_trajectory(tmp_path):
    from vlalab.apps.webapi.eval_service import list_eval_candidates, load_eval_view

    eval_dir = _make_eval_dir(tmp_path)

    candidates = list_eval_candidates(search_roots=[tmp_path])
    assert candidates["default_path"] == str(eval_dir)
    assert candidates["candidates"][0]["path"] == str(eval_dir)
    assert candidates["candidates"][0]["format"] == "vlalab-openloop"
    assert candidates["candidates"][0]["trajectory_count"] == 1
    assert candidates["candidates"][0]["json_count"] == 1
    assert candidates["candidates"][0]["png_count"] == 1

    view = load_eval_view(dir_path=str(eval_dir))
    assert view["selected_traj_id"] == 3
    assert view["available_traj_ids"] == [3]
    assert view["gt_actions"][0] == [0.0, 1.0, 2.0, 3.0]
    assert view["pred_actions"][0] == [0.1, 1.1, 2.1, 3.1]
    assert view["states"][0] == [0.0, 1.0, 2.0]
    assert view["static_pngs"][0]["name"] == "summary.png"
