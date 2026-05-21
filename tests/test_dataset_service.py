import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for color in colors:
        frame = np.full((48, 64, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_lerobot_dataset(root: Path) -> Path:
    dataset = root / "008_stack_bowls"
    features = {
        "action": {
            "dtype": "float32",
            "names": [
                "eef_pose.x",
                "eef_pose.y",
                "eef_pose.z",
                "eef_pose.qx",
                "eef_pose.qy",
                "eef_pose.qz",
                "eef_pose.qw",
                "gripper.pos",
            ],
            "shape": [8],
        },
        "observation.state": {
            "dtype": "float32",
            "names": [
                "eef_pose.x",
                "eef_pose.y",
                "eef_pose.z",
                "eef_pose.qx",
                "eef_pose.qy",
                "eef_pose.qz",
                "eef_pose.qw",
                "gripper.pos",
            ],
            "shape": [8],
        },
        "observation.images.ego_view": {"dtype": "video", "shape": [48, 64, 3]},
        "observation.images.front_view": {"dtype": "video", "shape": [48, 64, 3]},
    }
    _write_json(
        dataset / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "total_episodes": 1,
            "total_frames": 4,
            "fps": 10,
            "features": features,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        },
    )
    _write_jsonl(dataset / "meta" / "episodes.jsonl", [{"episode_index": 0, "length": 4, "tasks": ["Stack bowls"]}])
    _write_jsonl(dataset / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "Stack bowls"}])

    actions = [[float(step + dim / 10) for dim in range(8)] for step in range(4)]
    states = [[float(step * 10 + dim) for dim in range(8)] for step in range(4)]
    table = pa.table(
        {
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    parquet_path = dataset / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)

    _write_video(
        dataset / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4",
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)],
    )
    _write_video(
        dataset / "videos" / "chunk-000" / "observation.images.front_view" / "episode_000000.mp4",
        [(0, 0, 0), (60, 60, 60), (120, 120, 120), (180, 180, 180)],
    )
    return dataset


def test_lerobot_candidates_inspect_and_episode_view(tmp_path):
    from vlalab.apps.webapi.dataset_service import (
        inspect_dataset,
        list_dataset_candidates,
        load_dataset_episode_view,
    )

    dataset = _make_lerobot_dataset(tmp_path)

    candidates = list_dataset_candidates([tmp_path])
    assert candidates["default_path"] == str(dataset)
    assert candidates["candidates"][0]["path"] == str(dataset)
    assert candidates["candidates"][0]["format"] == "lerobot"

    info = inspect_dataset(dataset)
    assert info["format"] == "lerobot"
    assert info["episode_count"] == 1
    assert info["total_steps"] == 4
    assert info["image_keys"] == ["observation.images.ego_view", "observation.images.front_view"]
    assert info["action_dim"] == 8
    assert info["state_dim"] == 8
    assert info["task"] == "Stack bowls"

    view = load_dataset_episode_view(dataset, episode_idx=0, step_idx=2, step_interval=2, max_frames=3)
    assert view["format"] == "lerobot"
    assert view["episode_length"] == 4
    assert view["current_action"][:3] == pytest.approx([2.0, 2.1, 2.2])
    assert view["current_state"][:3] == [20.0, 21.0, 22.0]
    assert view["xyz_series"]["x"] == [0.0, 1.0, 2.0, 3.0]
    assert view["state_xyz_series"]["x"] == [0.0, 10.0, 20.0, 30.0]
    assert view["gripper_series"] == pytest.approx([0.7, 1.7, 2.7, 3.7])
    assert sorted(view["step_images"]) == ["observation.images.ego_view", "observation.images.front_view"]
    assert all(value.startswith("data:image/jpeg;base64,") for value in view["step_images"].values())
    assert len(view["image_grids"]["observation.images.ego_view"]) == 2
