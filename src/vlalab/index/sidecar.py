"""Rerun-like sidecar index for VLA-Lab run directories."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from vlalab.apps.webapi.replay_service import (
    _build_step_details,
    _get_action_dim_labels,
    _get_state_dim_labels,
    _split_xyz_quat_gripper,
)
from vlalab.apps.webapi.service import build_run_summary, latency_ms, load_meta

SCHEMA_VERSION = 1
INDEXER_VERSION = "sidecar-v1"
INDEX_DIR_NAME = "vlalab_index"
COLUMNS_FILE = "columns.npz"
MANIFEST_FILE = "manifest.json"
MEDIA_FILE = "media.json"
RERUN_DIR_NAME = "rerun"
RERUN_FILE = "recording.rrd"
RERUN_SERVER_STATE_FILE = "server.json"
DEFAULT_RERUN_WEB_PORT = 9090
DEFAULT_RERUN_RECORDING_PORT = 9876
DEFAULT_RERUN_BIND = "127.0.0.1"
DEFAULT_RERUN_LOCAL_WEB_URL = "http://127.0.0.1:19090"
DEFAULT_RERUN_LOCAL_RECORDING_URL = "rerun+http://127.0.0.1:19876/proxy"

LATENCY_ENTITIES = {
    "/latency/inference_ms": "inference_latency_ms",
    "/latency/transport_ms": "transport_latency_ms",
    "/latency/total_ms": "total_latency_ms",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sidecar_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "artifacts" / INDEX_DIR_NAME


def _manifest_path(run_dir: Path) -> Path:
    return _sidecar_dir(run_dir) / MANIFEST_FILE


def _columns_path(run_dir: Path) -> Path:
    return _sidecar_dir(run_dir) / COLUMNS_FILE


def _media_path(run_dir: Path) -> Path:
    return _sidecar_dir(run_dir) / MEDIA_FILE


def rerun_file_path(run_dir: Path) -> Path:
    return Path(run_dir) / "artifacts" / RERUN_DIR_NAME / RERUN_FILE


def _rerun_server_state_path(run_dir: Path) -> Path:
    return rerun_file_path(run_dir).parent / RERUN_SERVER_STATE_FILE


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _rerun_server_config() -> Dict[str, Any]:
    web_viewer_url = os.environ.get("VLALAB_RERUN_VIEWER_BASE_URL", DEFAULT_RERUN_LOCAL_WEB_URL).rstrip("/")
    recording_url = os.environ.get("VLALAB_RERUN_RECORDING_URL", DEFAULT_RERUN_LOCAL_RECORDING_URL)
    return {
        "bind": os.environ.get("VLALAB_RERUN_BIND", DEFAULT_RERUN_BIND),
        "web_port": _int_env("VLALAB_RERUN_WEB_VIEWER_PORT", DEFAULT_RERUN_WEB_PORT),
        "recording_port": _int_env("VLALAB_RERUN_PORT", DEFAULT_RERUN_RECORDING_PORT),
        "web_viewer_url": web_viewer_url,
        "recording_url": recording_url,
        "viewer_url": f"{web_viewer_url}/?url={quote(recording_url, safe='')}",
    }


def _is_pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _read_rerun_server_state(run_dir: Path) -> Dict[str, Any]:
    path = _rerun_server_state_path(run_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _rerun_rrd_mtime_ns(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def rerun_server_status(run_dir: Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    config = _rerun_server_config()
    state = _read_rerun_server_state(run_path)
    rrd_path = rerun_file_path(run_path)
    expected_rrd = str(rrd_path)
    expected_mtime_ns = _rerun_rrd_mtime_ns(rrd_path)
    state_mtime_ns = state.get("rrd_mtime_ns")
    mtime_matches = expected_mtime_ns is not None and state_mtime_ns == expected_mtime_ns
    running = bool(
        state.get("pid")
        and _is_pid_alive(state.get("pid"))
        and state.get("rrd_path") == expected_rrd
        and mtime_matches
    )
    web_ready = bool(running and state.get("web_ready", state.get("ready")))
    recording_ready = bool(running and state.get("recording_ready", state.get("ready")))
    return {
        "running": running,
        "ready": bool(web_ready and recording_ready),
        "web_ready": web_ready,
        "recording_ready": recording_ready,
        "pid": int(state["pid"]) if running and state.get("pid") is not None else None,
        "rrd_path": expected_rrd,
        "rrd_mtime_ns": expected_mtime_ns,
        "state_rrd_mtime_ns": state_mtime_ns,
        "log_path": state.get("log_path"),
        "started_at": state.get("started_at") if running else None,
        "web_viewer_url": config["web_viewer_url"],
        "recording_url": config["recording_url"],
        "viewer_url": config["viewer_url"],
        "web_port": config["web_port"],
        "recording_port": config["recording_port"],
    }



def _file_signature(path: Path) -> Dict[str, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"mtime_ns": 0, "size": 0}
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def source_signature(run_dir: Path) -> Dict[str, Dict[str, int]]:
    run_path = Path(run_dir)
    return {
        "meta_json": _file_signature(run_path / "meta.json"),
        "steps_jsonl": _file_signature(run_path / "steps.jsonl"),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_steps(run_dir: Path) -> List[Dict[str, Any]]:
    steps_path = Path(run_dir) / "steps.jsonl"
    if not steps_path.exists():
        return []
    steps: List[Dict[str, Any]] = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def _read_steps_window(run_dir: Path, offset: int, limit: int) -> List[Dict[str, Any]]:
    steps_path = Path(run_dir) / "steps.jsonl"
    if not steps_path.exists() or limit <= 0:
        return []
    items: List[Dict[str, Any]] = []
    with steps_path.open("r", encoding="utf-8") as handle:
        row_idx = 0
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if row_idx >= offset and len(items) < limit:
                items.append(json.loads(line))
            row_idx += 1
            if len(items) >= limit:
                break
    return items


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    return value


def _array_values(arr: np.ndarray) -> List[Any]:
    return _clean_value(arr.tolist())


def _latency_series(steps: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    key_base = key.removesuffix("_ms")
    values = [latency_ms(step.get("timing", {}), key_base) for step in steps]
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _first_wall_time(step: Dict[str, Any]) -> Optional[float]:
    timing = step.get("timing", {})
    for key in ("client_send", "server_recv", "infer_start", "send_timestamp"):
        value = _finite_or_none(timing.get(key))
        if value is not None:
            return value
    return None


def _camera_names(meta: Dict[str, Any], steps: Sequence[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for camera in meta.get("cameras", []) or []:
        if isinstance(camera, dict) and camera.get("name") and camera["name"] not in names:
            names.append(str(camera["name"]))
    for step in steps:
        for image_ref in step.get("obs", {}).get("images", []) or []:
            camera_name = image_ref.get("camera_name") or "default"
            if camera_name not in names:
                names.append(str(camera_name))
    return names


def _allocate_columns(steps: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    step_count = len(steps)
    state_dim = 0
    pose_dim = 0
    action_dim = 0
    action_horizon = 0
    for step in steps:
        obs = step.get("obs", {})
        action_values = step.get("action", {}).get("values", []) or []
        state_dim = max(state_dim, len(obs.get("state", []) or []))
        pose_dim = max(pose_dim, len(obs.get("pose", []) or []))
        action_horizon = max(action_horizon, len(action_values))
        for row in action_values:
            if isinstance(row, list):
                action_dim = max(action_dim, len(row))

    columns = {
        "step_idx": np.full((step_count,), -1, dtype=np.int64),
        "elapsed_s": np.full((step_count,), np.nan, dtype=float),
        "wall_time": np.full((step_count,), np.nan, dtype=float),
        "state": np.full((step_count, state_dim), np.nan, dtype=float),
        "eef_pose": np.full((step_count, pose_dim), np.nan, dtype=float),
        "gripper": np.full((step_count,), np.nan, dtype=float),
        "action_first": np.full((step_count, action_dim), np.nan, dtype=float),
        "action_chunk": np.full((step_count, action_horizon, action_dim), np.nan, dtype=float),
        "transport_latency_ms": np.full((step_count,), np.nan, dtype=float),
        "inference_latency_ms": np.full((step_count,), np.nan, dtype=float),
        "total_latency_ms": np.full((step_count,), np.nan, dtype=float),
    }
    return columns


def _populate_columns(columns: Dict[str, np.ndarray], steps: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    wall_times = [_first_wall_time(step) for step in steps]
    base_wall_time = next((value for value in wall_times if value is not None), None)
    inference_freq = _finite_or_none(meta.get("inference_freq"))

    for row_idx, step in enumerate(steps):
        step_idx = int(step.get("step_idx", row_idx))
        columns["step_idx"][row_idx] = step_idx
        wall_time = wall_times[row_idx]
        if wall_time is not None:
            columns["wall_time"][row_idx] = wall_time
            columns["elapsed_s"][row_idx] = wall_time - (base_wall_time or wall_time)
        elif inference_freq and inference_freq > 0:
            columns["elapsed_s"][row_idx] = step_idx / inference_freq
        else:
            columns["elapsed_s"][row_idx] = float(step_idx)

        obs = step.get("obs", {})
        state = obs.get("state", []) or []
        pose = obs.get("pose", []) or []
        if state and columns["state"].shape[1]:
            columns["state"][row_idx, : len(state)] = np.asarray(state, dtype=float)
        if pose and columns["eef_pose"].shape[1]:
            columns["eef_pose"][row_idx, : len(pose)] = np.asarray(pose, dtype=float)
        gripper = _finite_or_none(obs.get("gripper"))
        if gripper is not None:
            columns["gripper"][row_idx] = gripper

        action_values = step.get("action", {}).get("values", []) or []
        for horizon_idx, action in enumerate(action_values):
            if not isinstance(action, list):
                continue
            action_arr = np.asarray(action, dtype=float)
            columns["action_chunk"][row_idx, horizon_idx, : len(action_arr)] = action_arr
            if horizon_idx == 0:
                columns["action_first"][row_idx, : len(action_arr)] = action_arr

        timing = step.get("timing", {})
        for entity, column_name in LATENCY_ENTITIES.items():
            key_base = column_name.removesuffix("_ms")
            value = latency_ms(timing, key_base)
            if value is not None:
                columns[column_name][row_idx] = float(value)


def _build_media_index(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    media: List[Dict[str, Any]] = []
    for row_idx, step in enumerate(steps):
        step_idx = int(step.get("step_idx", row_idx))
        images = []
        for image_ref in step.get("obs", {}).get("images", []) or []:
            path = image_ref.get("path")
            if not path:
                continue
            images.append(
                {
                    "camera_name": image_ref.get("camera_name", "default"),
                    "path": path,
                    "shape": image_ref.get("shape"),
                    "encoding": image_ref.get("encoding", "jpeg"),
                }
            )
        media.append({"row_idx": row_idx, "step_idx": step_idx, "images": images})
    return media


def _build_entities(meta: Dict[str, Any], steps: Sequence[Dict[str, Any]], columns: Dict[str, np.ndarray]) -> Dict[str, Any]:
    state_labels = _get_state_dim_labels(meta, int(columns["state"].shape[1]))
    action_labels = _get_action_dim_labels(int(columns["action_first"].shape[1]))
    entities: Dict[str, Any] = {}
    for name in _camera_names(meta, steps):
        entities[f"/camera/{name}/image"] = {"type": "image", "media": True}
    entities["/robot/state"] = {"type": "scalar_batch", "column": "state", "labels": state_labels}
    entities["/robot/eef_pose"] = {
        "type": "scalar_batch",
        "column": "eef_pose",
        "labels": [f"pose_{idx}" for idx in range(columns["eef_pose"].shape[1])],
    }
    entities["/robot/gripper"] = {"type": "scalar", "column": "gripper", "labels": ["gripper"]}
    entities["/policy/action_first"] = {"type": "scalar_batch", "column": "action_first", "labels": action_labels}
    entities["/policy/action_chunk"] = {
        "type": "tensor",
        "column": "action_chunk",
        "labels": action_labels,
        "shape": list(columns["action_chunk"].shape[1:]),
    }
    for entity, column in LATENCY_ENTITIES.items():
        entities[entity] = {"type": "scalar", "column": column, "labels": [entity.rsplit("/", 1)[-1]]}
    return entities


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _status_from_manifest(run_dir: Path, manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sidecar = _sidecar_dir(run_dir)
    manifest_path = _manifest_path(run_dir)
    columns_path = _columns_path(run_dir)
    media_path = _media_path(run_dir)
    available = bool(manifest and manifest_path.exists() and columns_path.exists() and media_path.exists())
    stale = True
    reason = "missing"
    if available and manifest is not None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            reason = "schema_version"
        elif manifest.get("indexer_version") != INDEXER_VERSION:
            reason = "indexer_version"
        elif manifest.get("source_signature") != source_signature(run_dir):
            reason = "source_signature"
        else:
            stale = False
            reason = "current"
    return {
        "available": available,
        "stale": stale,
        "reason": reason,
        "path": str(sidecar),
        "manifest_path": str(manifest_path),
        "columns_path": str(columns_path),
        "media_path": str(media_path),
        "schema_version": SCHEMA_VERSION,
        "indexer_version": INDEXER_VERSION,
        "manifest": manifest,
    }


def index_status(run_dir: Path) -> Dict[str, Any]:
    manifest = _read_json(_manifest_path(run_dir)) if _manifest_path(run_dir).exists() else None
    return _status_from_manifest(Path(run_dir), manifest)


def build_index(run_dir: Path, force: bool = False) -> Dict[str, Any]:
    run_path = Path(run_dir)
    status = index_status(run_path)
    if status["available"] and not status["stale"] and not force:
        return status

    meta = _read_json(run_path / "meta.json")
    steps = _read_steps(run_path)
    columns = _allocate_columns(steps)
    _populate_columns(columns, steps, meta)
    entities = _build_entities(meta, steps, columns)
    media = _build_media_index(steps)
    state_labels = entities["/robot/state"]["labels"]
    action_labels = entities["/policy/action_first"]["labels"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "indexer_version": INDEXER_VERSION,
        "generated_at": _utc_now(),
        "source_signature": source_signature(run_path),
        "step_count": len(steps),
        "camera_names": _camera_names(meta, steps),
        "state_labels": state_labels,
        "action_labels": action_labels,
        "state_groups": _split_xyz_quat_gripper(state_labels),
        "action_groups": _split_xyz_quat_gripper(action_labels),
        "timelines": {
            "step_idx": {"type": "sequence", "column": "step_idx"},
            "elapsed_s": {"type": "duration", "unit": "s", "column": "elapsed_s"},
            "wall_time": {"type": "timestamp", "unit": "unix_s", "column": "wall_time"},
        },
        "entities": entities,
        "columns_file": COLUMNS_FILE,
        "media_file": MEDIA_FILE,
    }
    sidecar = _sidecar_dir(run_path)
    sidecar.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_columns_path(run_path), **columns)
    _write_json(_media_path(run_path), media)
    _write_json(_manifest_path(run_path), manifest)
    return _status_from_manifest(run_path, manifest)


def ensure_index(run_dir: Path) -> Dict[str, Any]:
    status = index_status(run_dir)
    if not status["available"] or status["stale"]:
        return build_index(run_dir)
    return status


def _load_columns(run_dir: Path) -> Dict[str, np.ndarray]:
    ensure_index(run_dir)
    with np.load(_columns_path(run_dir), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _manifest(run_dir: Path) -> Dict[str, Any]:
    return ensure_index(run_dir)["manifest"] or {}


def _slice_indices(total: int, start: Optional[int], end: Optional[int], stride: int) -> np.ndarray:
    safe_stride = max(1, int(stride or 1))
    safe_start = max(0, int(start or 0))
    safe_end = total if end is None else min(total, max(safe_start, int(end)))
    return np.arange(safe_start, safe_end, safe_stride, dtype=int)


def load_series(
    run_dir: Path,
    entities: Optional[Sequence[str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    stride: int = 1,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _manifest(run_path)
    columns = _load_columns(run_path)
    total = int(manifest.get("step_count", len(columns.get("step_idx", []))))
    indices = _slice_indices(total, start, end, stride)
    requested = list(entities or ["/robot/state", "/policy/action_first", "/latency/inference_ms", "/latency/transport_ms", "/latency/total_ms"])
    response_entities: Dict[str, Any] = {}
    entity_specs = manifest.get("entities", {})
    for entity in requested:
        spec = entity_specs.get(entity)
        if not spec or spec.get("media"):
            continue
        column_name = spec.get("column")
        if column_name not in columns:
            continue
        response_entities[entity] = {
            "type": spec.get("type"),
            "labels": spec.get("labels", []),
            "values": _array_values(columns[column_name][indices]),
        }
    return {
        "total_steps": total,
        "start": int(indices[0]) if indices.size else 0,
        "end": int(indices[-1] + 1) if indices.size else 0,
        "stride": max(1, int(stride or 1)),
        "timeline": {
            "step_idx": _array_values(columns["step_idx"][indices]),
            "elapsed_s": _array_values(columns["elapsed_s"][indices]),
            "wall_time": _array_values(columns["wall_time"][indices]),
        },
        "entities": response_entities,
    }


def load_replay_window(run_dir: Path, project: str, run_name: str, center: int, radius: int = 12) -> Dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _manifest(run_path)
    total = int(manifest.get("step_count", 0))
    if total <= 0:
        offset = 0
        limit = 0
    else:
        safe_center = min(max(0, int(center)), total - 1)
        safe_radius = max(0, int(radius))
        offset = max(0, safe_center - safe_radius)
        limit = min(total - offset, safe_radius * 2 + 1)
    steps = _read_steps_window(run_path, offset, limit)
    return {
        "run_id": f"{project}/{run_name}",
        "total_steps": total,
        "center": int(center),
        "radius": max(0, int(radius)),
        "offset": offset,
        "limit": limit,
        "camera_names": manifest.get("camera_names", []),
        "state_labels": manifest.get("state_labels", []),
        "action_labels": manifest.get("action_labels", []),
        "state_groups": manifest.get("state_groups", []),
        "action_groups": manifest.get("action_groups", []),
        "step_details": _build_step_details(project, run_name, steps),
        "index": index_status(run_path),
    }


def load_replay_summary(run_dir: Path, project: str, run_name: str) -> Dict[str, Any]:
    run_path = Path(run_dir)
    meta = load_meta(run_path)
    status = ensure_index(run_path)
    manifest = status.get("manifest") or {}
    summary = build_run_summary(run_path, meta=meta, steps=[], include_latency=False).model_dump()
    if manifest.get("step_count") is not None:
        summary["total_steps"] = int(manifest["step_count"])
    return {
        "summary": summary,
        "meta": meta,
        "camera_names": manifest.get("camera_names", []),
        "state_labels": manifest.get("state_labels", []),
        "action_labels": manifest.get("action_labels", []),
        "state_groups": manifest.get("state_groups", []),
        "action_groups": manifest.get("action_groups", []),
        "index": status,
    }


def rerun_status(run_dir: Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    path = rerun_file_path(run_path)
    idx_status = index_status(run_path)
    manifest_path = _manifest_path(run_path)
    available = path.exists()
    stale = not available or idx_status.get("stale", True)
    if available and manifest_path.exists():
        stale = stale or path.stat().st_mtime_ns < manifest_path.stat().st_mtime_ns
    server = rerun_server_status(run_path)
    return {
        "available": available,
        "stale": stale,
        "path": str(path),
        "url": f"/api/runs/{{project}}/{{run_name}}/rerun/file",
        "index_available": idx_status.get("available", False),
        "index_stale": idx_status.get("stale", True),
        "server": server,
        "web_viewer_url": server["web_viewer_url"],
        "recording_url": server["recording_url"],
        "viewer_url": server["viewer_url"],
    }


def _vector_labels(labels: Sequence[str], size: int, fallback: Sequence[str]) -> List[str]:
    if labels and len(labels) >= size:
        return [str(label) for label in labels[:size]]
    if len(fallback) >= size:
        return [str(label) for label in fallback[:size]]
    return [f"dim_{idx}" for idx in range(size)]


def _action_chunk_labels(action_labels: Sequence[str], horizon: int, action_dim: int) -> List[str]:
    return [f"h{horizon_idx:02d}:{action_labels[dim_idx]}" for horizon_idx in range(horizon) for dim_idx in range(action_dim)]


def _flatten_action_chunk(values: Any, horizon: int, action_dim: int) -> List[float]:
    rows = values.tolist() if isinstance(values, np.ndarray) else values
    if rows is None:
        return []
    flat: List[float] = []
    for horizon_idx in range(horizon):
        row = rows[horizon_idx] if horizon_idx < len(rows) else []
        for dim_idx in range(action_dim):
            value = row[dim_idx] if dim_idx < len(row) else np.nan
            flat.append(float(value) if _finite_or_none(value) is not None else np.nan)
    return flat


def _log_rerun_series_style(rec: Any, entity: str, labels: Sequence[str]) -> None:
    if not labels:
        return
    import rerun as rr

    rec.log(entity.lstrip("/"), rr.SeriesLines(names=[str(label) for label in labels]), static=True)


def _log_rerun_scalars(rec: Any, entity: str, values: Any) -> None:
    import rerun as rr

    if values is None:
        return
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, list) and not values:
        return
    rec.log(entity.lstrip("/"), rr.Scalars(values))


def _rerun_default_blueprint(camera_names: Sequence[str]) -> Any:
    try:
        import rerun.blueprint as rrb
    except Exception:
        return None

    camera_views = [
        rrb.Spatial2DView(origin=f"/camera/{camera_name}", name=str(camera_name))
        for camera_name in camera_names
    ] or [rrb.Spatial2DView(origin="/camera", name="Cameras")]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*camera_views, name="Cameras", row_shares=[1.0 for _ in camera_views]),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/robot", contents="/robot/**", name="Robot state"),
                rrb.TimeSeriesView(origin="/policy", contents="/policy/**", name="Policy actions"),
                rrb.TimeSeriesView(origin="/latency", contents="/latency/**", name="Latency"),
                name="Signals",
                row_shares=[1.0, 1.0, 0.7],
            ),
            name="VLA replay",
            column_shares=[1.4, 1.0],
        ),
        collapse_panels=False,
    )


def _external_rerun_env() -> Optional[Dict[str, str]]:
    rerun_root = Path(os.environ.get("VLALAB_RERUN_PYTHONPATH", "/data3/jikangye/.rerun-demo-py"))
    if not rerun_root.exists():
        return None
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{rerun_root}:{rerun_root / 'rerun_sdk'}"
    return env


def _build_rerun_recording_subprocess(run_path: Path, rrd_path: Path) -> None:
    python_exe = os.environ.get("VLALAB_RERUN_PYTHON", "/usr/bin/python3")
    env = _external_rerun_env()
    if env is None or not Path(python_exe).exists():
        raise RuntimeError("Rerun export requires rerun-sdk for this Python, or VLALAB_RERUN_PYTHON/PYTHONPATH")
    code = r'''
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import rerun as rr

run_path = Path(sys.argv[1])
rrd_path = Path(sys.argv[2])
steps_path = run_path / "steps.jsonl"
steps = []
if steps_path.exists():
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number

def latency_ms(timing, key):
    if timing.get(key) is not None:
        return finite(timing.get(key))
    base = key.removesuffix("_ms")
    if timing.get(base) is not None:
        value = finite(timing.get(base))
        return None if value is None else value * 1000.0
    if key == "total_latency_ms":
        parts = [latency_ms(timing, "transport_latency_ms"), latency_ms(timing, "inference_latency_ms")]
        valid = [part for part in parts if part is not None]
        return sum(valid) if valid else None
    return None

def labels(candidates, size, fallback):
    candidates = candidates or []
    if len(candidates) >= size:
        return [str(item) for item in candidates[:size]]
    if len(fallback) >= size:
        return [str(item) for item in fallback[:size]]
    return [f"dim_{idx}" for idx in range(size)]

def action_chunk_labels(action_labels, horizon, action_dim):
    return [f"h{horizon_idx:02d}:{action_labels[dim_idx]}" for horizon_idx in range(horizon) for dim_idx in range(action_dim)]

def flatten_action_chunk(rows, horizon, action_dim):
    rows = rows or []
    flat = []
    for horizon_idx in range(horizon):
        row = rows[horizon_idx] if horizon_idx < len(rows) else []
        for dim_idx in range(action_dim):
            value = row[dim_idx] if dim_idx < len(row) else float("nan")
            flat.append(float(value) if finite(value) is not None else float("nan"))
    return flat

def log_scalars(rec, entity, values):
    if values is None:
        return
    if isinstance(values, list) and not values:
        return
    rec.log(entity, rr.Scalars(values))

def log_series_style(rec, entity, names):
    if names:
        rec.log(entity, rr.SeriesLines(names=names), static=True)

def default_blueprint(camera_names):
    try:
        import rerun.blueprint as rrb
    except Exception:
        return None
    camera_views = [rrb.Spatial2DView(origin=f"/camera/{name}", name=str(name)) for name in camera_names]
    if not camera_views:
        camera_views = [rrb.Spatial2DView(origin="/camera", name="Cameras")]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*camera_views, name="Cameras", row_shares=[1.0 for _ in camera_views]),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/robot", contents="/robot/**", name="Robot state"),
                rrb.TimeSeriesView(origin="/policy", contents="/policy/**", name="Policy actions"),
                rrb.TimeSeriesView(origin="/latency", contents="/latency/**", name="Latency"),
                name="Signals",
                row_shares=[1.0, 1.0, 0.7],
            ),
            name="VLA replay",
            column_shares=[1.4, 1.0],
        ),
        collapse_panels=False,
    )

meta = {}
meta_path = run_path / "meta.json"
if meta_path.exists():
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

state_dim = max((len(step.get("obs", {}).get("state", []) or []) for step in steps), default=0)
action_horizon = max((len(step.get("action", {}).get("values", []) or []) for step in steps), default=0)
action_dim = 0
for step in steps:
    for row in step.get("action", {}).get("values", []) or []:
        action_dim = max(action_dim, len(row))
state_names = labels(meta.get("state_keys") or meta.get("state_labels"), state_dim, ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
action_names = labels(meta.get("action_keys") or meta.get("action_labels"), action_dim, ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
chunk_names = action_chunk_labels(action_names, action_horizon, action_dim)
camera_names = []
for step in steps:
    for image_ref in step.get("obs", {}).get("images", []) or []:
        name = image_ref.get("camera_name") or "default"
        if name not in camera_names:
            camera_names.append(name)

base_time = None
for step in steps:
    timing = step.get("timing", {})
    for key in ("client_send", "server_recv", "infer_start", "send_timestamp"):
        value = finite(timing.get(key))
        if value is not None:
            base_time = value
            break
    if base_time is not None:
        break

rrd_path.parent.mkdir(parents=True, exist_ok=True)
with rr.RecordingStream("vlalab", recording_id=run_path.name, send_properties=False) as rec:
    blueprint = default_blueprint(camera_names)
    if blueprint is None:
        rec.save(str(rrd_path))
    else:
        rec.save(str(rrd_path), default_blueprint=blueprint)
    log_series_style(rec, "robot/state", state_names)
    log_series_style(rec, "policy/action_first", action_names)
    log_series_style(rec, "policy/action_chunk", chunk_names)
    log_series_style(rec, "latency/transport_ms", ["transport_ms"])
    log_series_style(rec, "latency/inference_ms", ["inference_ms"])
    log_series_style(rec, "latency/total_ms", ["total_ms"])
    for row_idx, step in enumerate(steps):
        step_idx = int(step.get("step_idx", row_idx))
        rec.set_time("step_idx", sequence=step_idx)
        timing = step.get("timing", {})
        wall_time = None
        for key in ("client_send", "server_recv", "infer_start", "send_timestamp"):
            wall_time = finite(timing.get(key))
            if wall_time is not None:
                break
        if wall_time is not None and base_time is not None:
            rec.set_time("elapsed_s", duration=wall_time - base_time)
        obs = step.get("obs", {})
        for image_ref in obs.get("images", []) or []:
            rel_path = image_ref.get("path")
            if not rel_path:
                continue
            image_path = run_path / rel_path
            if not image_path.exists():
                continue
            try:
                image = np.asarray(Image.open(image_path).convert("RGB"))
            except Exception:
                continue
            rec.log(f"camera/{image_ref.get('camera_name', 'default')}/image", rr.Image(image))
        log_scalars(rec, "robot/state", obs.get("state", []))
        log_scalars(rec, "robot/eef_pose", obs.get("pose", []))
        log_scalars(rec, "robot/gripper", obs.get("gripper"))
        actions = step.get("action", {}).get("values", []) or []
        if actions:
            log_scalars(rec, "policy/action_first", actions[0])
            log_scalars(rec, "policy/action_chunk", flatten_action_chunk(actions, action_horizon, action_dim))
        log_scalars(rec, "latency/transport_ms", latency_ms(timing, "transport_latency_ms"))
        log_scalars(rec, "latency/inference_ms", latency_ms(timing, "inference_latency_ms"))
        log_scalars(rec, "latency/total_ms", latency_ms(timing, "total_latency_ms"))
'''
    proc = subprocess.run(
        [python_exe, "-c", code, str(run_path), str(rrd_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"Rerun export subprocess failed: {detail}")


def build_rerun_recording(run_dir: Path, force: bool = False) -> Dict[str, Any]:
    run_path = Path(run_dir)
    status = rerun_status(run_path)
    if status["available"] and not status["stale"] and not force:
        return status
    rrd_path = rerun_file_path(run_path)

    try:
        import cv2
        import rerun as rr
    except Exception:
        ensure_index(run_path)
        _build_rerun_recording_subprocess(run_path, rrd_path)
        return rerun_status(run_path)

    manifest = ensure_index(run_path)["manifest"] or {}
    columns = _load_columns(run_path)
    steps = _read_steps(run_path)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    camera_names = manifest.get("camera_names", [])
    state_labels = _vector_labels(manifest.get("state_labels", []), columns["state"].shape[1], ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
    action_labels = _vector_labels(manifest.get("action_labels", []), columns["action_first"].shape[1], ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
    action_chunk_labels = _action_chunk_labels(action_labels, columns["action_chunk"].shape[1], columns["action_chunk"].shape[2])

    with rr.RecordingStream("vlalab", recording_id=run_path.name, send_properties=False) as rec:
        blueprint = _rerun_default_blueprint(camera_names)
        if blueprint is None:
            rec.save(str(rrd_path))
        else:
            rec.save(str(rrd_path), default_blueprint=blueprint)
        _log_rerun_series_style(rec, "/robot/state", state_labels)
        _log_rerun_series_style(rec, "/policy/action_first", action_labels)
        _log_rerun_series_style(rec, "/policy/action_chunk", action_chunk_labels)
        _log_rerun_series_style(rec, "/latency/transport_ms", ["transport_ms"])
        _log_rerun_series_style(rec, "/latency/inference_ms", ["inference_ms"])
        _log_rerun_series_style(rec, "/latency/total_ms", ["total_ms"])
        for row_idx, step in enumerate(steps):
            step_idx = int(columns["step_idx"][row_idx])
            rec.set_time("step_idx", sequence=step_idx)
            elapsed = _finite_or_none(columns["elapsed_s"][row_idx])
            if elapsed is not None:
                rec.set_time("elapsed_s", duration=elapsed)
            for image_ref in step.get("obs", {}).get("images", []) or []:
                rel_path = image_ref.get("path")
                if not rel_path:
                    continue
                image_path = run_path / rel_path
                if not image_path.exists():
                    continue
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                camera_name = image_ref.get("camera_name", "default")
                rec.log(f"camera/{camera_name}/image", rr.Image(image_rgb))
            _log_rerun_scalars(rec, "/robot/state", columns["state"][row_idx])
            _log_rerun_scalars(rec, "/robot/eef_pose", columns["eef_pose"][row_idx])
            _log_rerun_scalars(rec, "/robot/gripper", columns["gripper"][row_idx])
            _log_rerun_scalars(rec, "/policy/action_first", columns["action_first"][row_idx])
            _log_rerun_scalars(
                rec,
                "/policy/action_chunk",
                _flatten_action_chunk(
                    columns["action_chunk"][row_idx],
                    columns["action_chunk"].shape[1],
                    columns["action_chunk"].shape[2],
                ),
            )
            for entity, column in LATENCY_ENTITIES.items():
                _log_rerun_scalars(rec, entity, columns[column][row_idx])
    return rerun_status(run_path)

def _rerun_cli_command() -> tuple[List[str], Dict[str, str]]:
    env = _external_rerun_env() or os.environ.copy()
    rerun_bin = os.environ.get("VLALAB_RERUN_BIN")
    if rerun_bin:
        return [rerun_bin], env

    rerun_root = Path(os.environ.get("VLALAB_RERUN_PYTHONPATH", "/data3/jikangye/.rerun-demo-py"))
    rerun_script = Path(os.environ.get("VLALAB_RERUN_SCRIPT", str(rerun_root / "bin" / "rerun")))
    python_exe = os.environ.get("VLALAB_RERUN_PYTHON", sys.executable)
    if rerun_script.exists() and Path(python_exe).exists():
        return [python_exe, str(rerun_script)], env

    discovered = shutil.which("rerun")
    if discovered:
        return [discovered], env
    raise RuntimeError("Rerun server requires the rerun CLI; set VLALAB_RERUN_BIN or VLALAB_RERUN_SCRIPT")


def _stop_rerun_ports(web_port: int, recording_port: int) -> None:
    ports = " ".join(str(int(port)) for port in (web_port, recording_port))
    terminate = f"""
for port in {ports}; do
  if command -v fuser >/dev/null 2>&1; then
    fuser -TERM -k "$port"/tcp >/dev/null 2>&1 || true
  elif command -v lsof >/dev/null 2>&1; then
    for pid in $(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null); do kill -TERM "$pid" >/dev/null 2>&1 || true; done
  fi
done
"""
    kill = terminate.replace("-TERM", "-KILL").replace("kill -TERM", "kill -KILL")
    subprocess.run(["bash", "-lc", terminate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(0.5)
    subprocess.run(["bash", "-lc", kill], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _wait_for_rerun_http(host: str, port: int, timeout: float = 8.0, path: str = "/") -> bool:
    deadline = time.time() + timeout
    safe_path = path if path.startswith("/") else f"/{path}"
    url = f"http://{host}:{int(port)}{safe_path}"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as response:
                if 200 <= int(response.status) < 500:
                    return True
        except HTTPError as exc:
            if 200 <= int(exc.code) < 500:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def _reuse_ready_rerun_server(run_path: Path, rrd_path: Path) -> Optional[Dict[str, Any]]:
    status = rerun_status(run_path)
    server = status.get("server") or {}
    if server.get("running") and server.get("ready") and server.get("rrd_path") == str(rrd_path):
        status["server"] = {**server, "reused": True}
        status["web_viewer_url"] = status["server"]["web_viewer_url"]
        status["recording_url"] = status["server"]["recording_url"]
        status["viewer_url"] = status["server"]["viewer_url"]
        return status
    return None


def serve_rerun_recording(run_dir: Path, force: bool = False) -> Dict[str, Any]:
    run_path = Path(run_dir)
    file_status = build_rerun_recording(run_path, force=force)
    rrd_path = Path(file_status["path"])
    if not rrd_path.exists():
        raise RuntimeError("Rerun recording has not been built")

    if not force:
        reusable = _reuse_ready_rerun_server(run_path, rrd_path)
        if reusable is not None:
            return reusable

    config = _rerun_server_config()
    cmd_prefix, env = _rerun_cli_command()
    _stop_rerun_ports(config["web_port"], config["recording_port"])

    log_path = rrd_path.parent / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *cmd_prefix,
        str(rrd_path),
        "--web-viewer",
        "--bind",
        config["bind"],
        "--web-viewer-port",
        str(config["web_port"]),
        "--port",
        str(config["recording_port"]),
    ]
    with log_path.open("ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

    web_ready = _wait_for_rerun_http(config["bind"], config["web_port"], path="/")
    recording_ready = _wait_for_rerun_http(config["bind"], config["recording_port"], path="/proxy")
    ready = web_ready and recording_ready
    state = {
        "running": True,
        "ready": ready,
        "web_ready": web_ready,
        "recording_ready": recording_ready,
        "pid": proc.pid,
        "rrd_path": str(rrd_path),
        "rrd_mtime_ns": _rerun_rrd_mtime_ns(rrd_path),
        "log_path": str(log_path),
        "started_at": _utc_now(),
        "web_viewer_url": config["web_viewer_url"],
        "recording_url": config["recording_url"],
        "viewer_url": config["viewer_url"],
        "web_port": config["web_port"],
        "recording_port": config["recording_port"],
        "command": cmd,
    }
    _rerun_server_state_path(run_path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    status = rerun_status(run_path)
    status["server"] = state
    status["web_viewer_url"] = state["web_viewer_url"]
    status["recording_url"] = state["recording_url"]
    status["viewer_url"] = state["viewer_url"]
    return status
