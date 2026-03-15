"""Rich replay, attention, and run-management helpers for the web API."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .service import (
    build_run_summary,
    image_url,
    latency_ms,
    timing_summary,
    load_meta,
    load_steps,
    resolve_run_path,
)

ACTION_DIM_LABELS = {
    8: ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
    7: ["x", "y", "z", "qx", "qy", "qz", "qw"],
    6: ["x", "y", "z", "rx", "ry", "rz"],
    3: ["x", "y", "z"],
}

_GROOT_ATTENTION_BACKEND = None
_GROOT_ATTENTION_PYTHON = None


def _get_action_dim_labels(action_dim: int) -> List[str]:
    if action_dim in ACTION_DIM_LABELS:
        return ACTION_DIM_LABELS[action_dim]
    return [f"dim_{idx}" for idx in range(action_dim)]


def _get_state_dim_labels(meta: Dict[str, Any], state_dim: int) -> List[str]:
    if state_dim <= 0:
        return []

    meta_state_keys = meta.get("state_keys")
    if isinstance(meta_state_keys, list) and len(meta_state_keys) >= state_dim:
        return [str(item) for item in meta_state_keys[:state_dim]]

    extra = meta.get("extra", {})
    if isinstance(extra, dict):
        extra_state_keys = extra.get("state_keys")
        if isinstance(extra_state_keys, list) and len(extra_state_keys) >= state_dim:
            return [str(item) for item in extra_state_keys[:state_dim]]

    if state_dim in ACTION_DIM_LABELS:
        return ACTION_DIM_LABELS[state_dim]

    return [f"dim_{idx}" for idx in range(state_dim)]


def _split_xyz_quat_gripper(dim_labels: List[str]) -> List[Dict[str, Any]]:
    xyz_aliases = {"x", "y", "z"}
    quat_aliases = {"qx", "qy", "qz", "qw", "rx", "ry", "rz"}
    gripper_aliases = {"gripper", "grip", "hand"}

    xyz_indices = []
    quat_indices = []
    gripper_indices = []
    other_indices = []

    for idx, label in enumerate(dim_labels):
        lower = label.lower()
        if lower in xyz_aliases:
            xyz_indices.append(idx)
        elif lower in quat_aliases:
            quat_indices.append(idx)
        elif lower in gripper_aliases:
            gripper_indices.append(idx)
        else:
            other_indices.append(idx)

    if not xyz_indices and dim_labels:
        xyz_indices = list(range(min(3, len(dim_labels))))
        other_indices = [idx for idx in other_indices if idx not in xyz_indices]

    if not quat_indices and len(dim_labels) >= 7:
        fallback = [3, 4, 5, 6]
        quat_indices = [idx for idx in fallback if idx < len(dim_labels)]
        other_indices = [idx for idx in other_indices if idx not in quat_indices]

    if not gripper_indices and len(dim_labels) >= 8:
        gripper_indices = [7]
        other_indices = [idx for idx in other_indices if idx not in gripper_indices]

    if other_indices:
        quat_indices.extend(other_indices)

    return [
        {"key": "xyz", "title": "XYZ", "indices": xyz_indices},
        {"key": "quat", "title": "Orientation", "indices": quat_indices},
        {"key": "gripper", "title": "Gripper", "indices": gripper_indices},
    ]


def _serialize_matrix(arr: np.ndarray) -> List[List[Optional[float]]]:
    if arr.size == 0:
        return []
    serialized: List[List[Optional[float]]] = []
    for row in arr.tolist():
        serialized.append([
            None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)
            for value in row
        ])
    return serialized


def _build_timeline_events(timing: Dict[str, Any]) -> List[Dict[str, Any]]:
    client_send = timing.get("client_send")
    server_recv = timing.get("server_recv")
    infer_start = timing.get("infer_start")
    infer_end = timing.get("infer_end")
    send_timestamp = timing.get("send_timestamp")

    timestamps = [
        value for value in [client_send, server_recv, infer_start, infer_end, send_timestamp] if value is not None
    ]
    if len(timestamps) < 2:
        return []

    base_time = client_send if client_send is not None else min(timestamps)
    events = []
    for label, timestamp, color in [
        ("Client Send", client_send, "#2563eb"),
        ("Server Recv", server_recv, "#0f766e"),
        ("Infer Start", infer_start, "#d97706"),
        ("Infer End", infer_end, "#dc2626"),
        ("Action Sent", send_timestamp, "#7c3aed"),
    ]:
        if timestamp is None:
            continue
        events.append(
            {
                "label": label,
                "offset_ms": float((timestamp - base_time) * 1000.0),
                "color": color,
            }
        )
    return events


def _extract_state_action_data(steps: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    step_count = len(steps)
    max_state_dim = 0
    max_action_dim = 0
    states_raw: List[List[float]] = []
    actions_raw: List[List[float]] = []

    for step in steps:
        state = step.get("obs", {}).get("state", []) or []
        action_values = step.get("action", {}).get("values", []) or []
        first_action = action_values[0] if action_values else []

        max_state_dim = max(max_state_dim, len(state))
        max_action_dim = max(max_action_dim, len(first_action))

        states_raw.append(state)
        actions_raw.append(first_action)

    states_arr = np.full((step_count, max_state_dim), np.nan) if max_state_dim else np.array([])
    actions_arr = np.full((step_count, max_action_dim), np.nan) if max_action_dim else np.array([])

    for idx, state in enumerate(states_raw):
        if state and max_state_dim:
            states_arr[idx, : len(state)] = state

    for idx, action in enumerate(actions_raw):
        if action and max_action_dim:
            actions_arr[idx, : len(action)] = action

    state_labels = _get_state_dim_labels(meta, max_state_dim)
    action_labels = _get_action_dim_labels(max_action_dim)

    return {
        "states_arr": states_arr,
        "actions_arr": actions_arr,
        "state_labels": state_labels,
        "action_labels": action_labels,
        "state_groups": _split_xyz_quat_gripper(state_labels),
        "action_groups": _split_xyz_quat_gripper(action_labels),
    }


def _extract_expanded_execution_data(steps: List[Dict[str, Any]], action_dim_labels: List[str]) -> Dict[str, Any]:
    expanded_actions_list = []
    chunk_boundaries: List[Dict[str, int]] = []
    max_action_dim = 0
    exec_idx = 0

    for step in steps:
        action_values = step.get("action", {}).get("values", []) or []
        if action_values:
            chunk = np.asarray(action_values, dtype=float)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            max_action_dim = max(max_action_dim, chunk.shape[1])
            start = exec_idx
            for action in chunk:
                expanded_actions_list.append(action)
                exec_idx += 1
            chunk_boundaries.append({"start": start, "end": exec_idx})
        else:
            chunk_boundaries.append({"start": exec_idx, "end": exec_idx})

    if not expanded_actions_list or max_action_dim == 0:
        return {
            "expanded_actions": [],
            "chunk_boundaries": chunk_boundaries,
            "action_labels": [],
            "action_groups": _split_xyz_quat_gripper([]),
        }

    expanded_actions = np.full((len(expanded_actions_list), max_action_dim), np.nan)
    for idx, action in enumerate(expanded_actions_list):
        expanded_actions[idx, : len(action)] = action

    labels = action_dim_labels[:max_action_dim] if action_dim_labels else _get_action_dim_labels(max_action_dim)
    return {
        "expanded_actions": _serialize_matrix(expanded_actions),
        "chunk_boundaries": chunk_boundaries,
        "action_labels": labels,
        "action_groups": _split_xyz_quat_gripper(labels),
    }


def _build_action_stats(actions_arr: np.ndarray, action_labels: List[str]) -> List[Dict[str, Any]]:
    if actions_arr.size == 0:
        return []
    stats = []
    for idx in range(actions_arr.shape[1]):
        column = actions_arr[:, idx]
        valid = column[~np.isnan(column)]
        if valid.size == 0:
            continue
        stats.append(
            {
                "label": action_labels[idx] if idx < len(action_labels) else f"dim_{idx}",
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            }
        )
    return stats


def _build_step_details(project: str, run_name: str, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for step in steps:
        obs = step.get("obs", {})
        action = step.get("action", {})
        timing = step.get("timing", {})
        values = action.get("values") or []
        details.append(
            {
                "step_idx": int(step.get("step_idx", 0)),
                "prompt": step.get("prompt"),
                "tags": step.get("tags", {}),
                "state": [float(v) for v in obs.get("state", [])],
                "action_chunk": [[float(v) for v in row] for row in values],
                "action_preview": [float(v) for v in values[0]] if values else [],
                "timing": {
                    "transport_latency_ms": latency_ms(timing, "transport_latency"),
                    "inference_latency_ms": latency_ms(timing, "inference_latency"),
                    "total_latency_ms": latency_ms(timing, "total_latency"),
                    "message_interval_ms": latency_ms(timing, "message_interval"),
                    "client_send": timing.get("client_send"),
                    "server_recv": timing.get("server_recv"),
                    "infer_start": timing.get("infer_start"),
                    "infer_end": timing.get("infer_end"),
                    "send_timestamp": timing.get("send_timestamp"),
                },
                "timeline_events": _build_timeline_events(timing),
                "images": [
                    {
                        "camera_name": image_ref.get("camera_name", "default"),
                        "path": image_ref.get("path", ""),
                        "url": image_url(project, run_name, image_ref.get("path", "")),
                    }
                    for image_ref in obs.get("images", [])
                    if image_ref.get("path")
                ],
                "raw_step": step,
            }
        )
    return details


def load_run_replay(runs_dir: Path, project: str, run_name: str) -> Dict[str, Any]:
    run_path = resolve_run_path(runs_dir, project, run_name)
    meta = load_meta(run_path)
    steps = load_steps(run_path)
    summary = build_run_summary(run_path, meta=meta, steps=steps, include_latency=True)

    extracted = _extract_state_action_data(steps, meta)
    states_arr = extracted["states_arr"]
    actions_arr = extracted["actions_arr"]
    action_labels = extracted["action_labels"]

    timing_series = {
        "transport_latency_ms": [latency_ms(step.get("timing", {}), "transport_latency") for step in steps],
        "inference_latency_ms": [latency_ms(step.get("timing", {}), "inference_latency") for step in steps],
        "total_latency_ms": [latency_ms(step.get("timing", {}), "total_latency") for step in steps],
        "message_interval_ms": [latency_ms(step.get("timing", {}), "message_interval") for step in steps],
    }

    cameras = meta.get("cameras", []) or []
    camera_names = [
        camera.get("name") for camera in cameras if isinstance(camera, dict) and camera.get("name")
    ]
    if not camera_names:
        for step in steps:
            for image_ref in step.get("obs", {}).get("images", []):
                camera_name = image_ref.get("camera_name")
                if camera_name and camera_name not in camera_names:
                    camera_names.append(camera_name)

    attention_caches = list_attention_caches(run_path)

    return {
        "summary": summary.model_dump(),
        "meta": meta,
        "camera_names": camera_names,
        "state_labels": extracted["state_labels"],
        "action_labels": action_labels,
        "state_groups": extracted["state_groups"],
        "action_groups": extracted["action_groups"],
        "states": _serialize_matrix(states_arr),
        "first_actions": _serialize_matrix(actions_arr),
        "timing_series": timing_series,
        "expanded_execution": _extract_expanded_execution_data(steps, action_labels),
        "latency_summary": {
            key: value.model_dump()
            for key, value in timing_summary(steps).items()
        },
        "action_stats": _build_action_stats(actions_arr, action_labels),
        "step_details": _build_step_details(project, run_name, steps),
        "attention_caches": attention_caches,
    }


def delete_run(runs_dir: Path, project: str, run_name: str) -> Dict[str, str]:
    run_path = resolve_run_path(runs_dir, project, run_name)
    shutil.rmtree(run_path)
    return {"status": "deleted", "run_id": f"{project}/{run_name}"}


def list_attention_caches(run_path: Path) -> List[Dict[str, Any]]:
    attention_root = run_path / "artifacts" / "attention"
    if not attention_root.exists():
        return []

    caches = []
    for summary_path in sorted(attention_root.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            continue
        caches.append(
            {
                "name": summary_path.parent.name,
                "summary_path": str(summary_path),
                "output_dir": str(summary_path.parent),
                "step_count": len(data.get("results", [])),
                "attention_layer": data.get("attention_layer"),
                "model_path": data.get("model_path"),
            }
        )
    return caches


def _load_groot_attention_backend():
    global _GROOT_ATTENTION_BACKEND
    if _GROOT_ATTENTION_BACKEND is not None:
        return _GROOT_ATTENTION_BACKEND

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "Isaac-GR00T" / "realworld_deploy" / "offline_attention" / "backend.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("groot_offline_attention_backend", candidate)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load attention backend from {candidate}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _GROOT_ATTENTION_BACKEND = module
            return module

    raise FileNotFoundError("Could not locate Isaac-GR00T offline attention backend in this workspace.")


def _python_has_module(python_executable: str, module_name: str) -> bool:
    result = subprocess.run(
        [python_executable, "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _find_groot_attention_python() -> Optional[str]:
    global _GROOT_ATTENTION_PYTHON
    if _GROOT_ATTENTION_PYTHON is not None:
        return _GROOT_ATTENTION_PYTHON

    backend = _load_groot_attention_backend()
    isaac_root = Path(getattr(backend, "ISAAC_ROOT", Path(backend.__file__).resolve().parents[2]))
    candidates = [Path(sys.executable), isaac_root / ".venv" / "bin" / "python"]

    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists() and _python_has_module(str(candidate), "torch"):
            _GROOT_ATTENTION_PYTHON = str(candidate)
            return _GROOT_ATTENTION_PYTHON
    return None


def load_attention_state(
    runs_dir: Path,
    project: str,
    run_name: str,
    step_idx: int,
    attention_layer: int = -1,
    model_path_override: Optional[str] = None,
    prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    run_path = resolve_run_path(runs_dir, project, run_name)
    backend = _load_groot_attention_backend()
    _, _, model_path, prompt = backend.resolve_run_context(
        run_path,
        model_path_override=model_path_override,
        prompt_override=prompt_override,
    )
    output_dir = backend.build_default_output_dir(run_path, model_path, attention_layer)
    summary = backend.load_cached_summary(output_dir)
    cached_result = backend.get_step_result(summary, step_idx)

    overlays = []
    for overlay in (cached_result or {}).get("overlay_files", []):
        overlay_path = Path(overlay.get("overlay_path", ""))
        if not overlay_path.exists():
            continue
        relative = overlay_path.resolve().relative_to(run_path.resolve())
        overlays.append(
            {
                "camera_name": overlay.get("camera_name", "attention"),
                "overlay_url": image_url(project, run_name, str(relative)),
                "overlay_path": str(relative),
                "heatmap_npy_path": overlay.get("heatmap_npy_path"),
            }
        )

    return {
        "step_idx": step_idx,
        "attention_layer": attention_layer,
        "model_path": model_path,
        "prompt": prompt,
        "output_dir": str(output_dir),
        "cached_step_count": len((summary or {}).get("results", [])),
        "cached_steps": [int(result.get("step_idx", -1)) for result in (summary or {}).get("results", [])],
        "current_cached": cached_result is not None,
        "current": cached_result,
        "overlays": overlays,
    }


def generate_attention(
    runs_dir: Path,
    project: str,
    run_name: str,
    requested_steps: List[int],
    focus_step: Optional[int] = None,
    device: str = "cuda:0",
    attention_layer: int = -1,
    model_path_override: Optional[str] = None,
    prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    run_path = resolve_run_path(runs_dir, project, run_name)
    backend = _load_groot_attention_backend()
    python_executable = _find_groot_attention_python()
    if python_executable is None:
        raise RuntimeError("Could not find a Python environment with torch for attention generation.")

    script_path = Path(backend.__file__).resolve().parent / "run_offline_attention.py"
    _, _, model_path, prompt = backend.resolve_run_context(
        run_path,
        model_path_override=model_path_override,
        prompt_override=prompt_override,
    )
    output_dir = backend.build_default_output_dir(run_path, model_path, attention_layer)

    cmd = [
        python_executable,
        str(script_path),
        "--run-dir",
        str(run_path),
        "--steps",
        ",".join(str(idx) for idx in requested_steps),
        "--device",
        device,
        "--attention-layer",
        str(attention_layer),
        "--output-dir",
        str(output_dir),
    ]
    if model_path_override:
        cmd.extend(["--model-path", model_path_override])
    if prompt_override:
        cmd.extend(["--prompt", prompt_override])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(error_text or f"Attention subprocess failed with code {result.returncode}")

    if focus_step is not None:
        step_to_return = int(focus_step)
    else:
        step_to_return = requested_steps[0] if requested_steps else 0

    if requested_steps and step_to_return not in requested_steps:
        step_to_return = requested_steps[0]

    state = load_attention_state(
        runs_dir,
        project,
        run_name,
        step_idx=step_to_return,
        attention_layer=attention_layer,
        model_path_override=model_path_override,
        prompt_override=prompt_override,
    )
    state["requested_steps"] = requested_steps
    state["focus_step"] = step_to_return
    state["stdout_tail"] = "\n".join((result.stdout or "").strip().splitlines()[-8:])
    return state
