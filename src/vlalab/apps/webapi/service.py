"""Helpers for loading VLA-Lab runs behind a web-friendly API."""

from __future__ import annotations

import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import vlalab

from .models import (
    CameraImage,
    LatencyCompareItem,
    LatencyCompareResponse,
    LatencyMetric,
    LatencySeries,
    RunDetail,
    RunSummary,
    StepPreview,
)


def get_runs_dir(dir_override: Optional[str] = None) -> Path:
    """Return the effective runs directory for the web API."""
    return vlalab.get_runs_dir(dir_override).resolve()


def stat_signature(path: Path) -> Tuple[int, int]:
    """Return a cheap (mtime_ns, size) pair for cache-key use."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=1024)
def read_json_cached(path_str: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    if not mtime_ns or not size:
        return {}
    with open(path_str, "r") as f:
        return json.load(f)


@lru_cache(maxsize=512)
def _read_jsonl_cached(path_str: str, mtime_ns: int, size: int) -> Tuple[Dict[str, Any], ...]:
    if not mtime_ns or not size:
        return ()
    with open(path_str, "r") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def load_meta(run_path: Path) -> Dict[str, Any]:
    meta_path = run_path / "meta.json"
    return read_json_cached(str(meta_path), *stat_signature(meta_path))


def load_steps(run_path: Path) -> List[Dict[str, Any]]:
    steps_path = run_path / "steps.jsonl"
    return list(_read_jsonl_cached(str(steps_path), *stat_signature(steps_path)))


def _iter_run_paths(runs_dir: Path, project: Optional[str] = None) -> List[Path]:
    runs: List[Path] = []

    def is_run_dir(path: Path) -> bool:
        return (path / "meta.json").exists() or (path / "steps.jsonl").exists()

    if project:
        project_dir = runs_dir / project
        if project_dir.exists():
            for item in project_dir.iterdir():
                if item.is_dir() and is_run_dir(item):
                    runs.append(item)
    else:
        for project_dir in runs_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                for item in project_dir.iterdir():
                    if item.is_dir() and is_run_dir(item):
                        runs.append(item)

    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def list_projects(runs_dir: Path) -> List[str]:
    return vlalab.list_projects(dir=str(runs_dir))


def count_runs(runs_dir: Path) -> int:
    """Count total runs without loading metadata (fast directory scan)."""
    return len(_iter_run_paths(runs_dir))


def _matches_search(meta: Dict[str, Any], run_path: Path, search: Optional[str]) -> bool:
    if not search:
        return True
    needle = search.strip().lower()
    haystacks = [
        run_path.name,
        run_path.parent.name,
        str(meta.get("model_name", "")),
        str(meta.get("task_name", "")),
        str(meta.get("robot_name", "")),
        str(meta.get("model_path", "")),
        str(meta.get("task_prompt", "")),
    ]
    return any(needle in item.lower() for item in haystacks if item)


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


def latency_ms(timing: Dict[str, Any], key_base: str) -> Optional[float]:
    ms_key = f"{key_base}_ms"
    if ms_key in timing and timing[ms_key] is not None:
        return float(timing[ms_key])
    if key_base in timing and timing[key_base] is not None:
        return float(timing[key_base]) * 1000.0
    return None


def summarize_metric(values: Sequence[Optional[float]]) -> LatencyMetric:
    filtered = np.asarray([value for value in values if value is not None], dtype=float)
    if filtered.size == 0:
        return LatencyMetric()
    return LatencyMetric(
        avg_ms=float(np.mean(filtered)),
        p95_ms=float(np.percentile(filtered, 95)),
        max_ms=float(np.max(filtered)),
    )


def timing_summary(steps: Sequence[Dict[str, Any]]) -> Dict[str, LatencyMetric]:
    transport = []
    inference = []
    total = []
    for step in steps:
        timing = step.get("timing", {})
        transport.append(latency_ms(timing, "transport_latency"))
        inference.append(latency_ms(timing, "inference_latency"))
        total.append(latency_ms(timing, "total_latency"))
    return {
        "transport_latency": summarize_metric(transport),
        "inference_latency": summarize_metric(inference),
        "total_latency": summarize_metric(total),
    }


def build_run_summary(
    run_path: Path,
    meta: Optional[Dict[str, Any]] = None,
    steps: Optional[Sequence[Dict[str, Any]]] = None,
    include_latency: bool = False,
) -> RunSummary:
    meta = meta or load_meta(run_path)
    if steps is None and include_latency:
        steps = load_steps(run_path)

    total_steps = int(meta.get("total_steps") or (len(steps) if steps is not None else 0))
    summary = RunSummary(
        run_id=f"{run_path.parent.name}/{run_path.name}",
        project=run_path.parent.name,
        run_name=run_path.name,
        path=str(run_path),
        start_time=meta.get("start_time"),
        end_time=meta.get("end_time"),
        updated_at=_iso_from_timestamp(run_path.stat().st_mtime),
        model_name=meta.get("model_name", "unknown"),
        model_type=meta.get("model_type"),
        task_name=meta.get("task_name", "unknown"),
        robot_name=meta.get("robot_name", "unknown"),
        total_steps=total_steps,
        action_dim=meta.get("action_dim"),
        action_horizon=meta.get("action_horizon"),
        inference_freq=meta.get("inference_freq"),
    )
    if include_latency and steps is not None:
        summary.latency = timing_summary(steps)
    return summary


def list_run_summaries(
    runs_dir: Path,
    project: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    include_latency: bool = False,
) -> List[RunSummary]:
    items: List[RunSummary] = []
    for run_path in _iter_run_paths(runs_dir, project=project):
        meta = load_meta(run_path)
        if not _matches_search(meta, run_path, search):
            continue
        steps = load_steps(run_path) if include_latency else None
        items.append(
            build_run_summary(
                run_path,
                meta=meta,
                steps=steps,
                include_latency=include_latency,
            )
        )
        if len(items) >= limit:
            break
    return items


def resolve_run_path(runs_dir: Path, project: str, run_name: str) -> Path:
    run_path = (runs_dir / project / run_name).resolve()
    project_path = (runs_dir / project).resolve()
    if project_path not in run_path.parents:
        raise FileNotFoundError(f"Invalid run path: {project}/{run_name}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run not found: {project}/{run_name}")
    return run_path


def image_url(project: str, run_name: str, image_path: str) -> str:
    return f"/api/runs/{project}/{run_name}/artifacts/{image_path}"


def _step_preview(project: str, run_name: str, step: Dict[str, Any]) -> StepPreview:
    obs = step.get("obs", {})
    action = step.get("action", {})
    timing = step.get("timing", {})
    values = action.get("values") or []
    action_preview: List[float] = []
    if values:
        first = values[0]
        if isinstance(first, list):
            action_preview = [float(v) for v in first]

    images = [
        CameraImage(
            camera_name=img.get("camera_name", "default"),
            path=img.get("path", ""),
            url=image_url(project, run_name, img.get("path", "")),
            shape=img.get("shape"),
            encoding=img.get("encoding"),
        )
        for img in obs.get("images", [])
        if img.get("path")
    ]

    return StepPreview(
        step_idx=int(step.get("step_idx", 0)),
        prompt=step.get("prompt"),
        state=[float(v) for v in obs.get("state", [])],
        action_preview=action_preview,
        action_chunk_size=len(values),
        timing={
            "transport_latency_ms": latency_ms(timing, "transport_latency"),
            "inference_latency_ms": latency_ms(timing, "inference_latency"),
            "total_latency_ms": latency_ms(timing, "total_latency"),
        },
        images=images,
        tags=step.get("tags", {}),
    )


def load_run_detail(
    runs_dir: Path,
    project: str,
    run_name: str,
    recent_steps: int = 12,
) -> RunDetail:
    run_path = resolve_run_path(runs_dir, project, run_name)
    meta = load_meta(run_path)
    steps = load_steps(run_path)
    summary = build_run_summary(run_path, meta=meta, steps=steps, include_latency=True)

    cameras = list(meta.get("cameras") or [])
    camera_names: List[str] = []
    for camera in cameras:
        if isinstance(camera, dict) and camera.get("name"):
            camera_names.append(camera["name"])
    if not camera_names and steps:
        image_refs = steps[-1].get("obs", {}).get("images", [])
        camera_names = [img.get("camera_name", "default") for img in image_refs if img.get("camera_name")]

    state_dim = None
    if steps:
        state_dim = len(steps[-1].get("obs", {}).get("state", []))

    recent = [_step_preview(project, run_name, step) for step in steps[-recent_steps:]]
    recent.reverse()

    return RunDetail(
        summary=summary,
        model_path=meta.get("model_path"),
        task_prompt=meta.get("task_prompt"),
        state_dim=state_dim,
        cameras=camera_names,
        timing_summary=timing_summary(steps),
        recent_steps=recent,
        extra=meta.get("extra", {}),
    )


def load_run_steps(
    runs_dir: Path,
    project: str,
    run_name: str,
    offset: int = 0,
    limit: int = 50,
) -> Tuple[int, List[StepPreview]]:
    run_path = resolve_run_path(runs_dir, project, run_name)
    steps = load_steps(run_path)
    sliced = steps[offset : offset + limit]
    return len(steps), [_step_preview(project, run_name, step) for step in sliced]


def resolve_artifact_path(
    runs_dir: Path,
    project: str,
    run_name: str,
    artifact_path: str,
) -> Path:
    run_path = resolve_run_path(runs_dir, project, run_name)
    candidate = (run_path / artifact_path).resolve()
    if run_path not in candidate.parents and candidate != run_path:
        raise FileNotFoundError(f"Invalid artifact path: {artifact_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return candidate


def _downsample(values: Sequence[Optional[float]], max_points: int) -> List[Optional[float]]:
    if len(values) <= max_points:
        return list(values)
    stride = max(1, math.ceil(len(values) / max_points))
    return list(values[::stride])


def _downsample_steps(step_indices: Sequence[int], max_points: int) -> List[int]:
    if len(step_indices) <= max_points:
        return list(step_indices)
    stride = max(1, math.ceil(len(step_indices) / max_points))
    return list(step_indices[::stride])


def build_latency_compare(
    runs_dir: Path,
    run_ids: Sequence[str],
    max_points: int = 300,
) -> LatencyCompareResponse:
    items: List[LatencyCompareItem] = []
    for run_id in run_ids:
        if "/" not in run_id:
            continue
        project, run_name = run_id.split("/", 1)
        run_path = resolve_run_path(runs_dir, project, run_name)
        meta = load_meta(run_path)
        steps = load_steps(run_path)
        summary = build_run_summary(run_path, meta=meta, steps=steps, include_latency=True)

        step_indices = [int(step.get("step_idx", idx)) for idx, step in enumerate(steps)]
        transport = [latency_ms(step.get("timing", {}), "transport_latency") for step in steps]
        inference = [latency_ms(step.get("timing", {}), "inference_latency") for step in steps]
        total = [latency_ms(step.get("timing", {}), "total_latency") for step in steps]

        sampled_steps = _downsample_steps(step_indices, max_points=max_points)
        items.append(
            LatencyCompareItem(
                run=summary,
                summary=timing_summary(steps),
                series=LatencySeries(
                    steps=sampled_steps,
                    transport_latency_ms=_downsample(transport, max_points=max_points),
                    inference_latency_ms=_downsample(inference, max_points=max_points),
                    total_latency_ms=_downsample(total, max_points=max_points),
                ),
            )
        )
    return LatencyCompareResponse(items=items)

