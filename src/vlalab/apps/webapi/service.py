"""Helpers for loading VLA-Lab runs behind a web-friendly API."""

from __future__ import annotations

import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import vlalab

from .deploy_service import resolve_dashboard_runs_config
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


@dataclass(frozen=True)
class RunsSource:
    path: str
    display_path: str
    local_path: Optional[Path] = None
    ssh_host: Optional[str] = None
    shell: str = "bash -lc"

    @property
    def is_remote(self) -> bool:
        return self.ssh_host is not None


RunsSourceLike = Union[RunsSource, Path]


def get_runs_source(dir_override: Optional[str] = None) -> RunsSource:
    """Return the effective runs source for the web API."""
    if dir_override is None:
        configured = resolve_dashboard_runs_config()
        if configured is not None:
            return RunsSource(
                path=configured.runs_dir,
                display_path=configured.workdir,
                local_path=None,
                ssh_host=configured.ssh_host,
                shell=configured.shell,
            )

    local_path = vlalab.get_runs_dir(dir_override).resolve()
    return RunsSource(path=str(local_path), display_path=str(local_path), local_path=local_path)


def get_runs_dir(dir_override: Optional[str] = None) -> Path:
    """Compatibility helper returning the configured runs path as a Path-like value."""
    return Path(get_runs_source(dir_override).path)


def _coerce_runs_source(runs_source: RunsSourceLike) -> RunsSource:
    if isinstance(runs_source, RunsSource):
        return runs_source
    local_path = Path(runs_source).resolve()
    return RunsSource(path=str(local_path), display_path=str(local_path), local_path=local_path)


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


def _ssh_prefix(ssh_host: str, connect_timeout: int = 8) -> List[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        ssh_host,
    ]


def _shell_command(shell: str, script: str) -> str:
    normalized_shell = str(shell or "bash -lc").strip() or "bash -lc"
    return f"{normalized_shell} {shlex.quote(script)}"


def _run_remote_shell(
    source: RunsSource,
    script: str,
    *,
    timeout: float = 30.0,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    if not source.is_remote or not source.ssh_host:
        raise RuntimeError("Remote shell requested for a local runs source")
    command = _ssh_prefix(source.ssh_host) + [_shell_command(source.shell, script)]
    return subprocess.run(
        command,
        capture_output=True,
        text=text,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )


def _run_remote_python(
    source: RunsSource,
    code: str,
    args: Sequence[str],
    *,
    timeout: float = 30.0,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    argv = " ".join(shlex.quote(str(arg)) for arg in args)
    script = f"""PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
  echo "Python not found on remote host" >&2
  exit 127
fi
"$PYTHON_BIN" - {argv} <<'PY'
{code}
PY"""
    return _run_remote_shell(source, script, timeout=timeout, text=text)


def _remote_error(result: subprocess.CompletedProcess[Any]) -> str:
    stderr = result.stderr if isinstance(result.stderr, str) else (result.stderr or b"").decode("utf-8", errors="replace")
    stdout = result.stdout if isinstance(result.stdout, str) else (result.stdout or b"").decode("utf-8", errors="replace")
    message = (stderr or stdout or f"remote command failed with code {result.returncode}").strip()
    return message


def _run_remote_json(
    source: RunsSource,
    code: str,
    args: Sequence[str],
    *,
    timeout: float = 30.0,
) -> Any:
    result = _run_remote_python(source, code, args, timeout=timeout, text=True)
    if result.returncode != 0:
        raise FileNotFoundError(_remote_error(result))
    output = (result.stdout or "").strip()
    if not output:
        return None
    return json.loads(output)


def _run_remote_bytes(
    source: RunsSource,
    code: str,
    args: Sequence[str],
    *,
    timeout: float = 30.0,
) -> bytes:
    result = _run_remote_python(source, code, args, timeout=timeout, text=False)
    if result.returncode != 0:
        raise FileNotFoundError(_remote_error(result))
    return bytes(result.stdout or b"")


def _remote_runs_listing(source: RunsSource, project: Optional[str] = None) -> List[Dict[str, Any]]:
    code = """
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
project = sys.argv[2] if len(sys.argv) > 2 else None
items = []

def is_run_dir(path: Path) -> bool:
    return (path / "meta.json").exists() or (path / "steps.jsonl").exists()

def add_run(path: Path) -> None:
    meta = {}
    meta_path = path / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    items.append(
        {
            "project": path.parent.name,
            "run_name": path.name,
            "path": str(path),
            "updated_at_ts": path.stat().st_mtime,
            "meta": meta,
        }
    )

if runs_dir.exists():
    if project:
        project_dir = runs_dir / project
        if project_dir.exists():
            for item in project_dir.iterdir():
                if item.is_dir() and is_run_dir(item):
                    add_run(item)
    else:
        for project_dir in runs_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                for item in project_dir.iterdir():
                    if item.is_dir() and is_run_dir(item):
                        add_run(item)

items.sort(key=lambda item: item["updated_at_ts"], reverse=True)
print(json.dumps(items))
"""
    args = [source.path]
    if project:
        args.append(project)
    payload = _run_remote_json(source, code, args, timeout=60.0)
    return payload or []


def _remote_load_meta(source: RunsSource, project: str, run_name: str) -> Dict[str, Any]:
    code = """
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1]).resolve()
project = sys.argv[2]
run_name = sys.argv[3]
project_path = (runs_dir / project).resolve()
run_path = (project_path / run_name).resolve()
meta_path = run_path / "meta.json"

if project_path not in run_path.parents:
    raise FileNotFoundError(f"Invalid run path: {project}/{run_name}")
if not run_path.exists():
    raise FileNotFoundError(f"Run not found: {project}/{run_name}")
if not meta_path.exists():
    print("{}")
    raise SystemExit(0)

print(meta_path.read_text())
"""
    payload = _run_remote_json(source, code, [source.path, project, run_name], timeout=30.0)
    return payload or {}


def _remote_load_steps(source: RunsSource, project: str, run_name: str) -> List[Dict[str, Any]]:
    code = """
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1]).resolve()
project = sys.argv[2]
run_name = sys.argv[3]
project_path = (runs_dir / project).resolve()
run_path = (project_path / run_name).resolve()
steps_path = run_path / "steps.jsonl"

if project_path not in run_path.parents:
    raise FileNotFoundError(f"Invalid run path: {project}/{run_name}")
if not run_path.exists():
    raise FileNotFoundError(f"Run not found: {project}/{run_name}")
if not steps_path.exists():
    print("[]")
    raise SystemExit(0)

items = []
with open(steps_path, "r") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
print(json.dumps(items))
"""
    payload = _run_remote_json(source, code, [source.path, project, run_name], timeout=60.0)
    return payload or []


def _remote_read_artifact_bytes(source: RunsSource, project: str, run_name: str, artifact_path: str) -> bytes:
    code = """
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1]).resolve()
project = sys.argv[2]
run_name = sys.argv[3]
artifact_path = sys.argv[4]
project_path = (runs_dir / project).resolve()
run_path = (project_path / run_name).resolve()
candidate = (run_path / artifact_path).resolve()

if project_path not in run_path.parents:
    raise FileNotFoundError(f"Invalid run path: {project}/{run_name}")
if run_path not in candidate.parents and candidate != run_path:
    raise FileNotFoundError(f"Invalid artifact path: {artifact_path}")
if not candidate.exists():
    raise FileNotFoundError(f"Artifact not found: {artifact_path}")

with open(candidate, "rb") as handle:
    sys.stdout.buffer.write(handle.read())
"""
    return _run_remote_bytes(source, code, [source.path, project, run_name, artifact_path], timeout=60.0)


def _remote_delete_run(source: RunsSource, project: str, run_name: str) -> None:
    code = """
import shutil
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1]).resolve()
project = sys.argv[2]
run_name = sys.argv[3]
project_path = (runs_dir / project).resolve()
run_path = (project_path / run_name).resolve()

if project_path not in run_path.parents:
    raise FileNotFoundError(f"Invalid run path: {project}/{run_name}")
if not run_path.exists():
    raise FileNotFoundError(f"Run not found: {project}/{run_name}")

shutil.rmtree(run_path)
print("{}")
"""
    _run_remote_json(source, code, [source.path, project, run_name], timeout=60.0)


def remote_list_attention_caches(source: RunsSource, project: str, run_name: str) -> List[Dict[str, Any]]:
    code = """
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1]).resolve()
project = sys.argv[2]
run_name = sys.argv[3]
project_path = (runs_dir / project).resolve()
run_path = (project_path / run_name).resolve()
attention_root = run_path / "artifacts" / "attention"

if project_path not in run_path.parents:
    raise FileNotFoundError(f"Invalid run path: {project}/{run_name}")
if not attention_root.exists():
    print("[]")
    raise SystemExit(0)

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

print(json.dumps(caches))
"""
    payload = _run_remote_json(source, code, [source.path, project, run_name], timeout=60.0)
    return payload or []


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
        if not runs_dir.exists():
            return []
        for project_dir in runs_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                for item in project_dir.iterdir():
                    if item.is_dir() and is_run_dir(item):
                        runs.append(item)

    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def list_projects(runs_source: RunsSourceLike) -> List[str]:
    source = _coerce_runs_source(runs_source)
    if source.is_remote:
        return sorted({item["project"] for item in _remote_runs_listing(source)})
    return vlalab.list_projects(dir=source.path)


def count_runs(runs_source: RunsSourceLike) -> int:
    """Count total runs without loading metadata (fast directory scan)."""
    source = _coerce_runs_source(runs_source)
    if source.is_remote:
        return len(_remote_runs_listing(source))
    if source.local_path is None:
        return 0
    return len(_iter_run_paths(source.local_path))


def _matches_search(
    meta: Dict[str, Any],
    *,
    project: str,
    run_name: str,
    path_label: str,
    search: Optional[str],
) -> bool:
    if not search:
        return True
    needle = search.strip().lower()
    haystacks = [
        run_name,
        project,
        path_label,
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
    if key_base == "total_latency":
        transport = None
        inference = None
        if timing.get("transport_latency_ms") is not None:
            transport = float(timing["transport_latency_ms"])
        elif timing.get("transport_latency") is not None:
            transport = float(timing["transport_latency"]) * 1000.0

        if timing.get("inference_latency_ms") is not None:
            inference = float(timing["inference_latency_ms"])
        elif timing.get("inference_latency") is not None:
            inference = float(timing["inference_latency"]) * 1000.0

        if transport is not None or inference is not None:
            return (transport or 0.0) + (inference or 0.0)
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
    run_path: Optional[Path] = None,
    *,
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    path_label: Optional[str] = None,
    updated_at: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    steps: Optional[Sequence[Dict[str, Any]]] = None,
    include_latency: bool = False,
) -> RunSummary:
    if run_path is not None:
        meta = meta or load_meta(run_path)
        if steps is None and include_latency:
            steps = load_steps(run_path)
        project = run_path.parent.name
        run_name = run_path.name
        path_label = str(run_path)
        updated_at = _iso_from_timestamp(run_path.stat().st_mtime)
    else:
        meta = meta or {}
        if not project or not run_name or not path_label or not updated_at:
            raise ValueError("Remote run summaries require project, run_name, path_label, and updated_at")

    total_steps = int(meta.get("total_steps") or (len(steps) if steps is not None else 0))
    summary = RunSummary(
        run_id=f"{project}/{run_name}",
        project=project,
        run_name=run_name,
        path=path_label,
        start_time=meta.get("start_time"),
        end_time=meta.get("end_time"),
        updated_at=updated_at,
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
    runs_source: RunsSourceLike,
    project: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    include_latency: bool = False,
) -> List[RunSummary]:
    source = _coerce_runs_source(runs_source)
    items: List[RunSummary] = []

    if source.is_remote:
        for run_info in _remote_runs_listing(source, project=project):
            meta = run_info.get("meta") or {}
            project_name = str(run_info.get("project") or "")
            run_name = str(run_info.get("run_name") or "")
            path_label = str(run_info.get("path") or "")
            if not _matches_search(
                meta,
                project=project_name,
                run_name=run_name,
                path_label=path_label,
                search=search,
            ):
                continue
            steps = _remote_load_steps(source, project_name, run_name) if include_latency else None
            items.append(
                build_run_summary(
                    None,
                    project=project_name,
                    run_name=run_name,
                    path_label=path_label,
                    updated_at=_iso_from_timestamp(float(run_info.get("updated_at_ts") or 0.0)),
                    meta=meta,
                    steps=steps,
                    include_latency=include_latency,
                )
            )
            if len(items) >= limit:
                break
        return items

    if source.local_path is None:
        return []

    for run_path in _iter_run_paths(source.local_path, project=project):
        meta = load_meta(run_path)
        if not _matches_search(
            meta,
            project=run_path.parent.name,
            run_name=run_path.name,
            path_label=str(run_path),
            search=search,
        ):
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


def resolve_run_path(runs_source: RunsSourceLike, project: str, run_name: str) -> Path:
    source = _coerce_runs_source(runs_source)
    if source.is_remote or source.local_path is None:
        raise RuntimeError("Remote runs source does not expose a local filesystem path")
    run_path = (source.local_path / project / run_name).resolve()
    project_path = (source.local_path / project).resolve()
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
    runs_source: RunsSourceLike,
    project: str,
    run_name: str,
    recent_steps: int = 12,
) -> RunDetail:
    source = _coerce_runs_source(runs_source)
    if source.is_remote:
        meta = _remote_load_meta(source, project, run_name)
        steps = _remote_load_steps(source, project, run_name)
        summary = build_run_summary(
            None,
            project=project,
            run_name=run_name,
            path_label=f"{source.path}/{project}/{run_name}",
            updated_at=_iso_from_timestamp(
                max([0.0] + [float(item.get("updated_at_ts") or 0.0) for item in _remote_runs_listing(source, project=project) if item.get("run_name") == run_name])
            ),
            meta=meta,
            steps=steps,
            include_latency=True,
        )
    else:
        run_path = resolve_run_path(source, project, run_name)
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
    runs_source: RunsSourceLike,
    project: str,
    run_name: str,
    offset: int = 0,
    limit: int = 50,
) -> Tuple[int, List[StepPreview]]:
    source = _coerce_runs_source(runs_source)
    steps = _remote_load_steps(source, project, run_name) if source.is_remote else load_steps(resolve_run_path(source, project, run_name))
    sliced = steps[offset : offset + limit]
    return len(steps), [_step_preview(project, run_name, step) for step in sliced]


def resolve_artifact_path(
    runs_source: RunsSourceLike,
    project: str,
    run_name: str,
    artifact_path: str,
) -> Path:
    run_path = resolve_run_path(runs_source, project, run_name)
    candidate = (run_path / artifact_path).resolve()
    if run_path not in candidate.parents and candidate != run_path:
        raise FileNotFoundError(f"Invalid artifact path: {artifact_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return candidate


def read_artifact_bytes(
    runs_source: RunsSourceLike,
    project: str,
    run_name: str,
    artifact_path: str,
) -> bytes:
    source = _coerce_runs_source(runs_source)
    if source.is_remote:
        return _remote_read_artifact_bytes(source, project, run_name, artifact_path)
    path = resolve_artifact_path(source, project, run_name, artifact_path)
    return path.read_bytes()


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
    runs_source: RunsSourceLike,
    run_ids: Sequence[str],
    max_points: int = 300,
) -> LatencyCompareResponse:
    source = _coerce_runs_source(runs_source)
    items: List[LatencyCompareItem] = []
    remote_listing: Dict[str, Dict[str, Any]] = {}
    if source.is_remote:
        remote_listing = {
            f"{item['project']}/{item['run_name']}": item
            for item in _remote_runs_listing(source)
        }

    for run_id in run_ids:
        if "/" not in run_id:
            continue
        project, run_name = run_id.split("/", 1)

        if source.is_remote:
            run_info = remote_listing.get(run_id)
            if run_info is None:
                raise FileNotFoundError(f"Run not found: {run_id}")
            meta = run_info.get("meta") or {}
            steps = _remote_load_steps(source, project, run_name)
            summary = build_run_summary(
                None,
                project=project,
                run_name=run_name,
                path_label=str(run_info.get("path") or f"{source.path}/{project}/{run_name}"),
                updated_at=_iso_from_timestamp(float(run_info.get("updated_at_ts") or 0.0)),
                meta=meta,
                steps=steps,
                include_latency=True,
            )
        else:
            run_path = resolve_run_path(source, project, run_name)
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


def delete_run_tree(runs_source: RunsSourceLike, project: str, run_name: str) -> None:
    source = _coerce_runs_source(runs_source)
    if source.is_remote:
        _remote_delete_run(source, project, run_name)
        return
    run_path = resolve_run_path(source, project, run_name)
    import shutil

    shutil.rmtree(run_path)
