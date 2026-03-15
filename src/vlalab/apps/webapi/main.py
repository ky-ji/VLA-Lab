"""FastAPI app for the VLA-Lab web UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import vlalab

from .models import (
    LatencyCompareResponse,
    OverviewResponse,
    ProjectListResponse,
    RunDetail,
    RunListResponse,
    RunStepsResponse,
)
from .dataset_service import inspect_dataset, load_dataset_episode_view
from .eval_service import load_eval_inline, load_eval_view
from .replay_service import delete_run, generate_attention, load_attention_state, load_run_replay
from .service import (
    build_latency_compare,
    get_runs_dir,
    list_projects,
    list_run_summaries,
    load_run_detail,
    load_run_steps,
    resolve_artifact_path,
)

app = FastAPI(
    title="VLA-Lab API",
    version=vlalab.__version__,
    description="FastAPI backend for the VLA-Lab Next.js frontend.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "VLALAB_WEB_CORS_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": vlalab.__version__}


@app.get("/api/overview", response_model=OverviewResponse)
def overview() -> OverviewResponse:
    runs_dir = get_runs_dir()
    projects = list_projects(runs_dir)
    latest_runs = list_run_summaries(runs_dir, limit=6)
    return OverviewResponse(
        runs_dir=str(runs_dir),
        project_count=len(projects),
        run_count=len(list_run_summaries(runs_dir, limit=10_000)),
        latest_runs=latest_runs,
    )


@app.get("/api/projects", response_model=ProjectListResponse)
def projects() -> ProjectListResponse:
    runs_dir = get_runs_dir()
    return ProjectListResponse(
        runs_dir=str(runs_dir),
        projects=list_projects(runs_dir),
    )


@app.get("/api/runs", response_model=RunListResponse)
def runs(
    project: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    include_latency: bool = False,
) -> RunListResponse:
    runs_dir = get_runs_dir()
    items = list_run_summaries(
        runs_dir,
        project=project,
        search=search,
        limit=limit,
        include_latency=include_latency,
    )
    return RunListResponse(
        runs_dir=str(runs_dir),
        total=len(items),
        items=items,
    )


@app.get("/api/runs/{project}/{run_name}", response_model=RunDetail)
def run_detail(project: str, run_name: str) -> RunDetail:
    runs_dir = get_runs_dir()
    try:
        return load_run_detail(runs_dir, project, run_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{project}/{run_name}/steps", response_model=RunStepsResponse)
def run_steps(
    project: str,
    run_name: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> RunStepsResponse:
    runs_dir = get_runs_dir()
    try:
        total, items = load_run_steps(runs_dir, project, run_name, offset=offset, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RunStepsResponse(
        run_id=f"{project}/{run_name}",
        total=total,
        offset=offset,
        limit=limit,
        items=items,
    )


@app.get("/api/runs/{project}/{run_name}/replay")
def run_replay(project: str, run_name: str) -> dict:
    runs_dir = get_runs_dir()
    try:
        return load_run_replay(runs_dir, project, run_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/runs/{project}/{run_name}")
def delete_run_route(project: str, run_name: str) -> dict:
    runs_dir = get_runs_dir()
    try:
        return delete_run(runs_dir, project, run_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{project}/{run_name}/artifacts/{artifact_path:path}")
def artifact(project: str, run_name: str, artifact_path: str) -> FileResponse:
    runs_dir = get_runs_dir()
    try:
        path = resolve_artifact_path(runs_dir, project, run_name, artifact_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


@app.get("/api/runs/{project}/{run_name}/attention")
def run_attention(
    project: str,
    run_name: str,
    step_idx: int = Query(default=0, ge=0),
    attention_layer: int = -1,
    model_path_override: Optional[str] = None,
    prompt_override: Optional[str] = None,
) -> dict:
    runs_dir = get_runs_dir()
    try:
        return load_attention_state(
            runs_dir,
            project,
            run_name,
            step_idx=step_idx,
            attention_layer=attention_layer,
            model_path_override=model_path_override,
            prompt_override=prompt_override,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{project}/{run_name}/attention/generate")
def run_attention_generate(
    project: str,
    run_name: str,
    payload: dict = Body(default_factory=dict),
) -> dict:
    runs_dir = get_runs_dir()
    requested_steps = payload.get("requested_steps") or []
    if not requested_steps:
        raise HTTPException(status_code=400, detail="requested_steps is required")
    try:
        return generate_attention(
            runs_dir,
            project,
            run_name,
            requested_steps=[int(idx) for idx in requested_steps],
            focus_step=payload.get("focus_step"),
            device=str(payload.get("device", "cuda:0")),
            attention_layer=int(payload.get("attention_layer", -1)),
            model_path_override=payload.get("model_path_override"),
            prompt_override=payload.get("prompt_override"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/latency/compare", response_model=LatencyCompareResponse)
def latency_compare(
    runs: List[str] = Query(default_factory=list),
    max_points: int = Query(default=300, ge=10, le=5000),
) -> LatencyCompareResponse:
    runs_dir = get_runs_dir()
    try:
        return build_latency_compare(runs_dir, runs, max_points=max_points)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/inspect")
def dataset_inspect(path: str) -> dict:
    try:
        return inspect_dataset(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/episode")
def dataset_episode(
    path: str,
    episode_idx: int = Query(default=0, ge=0),
    step_idx: int = Query(default=0, ge=0),
    step_interval: int = Query(default=5, ge=1, le=50),
    max_frames: int = Query(default=20, ge=1, le=100),
    workspace_ratio: float = Query(default=0.1, ge=0.01, le=1.0),
) -> dict:
    try:
        return load_dataset_episode_view(
            path=path,
            episode_idx=episode_idx,
            step_idx=step_idx,
            step_interval=step_interval,
            max_frames=max_frames,
            workspace_ratio=workspace_ratio,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/eval/view")
def eval_view(
    source: str = Query(default="dir"),
    dir_path: Optional[str] = None,
    results_file: Optional[str] = None,
    traj_id: Optional[int] = None,
) -> dict:
    try:
        return load_eval_view(
            source=source,
            dir_path=dir_path,
            results_file=results_file,
            traj_id=traj_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/eval/inline")
def eval_inline(payload: dict = Body(default_factory=dict)) -> dict:
    try:
        return load_eval_inline(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/eval/static-image")
def eval_static_image(path: str) -> FileResponse:
    image_path = Path(path).expanduser().resolve()
    if image_path.suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Only PNG images are supported")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Eval image not found: {image_path}")
    return FileResponse(image_path)
