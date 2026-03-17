"""FastAPI app for the VLA-Lab web UI."""

from __future__ import annotations

import os
import mimetypes
from typing import List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

import vlalab

from .models import (
    DeployJobLogsResponse,
    DeployInputValuesRequest,
    DeployInputValuesResponse,
    DeployJobsResponse,
    DeployOverviewResponse,
    DeployRunRequest,
    DeployRunResponse,
    DeployStopResponse,
    LatencyCompareResponse,
    OverviewResponse,
    ProjectListResponse,
    RunDetail,
    RunListResponse,
    RunStepsResponse,
)
from .dataset_service import inspect_dataset, load_dataset_episode_view
from .deploy_service import (
    build_deploy_overview,
    get_deploy_job_logs,
    list_deploy_jobs,
    run_deploy_command,
    save_deploy_input_values,
    stop_deploy_job,
)
from .eval_service import load_eval_inline, load_eval_view
from .replay_service import delete_run, generate_attention, load_attention_state, load_run_replay
from .service import (
    build_latency_compare,
    count_runs,
    get_runs_source,
    list_projects,
    list_run_summaries,
    load_run_detail,
    load_run_steps,
    read_artifact_bytes,
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


@app.get("/")
def root() -> dict:
    return {
        "service": "VLA-Lab API",
        "status": "ok",
        "version": vlalab.__version__,
        "docs": "/docs",
        "health": "/api/health",
        "deploy_overview": "/api/deploy/overview",
    }


@app.get("/api/overview", response_model=OverviewResponse)
def overview() -> OverviewResponse:
    runs_source = get_runs_source()
    projects = list_projects(runs_source)
    run_count = count_runs(runs_source)
    latest_runs = list_run_summaries(runs_source, limit=6)
    return OverviewResponse(
        runs_dir=runs_source.display_path,
        project_count=len(projects),
        run_count=run_count,
        latest_runs=latest_runs,
    )


@app.get("/api/projects", response_model=ProjectListResponse)
def projects() -> ProjectListResponse:
    runs_source = get_runs_source()
    return ProjectListResponse(
        runs_dir=runs_source.display_path,
        projects=list_projects(runs_source),
    )


@app.get("/api/runs", response_model=RunListResponse)
def runs(
    project: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    include_latency: bool = False,
) -> RunListResponse:
    runs_source = get_runs_source()
    items = list_run_summaries(
        runs_source,
        project=project,
        search=search,
        limit=limit,
        include_latency=include_latency,
    )
    return RunListResponse(
        runs_dir=runs_source.display_path,
        total=len(items),
        items=items,
    )


@app.get("/api/runs/{project}/{run_name}", response_model=RunDetail)
def run_detail(project: str, run_name: str) -> RunDetail:
    runs_source = get_runs_source()
    try:
        return load_run_detail(runs_source, project, run_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{project}/{run_name}/steps", response_model=RunStepsResponse)
def run_steps(
    project: str,
    run_name: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> RunStepsResponse:
    runs_source = get_runs_source()
    try:
        total, items = load_run_steps(runs_source, project, run_name, offset=offset, limit=limit)
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
    runs_source = get_runs_source()
    try:
        return load_run_replay(runs_source, project, run_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/runs/{project}/{run_name}")
def delete_run_route(project: str, run_name: str) -> dict:
    runs_source = get_runs_source()
    try:
        return delete_run(runs_source, project, run_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{project}/{run_name}/artifacts/{artifact_path:path}")
def artifact(project: str, run_name: str, artifact_path: str):
    runs_source = get_runs_source()
    try:
        if runs_source.is_remote:
            content = read_artifact_bytes(runs_source, project, run_name, artifact_path)
            media_type = mimetypes.guess_type(artifact_path)[0] or "application/octet-stream"
            return Response(content=content, media_type=media_type)
        path = resolve_artifact_path(runs_source, project, run_name, artifact_path)
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
    runs_source = get_runs_source()
    try:
        return load_attention_state(
            runs_source,
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
    runs_source = get_runs_source()
    requested_steps = payload.get("requested_steps") or []
    if not requested_steps:
        raise HTTPException(status_code=400, detail="requested_steps is required")
    try:
        return generate_attention(
            runs_source,
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
    runs_source = get_runs_source()
    try:
        return build_latency_compare(runs_source, runs, max_points=max_points)
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
def eval_static_image(path: str, dir_path: Optional[str] = None) -> FileResponse:
    image_path = Path(path).expanduser().resolve()
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Only PNG/JPEG images are supported")
    if dir_path:
        allowed_root = Path(dir_path).expanduser().resolve()
        if allowed_root not in image_path.parents and image_path != allowed_root:
            raise HTTPException(status_code=403, detail="Image path outside allowed directory")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Eval image not found: {image_path}")
    return FileResponse(image_path)


@app.get("/api/deploy/overview", response_model=DeployOverviewResponse)
def deploy_overview(
    model_server_config_path: Optional[str] = None,
    inference_client_config_path: Optional[str] = None,
) -> DeployOverviewResponse:
    try:
        return build_deploy_overview(
            {
                "model_server_config_path": model_server_config_path,
                "inference_client_config_path": inference_client_config_path,
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/deploy/run", response_model=DeployRunResponse)
def deploy_run(payload: DeployRunRequest = Body(...)) -> DeployRunResponse:
    try:
        return run_deploy_command(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/deploy/inputs", response_model=DeployInputValuesResponse)
def deploy_save_inputs(payload: DeployInputValuesRequest = Body(...)) -> DeployInputValuesResponse:
    try:
        values = save_deploy_input_values(payload.values)
        return DeployInputValuesResponse(ok=True, message="输入参数已保存", values=values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/deploy/jobs", response_model=DeployJobsResponse)
def deploy_jobs() -> DeployJobsResponse:
    try:
        return list_deploy_jobs()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/deploy/jobs/{job_id}/logs", response_model=DeployJobLogsResponse)
def deploy_job_logs(
    job_id: str,
    stream: str = Query(default="stdout"),
    lines: int = Query(default=200, ge=1, le=500),
) -> DeployJobLogsResponse:
    try:
        return get_deploy_job_logs(job_id, stream=stream, lines=lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/deploy/jobs/{job_id}/stop", response_model=DeployStopResponse)
def deploy_stop_job(job_id: str) -> DeployStopResponse:
    try:
        job = stop_deploy_job(job_id)
        return DeployStopResponse(ok=True, message="停止请求已发送", job=job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
