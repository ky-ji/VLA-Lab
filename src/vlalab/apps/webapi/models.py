"""Pydantic response models for the VLA-Lab web API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LatencyMetric(BaseModel):
    avg_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    max_ms: Optional[float] = None


class CameraImage(BaseModel):
    camera_name: str
    path: str
    url: str
    shape: Optional[List[int]] = None
    encoding: Optional[str] = None


class StepPreview(BaseModel):
    step_idx: int
    prompt: Optional[str] = None
    state: List[float] = Field(default_factory=list)
    action_preview: List[float] = Field(default_factory=list)
    action_chunk_size: int = 0
    timing: Dict[str, Optional[float]] = Field(default_factory=dict)
    images: List[CameraImage] = Field(default_factory=list)
    tags: Dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    project: str
    run_name: str
    path: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    updated_at: str
    model_name: str = "unknown"
    model_type: Optional[str] = None
    task_name: str = "unknown"
    robot_name: str = "unknown"
    total_steps: int = 0
    action_dim: Optional[int] = None
    action_horizon: Optional[int] = None
    inference_freq: Optional[float] = None
    latency: Optional[Dict[str, LatencyMetric]] = None


class RunDetail(BaseModel):
    summary: RunSummary
    model_path: Optional[str] = None
    task_prompt: Optional[str] = None
    state_dim: Optional[int] = None
    cameras: List[str] = Field(default_factory=list)
    timing_summary: Dict[str, LatencyMetric] = Field(default_factory=dict)
    recent_steps: List[StepPreview] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ProjectListResponse(BaseModel):
    runs_dir: str
    projects: List[str]


class RunListResponse(BaseModel):
    runs_dir: str
    total: int
    items: List[RunSummary]


class OverviewResponse(BaseModel):
    runs_dir: str
    project_count: int
    run_count: int
    latest_runs: List[RunSummary]


class RunStepsResponse(BaseModel):
    run_id: str
    total: int
    offset: int
    limit: int
    items: List[StepPreview]


class LatencySeries(BaseModel):
    steps: List[int]
    transport_latency_ms: List[Optional[float]]
    inference_latency_ms: List[Optional[float]]
    total_latency_ms: List[Optional[float]]


class LatencyCompareItem(BaseModel):
    run: RunSummary
    summary: Dict[str, LatencyMetric]
    series: LatencySeries


class LatencyCompareResponse(BaseModel):
    items: List[LatencyCompareItem]


class DeployTargetConnection(BaseModel):
    id: str
    label: str
    connected: bool = False
    last_error: Optional[str] = None


class DeployInputSpec(BaseModel):
    id: str
    label: str
    type: str
    required: bool = True
    default: Optional[str] = None
    options: List[str] = Field(default_factory=list)


class DeployCommandSpec(BaseModel):
    id: str
    label: str
    target_id: str
    background: bool = False
    required_inputs: List[str] = Field(default_factory=list)
    resolved_preview: str = ""


class DeployJob(BaseModel):
    id: str
    command_id: str
    target_id: str
    state: str
    remote_pid: Optional[int] = None
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None
    last_stdout: Optional[str] = None
    last_stderr: Optional[str] = None
    submitted_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class DeployOverviewResponse(BaseModel):
    refreshed_at: str
    targets: List[DeployTargetConnection] = Field(default_factory=list)
    inputs: List[DeployInputSpec] = Field(default_factory=list)
    commands: List[DeployCommandSpec] = Field(default_factory=list)
    jobs: List[DeployJob] = Field(default_factory=list)


class DeployRunRequest(BaseModel):
    command_id: str
    values: Dict[str, Any] = Field(default_factory=dict)


class DeployRunResponse(BaseModel):
    ok: bool
    message: str
    job: DeployJob


class DeployJobsResponse(BaseModel):
    refreshed_at: str
    jobs: List[DeployJob] = Field(default_factory=list)


class DeployJobLogsResponse(BaseModel):
    job_id: str
    stream: str
    path: Optional[str] = None
    content: str = ""
    updated_at: str
