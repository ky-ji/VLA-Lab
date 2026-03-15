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

