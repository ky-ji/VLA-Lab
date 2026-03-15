"""
Shared Streamlit caching helpers for VLA-Lab pages.

The app uses Streamlit's rerun model, so page switches can easily trigger
the same directory scans and file parsing work again. This module centralizes
those expensive operations behind file-aware caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

import vlalab


def _stat_signature(path: Path) -> Tuple[int, int]:
    """Return a cheap cache key derived from file metadata."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(ttl=2, show_spinner=False)
def list_projects_cached(runs_dir: str) -> List[str]:
    """Cache project discovery for a short window."""
    return vlalab.list_projects(dir=runs_dir)


@st.cache_data(ttl=2, show_spinner=False)
def list_runs_cached(runs_dir: str, project: Optional[str]) -> List[str]:
    """Cache run discovery for a short window."""
    return [str(path) for path in vlalab.list_runs(project=project, dir=runs_dir)]


def load_run_bundle(run_dir: Path | str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a run's meta.json and steps.jsonl with file-aware caching."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    steps_path = run_dir / "steps.jsonl"
    meta_sig = _stat_signature(meta_path)
    steps_sig = _stat_signature(steps_path)
    return _load_run_bundle_cached(str(run_dir), *meta_sig, *steps_sig)


@st.cache_data(show_spinner=False)
def _load_run_bundle_cached(
    run_dir: str,
    meta_mtime_ns: int,
    meta_size: int,
    steps_mtime_ns: int,
    steps_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []

    meta_path = Path(run_dir) / "meta.json"
    if meta_mtime_ns and meta_size and meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    steps_path = Path(run_dir) / "steps.jsonl"
    if steps_mtime_ns and steps_size and steps_path.exists():
        with open(steps_path, "r") as f:
            steps = [json.loads(line) for line in f if line.strip()]

    return meta, steps


def load_json_file(path: Path | str) -> Dict[str, Any]:
    """Load a JSON file with cache invalidation based on file metadata."""
    path = Path(path)
    return _load_json_file_cached(str(path), *_stat_signature(path))


@st.cache_data(show_spinner=False)
def _load_json_file_cached(
    path: str,
    mtime_ns: int,
    size: int,
) -> Dict[str, Any]:
    if not mtime_ns or not size:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_trajectory_arrays(results_dir: Path | str, traj_id: int) -> Dict[str, np.ndarray]:
    """Load cached trajectory arrays for an eval result directory."""
    results_dir = Path(results_dir)
    gt_path = results_dir / f"traj_{traj_id}_gt.npy"
    pred_path = results_dir / f"traj_{traj_id}_pred.npy"
    states_path = results_dir / f"traj_{traj_id}_states.npy"
    return _load_trajectory_arrays_cached(
        str(gt_path),
        *_stat_signature(gt_path),
        str(pred_path),
        *_stat_signature(pred_path),
        str(states_path),
        *_stat_signature(states_path),
    )


@st.cache_data(show_spinner=False)
def _load_trajectory_arrays_cached(
    gt_path: str,
    gt_mtime_ns: int,
    gt_size: int,
    pred_path: str,
    pred_mtime_ns: int,
    pred_size: int,
    states_path: str,
    states_mtime_ns: int,
    states_size: int,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}

    if gt_mtime_ns and gt_size and Path(gt_path).exists():
        arrays["gt_actions"] = np.load(gt_path)
    if pred_mtime_ns and pred_size and Path(pred_path).exists():
        arrays["pred_actions"] = np.load(pred_path)
    if states_mtime_ns and states_size and Path(states_path).exists():
        arrays["states"] = np.load(states_path)

    return arrays


def clear_streamlit_data_caches() -> None:
    """Clear page-level caches when the user explicitly asks for a refresh."""
    list_projects_cached.clear()
    list_runs_cached.clear()
    _load_run_bundle_cached.clear()
    _load_json_file_cached.clear()
    _load_trajectory_arrays_cached.clear()
