"""Shared attention backend discovery and execution helpers."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional


_ATTENTION_BACKEND = None
_ATTENTION_BACKEND_KEY = None
_ATTENTION_PYTHON = None
_ATTENTION_PYTHON_KEY = None

_REQUIRED_BACKEND_CALLABLES = (
    "resolve_run_context",
    "build_default_output_dir",
    "load_cached_summary",
    "get_step_result",
    "generate_attention_for_steps",
)


def _module_cache_key(spec: str) -> str:
    return spec.strip()


def _sanitize_module_name(spec: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in spec)
    return f"vlalab_attention_backend_{cleaned}"


def _load_module_from_path(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(_sanitize_module_name(str(resolved)), resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load attention backend from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_backend_from_spec(spec: str) -> ModuleType:
    candidate = Path(spec).expanduser()
    if candidate.exists():
        return _load_module_from_path(candidate)

    try:
        return importlib.import_module(spec)
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Attention backend '{spec}' was not found. "
            "Set VLALAB_ATTENTION_BACKEND to a valid backend.py path or importable module."
        ) from exc


def _iter_default_backend_specs() -> Iterable[str]:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "Isaac-GR00T" / "realworld_deploy" / "offline_attention" / "backend.py"
        if candidate.exists():
            yield str(candidate)


def _discover_backend_spec() -> str:
    env_spec = os.getenv("VLALAB_ATTENTION_BACKEND", "").strip()
    if env_spec:
        return env_spec

    for candidate in _iter_default_backend_specs():
        return candidate

    raise FileNotFoundError(
        "Could not locate an attention backend. "
        "Set VLALAB_ATTENTION_BACKEND to your backend.py path or importable module."
    )


def _validate_backend(module: ModuleType) -> ModuleType:
    missing = []
    for name in _REQUIRED_BACKEND_CALLABLES:
        value = getattr(module, name, None)
        if not callable(value):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Invalid attention backend "
            f"'{get_attention_backend_label(module)}'. Missing callables: {', '.join(missing)}."
        )
    return module


def load_attention_backend() -> ModuleType:
    global _ATTENTION_BACKEND, _ATTENTION_BACKEND_KEY

    backend_spec = _discover_backend_spec()
    cache_key = _module_cache_key(backend_spec)
    if _ATTENTION_BACKEND is not None and _ATTENTION_BACKEND_KEY == cache_key:
        return _ATTENTION_BACKEND

    _ATTENTION_BACKEND = _validate_backend(_load_backend_from_spec(backend_spec))
    _ATTENTION_BACKEND_KEY = cache_key
    return _ATTENTION_BACKEND


def get_attention_backend_label(backend: Optional[ModuleType] = None) -> str:
    backend = backend or load_attention_backend()
    label = getattr(backend, "BACKEND_NAME", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    name = getattr(backend, "__name__", "attention_backend").split(".")[-1]
    return name or "attention_backend"


def _python_has_module(python_executable: str, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [python_executable, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _iter_python_candidates(backend: ModuleType) -> Iterable[Path]:
    env_python = os.getenv("VLALAB_ATTENTION_PYTHON", "").strip()
    if env_python:
        yield Path(env_python)

    yield Path(sys.executable)

    get_candidates = getattr(backend, "get_runtime_python_candidates", None)
    if callable(get_candidates):
        for candidate in get_candidates():
            if candidate:
                yield Path(candidate)
        return

    backend_root = getattr(backend, "ATTENTION_BACKEND_ROOT", None) or getattr(backend, "ISAAC_ROOT", None)
    if backend_root:
        yield Path(backend_root) / ".venv" / "bin" / "python"


def find_attention_python(
    backend: Optional[ModuleType] = None,
    required_module: str = "torch",
) -> Optional[str]:
    global _ATTENTION_PYTHON, _ATTENTION_PYTHON_KEY

    backend = backend or load_attention_backend()
    cache_key = (
        _module_cache_key(_discover_backend_spec()),
        os.getenv("VLALAB_ATTENTION_PYTHON", "").strip(),
        required_module,
    )
    if _ATTENTION_PYTHON is not None and _ATTENTION_PYTHON_KEY == cache_key:
        return _ATTENTION_PYTHON

    seen = set()
    for candidate in _iter_python_candidates(backend):
        candidate = candidate.expanduser()
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if not candidate.exists():
            continue
        if required_module and not _python_has_module(candidate_str, required_module):
            continue
        _ATTENTION_PYTHON = candidate_str
        _ATTENTION_PYTHON_KEY = cache_key
        return _ATTENTION_PYTHON

    _ATTENTION_PYTHON = None
    _ATTENTION_PYTHON_KEY = cache_key
    return None


def get_attention_script_path(backend: Optional[ModuleType] = None) -> Path:
    backend = backend or load_attention_backend()
    script = getattr(backend, "ATTENTION_SCRIPT_PATH", None)
    if script:
        candidate = Path(script).expanduser().resolve()
    else:
        candidate = Path(backend.__file__).resolve().parent / "run_offline_attention.py"

    if not candidate.exists():
        raise FileNotFoundError(
            f"Attention entry script not found for backend '{get_attention_backend_label(backend)}': {candidate}"
        )
    return candidate


def run_attention_generation_subprocess(
    *,
    run_dir: Path,
    requested_steps: list[int],
    device: str,
    attention_layer: int,
    output_dir: Path,
    model_path_override: Optional[str],
    prompt_override: Optional[str],
    backend: Optional[ModuleType] = None,
) -> tuple[dict, str]:
    backend = backend or load_attention_backend()
    python_executable = find_attention_python(backend)
    if python_executable is None:
        backend_name = get_attention_backend_label(backend)
        raise RuntimeError(
            f"Could not find a Python environment with torch for {backend_name} attention generation. "
            "Set VLALAB_ATTENTION_PYTHON if you need to point VLA-Lab to a dedicated runtime."
        )

    script_path = get_attention_script_path(backend)
    cmd = [
        python_executable,
        str(script_path),
        "--run-dir",
        str(run_dir),
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
        if error_text:
            error_text = "\n".join(error_text.splitlines()[-12:])
        raise RuntimeError(error_text or f"Attention subprocess failed with code {result.returncode}")

    summary = backend.load_cached_summary(output_dir)
    if summary is None:
        raise RuntimeError("Attention generation finished but summary.json was not found.")

    stdout_tail = (result.stdout or "").strip()
    if stdout_tail:
        stdout_tail = "\n".join(stdout_tail.splitlines()[-8:])
    return summary, stdout_tail
