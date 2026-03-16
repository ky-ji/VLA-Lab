# Attention Backend Guide

`VLA-Lab` owns the attention visualization UI, cache browsing, and replay linkage.  
The model repo owns the actual attention extraction logic.

This split is intentional: when you migrate to another model, you should only need to replace the backend module, not rewrite the `VLA-Lab` UI.

## Responsibility Boundary

`VLA-Lab` is responsible for:

- locating the attention backend
- launching generation from Web/Streamlit
- reading cached summaries and displaying overlays
- binding attention results to replay step selection

The model backend is responsible for:

- loading model-specific checkpoints and runtime deps
- deciding how observations are mapped to model inputs
- extracting attention maps
- writing cached outputs in the agreed format

## Backend Registration

Default behavior:

- `VLA-Lab` will auto-discover the existing `Isaac-GR00T/realworld_deploy/offline_attention/backend.py` when that repo is present as a sibling workspace.

Recommended behavior for new models:

```bash
export VLALAB_ATTENTION_BACKEND=/abs/path/to/your_model/offline_attention/backend.py
```

Optional runtime override:

```bash
export VLALAB_ATTENTION_PYTHON=/abs/path/to/python
```

Use `VLALAB_ATTENTION_PYTHON` when attention generation must run in a dedicated environment with `torch` or model-specific dependencies installed.

## Required Backend Contract

Your backend module must provide these callables:

```python
def resolve_run_context(
    run_dir: Path,
    *,
    model_path_override: str | None = None,
    prompt_override: str | None = None,
) -> tuple[dict, list[dict], str, str]: ...

def build_default_output_dir(
    run_dir: Path,
    model_path: str,
    attention_layer: int,
) -> Path: ...

def load_cached_summary(output_dir: Path) -> dict | None: ...

def get_step_result(summary: dict | None, step_idx: int) -> dict | None: ...

def generate_attention_for_steps(
    run_dir: Path,
    step_indices: Iterable[int],
    *,
    device: str = "cuda:0",
    attention_layer: int = -1,
    output_dir: Path | None = None,
    model_path_override: str | None = None,
    prompt_override: str | None = None,
    **kwargs,
) -> tuple[dict, Path]: ...
```

Semantics:

- `resolve_run_context`: read `meta.json` and `steps.jsonl`, resolve the actual model path and prompt used for generation.
- `build_default_output_dir`: return a stable cache directory under the run, usually `artifacts/attention/<cache_id>`.
- `load_cached_summary`: load `summary.json`.
- `get_step_result`: return one step entry from the cached summary.
- `generate_attention_for_steps`: generate or refresh cache files for the requested steps and return the merged summary.

## Optional Backend Hooks

These are optional but recommended:

```python
BACKEND_NAME = "MyModel"
ATTENTION_SCRIPT_PATH = "/abs/path/to/run_offline_attention.py"
ATTENTION_BACKEND_ROOT = "/abs/path/to/model_repo"

def create_engine(...): ...

def get_runtime_python_candidates() -> Iterable[str | Path]: ...
```

Notes:

- `create_engine` enables faster in-process generation in Streamlit when the current Python already has the required dependencies.
- `get_runtime_python_candidates` lets the backend suggest model-specific virtualenv interpreters.
- `BACKEND_NAME` is shown in the UI/API for debugging and migration clarity.

## Cache Format

The cache directory should contain:

```text
artifacts/
  attention/
    <cache_id>/
      summary.json
      step_000123_front_overlay.jpg
      step_000123_front_heatmap.npy
      ...
```

Top-level `summary.json` fields:

```json
{
  "run_dir": "...",
  "model_path": "...",
  "prompt": "...",
  "device": "cuda:0",
  "attention_layer": -1,
  "output_dir": "...",
  "results": []
}
```

Each item in `results` should look like:

```json
{
  "step_idx": 123,
  "overlay_files": [
    {
      "camera_name": "front",
      "source_image": "artifacts/images/step_000123_front.jpg",
      "overlay_path": "/abs/path/to/run/artifacts/attention/.../step_000123_front_overlay.jpg",
      "heatmap_npy_path": "/abs/path/to/run/artifacts/attention/.../step_000123_front_heatmap.npy"
    }
  ]
}
```

Required compatibility rules:

- `overlay_path` should live inside the run directory so `VLA-Lab` can expose it as a run artifact.
- `step_idx` must match the step index in `steps.jsonl`.
- `camera_name` should match the replay camera naming as closely as possible.
- repeated generation should merge into the existing `summary.json` instead of overwriting unrelated cached steps.

Extra backend-specific fields are allowed. `VLA-Lab` will ignore fields it does not understand.

## Migration Checklist

When adapting a new model, keep this order:

1. Reuse the run's `meta.json` and `steps.jsonl` as the source of truth.
2. Implement the five required backend callables.
3. Make sure overlays are saved under `run_dir/artifacts/attention/...`.
4. Verify one step can be generated from `VLA-Lab` with `VLALAB_ATTENTION_BACKEND` set.
5. Only add model-specific knobs after the base contract is stable.

## Recommended Rule of Thumb

Keep model logic in the model repo, and keep visualization protocol in `VLA-Lab`.

That gives us a stable contract:

- model changes do not require UI rewrites
- `VLA-Lab` upgrades do not require backend rewrites unless the contract changes
- cross-model attention support stays additive instead of one-off hardcoding
