from pathlib import Path


def _write_minimal_backend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
from pathlib import Path

BACKEND_NAME = "fake-attention"

def resolve_run_context(run_dir, *, model_path_override=None, prompt_override=None):
    return {}, [], model_path_override or "model", prompt_override or "prompt"

def build_default_output_dir(run_dir, model_path, attention_layer):
    return Path(run_dir) / "artifacts" / "attention" / "fake"

def load_cached_summary(output_dir):
    return {"results": []}

def get_step_result(summary, step_idx):
    return None

def generate_attention_for_steps(*args, **kwargs):
    return {"results": []}
""",
        encoding="utf-8",
    )


def _reset_attention_module(module) -> None:
    module._ATTENTION_BACKEND = None
    module._ATTENTION_BACKEND_KEY = None
    module._ATTENTION_PYTHON = None
    module._ATTENTION_PYTHON_KEY = None


def test_discovers_attention_backend_from_vlalab_dir(monkeypatch, tmp_path):
    from vlalab.attention import backend

    isaac_root = tmp_path / "Isaac-GR00T"
    backend_path = isaac_root / "realworld_deploy" / "offline_attention" / "backend.py"
    runs_dir = isaac_root / "vlalab_runs"
    runs_dir.mkdir(parents=True)
    _write_minimal_backend(backend_path)

    monkeypatch.delenv("VLALAB_ATTENTION_BACKEND", raising=False)
    monkeypatch.setenv("VLALAB_DIR", str(runs_dir))
    _reset_attention_module(backend)

    loaded = backend.load_attention_backend()

    assert Path(loaded.__file__).resolve() == backend_path.resolve()
    assert backend.get_attention_backend_label(loaded) == "fake-attention"
