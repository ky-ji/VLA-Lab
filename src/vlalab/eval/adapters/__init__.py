"""
VLA-Lab Policy Adapters

Adapters that wrap specific VLA model implementations to conform to
the unified EvalPolicy interface.

All adapters are lazy-imported to avoid pulling in heavy dependencies
(torch, omegaconf, etc.) at module load time.
"""

_ADAPTER_MAP = {
    "GR00TAdapter": "vlalab.eval.adapters.groot_adapter",
    "DiffusionPolicyAdapter": "vlalab.eval.adapters.dp_adapter",
    "VitaAdapter": "vlalab.eval.adapters.vita_adapter",
    "load_vita_policy": "vlalab.eval.adapters.vita_adapter",
}

__all__ = list(_ADAPTER_MAP.keys())


def __getattr__(name: str):
    if name in _ADAPTER_MAP:
        import importlib
        mod = importlib.import_module(_ADAPTER_MAP[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'vlalab.eval.adapters' has no attribute {name}")
