"""Cloud training pipeline for vast.ai."""


def __getattr__(name: str):
    """Lazy import to avoid pulling in torch/numpy at import time."""
    if name in ("launch", "monitor", "status", "destroy"):
        from . import provisioner
        return getattr(provisioner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["launch", "monitor", "status", "destroy"]
