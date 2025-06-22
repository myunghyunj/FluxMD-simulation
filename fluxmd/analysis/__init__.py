"""
Analysis modules for FluxMD flux calculations
"""

"""Analysis subpackage exports.

If PyTorch (or GPU libs) are unavailable we still want CPU functionality to
work so tests that don't need GPU can run.  Therefore the UMA-optimised
analyzer is imported conditionally; when missing we expose a stub that raises
on use but doesn't break module import time.
"""

from .flux_analyzer import TrajectoryFluxAnalyzer  # CPU version (always works)

# Attempt to import the GPU / UMA implementation; skip gracefully if torch missing

try:
    from .flux_analyzer_uma import TrajectoryFluxAnalyzer as UMAFluxAnalyzer  # type: ignore
except (ImportError, ModuleNotFoundError):  # torch or cuda libs absent
    import warnings

    class _UnavailableUMA:
        """Placeholder that warns on first use."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "UMA-optimised TrajectoryFluxAnalyzer requires PyTorch. "
                "Install FluxMD with the 'gpu' extra or ensure torch is importable."
            )

    UMAFluxAnalyzer = _UnavailableUMA  # type: ignore
    warnings.warn("PyTorch not available – UMA flux analyzer disabled.", ImportWarning)

__all__ = ["TrajectoryFluxAnalyzer", "UMAFluxAnalyzer"]
