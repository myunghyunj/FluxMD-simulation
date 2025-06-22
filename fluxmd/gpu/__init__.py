"""
GPU acceleration modules for FluxMD
"""

try:
    from .gpu_accelerated_flux import GPUAcceleratedInteractionCalculator, get_device
    from .gpu_accelerated_flux_uma import (
        GPUAcceleratedInteractionCalculator as UMACalculator,
        InteractionResult
    )

    __all__ = [
        'GPUAcceleratedInteractionCalculator',
        'UMACalculator',
        'InteractionResult',
        'get_device'
    ]
except (ImportError, ModuleNotFoundError):
    # This module depends on torch, which may not be installed in CPU-only test environments.
    # Fail gracefully so that test discovery does not crash.
    __all__ = []