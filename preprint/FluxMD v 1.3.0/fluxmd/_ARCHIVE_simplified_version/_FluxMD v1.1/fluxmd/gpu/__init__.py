"""
GPU acceleration modules for FluxMD
"""

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