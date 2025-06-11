"""
FluxMD: GPU-accelerated binding site prediction using flux differential analysis
"""

from .__version__ import __version__

# Import key classes for easier access
from .core.trajectory_generator import ProteinLigandFluxAnalyzer
from .analysis.flux_analyzer import TrajectoryFluxAnalyzer
from .gpu.gpu_accelerated_flux import get_device

__all__ = [
    '__version__',
    'ProteinLigandFluxAnalyzer',
    'TrajectoryFluxAnalyzer',
    'get_device'
]