"""
FluxMD: Energy Flux Differential Analysis for Protein-Ligand Binding Sites

A GPU-accelerated tool for identifying binding sites through molecular dynamics
and energy flux analysis.
"""

from .__version__ import __version__, __version_info__

# Core modules
from .core.trajectory_generator import ProteinLigandFluxAnalyzer
from .core.intra_protein_interactions import IntraProteinInteractions
from .core.protonation_aware_interactions import ProtonationAwareInteractionDetector

# Analysis modules
from .analysis.flux_analyzer import TrajectoryFluxAnalyzer

# GPU modules
from .gpu.gpu_accelerated_flux import (
    GPUAcceleratedInteractionCalculator,
    GPUFluxCalculator,
    get_device
)

# Utilities
from .utils.dna_to_pdb import DNAStructureGenerator

__all__ = [
    "__version__",
    "__version_info__",
    "ProteinLigandFluxAnalyzer",
    "IntraProteinInteractions",
    "ProtonationAwareInteractionDetector",
    "TrajectoryFluxAnalyzer",
    "GPUAcceleratedInteractionCalculator",
    "GPUFluxCalculator",
    "get_device",
    "DNAStructureGenerator",
]