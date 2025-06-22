"""
FluxMD: GPU-accelerated molecular dynamics simulation tool for mapping binding interfaces.

FluxMD uses dynamic energy flux analysis to identify binding sites between biomolecules,
supporting protein-protein, protein-ligand, and protein-DNA interactions.
"""

from .__version__ import __version__
from .analysis.flux_analyzer import TrajectoryFluxAnalyzer
from .core.matryoshka_generator import MatryoshkaTrajectoryGenerator

# Core functionality
from .core.trajectory_generator import ProteinLigandFluxAnalyzer
from .utils.dna_to_pdb import dna_to_pdb_structure

# Utilities
from .utils.pdb_parser import PDBParser


def get_version() -> str:
    """Return package version."""
    return __version__


__all__ = [
    "__version__",
    "get_version",
    "ProteinLigandFluxAnalyzer",
    "MatryoshkaTrajectoryGenerator",
    "TrajectoryFluxAnalyzer",
    "PDBParser",
    "dna_to_pdb_structure",
]
