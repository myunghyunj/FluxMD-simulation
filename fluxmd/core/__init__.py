"""
Core modules for FluxMD trajectory generation and interaction calculations.
"""

try:
    from .intra_protein_interactions import IntraProteinInteractions
    from .matryoshka_generator import MatryoshkaTrajectoryGenerator
    from .protonation_aware_interactions import (
        ProtonationAwareInteractionDetector,
        calculate_interactions_with_protonation,
    )
    from .ref15_energy import REF15EnergyCalculator, get_ref15_calculator
    from .trajectory_generator import ProteinLigandFluxAnalyzer
except Exception:  # pragma: no cover
    ProteinLigandFluxAnalyzer = None
    IntraProteinInteractions = None
    ProtonationAwareInteractionDetector = None
    calculate_interactions_with_protonation = None
    MatryoshkaTrajectoryGenerator = None
    REF15EnergyCalculator = None
    get_ref15_calculator = None

# Submodules
from . import dynamics, geometry, surface


__all__ = [
    "ProteinLigandFluxAnalyzer",
    "IntraProteinInteractions",
    "ProtonationAwareInteractionDetector",
    "calculate_interactions_with_protonation",
    "MatryoshkaTrajectoryGenerator",
    "REF15EnergyCalculator",
    "get_ref15_calculator",
    "surface",
    "geometry",
    "dynamics",
]
