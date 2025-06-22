"""
Surface generation and manipulation for Matryoshka trajectories.

This module provides tools for:
- Solvent-excluded surface (SES) generation
- Multi-layer surface construction
- DNA groove detection and labeling
"""

from .dna_groove_detector import DNAGrooveDetector
from .layer_stream import MatryoshkaLayerGenerator
from .ses_builder import SESBuilder, SurfaceMesh

__all__ = [
    "SESBuilder",
    "SurfaceMesh",
    "MatryoshkaLayerGenerator",
    "DNAGrooveDetector",
]
