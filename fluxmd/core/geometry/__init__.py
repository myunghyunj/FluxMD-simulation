"""
Geometric analysis tools for trajectory generation.

This module provides:
- PCA-based anchor point detection
- Protein extremity identification
- Nucleic acid backbone detection
"""

from .pca_anchors import extreme_calpha_pairs

__all__ = [
    "PCAAnchors",
    "extreme_calpha_pairs",
]
