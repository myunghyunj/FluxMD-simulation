# fluxmd/core/geometry/pca_anchors.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class PCAAnchors:
    """Wrapper class for extreme CA/P atom detection via PCA."""

    def extreme_calpha_pairs(
        self, coords: np.ndarray, atom_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate to :func:`extreme_calpha_pairs`."""
        return extreme_calpha_pairs(coords, atom_names)


__all__ = ["PCAAnchors", "extreme_calpha_pairs"]


def extreme_calpha_pairs(
    coords: np.ndarray, atom_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Find trajectory anchors at PCA extremes of backbone atoms.

    Args:
        coords: (N,3) array of all atom coordinates
        atom_names: List of atom names (CA for protein, P for DNA)

    Returns:
        Tuple of (start_pos, end_pos) 3D coordinates
    """
    # Extract backbone coordinates (CA for protein, P for DNA)
    backbone_mask = np.array([name == "CA" or name == "P" for name in atom_names])

    if not np.any(backbone_mask):
        raise ValueError("No backbone atoms (CA or P) found in structure")

    backbone_coords = coords[backbone_mask]

    # Center coordinates
    center = backbone_coords.mean(axis=0)
    centered_coords = backbone_coords - center

    # Compute covariance matrix
    cov = np.cov(centered_coords.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract principal components
    λ1, λ2 = eigenvalues[:2]
    v1 = eigenvectors[:, 0]

    # Check for degeneracy
    if λ1 > 0 and (λ1 - λ2) / λ1 < 0.05:
        # Degenerate case: use both PC1 and PC2
        print(f"PCA degeneracy detected: (λ1-λ2)/λ1 = {(λ1 - λ2) / λ1:.3f}")

        # Project onto PC1
        proj1 = centered_coords @ v1
        idx1_min, idx1_max = proj1.argmin(), proj1.argmax()

        # Also project onto PC2
        v2 = eigenvectors[:, 1]
        proj2 = centered_coords @ v2
        idx2_min, idx2_max = proj2.argmin(), proj2.argmax()

        # Choose the pair with maximum separation
        sep1 = np.linalg.norm(backbone_coords[idx1_max] - backbone_coords[idx1_min])
        sep2 = np.linalg.norm(backbone_coords[idx2_max] - backbone_coords[idx2_min])

        if sep1 >= sep2:
            return backbone_coords[idx1_min].copy(), backbone_coords[idx1_max].copy()
        else:
            return backbone_coords[idx2_min].copy(), backbone_coords[idx2_max].copy()

    else:
        # Non-degenerate case: use PC1 only
        projections = centered_coords @ v1
        idx_min = projections.argmin()
        idx_max = projections.argmax()

        return backbone_coords[idx_min].copy(), backbone_coords[idx_max].copy()
