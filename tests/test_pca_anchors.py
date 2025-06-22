# tests/test_pca_anchors.py
"""Test PCA-based anchor detection."""

import numpy as np
import pytest

from fluxmd.core.geometry import extreme_calpha_pairs


def test_linear_protein():
    """Test anchor detection on a linear protein structure."""
    # Create a linear protein with 10 CA atoms
    n_atoms = 10
    coords = np.zeros((n_atoms, 3))
    coords[:, 0] = np.linspace(0, 30, n_atoms)  # Linear along X-axis
    atom_names = ['CA'] * n_atoms
    
    start, end = extreme_calpha_pairs(coords, atom_names)
    
    # Should find first and last atoms
    assert np.allclose(start, [0, 0, 0])
    assert np.allclose(end, [30, 0, 0])


def test_bent_protein():
    """Test anchor detection on a bent protein structure."""
    # Create an L-shaped protein
    coords = []
    atom_names = []
    
    # First segment along X
    for i in range(5):
        coords.append([i * 3.8, 0, 0])
        atom_names.append('CA')
    
    # Second segment along Y
    for i in range(5):
        coords.append([4 * 3.8, i * 3.8, 0])
        atom_names.append('CA')
    
    coords = np.array(coords)
    start, end = extreme_calpha_pairs(coords, atom_names)
    
    # Should find corners of the L
    dist = np.linalg.norm(end - start)
    assert dist > 20  # Reasonable separation


def test_degenerate_case():
    """Test handling of degenerate (spherical) structures."""
    # Create a roughly spherical arrangement
    n_atoms = 20
    theta = np.linspace(0, np.pi, n_atoms)
    phi = np.linspace(0, 2*np.pi, n_atoms)
    
    coords = np.zeros((n_atoms, 3))
    coords[:, 0] = 10 * np.sin(theta) * np.cos(phi)
    coords[:, 1] = 10 * np.sin(theta) * np.sin(phi)
    coords[:, 2] = 10 * np.cos(theta)
    
    atom_names = ['CA'] * n_atoms
    
    # Add small perturbation to avoid exact sphere
    coords += np.random.randn(*coords.shape) * 0.1
    
    start, end = extreme_calpha_pairs(coords, atom_names)
    
    # Should still find two well-separated points
    dist = np.linalg.norm(end - start)
    assert dist > 15  # At least 75% of diameter


def test_dna_backbone():
    """Test anchor detection on DNA with P atoms."""
    # Create a simple DNA-like structure
    n_atoms = 20
    coords = np.zeros((n_atoms, 3))
    coords[:, 2] = np.linspace(0, 50, n_atoms)  # Helix along Z
    coords[:, 0] = 5 * np.cos(np.linspace(0, 4*np.pi, n_atoms))
    coords[:, 1] = 5 * np.sin(np.linspace(0, 4*np.pi, n_atoms))
    
    atom_names = ['P'] * n_atoms
    
    start, end = extreme_calpha_pairs(coords, atom_names)
    
    # Should find top and bottom of helix
    z_diff = abs(end[2] - start[2])
    assert z_diff > 45  # Most of the length


def test_mixed_atoms():
    """Test that only CA and P atoms are considered."""
    coords = np.array([
        [0, 0, 0],    # CA
        [5, 0, 0],    # CB (ignored)
        [10, 0, 0],   # CA
        [15, 0, 0],   # O (ignored)
        [20, 0, 0],   # P
    ])
    atom_names = ['CA', 'CB', 'CA', 'O', 'P']
    
    start, end = extreme_calpha_pairs(coords, atom_names)
    
    # Should use only CA and P atoms
    assert np.allclose(start, [0, 0, 0]) or np.allclose(start, [20, 0, 0])
    assert np.allclose(end, [0, 0, 0]) or np.allclose(end, [20, 0, 0])
    assert not np.allclose(start, end)


def test_no_backbone_atoms():
    """Test error handling when no backbone atoms present."""
    coords = np.array([[0, 0, 0], [1, 1, 1]])
    atom_names = ['CB', 'CG']
    
    with pytest.raises(ValueError, match="No backbone atoms"):
        extreme_calpha_pairs(coords, atom_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])