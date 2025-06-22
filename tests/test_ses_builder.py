# tests/test_ses_builder.py
"""Test SES (Solvent Excluded Surface) construction."""

import numpy as np
import pytest

from fluxmd.core.surface import SESBuilder, SurfaceMesh


def test_surface_mesh_creation():
    """Test SurfaceMesh dataclass creation."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    
    mesh = SurfaceMesh(vertices, faces)
    
    assert mesh.vertices.dtype == np.float32
    assert mesh.faces.dtype == np.int32
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)


def test_ses_builder_init():
    """Test SESBuilder initialization."""
    coords = np.array([[0, 0, 0], [5, 0, 0]])
    radii = np.array([1.5, 1.7])
    
    builder = SESBuilder(coords, radii)
    
    assert builder.probe == 0.75  # Default
    assert np.array_equal(builder.coords, coords)
    assert np.array_equal(builder.radii, radii)


def test_distance_field_single_atom():
    """Test distance field for a single atom."""
    coords = np.array([[0, 0, 0]])
    radii = np.array([2.0])
    
    builder = SESBuilder(coords, radii, probe_radius=1.0)
    distance_field, origin, spacing = builder.build_distance_field(grid_spacing=1.0)
    
    # Check field properties
    assert distance_field.ndim == 3
    assert np.all(spacing == 1.0)
    
    # Check center point (should be inside)
    center_idx = tuple(s // 2 for s in distance_field.shape)
    center_dist = distance_field[center_idx]
    assert center_dist < 0  # Inside the SES
    
    # Check far point (should be outside)
    corner_dist = distance_field[0, 0, 0]
    assert corner_dist > 0  # Outside the SES


def test_distance_field_two_atoms():
    """Test distance field for two overlapping atoms."""
    coords = np.array([[0, 0, 0], [3, 0, 0]])
    radii = np.array([2.0, 2.0])
    
    builder = SESBuilder(coords, radii, probe_radius=0.5)
    distance_field, origin, spacing = builder.build_distance_field(grid_spacing=0.5)
    
    # Point between atoms should be inside
    mid_x = int((3.0 - origin[0]) / spacing[0])
    mid_y = int((0.0 - origin[1]) / spacing[1])
    mid_z = int((0.0 - origin[2]) / spacing[2])
    
    # Make sure indices are valid
    mid_x = np.clip(mid_x, 0, distance_field.shape[0] - 1)
    mid_y = np.clip(mid_y, 0, distance_field.shape[1] - 1)
    mid_z = np.clip(mid_z, 0, distance_field.shape[2] - 1)
    
    assert distance_field[mid_x, mid_y, mid_z] < 0


def test_build_ses0_requires_skimage():
    """Test that marching_cubes properly reports missing dependency."""
    coords = np.array([[0, 0, 0]])
    radii = np.array([2.0])
    builder = SESBuilder(coords, radii)
    
    try:
        import skimage
        # If skimage is available, test should work
        mesh = builder.build_ses0()
        assert isinstance(mesh, SurfaceMesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    except ImportError:
        # If skimage not available, should get clear error
        with pytest.raises(ImportError, match="scikit-image is required"):
            builder.build_ses0()


def test_distance_field_performance():
    """Test that vectorized implementation handles reasonable molecule size."""
    # Create a small protein-like structure (100 atoms)
    n_atoms = 100
    coords = np.random.randn(n_atoms, 3) * 20  # ~40Å protein
    radii = np.random.uniform(1.4, 2.0, n_atoms)
    
    builder = SESBuilder(coords, radii)
    
    # Should complete in reasonable time
    distance_field, origin, spacing = builder.build_distance_field(grid_spacing=2.0)
    
    # Basic sanity checks
    assert distance_field.ndim == 3
    assert not np.all(distance_field > 0)  # Some points inside
    assert not np.all(distance_field < 0)  # Some points outside


if __name__ == "__main__":
    pytest.main([__file__, "-v"])