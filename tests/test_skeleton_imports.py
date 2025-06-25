# tests/test_skeleton_imports.py
"""Test that all Matryoshka skeleton modules can be imported."""

import pytest


def test_surface_imports():
    """Test surface module imports."""
    from fluxmd.core.surface import MatryoshkaLayerGenerator, SESBuilder
    from fluxmd.core.surface.layer_stream import MatryoshkaLayerGenerator as MLG2
    from fluxmd.core.surface.ses_builder import SESBuilder as SESBuilder2

    assert SESBuilder is SESBuilder2
    assert MatryoshkaLayerGenerator is MLG2


def test_geometry_imports():
    """Test geometry module imports."""
    from fluxmd.core.geometry import extreme_calpha_pairs
    from fluxmd.core.geometry.pca_anchors import extreme_calpha_pairs as ecp2

    assert extreme_calpha_pairs is ecp2


def test_dynamics_imports():
    """Test dynamics module imports."""
    from fluxmd.core.dynamics import BrownianSurfaceRoller
    from fluxmd.core.dynamics.brownian_roller import BrownianSurfaceRoller as BSR2

    assert BrownianSurfaceRoller is BSR2


def test_generator_import():
    """Test main generator import."""
    from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator

    assert MatryoshkaTrajectoryGenerator is not None


def test_instantiation_signatures():
    """Test that classes can be instantiated with minimal args."""
    import numpy as np

    from fluxmd.core.surface import SESBuilder, SurfaceMesh

    # Test SurfaceMesh
    vertices = np.zeros((10, 3), dtype=np.float32)
    faces = np.zeros((5, 3), dtype=np.int32)
    mesh = SurfaceMesh(vertices, faces)
    assert mesh.vertices.shape == (10, 3)
    assert mesh.faces.shape == (5, 3)

    # Test SESBuilder
    coords = np.zeros((5, 3))
    radii = np.ones(5)
    builder = SESBuilder(coords, radii)
    assert builder.probe == 0.75  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
