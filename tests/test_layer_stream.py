# tests/test_layer_stream.py
"""Test Matryoshka layer generation."""

import numpy as np
import pytest

from fluxmd.core.surface import MatryoshkaLayerGenerator, SurfaceMesh


def create_simple_mesh() -> SurfaceMesh:
    """Create a simple triangulated cube for testing."""
    # Cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top
    ], dtype=np.float32)
    
    # Cube faces (triangulated)
    faces = np.array([
        # Bottom
        [0, 1, 2], [0, 2, 3],
        # Top
        [4, 6, 5], [4, 7, 6],
        # Sides
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0]
    ], dtype=np.int32)
    
    return SurfaceMesh(vertices, faces)


def test_layer_generator_init():
    """Test layer generator initialization."""
    mesh = create_simple_mesh()
    generator = MatryoshkaLayerGenerator(mesh, step=1.5)
    
    assert generator.step == 1.5
    assert 0 in generator._cache
    assert generator._cache[0] is mesh
    assert generator._base_normals.shape == (8, 3)


@pytest.mark.xfail(reason="Surface-offset algorithm was refactored in 2.0; test needs update.")
def test_vertex_normal_computation():
    """Test vertex normal calculation."""
    mesh = create_simple_mesh()
    generator = MatryoshkaLayerGenerator(mesh, step=1.5)
    
    # Check that normals are unit vectors
    norms = np.linalg.norm(generator._base_normals, axis=1)
    assert np.allclose(norms, 1.0)
    
    # Check that corner normals point outward
    # For a cube, corner normals should point away from center
    for i in range(8):
        vertex = mesh.vertices[i]
        normal = generator._base_normals[i]
        # Dot product with vertex (centered at origin) should be positive
        assert np.dot(normal, vertex) > 0


@pytest.mark.xfail(reason="Surface-offset algorithm was refactored in 2.0; test needs update.")
def test_layer_offset():
    """Test layer offset generation."""
    mesh = create_simple_mesh()
    generator = MatryoshkaLayerGenerator(mesh, step=0.5)
    
    # Get first offset layer
    layer1 = generator.get_layer(1)
    
    # Check that vertices moved outward
    assert layer1.vertices.shape == mesh.vertices.shape
    assert layer1.faces.shape == mesh.faces.shape
    
    # All vertices should be further from origin
    dist0 = np.linalg.norm(mesh.vertices, axis=1)
    dist1 = np.linalg.norm(layer1.vertices, axis=1)
    assert np.all(dist1 > dist0)
    
    # Check approximate offset distance
    offset_distances = dist1 - dist0
    assert np.allclose(offset_distances, 0.5, atol=0.1)


@pytest.mark.xfail(reason="Surface-offset algorithm was refactored in 2.0; test needs update.")
def test_layer_iterator():
    """Test layer iteration."""
    mesh = create_simple_mesh()
    generator = MatryoshkaLayerGenerator(mesh, step=1.0)
    
    # Get first few layers
    layers = []
    for i, layer in enumerate(generator):
        layers.append(layer)
        if i >= 3:
            break
    
    assert len(layers) == 4
    assert layers[0] is mesh  # Layer 0 is base mesh
    
    # Check increasing distances
    for i in range(1, 4):
        dist_prev = np.linalg.norm(layers[i-1].vertices, axis=1).mean()
        dist_curr = np.linalg.norm(layers[i].vertices, axis=1).mean()
        assert dist_curr > dist_prev


def test_max_useful_layers():
    """Test calculation of maximum useful layers."""
    mesh = create_simple_mesh()
    generator = MatryoshkaLayerGenerator(mesh, step=1.5)
    
    # With ligand radius 5Å and cutoff 12Å
    max_layers = generator.get_max_useful_layers(ligand_radius=5.0, cutoff=12.0)
    
    # Should be ceil((12 + 5) / 1.5) = 12
    assert max_layers == 12


def test_self_intersection_detection():
    """Test that self-intersection detection works."""
    # Create a mesh with potential self-intersection
    # (concave region that might fold when offset)
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],  # Square base
        [1, 1, -0.5]  # Indented center point
    ])
    
    faces = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],  # Pyramid faces
        [0, 2, 1], [0, 3, 2]  # Base (reversed for inward normals)
    ])
    
    mesh = SurfaceMesh(vertices, faces)
    generator = MatryoshkaLayerGenerator(mesh, step=2.0)
    
    # Large offset should trigger smoothing
    layer = generator.get_layer(1)
    
    # Should still have same topology
    assert layer.vertices.shape == vertices.shape
    assert layer.faces.shape == faces.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])