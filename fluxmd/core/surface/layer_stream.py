# fluxmd/core/surface/layer_stream.py
from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .ses_builder import SurfaceMesh

__all__ = ["MatryoshkaLayerGenerator"]


class MatryoshkaLayerGenerator:
    """Memory-efficient generator for nested molecular surface layers."""
    
    def __init__(self, base_surface: SurfaceMesh, step: float) -> None:
        """Initialize with base surface and layer spacing.
        
        Args:
            base_surface: Innermost SES mesh
            step: Distance between layers in Angstroms
        """
        self.base = base_surface
        self.step = step
        self._cache: Dict[int, SurfaceMesh] = {0: base_surface}
        
        # Precompute vertex normals for base surface
        self._base_normals = self._compute_vertex_normals(base_surface)
    
    def _compute_vertex_normals(self, mesh: SurfaceMesh) -> np.ndarray:
        """Compute vertex normals via area-weighted face normals.
        
        Args:
            mesh: Surface mesh
            
        Returns:
            (N, 3) array of unit normal vectors
        """
        vertices = mesh.vertices
        faces = mesh.faces
        n_vertices = len(vertices)
        
        # Initialize normals accumulator
        vertex_normals = np.zeros((n_vertices, 3), dtype=np.float32)
        
        # Compute face normals and areas
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # Face edges
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # Face normal (not normalized yet)
            face_normal = np.cross(edge1, edge2)
            
            # Face area is half the magnitude of cross product
            face_normal_magnitude = np.linalg.norm(face_normal)
            area = 0.5 * face_normal_magnitude
            
            if area > 1e-10:  # Skip degenerate faces
                # Normalize face normal
                face_normal /= face_normal_magnitude
                
                # Add area-weighted normal to each vertex
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += area * face_normal
        
        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        vertex_normals /= norms
        
        return vertex_normals
    
    def _detect_self_intersections(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Simple self-intersection detection via local neighborhood check.
        
        Args:
            vertices: Vertex positions
            faces: Face connectivity
            
        Returns:
            Boolean mask of problematic vertices
        """
        # Build vertex adjacency
        n_vertices = len(vertices)
        adjacency = [set() for _ in range(n_vertices)]
        
        for face in faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # Check for folded regions (negative dot products with neighbors)
        problematic = np.zeros(n_vertices, dtype=bool)
        
        for i in range(n_vertices):
            if len(adjacency[i]) < 3:  # Boundary vertex
                continue
                
            # Check angles with neighbors
            neighbors = list(adjacency[i])
            v_center = vertices[i]
            
            angles = []
            for j in neighbors:
                edge = vertices[j] - v_center
                edge_norm = np.linalg.norm(edge)
                if edge_norm > 1e-10:
                    # Approximate local normal from neighbors
                    local_normal = np.zeros(3)
                    for k in neighbors:
                        if k != j:
                            e1 = vertices[j] - v_center
                            e2 = vertices[k] - v_center
                            local_normal += np.cross(e1, e2)
                    
                    if np.linalg.norm(local_normal) > 1e-10:
                        local_normal /= np.linalg.norm(local_normal)
                        # Check if edge points inward
                        if np.dot(edge / edge_norm, local_normal) < -0.5:
                            problematic[i] = True
                            break
        
        return problematic
    
    def _smooth_problematic_regions(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray, 
        problematic: np.ndarray,
        iterations: int = 3
    ) -> np.ndarray:
        """Apply Laplacian smoothing to problematic regions.
        
        Args:
            vertices: Vertex positions
            faces: Face connectivity
            problematic: Boolean mask of vertices to smooth
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed vertex positions
        """
        # Build vertex adjacency
        n_vertices = len(vertices)
        adjacency = [set() for _ in range(n_vertices)]
        
        for face in faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # Apply smoothing
        smoothed_vertices = vertices.copy()
        
        for _ in range(iterations):
            new_positions = smoothed_vertices.copy()
            
            for i in np.where(problematic)[0]:
                if len(adjacency[i]) > 0:
                    # Average position of neighbors
                    neighbor_positions = smoothed_vertices[list(adjacency[i])]
                    new_positions[i] = neighbor_positions.mean(axis=0)
            
            smoothed_vertices = new_positions
        
        return smoothed_vertices
    
    def _offset_layer(self, layer_idx: int) -> SurfaceMesh:
        """Generate offset surface by vertex displacement.
        
        Args:
            layer_idx: Layer index (0 = base)
            
        Returns:
            Offset surface mesh
        """
        if layer_idx == 0:
            return self.base
        
        # Start from base surface
        vertices = self.base.vertices.copy()
        faces = self.base.faces.copy()
        
        # Offset along normals
        offset_distance = layer_idx * self.step
        offset_vertices = vertices + offset_distance * self._base_normals
        
        # Detect self-intersections
        problematic = self._detect_self_intersections(offset_vertices, faces)
        
        if np.any(problematic):
            # Smooth problematic regions
            offset_vertices = self._smooth_problematic_regions(
                offset_vertices, faces, problematic
            )
        
        return SurfaceMesh(offset_vertices, faces)
    
    def get_layer(self, i: int) -> SurfaceMesh:
        """Get layer i, generating if needed and managing cache.
        
        Args:
            i: Layer index
            
        Returns:
            Surface mesh for layer i
        """
        if i not in self._cache:
            self._cache[i] = self._offset_layer(i)
        
        # Keep only adjacent layers in memory
        for k in list(self._cache.keys()):
            if abs(k - i) > 1:
                del self._cache[k]
        
        return self._cache[i]
    
    def __iter__(self) -> Iterator[SurfaceMesh]:
        """Iterate through layers starting from base."""
        i = 0
        while True:
            yield self.get_layer(i)
            i += 1
    
    def get_max_useful_layers(self, ligand_radius: float, cutoff: float = 12.0) -> int:
        """Calculate maximum useful number of layers.
        
        Args:
            ligand_radius: Radius of ligand pseudo-sphere
            cutoff: VdW cutoff distance
            
        Returns:
            Maximum layer index before exceeding cutoff everywhere
        """
        max_offset = cutoff + ligand_radius
        return int(np.ceil(max_offset / self.step))