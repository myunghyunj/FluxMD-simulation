# fluxmd/core/surface/ses_builder.py
from __future__ import annotations

from typing import Tuple, Optional, Dict

import numpy as np

from .dna_groove_detector import DNAGrooveDetector

__all__ = ["SurfaceMesh", "SESBuilder"]


class SurfaceMesh:
    """Triangulated mesh representation for molecular surfaces."""
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, 
                 groove_labels: Optional[Dict[int, str]] = None) -> None:
        """Initialize mesh with vertices and face indices.
        
        Args:
            vertices: (N,3) array of vertex coordinates
            faces: (M,3) array of triangle vertex indices
            groove_labels: Optional dict mapping vertex index to groove type
        """
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.int32)
        self.groove_labels = groove_labels or {}


class SESBuilder:
    """Builds Solvent-Excluded Surface (SES) via distance field + marching cubes."""
    
    def __init__(
        self,
        atom_coords: np.ndarray,
        atom_radii: np.ndarray,
        probe_radius: float = 0.75,
        atom_names: Optional[np.ndarray] = None,
        resnames: Optional[np.ndarray] = None
    ) -> None:
        """Initialize builder with atomic structure.
        
        Args:
            atom_coords: (N,3) array of atom positions
            atom_radii: (N,) array of VDW radii
            probe_radius: Solvent probe radius in Angstroms
            atom_names: Optional atom names for DNA detection
            resnames: Optional residue names for DNA detection
        """
        self.coords = atom_coords
        self.radii = atom_radii
        self.probe = probe_radius
        
        # Initialize DNA groove detector if atom info provided
        self.groove_detector = None
        if atom_names is not None:
            atoms_dict = {
                'coords': atom_coords,
                'names': atom_names,
                'resnames': resnames if resnames is not None else [''] * len(atom_coords)
            }
            self.groove_detector = DNAGrooveDetector(atoms_dict)
    
    def build_distance_field(
        self,
        grid_spacing: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute signed distance field on regular grid.
        
        Args:
            grid_spacing: Grid resolution in Angstroms
            
        Returns:
            Tuple of (distance_field, origin, spacing)
        """
        # Create bounding box with padding
        max_radius = self.radii.max()
        padding = max_radius + self.probe + 2.0
        
        mins = self.coords.min(axis=0) - padding
        maxs = self.coords.max(axis=0) + padding
        
        # Generate 3D grid
        nx = int(np.ceil((maxs[0] - mins[0]) / grid_spacing))
        ny = int(np.ceil((maxs[1] - mins[1]) / grid_spacing))
        nz = int(np.ceil((maxs[2] - mins[2]) / grid_spacing))
        
        # Create meshgrid
        x = np.linspace(mins[0], maxs[0], nx)
        y = np.linspace(mins[1], maxs[1], ny)
        z = np.linspace(mins[2], maxs[2], nz)
        
        # Initialize distance field
        distance_field = np.full((nx, ny, nz), np.inf, dtype=np.float32)
        
        # Compute signed distance to expanded VDW spheres
        # Vectorized implementation for better performance
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        # Process in chunks to manage memory
        chunk_size = 10000
        n_points = len(grid_points)
        
        for start_idx in range(0, n_points, chunk_size):
            end_idx = min(start_idx + chunk_size, n_points)
            chunk_points = grid_points[start_idx:end_idx]
            
            # Compute distances to all atoms for this chunk
            # Shape: (chunk_size, n_atoms)
            distances = np.linalg.norm(
                chunk_points[:, np.newaxis, :] - self.coords[np.newaxis, :, :],
                axis=2
            )
            
            # Distance to surface for each atom
            surface_distances = distances - self.radii[np.newaxis, :]
            
            # Minimum distance to any atom surface
            min_distances = surface_distances.min(axis=1)
            
            # Store in distance field (subtract probe radius for SES)
            flat_indices = np.arange(start_idx, end_idx)
            i_indices = flat_indices // (ny * nz)
            j_indices = (flat_indices % (ny * nz)) // nz
            k_indices = flat_indices % nz
            
            distance_field[i_indices, j_indices, k_indices] = min_distances - self.probe
        
        return distance_field, mins, np.array([grid_spacing, grid_spacing, grid_spacing])
    
    def marching_cubes(
        self,
        distance_field: np.ndarray,
        iso_value: float
    ) -> SurfaceMesh:
        """Extract isosurface mesh at given distance value.
        
        Args:
            distance_field: 3D scalar field
            iso_value: Distance value for surface extraction
            
        Returns:
            Triangulated surface mesh
        """
        try:
            from skimage import measure
        except ImportError:
            raise ImportError(
                "scikit-image is required for marching cubes. "
                "Install with: pip install scikit-image"
            )
        
        # Extract isosurface
        vertices, faces, _, _ = measure.marching_cubes(
            distance_field, 
            level=iso_value,
            spacing=(1.0, 1.0, 1.0)  # We'll scale vertices manually
        )
        
        # The vertices are in grid coordinates, need to transform to world coordinates
        # This transformation was stored from build_distance_field
        # For now, we'll require the caller to handle transformation
        
        return SurfaceMesh(vertices, faces)
    
    def build_ses0(self) -> SurfaceMesh:
        """Build primary SES at probe radius.
        
        Returns:
            SES mesh (Connolly surface) with optional groove labels
        """
        # Build distance field
        distance_field, origin, spacing = self.build_distance_field()
        
        # Extract surface at iso_value = 0 (SES is where distance = 0)
        mesh = self.marching_cubes(distance_field, iso_value=0.0)
        
        # Transform vertices from grid coordinates to world coordinates
        # vertices are in grid indices, need to multiply by spacing and add origin
        mesh.vertices = mesh.vertices * spacing + origin
        
        # Label DNA grooves if detector available
        if self.groove_detector is not None and self.groove_detector.has_dna:
            print("  Detecting DNA grooves on surface...")
            groove_labels = self.groove_detector.label_surface_vertices(
                mesh.vertices, radius_cutoff=5.0
            )
            mesh.groove_labels = groove_labels
        
        return mesh