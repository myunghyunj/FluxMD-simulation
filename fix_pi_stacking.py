#!/usr/bin/env python
"""
fix_pi_stacking.py - Proper pi-stacking detection implementation

This shows how pi-stacking detection SHOULD be implemented.
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Set

# Proper distance cutoffs (Angstroms)
PI_STACKING_CUTOFFS = {
    'centroid_distance': 4.5,  # Max distance between ring centers
    'perpendicular_distance': 3.8,  # Max perpendicular distance
    'angle_parallel': 30,  # Max angle for parallel stacking (degrees)
    'angle_t_shaped': (60, 120),  # Angle range for T-shaped
}

def detect_aromatic_rings_ligand(atoms: List[Dict]) -> List[Set[int]]:
    """
    Detect aromatic rings in ligand using graph-based approach
    
    Parameters:
    -----------
    atoms : list of dict
        Each dict has 'element', 'coords', 'bonds' (list of bonded atom indices)
    
    Returns:
    --------
    list of sets
        Each set contains indices of atoms forming an aromatic ring
    """
    # Build molecular graph
    G = nx.Graph()
    
    for i, atom in enumerate(atoms):
        G.add_node(i, element=atom['element'])
    
    # Add bonds
    for i, atom in enumerate(atoms):
        for j in atom.get('bonds', []):
            if j > i:  # Avoid duplicate edges
                G.add_edge(i, j)
    
    # Find all cycles
    aromatic_rings = []
    
    # Look for 5 and 6-membered rings
    for ring_size in [5, 6]:
        for cycle in nx.simple_cycles(G.to_directed(), length_bound=ring_size):
            if len(cycle) == ring_size:
                # Check if ring could be aromatic
                if is_aromatic_ring(cycle, atoms):
                    aromatic_rings.append(set(cycle))
    
    return aromatic_rings

def is_aromatic_ring(ring_atoms: List[int], atoms: List[Dict]) -> bool:
    """
    Check if a ring is aromatic based on:
    1. Planarity
    2. Atom types (C, N, O, S in conjugated system)
    3. Huckel's rule approximation
    """
    # Get coordinates
    coords = np.array([atoms[i]['coords'] for i in ring_atoms])
    
    # Check planarity - all atoms should be within 0.5 Å of best-fit plane
    if not is_planar(coords, tolerance=0.5):
        return False
    
    # Check atom types - must be sp2 hybridized
    aromatic_elements = {'C', 'N', 'O', 'S'}
    for i in ring_atoms:
        if atoms[i]['element'] not in aromatic_elements:
            return False
    
    # Simple aromaticity check - all carbons/nitrogens in ring
    # (More sophisticated: check hybridization, electron count)
    return True

def is_planar(coords: np.ndarray, tolerance: float = 0.5) -> bool:
    """Check if points are coplanar within tolerance"""
    if len(coords) < 4:
        return True
    
    # Fit plane using SVD
    centroid = coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords - centroid)
    normal = vh[2]
    
    # Calculate distances from plane
    distances = np.abs(np.dot(coords - centroid, normal))
    
    return np.max(distances) < tolerance

def calculate_ring_properties(ring_atoms: Set[int], atoms: List[Dict]) -> Dict:
    """Calculate ring centroid and normal vector"""
    coords = np.array([atoms[i]['coords'] for i in ring_atoms])
    
    centroid = coords.mean(axis=0)
    
    # Calculate normal using SVD
    _, _, vh = np.linalg.svd(coords - centroid)
    normal = vh[2]
    
    return {
        'centroid': centroid,
        'normal': normal,
        'atoms': ring_atoms
    }

def detect_pi_stacking(protein_rings: List[Dict], ligand_rings: List[Dict]) -> List[Dict]:
    """
    Detect pi-stacking interactions between protein and ligand rings
    
    Returns list of interactions with proper classification
    """
    interactions = []
    
    for p_ring in protein_rings:
        p_centroid = p_ring['centroid']
        p_normal = p_ring['normal']
        
        for l_ring in ligand_rings:
            l_centroid = l_ring['centroid']
            l_normal = l_ring['normal']
            
            # Calculate centroid distance
            distance = np.linalg.norm(p_centroid - l_centroid)
            
            # Apply proper distance cutoff
            if distance > PI_STACKING_CUTOFFS['centroid_distance']:
                continue
            
            # Calculate angle between normals
            cos_angle = np.abs(np.dot(p_normal, l_normal))
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            
            # Calculate perpendicular distance
            offset_vector = l_centroid - p_centroid
            perpendicular_dist = np.abs(np.dot(offset_vector, p_normal))
            
            # Classify interaction type
            if angle < PI_STACKING_CUTOFFS['angle_parallel']:
                if perpendicular_dist < PI_STACKING_CUTOFFS['perpendicular_distance']:
                    interaction_type = 'parallel'
                    energy = -5.0
                else:
                    continue  # Too far for parallel stacking
            elif PI_STACKING_CUTOFFS['angle_t_shaped'][0] < angle < PI_STACKING_CUTOFFS['angle_t_shaped'][1]:
                interaction_type = 't-shaped'
                energy = -4.0
            else:
                interaction_type = 'angled'
                energy = -2.5
            
            interactions.append({
                'protein_ring': p_ring['atoms'],
                'ligand_ring': l_ring['atoms'],
                'distance': distance,
                'angle': angle,
                'type': interaction_type,
                'energy': energy,
                'perpendicular_distance': perpendicular_dist
            })
    
    return interactions

# Example fix for gpu_accelerated_flux.py
def get_proper_pi_stacking_parameters():
    """Return proper parameters for pi-stacking detection"""
    return {
        'distance_cutoff': 4.5,  # Not 7.0!
        'parallel_angle_cutoff': 30,
        'perpendicular_distance_cutoff': 3.8,
        'min_ring_size': 5,
        'max_ring_size': 6,
        'planarity_tolerance': 0.5
    }