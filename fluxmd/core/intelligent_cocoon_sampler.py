"""
Intelligent Cocoon Sampler for REF15-Enhanced FluxMD
Uses energy-aware approach trajectory generation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import logging

from .ref15_energy import get_ref15_calculator, AtomContext
from .rosetta_atom_types import get_atom_typer

logger = logging.getLogger(__name__)


class IntelligentCocoonSampler:
    """
    Enhanced trajectory generation using REF15 energy guidance
    Replaces random spherical sampling with physics-informed approaches
    """
    
    def __init__(self, protein_surface_coords: np.ndarray, 
                 protein_atoms: List[Dict],
                 ref15_calculator=None,
                 physiological_pH: float = 7.4):
        """
        Initialize intelligent sampler
        
        Args:
            protein_surface_coords: Coordinates of protein surface atoms
            protein_atoms: List of protein atom dictionaries
            ref15_calculator: REF15 energy calculator instance
            physiological_pH: pH for calculations
        """
        self.protein_coords = protein_surface_coords
        self.protein_atoms = protein_atoms
        self.protein_center = protein_surface_coords.mean(axis=0)
        
        # Initialize energy calculator
        self.ref15 = ref15_calculator or get_ref15_calculator(physiological_pH)
        self.atom_typer = get_atom_typer()
        
        # Calculate protein properties
        self._analyze_protein_surface()
        
    def _analyze_protein_surface(self):
        """Analyze protein surface for intelligent sampling"""
        logger.info("Analyzing protein surface properties...")
        
        # Calculate electrostatic potential grid
        self._calculate_electrostatic_surface()
        
        # Identify hydrophobic patches
        self._identify_hydrophobic_regions()
        
        # Find hydrogen bond donors/acceptors
        self._map_hbond_sites()
        
        # Locate aromatic rings for pi-stacking
        self._find_aromatic_patches()
        
    def _calculate_electrostatic_surface(self):
        """Calculate simplified electrostatic potential on protein surface"""
        # Grid around protein
        margin = 20.0  # Angstroms
        
        min_coords = self.protein_coords.min(axis=0) - margin
        max_coords = self.protein_coords.max(axis=0) + margin
        
        # Create sparse grid for efficiency
        grid_spacing = 2.0  # Angstroms
        nx = int((max_coords[0] - min_coords[0]) / grid_spacing) + 1
        ny = int((max_coords[1] - min_coords[1]) / grid_spacing) + 1
        nz = int((max_coords[2] - min_coords[2]) / grid_spacing) + 1
        
        # Store grid points and potentials
        self.elec_grid_points = []
        self.elec_potentials = []
        
        # Calculate potential at surface points only
        try:
            hull = ConvexHull(self.protein_coords)
            surface_indices = np.unique(hull.simplices.flatten())
            surface_coords = self.protein_coords[surface_indices]
            
            # Extend surface outward
            center = self.protein_coords.mean(axis=0)
            for coord in surface_coords:
                direction = coord - center
                direction /= np.linalg.norm(direction)
                
                # Sample points along ray
                for r in [5.0, 10.0, 15.0, 20.0]:
                    point = coord + direction * r
                    
                    # Calculate potential from charged atoms
                    potential = 0.0
                    for atom in self.protein_atoms:
                        if 'formal_charge' in atom and atom['formal_charge'] != 0:
                            atom_coord = np.array([atom['x'], atom['y'], atom['z']])
                            distance = np.linalg.norm(point - atom_coord)
                            if distance > 0.1:
                                # Simple Coulomb potential
                                potential += 332.0 * atom['formal_charge'] / (10.0 * distance)
                    
                    self.elec_grid_points.append(point)
                    self.elec_potentials.append(potential)
                    
        except Exception as e:
            logger.warning(f"Electrostatic surface calculation failed: {e}")
            self.elec_grid_points = []
            self.elec_potentials = []
            
        self.elec_grid_points = np.array(self.elec_grid_points) if self.elec_grid_points else np.array([])
        self.elec_potentials = np.array(self.elec_potentials) if self.elec_potentials else np.array([])
        
        logger.info(f"  Calculated potential at {len(self.elec_potentials)} surface points")
        
    def _identify_hydrophobic_regions(self):
        """Identify hydrophobic patches on protein surface"""
        hydrophobic_atoms = {'C', 'S'}  # Simplified
        hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        
        self.hydrophobic_centers = []
        
        # Group hydrophobic atoms by proximity
        hydrophobic_coords = []
        for atom in self.protein_atoms:
            if (atom.get('element', 'C') in hydrophobic_atoms and 
                atom.get('resname', '') in hydrophobic_residues):
                hydrophobic_coords.append([atom['x'], atom['y'], atom['z']])
                
        if hydrophobic_coords:
            hydrophobic_coords = np.array(hydrophobic_coords)
            
            # Simple clustering - find dense regions
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=5.0, min_samples=3).fit(hydrophobic_coords)
                
                # Get cluster centers
                for label in set(clustering.labels_):
                    if label != -1:  # Skip noise
                        cluster_coords = hydrophobic_coords[clustering.labels_ == label]
                        center = cluster_coords.mean(axis=0)
                        self.hydrophobic_centers.append(center)
            except ImportError:
                # Fallback: use simple grid-based clustering
                logger.warning("sklearn not available, using simple hydrophobic center detection")
                # Just use center of mass of hydrophobic atoms
                if len(hydrophobic_coords) > 0:
                    self.hydrophobic_centers.append(hydrophobic_coords.mean(axis=0))
                    
        logger.info(f"  Found {len(self.hydrophobic_centers)} hydrophobic patches")
        
    def _map_hbond_sites(self):
        """Map hydrogen bond donor/acceptor sites"""
        self.hbond_donors = []
        self.hbond_acceptors = []
        
        for atom in self.protein_atoms:
            coord = np.array([atom['x'], atom['y'], atom['z']])
            
            # Check if surface exposed (simple distance check)
            min_dist_to_surface = np.min(cdist([coord], self.protein_coords)[0])
            if min_dist_to_surface < 5.0:  # Near surface
                
                # Create context for REF15
                context = self.ref15.create_atom_context(atom)
                
                if context.is_donor:
                    self.hbond_donors.append({
                        'coord': coord,
                        'atom_type': context.atom_type,
                        'residue': atom.get('resname', '')
                    })
                    
                if context.is_acceptor:
                    self.hbond_acceptors.append({
                        'coord': coord,
                        'atom_type': context.atom_type,
                        'residue': atom.get('resname', '')
                    })
                    
        logger.info(f"  Found {len(self.hbond_donors)} H-bond donors, {len(self.hbond_acceptors)} acceptors")
        
    def _find_aromatic_patches(self):
        """Find aromatic rings for pi-stacking"""
        aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        self.aromatic_centers = []
        
        # Group by residue
        residue_atoms = {}
        for atom in self.protein_atoms:
            if atom.get('resname', '') in aromatic_residues:
                key = (atom['resname'], atom.get('resSeq', 0), atom.get('chain', 'A'))
                if key not in residue_atoms:
                    residue_atoms[key] = []
                residue_atoms[key].append(atom)
                
        # Calculate ring centers
        for (resname, resid, chain), atoms in residue_atoms.items():
            ring_coords = []
            
            # Get ring atoms based on residue type
            ring_atom_names = {
                'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
                'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
            }
            
            for atom in atoms:
                if atom['name'] in ring_atom_names.get(resname, []):
                    ring_coords.append([atom['x'], atom['y'], atom['z']])
                    
            if len(ring_coords) >= 3:
                center = np.mean(ring_coords, axis=0)
                self.aromatic_centers.append({
                    'coord': center,
                    'residue': resname,
                    'resid': resid
                })
                
        logger.info(f"  Found {len(self.aromatic_centers)} aromatic rings")
        
    def generate_approach_points(self, n_trajectories: int, 
                               ligand_charge: float = 0.0,
                               ligand_has_aromatic: bool = False,
                               ligand_has_hbond: bool = True) -> List[np.ndarray]:
        """
        Generate intelligent approach points based on ligand properties
        
        Args:
            n_trajectories: Number of approach trajectories
            ligand_charge: Net charge of ligand
            ligand_has_aromatic: Whether ligand has aromatic rings
            ligand_has_hbond: Whether ligand can form H-bonds
            
        Returns:
            List of approach starting points
        """
        logger.info(f"\nGenerating {n_trajectories} intelligent approach points...")
        logger.info(f"  Ligand properties: charge={ligand_charge:.1f}, aromatic={ligand_has_aromatic}, H-bond={ligand_has_hbond}")
        
        approach_points = []
        
        # Allocate trajectories based on ligand properties
        n_per_type = self._allocate_trajectory_types(
            n_trajectories, ligand_charge, ligand_has_aromatic, ligand_has_hbond
        )
        
        # 1. Electrostatic-guided approaches
        if n_per_type['electrostatic'] > 0:
            elec_points = self._electrostatic_guided_points(
                n_per_type['electrostatic'], ligand_charge
            )
            approach_points.extend(elec_points)
            
        # 2. H-bond targeted approaches
        if n_per_type['hbond'] > 0:
            hbond_points = self._hbond_guided_points(n_per_type['hbond'])
            approach_points.extend(hbond_points)
            
        # 3. Hydrophobic approaches
        if n_per_type['hydrophobic'] > 0:
            hydro_points = self._hydrophobic_guided_points(n_per_type['hydrophobic'])
            approach_points.extend(hydro_points)
            
        # 4. Pi-stacking approaches
        if n_per_type['aromatic'] > 0:
            pi_points = self._aromatic_guided_points(n_per_type['aromatic'])
            approach_points.extend(pi_points)
            
        # 5. Random/exploratory approaches
        if n_per_type['random'] > 0:
            random_points = self._random_surface_points(n_per_type['random'])
            approach_points.extend(random_points)
            
        # Ensure we have exactly n_trajectories points
        if len(approach_points) < n_trajectories:
            # Add more random points if needed
            extra = self._random_surface_points(n_trajectories - len(approach_points))
            approach_points.extend(extra)
        elif len(approach_points) > n_trajectories:
            # Randomly sample if we have too many
            indices = np.random.choice(len(approach_points), n_trajectories, replace=False)
            approach_points = [approach_points[i] for i in indices]
            
        logger.info(f"  Generated {len(approach_points)} approach points")
        return approach_points
        
    def _allocate_trajectory_types(self, n_total: int, ligand_charge: float,
                                  has_aromatic: bool, has_hbond: bool) -> Dict[str, int]:
        """Intelligently allocate trajectory types based on ligand properties"""
        allocations = {
            'electrostatic': 0,
            'hbond': 0,
            'hydrophobic': 0,
            'aromatic': 0,
            'random': 0
        }
        
        # Base allocation
        if abs(ligand_charge) > 0.5:
            allocations['electrostatic'] = int(0.3 * n_total)
        if has_hbond:
            allocations['hbond'] = int(0.3 * n_total)
        if has_aromatic:
            allocations['aromatic'] = int(0.2 * n_total)
            
        # Always include some hydrophobic and random
        allocations['hydrophobic'] = int(0.1 * n_total)
        allocations['random'] = n_total - sum(allocations.values())
        
        # Ensure at least some random exploration
        if allocations['random'] < int(0.1 * n_total):
            allocations['random'] = int(0.1 * n_total)
            # Reduce others proportionally
            total_others = sum(v for k, v in allocations.items() if k != 'random')
            if total_others > 0:
                scale = (n_total - allocations['random']) / total_others
                for k in allocations:
                    if k != 'random':
                        allocations[k] = int(allocations[k] * scale)
                        
        return allocations
        
    def _electrostatic_guided_points(self, n_points: int, ligand_charge: float) -> List[np.ndarray]:
        """Generate approach points guided by electrostatic complementarity"""
        points = []
        
        if len(self.elec_potentials) == 0:
            return self._random_surface_points(n_points)
            
        # Find favorable electrostatic regions
        if ligand_charge > 0:
            # Positive ligand seeks negative potential
            favorable_mask = self.elec_potentials < -5.0
        elif ligand_charge < 0:
            # Negative ligand seeks positive potential
            favorable_mask = self.elec_potentials > 5.0
        else:
            # Neutral ligand - use random
            return self._random_surface_points(n_points)
            
        if np.any(favorable_mask):
            favorable_points = self.elec_grid_points[favorable_mask]
            
            # Sample from favorable regions
            n_favorable = len(favorable_points)
            for i in range(n_points):
                if n_favorable > 0:
                    idx = np.random.randint(n_favorable)
                    point = favorable_points[idx]
                    
                    # Add some randomness
                    point += np.random.randn(3) * 2.0
                    points.append(point)
                else:
                    # Fallback to random
                    points.extend(self._random_surface_points(1))
        else:
            # No favorable regions found
            points = self._random_surface_points(n_points)
            
        return points
        
    def _hbond_guided_points(self, n_points: int) -> List[np.ndarray]:
        """Generate approach points targeting H-bond sites"""
        points = []
        
        # Combine donors and acceptors
        hbond_sites = self.hbond_donors + self.hbond_acceptors
        
        if not hbond_sites:
            return self._random_surface_points(n_points)
            
        for i in range(n_points):
            # Pick a random H-bond site
            site = hbond_sites[np.random.randint(len(hbond_sites))]
            site_coord = site['coord']
            
            # Approach from optimal H-bond geometry
            # Add randomness but bias toward good angles
            
            # Random direction with bias
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Distance for H-bond approach
            distance = np.random.uniform(15.0, 20.0)
            
            point = site_coord + direction * distance
            points.append(point)
            
        return points
        
    def _hydrophobic_guided_points(self, n_points: int) -> List[np.ndarray]:
        """Generate approach points targeting hydrophobic patches"""
        points = []
        
        if not self.hydrophobic_centers:
            return self._random_surface_points(n_points)
            
        for i in range(n_points):
            # Pick a random hydrophobic center
            center = self.hydrophobic_centers[np.random.randint(len(self.hydrophobic_centers))]
            
            # Approach from various angles
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            distance = np.random.uniform(10.0, 15.0)
            
            point = center + direction * distance
            points.append(point)
            
        return points
        
    def _aromatic_guided_points(self, n_points: int) -> List[np.ndarray]:
        """Generate approach points for pi-stacking"""
        points = []
        
        if not self.aromatic_centers:
            return self._random_surface_points(n_points)
            
        for i in range(n_points):
            # Pick a random aromatic ring
            ring = self.aromatic_centers[np.random.randint(len(self.aromatic_centers))]
            ring_center = ring['coord']
            
            # Approach from above/below ring plane (simplified)
            # Would need ring normal for accurate geometry
            
            # Random but biased toward perpendicular approach
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.choice([np.pi/4, 3*np.pi/4])  # Above or below
            
            distance = np.random.uniform(10.0, 15.0)
            
            x = distance * np.sin(phi) * np.cos(theta)
            y = distance * np.sin(phi) * np.sin(theta) 
            z = distance * np.cos(phi)
            
            point = ring_center + np.array([x, y, z])
            points.append(point)
            
        return points
        
    def _random_surface_points(self, n_points: int) -> List[np.ndarray]:
        """Generate random points around protein surface"""
        points = []
        
        center = self.protein_center
        max_radius = np.max(cdist([center], self.protein_coords)[0])
        
        for i in range(n_points):
            # Random spherical coordinates
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(1 - 2 * np.random.random())
            
            # Random distance (biased toward surface)
            distance = max_radius + np.random.uniform(5.0, 20.0)
            
            x = distance * np.sin(phi) * np.cos(theta)
            y = distance * np.sin(phi) * np.sin(theta)
            z = distance * np.cos(phi)
            
            point = center + np.array([x, y, z])
            points.append(point)
            
        return points
        
    def sample_potential(self, point: np.ndarray) -> float:
        """Sample electrostatic potential at a point"""
        if len(self.elec_potentials) == 0:
            return 0.0
            
        # Find nearest grid point
        distances = cdist([point], self.elec_grid_points)[0]
        nearest_idx = np.argmin(distances)
        
        # Simple nearest neighbor - could interpolate for smoothness
        return self.elec_potentials[nearest_idx]