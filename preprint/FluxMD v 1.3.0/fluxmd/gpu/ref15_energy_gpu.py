"""
GPU-Accelerated REF15 Energy Calculator
Optimized for Apple Silicon (Metal) and NVIDIA CUDA
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import logging
from dataclasses import dataclass

from ..core.ref15_params import get_ref15_params, REF15Parameters
from ..core.rosetta_atom_types import get_atom_typer

logger = logging.getLogger(__name__)


@dataclass 
class REF15GPUArrays:
    """Pre-computed GPU arrays for REF15 calculations"""
    # Atom properties
    atom_types: torch.Tensor      # [N] atom type indices
    coords: torch.Tensor          # [N, 3] coordinates
    charges: torch.Tensor         # [N] formal charges
    is_donor: torch.Tensor        # [N] H-bond donor flags
    is_acceptor: torch.Tensor     # [N] H-bond acceptor flags
    is_aromatic: torch.Tensor     # [N] aromatic flags
    is_backbone: torch.Tensor     # [N] backbone flags
    residue_ids: torch.Tensor     # [N] residue IDs
    
    # LJ parameter matrices (pre-computed for all type pairs)
    lj_radius_matrix: torch.Tensor    # [n_types, n_types]
    lj_wdepth_matrix: torch.Tensor    # [n_types, n_types]
    
    # Solvation parameters
    lk_dgfree: torch.Tensor       # [n_types] solvation free energies
    lk_lambda: torch.Tensor       # [n_types] correlation lengths
    lk_volume: torch.Tensor       # [n_types] volumes
    
    # Neighbor counts for burial
    neighbor_counts: Optional[torch.Tensor] = None  # [N]


class REF15GPUCalculator:
    """
    Fully vectorized REF15 energy calculation on GPU
    Handles thousands of interactions in parallel
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.params = get_ref15_params()
        self.atom_typer = get_atom_typer()
        
        # Build atom type index mapping
        self._build_type_indices()
        
        # Pre-compute parameter matrices on GPU
        self._build_parameter_matrices()
        
        logger.info(f"REF15 GPU calculator initialized on {device}")
        
    def _build_type_indices(self):
        """Build mapping from atom type names to indices"""
        # Get all unique atom types
        all_types = list(self.atom_typer.atom_types.keys())
        self.type_to_idx = {t: i for i, t in enumerate(all_types)}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}
        self.n_types = len(all_types)
        
    def _build_parameter_matrices(self):
        """Pre-compute all parameter matrices on GPU"""
        # LJ parameters matrix
        lj_radius = torch.zeros((self.n_types, self.n_types), device=self.device)
        lj_wdepth = torch.zeros((self.n_types, self.n_types), device=self.device)
        
        for i, type1 in enumerate(self.idx_to_type.values()):
            for j, type2 in enumerate(self.idx_to_type.values()):
                radius, wdepth = self.params.get_lj_params(type1, type2)
                lj_radius[i, j] = radius
                lj_wdepth[i, j] = wdepth
                
        self.lj_radius_matrix = lj_radius
        self.lj_wdepth_matrix = lj_wdepth
        
        # Solvation parameters
        lk_dgfree = torch.zeros(self.n_types, device=self.device)
        lk_lambda = torch.zeros(self.n_types, device=self.device)
        lk_volume = torch.zeros(self.n_types, device=self.device)
        
        for i, atom_type in enumerate(self.idx_to_type.values()):
            dgfree, lambda_val, volume = self.params.get_lk_params(atom_type)
            lk_dgfree[i] = dgfree
            lk_lambda[i] = lambda_val
            lk_volume[i] = volume
            
        self.lk_dgfree_vec = lk_dgfree
        self.lk_lambda_vec = lk_lambda
        self.lk_volume_vec = lk_volume
        
        # H-bond polynomials (simplified - would need full tensor for all pairs)
        # For now, use average polynomial
        default_poly = self.params.get_hbond_poly('default', 'default')
        self.hbond_poly = torch.tensor(default_poly, device=self.device)
        
        # Weights
        self.weights = {k: torch.tensor(v, device=self.device) 
                       for k, v in self.params.weights.items()}
                       
    def prepare_atoms_gpu(self, atoms_df, is_protein=True) -> REF15GPUArrays:
        """
        Convert atom dataframe to GPU arrays with REF15 typing
        
        Args:
            atoms_df: Pandas DataFrame with atom information
            is_protein: Whether these are protein atoms
            
        Returns:
            REF15GPUArrays with all pre-computed properties
        """
        n_atoms = len(atoms_df)
        
        # Initialize arrays
        atom_types = torch.zeros(n_atoms, dtype=torch.long, device=self.device)
        coords = torch.zeros((n_atoms, 3), device=self.device)
        charges = torch.zeros(n_atoms, device=self.device)
        is_donor = torch.zeros(n_atoms, dtype=torch.bool, device=self.device)
        is_acceptor = torch.zeros(n_atoms, dtype=torch.bool, device=self.device)
        is_aromatic = torch.zeros(n_atoms, dtype=torch.bool, device=self.device)
        is_backbone = torch.zeros(n_atoms, dtype=torch.bool, device=self.device)
        residue_ids = torch.zeros(n_atoms, dtype=torch.long, device=self.device)
        
        # Process each atom
        for i, (_, atom) in enumerate(atoms_df.iterrows()):
            # Get Rosetta atom type
            atom_dict = atom.to_dict()
            rosetta_type = self.atom_typer.get_atom_type(atom_dict)
            type_info = self.atom_typer.get_type_info(rosetta_type)
            
            # Store type index
            atom_types[i] = self.type_to_idx.get(rosetta_type, 0)
            
            # Coordinates
            coords[i] = torch.tensor([atom['x'], atom['y'], atom['z']], device=self.device)
            
            # Properties
            charges[i] = atom.get('formal_charge', type_info.charge)
            is_donor[i] = type_info.is_donor
            is_acceptor[i] = type_info.is_acceptor
            is_aromatic[i] = type_info.is_aromatic
            
            if is_protein:
                is_backbone[i] = atom.get('name', '') in ['N', 'CA', 'C', 'O', 'H']
                residue_ids[i] = atom.get('resSeq', i)
            else:
                residue_ids[i] = 0  # All ligand atoms in same "residue"
                
        # Build arrays object
        arrays = REF15GPUArrays(
            atom_types=atom_types,
            coords=coords,
            charges=charges,
            is_donor=is_donor,
            is_acceptor=is_acceptor,
            is_aromatic=is_aromatic,
            is_backbone=is_backbone,
            residue_ids=residue_ids,
            lj_radius_matrix=self.lj_radius_matrix,
            lj_wdepth_matrix=self.lj_wdepth_matrix,
            lk_dgfree=self.lk_dgfree_vec,
            lk_lambda=self.lk_lambda_vec,
            lk_volume=self.lk_volume_vec
        )
        
        # Calculate neighbor counts for burial
        arrays.neighbor_counts = self._calculate_neighbor_counts(coords)
        
        return arrays
        
    def _calculate_neighbor_counts(self, coords: torch.Tensor, radius: float = 5.2) -> torch.Tensor:
        """Calculate number of neighbors within radius for each atom"""
        # Distance matrix
        dist_matrix = torch.cdist(coords, coords)
        
        # Count neighbors (excluding self)
        within_radius = (dist_matrix < radius) & (dist_matrix > 0.1)
        neighbor_counts = within_radius.sum(dim=1)
        
        return neighbor_counts
        
    def calculate_energy_batch(self, 
                             protein_arrays: REF15GPUArrays,
                             ligand_arrays: REF15GPUArrays,
                             cutoff: float = 6.0) -> Dict[str, torch.Tensor]:
        """
        Calculate all REF15 energy components for protein-ligand interactions
        
        Returns dictionary with energy tensors for each component
        """
        # Get all pairwise distances
        dist_matrix = torch.cdist(protein_arrays.coords, ligand_arrays.coords)
        
        # Find pairs within cutoff
        mask = dist_matrix <= cutoff
        
        # Get indices of interacting pairs
        p_indices, l_indices = torch.where(mask)
        distances = dist_matrix[mask]
        
        if len(distances) == 0:
            return self._empty_energy_dict()
            
        # Get atom types for pairs
        p_types = protein_arrays.atom_types[p_indices]
        l_types = ligand_arrays.atom_types[l_indices]
        
        # Initialize energy components
        energy_components = {}
        
        # 1. Lennard-Jones (attractive + repulsive)
        e_atr, e_rep = self._calculate_lj_batch(p_types, l_types, distances)
        energy_components['fa_atr'] = e_atr * self.weights['fa_atr']
        energy_components['fa_rep'] = e_rep * self.weights['fa_rep']
        
        # 2. Solvation
        e_sol = self._calculate_solvation_batch(
            p_indices, l_indices, distances,
            protein_arrays, ligand_arrays
        )
        energy_components['fa_sol'] = e_sol * self.weights['fa_sol']
        
        # 3. Electrostatics
        p_charges = protein_arrays.charges[p_indices]
        l_charges = ligand_arrays.charges[l_indices]
        e_elec = self._calculate_electrostatics_batch(p_charges, l_charges, distances)
        energy_components['fa_elec'] = e_elec * self.weights['fa_elec']
        
        # 4. Hydrogen bonds (simplified without angles)
        e_hbond = self._calculate_hbond_batch(
            p_indices, l_indices, distances,
            protein_arrays, ligand_arrays
        )
        energy_components['hbond'] = e_hbond  # Weight applied internally
        
        # Store indices for reference
        energy_components['protein_indices'] = p_indices
        energy_components['ligand_indices'] = l_indices
        energy_components['distances'] = distances
        
        return energy_components
        
    def _calculate_lj_batch(self, type1_idx: torch.Tensor, type2_idx: torch.Tensor,
                           distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized Lennard-Jones calculation"""
        # Lookup parameters
        lj_radius = self.lj_radius_matrix[type1_idx, type2_idx]
        lj_wdepth = self.lj_wdepth_matrix[type1_idx, type2_idx]
        
        # Reduced distance
        x = distances / lj_radius
        
        # Initialize energy arrays
        e_atr = torch.zeros_like(distances)
        e_rep = torch.zeros_like(distances)
        
        # Region 1: x < 0.6 (linear repulsive)
        mask1 = x < 0.6
        e_rep[mask1] = 8.666 * (0.6 - x[mask1])
        
        # Region 2: 0.6 <= x < 0.67 (cubic transition)
        mask2 = (x >= 0.6) & (x < 0.67)
        dx = x[mask2] - 0.6
        e_rep[mask2] = -46.592 * dx**3 + 9.482 * dx**2 - 0.648 * dx
        
        # Region 3: x >= 0.67 (standard LJ)
        mask3 = x >= 0.67
        x3_vals = x[mask3]
        x6 = x3_vals**6
        x12 = x6**2
        
        # Repulsive (x < 1.0)
        rep_mask = mask3 & (x < 1.0)
        e_rep[rep_mask] = 1.0 / x12[x[mask3] < 1.0] - 2.0 / x6[x[mask3] < 1.0] + 1.0
        
        # Attractive (x < 2.0)
        atr_mask = mask3 & (x < 2.0)
        x_atr = x[atr_mask]
        x6_atr = x_atr**6
        x3_atr = x_atr**3
        e_atr_base = -2.0 * (1.0 / x6_atr - 1.0 / x3_atr)
        
        # Apply switching function
        switch = self._switch_function(distances[atr_mask], 5.5, 6.0)
        e_atr[atr_mask] = e_atr_base * switch * lj_wdepth[atr_mask]
        
        return e_atr, e_rep
        
    def _calculate_solvation_batch(self, p_indices: torch.Tensor, l_indices: torch.Tensor,
                                   distances: torch.Tensor, p_arrays: REF15GPUArrays,
                                   l_arrays: REF15GPUArrays) -> torch.Tensor:
        """Vectorized Lazaridis-Karplus solvation"""
        # Get solvation parameters
        p_types = p_arrays.atom_types[p_indices]
        l_types = l_arrays.atom_types[l_indices]
        
        dgfree_p = p_arrays.lk_dgfree[p_types]
        dgfree_l = l_arrays.lk_dgfree[l_types]
        lambda_p = p_arrays.lk_lambda[p_types]
        lambda_l = l_arrays.lk_lambda[l_types]
        volume_p = p_arrays.lk_volume[p_types]
        volume_l = l_arrays.lk_volume[l_types]
        
        # Gaussian exclusion
        gaussian_p = torch.exp(-(distances / lambda_l)**2)
        gaussian_l = torch.exp(-(distances / lambda_p)**2)
        
        # Burial functions
        if p_arrays.neighbor_counts is not None:
            burial_p = self._burial_function_gpu(p_arrays.neighbor_counts[p_indices])
            burial_l = self._burial_function_gpu(l_arrays.neighbor_counts[l_indices])
        else:
            burial_p = torch.ones_like(distances)
            burial_l = torch.ones_like(distances)
            
        # Solvation energy
        e_sol = (dgfree_p * burial_p * gaussian_l * (volume_l / (volume_p + 1e-6)) +
                 dgfree_l * burial_l * gaussian_p * (volume_p / (volume_l + 1e-6)))
        
        # Apply switching
        switch = self._switch_function(distances, 5.2, 5.5)
        
        return e_sol * switch
        
    def _burial_function_gpu(self, neighbor_counts: torch.Tensor) -> torch.Tensor:
        """GPU burial function"""
        params = self.params.burial_params
        
        # Linear interpolation
        frac = (neighbor_counts - params['burial_min']) / \
               (params['burial_max'] - params['burial_min'])
        frac = torch.clamp(frac, 0.0, 1.0)
        
        return 1.0 - frac
        
    def _calculate_electrostatics_batch(self, charges1: torch.Tensor, charges2: torch.Tensor,
                                       distances: torch.Tensor) -> torch.Tensor:
        """Vectorized electrostatic calculation"""
        # Skip if no charges
        charge_mask = (charges1 != 0) & (charges2 != 0)
        if not charge_mask.any():
            return torch.zeros_like(distances)
            
        e_elec = torch.zeros_like(distances)
        
        # Distance-dependent dielectric
        dielectric = self._sigmoidal_dielectric_gpu(distances[charge_mask])
        
        # Coulomb's law
        e_elec[charge_mask] = 332.0637 * charges1[charge_mask] * charges2[charge_mask] / \
                              (dielectric * distances[charge_mask])
        
        # Counterpair correction
        cp_mask = distances < 5.5
        cp_factor = distances[cp_mask] / 5.5
        e_elec[cp_mask] *= cp_factor
        
        # Apply switching
        switch = self._switch_function(distances, 5.5, 6.0)
        
        return e_elec * switch
        
    def _sigmoidal_dielectric_gpu(self, distances: torch.Tensor) -> torch.Tensor:
        """GPU sigmoidal dielectric function"""
        params = self.params.elec_params
        
        x = (distances - params['die_offset']) * params['die_slope']
        sigmoid = 1.0 / (1.0 + torch.exp(-x))
        
        die = params['die_min'] + (params['die_max'] - params['die_min']) * sigmoid
        
        return die
        
    def _calculate_hbond_batch(self, p_indices: torch.Tensor, l_indices: torch.Tensor,
                               distances: torch.Tensor, p_arrays: REF15GPUArrays,
                               l_arrays: REF15GPUArrays) -> torch.Tensor:
        """Simplified H-bond calculation without full geometry"""
        # Check donor/acceptor pairs
        p_donor = p_arrays.is_donor[p_indices]
        p_acceptor = p_arrays.is_acceptor[p_indices]
        l_donor = l_arrays.is_donor[l_indices]
        l_acceptor = l_arrays.is_acceptor[l_indices]
        
        # Valid H-bond pairs
        hbond_mask = ((p_donor & l_acceptor) | (l_donor & p_acceptor)) & (distances <= 3.3)
        
        if not hbond_mask.any():
            return torch.zeros_like(distances)
            
        e_hbond = torch.zeros_like(distances)
        
        # Polynomial distance dependence
        r = distances[hbond_mask]
        poly = self.hbond_poly
        e_dist = poly[0] + poly[1]*r + poly[2]*r**2 + poly[3]*r**3 + poly[4]*r**4
        
        # Simplified angular factor
        angular_factor = 0.8
        
        # Apply switching
        switch = self._switch_function(r, 1.9, 3.3)
        
        e_hbond[hbond_mask] = e_dist * switch * angular_factor
        
        # Weight based on backbone/sidechain
        # Simplified - would need to track H-bond types
        weight = self.weights['hbond_sc']
        
        return e_hbond * weight
        
    def _switch_function(self, distance: torch.Tensor, switch_dis: float, 
                        cutoff_dis: float) -> torch.Tensor:
        """GPU switching function"""
        switch = torch.ones_like(distance)
        
        mask = (distance > switch_dis) & (distance < cutoff_dis)
        x = (distance[mask] - switch_dis) / (cutoff_dis - switch_dis)
        switch[mask] = 1.0 - 3.0 * x**2 + 2.0 * x**3
        
        switch[distance >= cutoff_dis] = 0.0
        
        return switch
        
    def _empty_energy_dict(self) -> Dict[str, torch.Tensor]:
        """Return empty energy dictionary"""
        return {
            'fa_atr': torch.tensor(0.0, device=self.device),
            'fa_rep': torch.tensor(0.0, device=self.device),
            'fa_sol': torch.tensor(0.0, device=self.device),
            'fa_elec': torch.tensor(0.0, device=self.device),
            'hbond': torch.tensor(0.0, device=self.device),
            'protein_indices': torch.tensor([], dtype=torch.long, device=self.device),
            'ligand_indices': torch.tensor([], dtype=torch.long, device=self.device),
            'distances': torch.tensor([], device=self.device)
        }
        
    def calculate_total_energy(self, energy_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum all energy components"""
        total = torch.tensor(0.0, device=self.device)
        
        for term in ['fa_atr', 'fa_rep', 'fa_sol', 'fa_elec', 'hbond']:
            if term in energy_components:
                total += energy_components[term].sum()
                
        return total
        
    def get_residue_energies(self, energy_components: Dict[str, torch.Tensor],
                            protein_arrays: REF15GPUArrays) -> Dict[int, float]:
        """Aggregate energies by protein residue"""
        residue_energies = {}
        
        if len(energy_components['protein_indices']) == 0:
            return residue_energies
            
        # Get residue IDs for interacting atoms
        p_indices = energy_components['protein_indices']
        res_ids = protein_arrays.residue_ids[p_indices]
        
        # Sum energies for all terms
        total_energies = torch.zeros_like(energy_components['distances'])
        
        for term in ['fa_atr', 'fa_rep', 'fa_sol', 'fa_elec', 'hbond']:
            if term in energy_components and len(energy_components[term]) == len(total_energies):
                total_energies += energy_components[term]
                
        # Aggregate by residue
        unique_res = torch.unique(res_ids)
        
        for res_id in unique_res:
            mask = res_ids == res_id
            res_energy = total_energies[mask].sum().item()
            residue_energies[res_id.item()] = res_energy
            
        return residue_energies


# Utility functions
def get_ref15_gpu_calculator(device: torch.device) -> REF15GPUCalculator:
    """Get or create REF15 GPU calculator for device"""
    return REF15GPUCalculator(device)