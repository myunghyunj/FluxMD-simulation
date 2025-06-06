#!/usr/bin/env python3
"""
GPU-Accelerated Interaction Calculator for FluxMD.
This module is responsible for calculating raw interaction data on the GPU.
The aggregation and flux calculation is handled by the FluxAnalyzer.

Optimized for Unified Memory Architecture (UMA) - keeps everything on GPU.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import platform
from dataclasses import dataclass
from protonation_aware_interactions import ProtonationAwareInteractionDetector

def get_device():
    """Detects and returns the best available PyTorch device."""
    if platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🚀 Apple Silicon GPU (Metal Performance Shaders) detected!")
        print("   Unified Memory Architecture - zero-copy data transfer")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"🚀 NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("💻 Using CPU (GPU not available).")
    return torch.device("cpu")

@dataclass
class InteractionResult:
    """A dataclass to hold raw interaction tensors on the GPU."""
    protein_indices: torch.Tensor
    residue_ids: torch.Tensor
    inter_vectors: torch.Tensor
    energies: torch.Tensor

class GPUAcceleratedInteractionCalculator:
    """Calculates non-covalent interactions for a given state on the GPU."""

    def __init__(self, device=None, physiological_pH=7.4):
        self.device = device or get_device()
        self.physiological_pH = physiological_pH
        self.protonation_detector = ProtonationAwareInteractionDetector(pH=self.physiological_pH)
        
        # Interaction cutoffs
        self.cutoffs = {
            'hbond': 3.5,
            'salt_bridge': 5.0,
            'pi_pi': 4.5,
            'pi_cation': 6.0,
            'vdw': 5.0
        }
        
        self.protein_properties = None
        self.intra_protein_vectors_gpu = None

    def set_intra_protein_vectors(self, intra_vectors_dict):
        """Store pre-computed intra-protein force field vectors on GPU."""
        if self.protein_properties and intra_vectors_dict:
            n_atoms = len(self.protein_properties['coords'])
            self.intra_protein_vectors_gpu = torch.zeros((n_atoms, 3), device=self.device, dtype=torch.float32)
            
            for i in range(n_atoms):
                res_id = self.protein_properties['residue_ids'][i].item()
                res_key = f"A:{res_id}"  # Assuming chain A
                if res_key in intra_vectors_dict:
                    vector = intra_vectors_dict[res_key]
                    self.intra_protein_vectors_gpu[i] = torch.tensor(vector, device=self.device, dtype=torch.float32)
            
            print(f"   ✓ Loaded intra-protein vectors for {len(intra_vectors_dict)} residues onto GPU")

    def precompute_protein_properties(self, protein_atoms: pd.DataFrame):
        """Pre-computes protein properties and stores them on the GPU."""
        n_atoms = len(protein_atoms)
        print(f"   Pre-computing properties for {n_atoms} protein atoms on {self.device}...")
        
        properties = {
            'coords': torch.tensor(protein_atoms[['x', 'y', 'z']].values, device=self.device, dtype=torch.float32),
            'residue_ids': torch.tensor(protein_atoms.get('residue_id', protein_atoms.get('resSeq', range(n_atoms))).values, 
                                       device=self.device, dtype=torch.long),
            'is_donor': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_acceptor': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_charged_pos': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_charged_neg': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_aromatic': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
        }

        # Process each atom for properties
        for i, (_, atom) in enumerate(protein_atoms.iterrows()):
            atom_dict = {
                'resname': atom['resname'],
                'name': atom['name'],
                'element': atom.get('element', atom['name'][0]).upper(),
                'x': atom['x'],
                'y': atom['y'],
                'z': atom['z'],
                'chain': atom.get('chain', 'A'),
                'resSeq': atom.get('residue_id', atom.get('resSeq', i)),
                'atom_id': i
            }
            
            pa_atom = self.protonation_detector.determine_atom_protonation(atom_dict)
            properties['is_donor'][i] = pa_atom.can_donate_hbond
            properties['is_acceptor'][i] = pa_atom.can_accept_hbond
            properties['is_charged_pos'][i] = pa_atom.formal_charge > 0
            properties['is_charged_neg'][i] = pa_atom.formal_charge < 0
            
            # Mark aromatic residues
            if atom['resname'] in ['PHE', 'TYR', 'TRP', 'HIS']:
                properties['is_aromatic'][i] = True
            
        self.protein_properties = properties
        print("   ✓ Protein properties pre-computed and stored on GPU")

    def precompute_ligand_properties(self, ligand_atoms: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Pre-computes ligand properties and returns them as GPU tensors."""
        n_atoms = len(ligand_atoms)
        
        properties = {
            'coords': torch.tensor(ligand_atoms[['x', 'y', 'z']].values, device=self.device, dtype=torch.float32),
            'is_donor': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_acceptor': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_charged_pos': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_charged_neg': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            'is_aromatic': torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
        }
        
        for i, (_, atom) in enumerate(ligand_atoms.iterrows()):
            element = str(atom.get('element', '')).strip().upper()
            if not element and atom.get('name'):
                element = atom['name'][0].upper()
            
            atom_dict = {
                'name': atom.get('name', ''),
                'element': element,
                'x': atom['x'],
                'y': atom['y'],
                'z': atom['z'],
                'atom_id': i
            }
            
            pa_atom = self.protonation_detector.process_ligand_atom(atom_dict)
            properties['is_donor'][i] = pa_atom.can_donate_hbond
            properties['is_acceptor'][i] = pa_atom.can_accept_hbond
            properties['is_charged_pos'][i] = pa_atom.formal_charge > 0
            properties['is_charged_neg'][i] = pa_atom.formal_charge < 0
            
            # Mark potential aromatic atoms
            if element in ['C', 'N', 'O', 'S']:
                properties['is_aromatic'][i] = True
        
        return properties

    def calculate_interactions_for_frame(self, ligand_coords_gpu: torch.Tensor, 
                                       ligand_properties: Dict[str, torch.Tensor]) -> Optional[InteractionResult]:
        """Calculates all interactions for a single frame (ligand position) entirely on the GPU."""
        max_dist = max(self.cutoffs.values())
        
        # Calculate distance matrix
        dist_matrix = torch.cdist(self.protein_properties['coords'], ligand_coords_gpu)
        
        # Find interacting pairs
        interacting_pairs = torch.nonzero(dist_matrix <= max_dist, as_tuple=False)
        
        if interacting_pairs.shape[0] == 0:
            return None

        p_idx, l_idx = interacting_pairs[:, 0], interacting_pairs[:, 1]
        distances = dist_matrix[p_idx, l_idx]

        # Extract properties for interacting atoms
        p_donor = self.protein_properties['is_donor'][p_idx]
        p_acceptor = self.protein_properties['is_acceptor'][p_idx]
        p_pos = self.protein_properties['is_charged_pos'][p_idx]
        p_neg = self.protein_properties['is_charged_neg'][p_idx]
        p_aromatic = self.protein_properties['is_aromatic'][p_idx]
        
        l_donor = ligand_properties['is_donor'][l_idx]
        l_acceptor = ligand_properties['is_acceptor'][l_idx]
        l_pos = ligand_properties['is_charged_pos'][l_idx]
        l_neg = ligand_properties['is_charged_neg'][l_idx]
        l_aromatic = ligand_properties['is_aromatic'][l_idx]

        # Detect interaction types
        hbond_mask = ((p_donor & l_acceptor) | (p_acceptor & l_donor)) & (distances <= self.cutoffs['hbond'])
        salt_mask = ((p_pos & l_neg) | (p_neg & l_pos)) & (distances <= self.cutoffs['salt_bridge'])
        pi_pi_mask = (p_aromatic & l_aromatic) & (distances <= self.cutoffs['pi_pi'])
        pi_cation_mask = ((p_aromatic & l_pos) | (p_pos & l_aromatic)) & (distances <= self.cutoffs['pi_cation'])
        
        # Calculate energies
        energies = torch.zeros_like(distances)
        
        # H-bonds
        if hbond_mask.any():
            d0 = 2.8
            r = distances[hbond_mask]
            energies[hbond_mask] = -5.0 * (5*(d0/r)**12 - 6*(d0/r)**10)
            energies[hbond_mask] = torch.clamp(energies[hbond_mask], -8.0, -0.5)
        
        # Salt bridges
        if salt_mask.any():
            r = distances[salt_mask]
            energies[salt_mask] = -332.0 / (4.0 * r * r)
            energies[salt_mask] = torch.clamp(energies[salt_mask], -10.0, -1.0)
        
        # Pi-pi stacking
        if pi_pi_mask.any():
            r = distances[pi_pi_mask]
            energies[pi_pi_mask] = -4.0 * torch.exp(-((r - 3.8) / 1.5)**2)
        
        # Pi-cation
        if pi_cation_mask.any():
            r = distances[pi_cation_mask]
            energies[pi_cation_mask] = -3.0 * torch.exp(-((r - 3.5) / 1.5)**2)
        
        # VDW for remaining close contacts
        vdw_mask = (distances <= self.cutoffs['vdw']) & (energies == 0)
        if vdw_mask.any():
            sigma = 3.4
            epsilon = 0.238
            r = distances[vdw_mask]
            sigma_over_r = sigma / r
            energies[vdw_mask] = 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)
            energies[vdw_mask] = torch.clamp(energies[vdw_mask], -10, 10)

        # Filter out non-interacting pairs
        active_mask = energies != 0
        if not active_mask.any():
            return None
            
        # Calculate interaction vectors (ligand -> protein direction)
        inter_vectors = ligand_coords_gpu[l_idx[active_mask]] - self.protein_properties['coords'][p_idx[active_mask]]

        return InteractionResult(
            protein_indices=p_idx[active_mask],
            residue_ids=self.protein_properties['residue_ids'][p_idx[active_mask]],
            inter_vectors=inter_vectors,
            energies=energies[active_mask]
        )

    def process_trajectory_batch(self, trajectory: np.ndarray, ligand_base_coords: np.ndarray,
                               n_rotations: int = 36) -> List[InteractionResult]:
        """Process a trajectory batch, returning raw GPU interaction results."""
        trajectory_gpu = torch.tensor(trajectory, device=self.device, dtype=torch.float32)
        ligand_base_gpu = torch.tensor(ligand_base_coords, device=self.device, dtype=torch.float32)
        ligand_center = ligand_base_gpu.mean(dim=0)
        
        # Pre-generate rotation matrices
        angles = torch.linspace(0, 2*np.pi, n_rotations, device=self.device)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        rotation_matrices = torch.zeros((n_rotations, 3, 3), device=self.device)
        rotation_matrices[:, 0, 0] = cos_angles
        rotation_matrices[:, 0, 1] = -sin_angles
        rotation_matrices[:, 1, 0] = sin_angles
        rotation_matrices[:, 1, 1] = cos_angles
        rotation_matrices[:, 2, 2] = 1.0
        
        # Center ligand
        ligand_centered = ligand_base_gpu - ligand_center
        
        # Pre-compute base ligand properties
        ligand_atoms_df = pd.DataFrame({
            'x': ligand_base_coords[:, 0],
            'y': ligand_base_coords[:, 1],
            'z': ligand_base_coords[:, 2],
            'element': ['C'] * len(ligand_base_coords),  # Placeholder
            'name': [f'L{i}' for i in range(len(ligand_base_coords))]
        })
        base_ligand_properties = self.precompute_ligand_properties(ligand_atoms_df)
        
        all_results = []
        
        for position in trajectory_gpu:
            # Apply all rotations at once
            rotated_ligands = torch.matmul(ligand_centered.unsqueeze(0), rotation_matrices.transpose(1, 2))
            transformed_ligands = rotated_ligands + position.unsqueeze(0).unsqueeze(0)
            
            # Find best rotation
            best_energy = float('inf')
            best_result = None
            
            for rot_idx in range(n_rotations):
                # Update ligand coordinates
                current_ligand_properties = base_ligand_properties.copy()
                current_ligand_properties['coords'] = transformed_ligands[rot_idx]
                
                # Calculate interactions
                result = self.calculate_interactions_for_frame(
                    transformed_ligands[rot_idx], 
                    current_ligand_properties
                )
                
                if result is not None and len(result.energies) > 0:
                    total_energy = result.energies.sum().item()
                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_result = result
            
            if best_result is not None:
                all_results.append(best_result)
        
        return all_results