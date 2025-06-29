#!/usr/bin/env python3
"""
GPU-Accelerated Interaction Calculator for FluxMD.
This module is responsible for calculating raw interaction data on the GPU.
The aggregation and flux calculation is handled by the FluxAnalyzer.

Optimized for Unified Memory Architecture (UMA) - keeps everything on GPU.
FIXED: Now uses the same energy functions as the reference implementation.
"""

import platform
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from fluxmd.core.energy_config import ENERGY_BOUNDS

from ..core.protonation_aware_interactions import ProtonationAwareInteractionDetector


def get_device():
    """Detects and returns the best available PyTorch device."""
    if (
        platform.system() == "Darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
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
    interaction_types: torch.Tensor  # New field for tracking interaction types

    # Interaction type constants
    HBOND = 0
    SALT_BRIDGE = 1
    PI_PI = 2
    PI_CATION = 3
    VDW = 4

    @staticmethod
    def get_interaction_name(itype: int) -> str:
        """Get human-readable name for interaction type."""
        names = {0: "H-bond", 1: "Salt bridge", 2: "Pi-pi", 3: "Pi-cation", 4: "VDW"}
        return names.get(itype, "Unknown")


class GPUAcceleratedInteractionCalculator:
    """Calculates non-covalent interactions for a given state on the GPU."""

    def __init__(self, device=None, physiological_pH=7.4, target_is_dna=False):
        self.device = device or get_device()
        self.physiological_pH = physiological_pH
        self.target_is_dna = target_is_dna
        self.protonation_detector = ProtonationAwareInteractionDetector(pH=self.physiological_pH)

        # Interaction cutoffs
        self.cutoffs = {
            "hbond": 3.5,
            "salt_bridge": 5.0,
            "pi_pi": 4.5,
            "pi_cation": 6.0,
            "vdw": 5.0,
        }

        self.protein_properties = None
        self.target_properties = None  # Generic name for protein or DNA
        self.intra_protein_vectors_gpu = None

        # DNA-specific properties
        if self.target_is_dna:
            self._init_dna_properties()

    def _init_dna_properties(self):
        """Initialize DNA-specific residue properties."""
        # DNA base properties for interaction detection
        self.DNA_DONORS = {
            "DA": {"N6"},  # Adenine amino group
            "DG": {"N1", "N2"},  # Guanine amino and imino groups
            "DC": {"N4"},  # Cytosine amino group
            "DT": set(),  # Thymine has no H-bond donors
        }

        self.DNA_ACCEPTORS = {
            "DA": {"N1", "N3", "N7"},  # Adenine nitrogens
            "DG": {"O6", "N3", "N7"},  # Guanine carbonyl and nitrogens
            "DC": {"O2", "N3"},  # Cytosine carbonyl and nitrogen
            "DT": {"O2", "O4", "N3"},  # Thymine carbonyls and nitrogen
        }

        # All DNA bases have aromatic rings
        self.DNA_AROMATIC = {"DA", "DG", "DC", "DT"}

        # Phosphate backbone atoms (always acceptors due to negative charge)
        self.DNA_BACKBONE_ACCEPTORS = {"OP1", "OP2", "O1P", "O2P", "O3'", "O5'"}

    def set_intra_protein_vectors(self, intra_vectors_dict):
        """Store pre-computed intra-protein force field vectors on GPU."""
        if self.protein_properties and intra_vectors_dict:
            n_atoms = len(self.protein_properties["coords"])
            self.intra_protein_vectors_gpu = torch.zeros(
                (n_atoms, 3), device=self.device, dtype=torch.float32
            )

            for i in range(n_atoms):
                res_id = self.protein_properties["residue_ids"][i].item()
                res_key = f"A:{res_id}"  # Assuming chain A
                if res_key in intra_vectors_dict:
                    vector = intra_vectors_dict[res_key]
                    self.intra_protein_vectors_gpu[i] = torch.tensor(
                        vector, device=self.device, dtype=torch.float32
                    )

            print(
                f"   ✓ Loaded intra-protein vectors for {len(intra_vectors_dict)} residues onto GPU"
            )

    def precompute_target_properties_gpu(self, target_atoms: pd.DataFrame):
        """Pre-computes target molecule properties (protein or DNA) and stores them on the GPU."""
        if self.target_is_dna:
            self._precompute_dna_properties(target_atoms)
        else:
            self.precompute_protein_properties(target_atoms)

    def precompute_mobile_properties_gpu(self, mobile_atoms: pd.DataFrame):
        """Pre-computes mobile molecule properties (ligand or protein) and returns them as GPU tensors."""
        if self.target_is_dna:
            # When target is DNA, mobile is protein
            return self._precompute_protein_properties_for_mobile(mobile_atoms)
        else:
            # When target is protein, mobile is ligand
            return self.precompute_ligand_properties(mobile_atoms)

    def _precompute_dna_properties(self, dna_atoms: pd.DataFrame):
        """Pre-computes DNA properties and stores them on the GPU."""
        n_atoms = len(dna_atoms)
        print(f"   Pre-computing properties for {n_atoms} DNA atoms on {self.device}...")

        properties = {
            "coords": torch.tensor(
                dna_atoms[["x", "y", "z"]].values, device=self.device, dtype=torch.float32
            ),
            "residue_ids": torch.tensor(
                dna_atoms.get("residue_id", dna_atoms.get("resSeq", range(n_atoms))).values,
                device=self.device,
                dtype=torch.long,
            ),
            "is_donor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_acceptor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_pos": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_neg": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_aromatic": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "formal_charges": torch.zeros(n_atoms, device=self.device, dtype=torch.float32),
        }

        # Process each atom for DNA-specific properties
        for i, (_, atom) in enumerate(dna_atoms.iterrows()):
            resname = atom["resname"]
            atom_name = atom["name"]

            # Check if it's a DNA base
            if resname in self.DNA_AROMATIC:
                # Mark aromatic for all base atoms
                properties["is_aromatic"][i] = True

                # Check donors
                if atom_name in self.DNA_DONORS.get(resname, set()):
                    properties["is_donor"][i] = True

                # Check acceptors
                if atom_name in self.DNA_ACCEPTORS.get(resname, set()):
                    properties["is_acceptor"][i] = True

            # Check backbone atoms
            if atom_name in self.DNA_BACKBONE_ACCEPTORS:
                properties["is_acceptor"][i] = True

            # Phosphate oxygens are negatively charged
            if atom_name in {"OP1", "OP2", "O1P", "O2P"}:
                properties["is_charged_neg"][i] = True
                properties["formal_charges"][i] = -0.5  # Each oxygen carries partial charge

        self.target_properties = properties
        self.protein_properties = properties  # Keep for compatibility
        print("   ✓ DNA properties pre-computed and stored on GPU")

        # Debug output
        print("\n   DNA Donor/Acceptor Statistics:")
        print(f"     Total donors: {properties['is_donor'].sum().item()}")
        print(f"     Total acceptors: {properties['is_acceptor'].sum().item()}")
        print(f"     Charged negative (phosphates): {properties['is_charged_neg'].sum().item()}")
        print(f"     Aromatic atoms: {properties['is_aromatic'].sum().item()}")

    def _precompute_protein_properties_for_mobile(
        self, protein_atoms: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """Pre-computes protein properties when protein is the mobile molecule (DNA is target)."""
        # Reuse the existing protein property computation logic
        # but return as a dictionary instead of storing
        n_atoms = len(protein_atoms)
        print(
            f"   Pre-computing properties for {n_atoms} protein atoms (mobile) on {self.device}..."
        )

        properties = {
            "coords": torch.tensor(
                protein_atoms[["x", "y", "z"]].values, device=self.device, dtype=torch.float32
            ),
            "is_donor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_acceptor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_pos": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_neg": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_aromatic": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "formal_charges": torch.zeros(n_atoms, device=self.device, dtype=torch.float32),
        }

        # Process each atom for properties
        for i, (_, atom) in enumerate(protein_atoms.iterrows()):
            atom_dict = {
                "resname": atom["resname"],
                "name": atom["name"],
                "element": atom.get("element", atom["name"][0]).upper(),
                "x": atom["x"],
                "y": atom["y"],
                "z": atom["z"],
                "chain": atom.get("chain", "A"),
                "resSeq": atom.get("residue_id", atom.get("resSeq", i)),
                "atom_id": i,
            }

            pa_atom = self.protonation_detector.determine_atom_protonation(atom_dict)
            properties["is_donor"][i] = pa_atom.can_donate_hbond
            properties["is_acceptor"][i] = pa_atom.can_accept_hbond
            properties["is_charged_pos"][i] = pa_atom.formal_charge > 0
            properties["is_charged_neg"][i] = pa_atom.formal_charge < 0
            properties["formal_charges"][i] = pa_atom.formal_charge

            # Mark aromatic residues
            if atom["resname"] in ["PHE", "TYR", "TRP", "HIS"]:
                properties["is_aromatic"][i] = True

        print("   ✓ Protein properties (mobile) pre-computed")
        return properties

    def precompute_protein_properties(self, protein_atoms: pd.DataFrame):
        """Pre-computes protein properties and stores them on the GPU."""
        n_atoms = len(protein_atoms)
        print(f"   Pre-computing properties for {n_atoms} protein atoms on {self.device}...")

        properties = {
            "coords": torch.tensor(
                protein_atoms[["x", "y", "z"]].values, device=self.device, dtype=torch.float32
            ),
            "residue_ids": torch.tensor(
                protein_atoms.get("residue_id", protein_atoms.get("resSeq", range(n_atoms))).values,
                device=self.device,
                dtype=torch.long,
            ),
            "is_donor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_acceptor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_pos": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_neg": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_aromatic": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "formal_charges": torch.zeros(
                n_atoms, device=self.device, dtype=torch.float32
            ),  # NEW: Track actual charges
        }

        # Process each atom for properties
        for i, (_, atom) in enumerate(protein_atoms.iterrows()):
            atom_dict = {
                "resname": atom["resname"],
                "name": atom["name"],
                "element": atom.get("element", atom["name"][0]).upper(),
                "x": atom["x"],
                "y": atom["y"],
                "z": atom["z"],
                "chain": atom.get("chain", "A"),
                "resSeq": atom.get("residue_id", atom.get("resSeq", i)),
                "atom_id": i,
            }

            pa_atom = self.protonation_detector.determine_atom_protonation(atom_dict)
            properties["is_donor"][i] = pa_atom.can_donate_hbond
            properties["is_acceptor"][i] = pa_atom.can_accept_hbond
            properties["is_charged_pos"][i] = pa_atom.formal_charge > 0
            properties["is_charged_neg"][i] = pa_atom.formal_charge < 0
            properties["formal_charges"][i] = pa_atom.formal_charge  # Store actual charge value

            # Mark aromatic residues
            if atom["resname"] in ["PHE", "TYR", "TRP", "HIS"]:
                properties["is_aromatic"][i] = True

        self.protein_properties = properties
        print("   ✓ Protein properties pre-computed and stored on GPU")

        # FIXED: Add debug output for donor/acceptor detection
        print("\n   Donor/Acceptor Statistics:")
        print(f"     Total donors: {properties['is_donor'].sum().item()}")
        print(f"     Total acceptors: {properties['is_acceptor'].sum().item()}")
        print(f"     Charged positive: {properties['is_charged_pos'].sum().item()}")
        print(f"     Charged negative: {properties['is_charged_neg'].sum().item()}")
        print(f"     Aromatic atoms: {properties['is_aromatic'].sum().item()}")

    def precompute_ligand_properties(self, ligand_atoms: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Pre-computes ligand properties and returns them as GPU tensors."""
        n_atoms = len(ligand_atoms)

        # FIXED: Debug element detection
        print("\n   Ligand element detection:")
        element_counts = {}

        properties = {
            "coords": torch.tensor(
                ligand_atoms[["x", "y", "z"]].values, device=self.device, dtype=torch.float32
            ),
            "is_donor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_acceptor": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_pos": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_charged_neg": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "is_aromatic": torch.zeros(n_atoms, device=self.device, dtype=torch.bool),
            "formal_charges": torch.zeros(
                n_atoms, device=self.device, dtype=torch.float32
            ),  # NEW: Track actual charges
        }

        for i, (_, atom) in enumerate(ligand_atoms.iterrows()):
            # FIXED: Properly handle element extraction
            element = atom.get("element", "")
            if element:
                element = str(element).strip().upper()
            else:
                # If element is missing, check atom name
                atom_name = str(atom.get("name", "")).strip()
                if atom_name in ["N", "O", "S", "P", "F", "Cl", "Br", "I"]:
                    element = atom_name.upper()
                elif atom_name.startswith("CL"):
                    element = "CL"
                elif atom_name.startswith("BR"):
                    element = "BR"
                elif atom_name and atom_name[0] in ["C", "N", "O", "S", "P", "H", "F"]:
                    element = atom_name[0].upper()
                else:
                    element = "C"  # Default
                    print(
                        f"WARNING: Could not determine element for atom {i} with name '{atom_name}', defaulting to C"
                    )

            # Track element counts for debug
            element_counts[element] = element_counts.get(element, 0) + 1

            atom_dict = {
                "name": atom.get("name", ""),
                "element": element,
                "x": atom["x"],
                "y": atom["y"],
                "z": atom["z"],
                "atom_id": i,
            }

            pa_atom = self.protonation_detector.process_ligand_atom(atom_dict)
            properties["is_donor"][i] = pa_atom.can_donate_hbond
            properties["is_acceptor"][i] = pa_atom.can_accept_hbond
            properties["is_charged_pos"][i] = pa_atom.formal_charge > 0
            properties["is_charged_neg"][i] = pa_atom.formal_charge < 0
            properties["formal_charges"][i] = pa_atom.formal_charge  # Store actual charge value

            # Mark potential aromatic atoms
            # FIXED: Don't mark ALL C,N,O,S as aromatic
            # Only mark if specifically tagged or in aromatic residues
            # For now, leave aromatic detection to PDBQT atom types or explicit marking
            if atom.get("is_aromatic", False):
                properties["is_aromatic"][i] = True

        # Print element counts
        print(f"     Elements found: {dict(sorted(element_counts.items()))}")

        # FIXED: Add debug output for ligand donor/acceptor detection
        print("\n   Ligand Donor/Acceptor Statistics:")
        print(f"     Total donors: {properties['is_donor'].sum().item()}")
        print(f"     Total acceptors: {properties['is_acceptor'].sum().item()}")
        print(f"     Charged positive: {properties['is_charged_pos'].sum().item()}")
        print(f"     Charged negative: {properties['is_charged_neg'].sum().item()}")

        return properties

    def calculate_interactions_for_frame(
        self, ligand_coords_gpu: torch.Tensor, ligand_properties: Dict[str, torch.Tensor]
    ) -> Optional[InteractionResult]:
        """Calculates all interactions for a single frame (ligand position) entirely on the GPU."""
        max_dist = max(self.cutoffs.values())

        # Calculate distance matrix
        dist_matrix = torch.cdist(self.protein_properties["coords"], ligand_coords_gpu)

        # Find interacting pairs
        interacting_pairs = torch.nonzero(dist_matrix <= max_dist, as_tuple=False)

        if interacting_pairs.shape[0] == 0:
            return None

        p_idx, l_idx = interacting_pairs[:, 0], interacting_pairs[:, 1]
        distances = dist_matrix[p_idx, l_idx]

        # Extract properties for interacting atoms
        p_donor = self.protein_properties["is_donor"][p_idx]
        p_acceptor = self.protein_properties["is_acceptor"][p_idx]
        p_pos = self.protein_properties["is_charged_pos"][p_idx]
        p_neg = self.protein_properties["is_charged_neg"][p_idx]
        p_aromatic = self.protein_properties["is_aromatic"][p_idx]
        p_charges = self.protein_properties["formal_charges"][p_idx]  # Get actual charges

        l_donor = ligand_properties["is_donor"][l_idx]
        l_acceptor = ligand_properties["is_acceptor"][l_idx]
        l_pos = ligand_properties["is_charged_pos"][l_idx]
        l_neg = ligand_properties["is_charged_neg"][l_idx]
        l_aromatic = ligand_properties["is_aromatic"][l_idx]
        l_charges = ligand_properties["formal_charges"][l_idx]  # Get actual charges

        # Detect interaction types and initialize type tracking
        hbond_mask = ((p_donor & l_acceptor) | (p_acceptor & l_donor)) & (
            distances <= self.cutoffs["hbond"]
        )
        salt_mask = ((p_pos & l_neg) | (p_neg & l_pos)) & (distances <= self.cutoffs["salt_bridge"])
        pi_pi_mask = (p_aromatic & l_aromatic) & (distances <= self.cutoffs["pi_pi"])
        pi_cation_mask = ((p_aromatic & l_pos) | (p_pos & l_aromatic)) & (
            distances <= self.cutoffs["pi_cation"]
        )

        # Initialize interaction type tensor
        interaction_types = torch.full_like(
            distances, -1, dtype=torch.int8
        )  # -1 for no interaction

        # Calculate energies
        energies = torch.zeros_like(distances)

        # H-bonds - FIXED: Now using Gaussian potential like reference implementation
        if hbond_mask.any():
            interaction_types[hbond_mask] = InteractionResult.HBOND
            r = distances[hbond_mask]
            # Gaussian potential: -5.0 * exp(-((distance - 2.8) / 0.5)^2)
            energies[hbond_mask] = -5.0 * torch.exp(-(((r - 2.8) / 0.5) ** 2))

            # Apply 1.5x multiplier for charged groups (matching reference)
            charged_hbond_mask = hbond_mask & ((p_charges != 0) | (l_charges != 0))
            if charged_hbond_mask.any():
                energies[charged_hbond_mask] *= 1.5

        # Salt bridges - FIXED: Now includes actual charge magnitudes
        if salt_mask.any():
            interaction_types[salt_mask] = InteractionResult.SALT_BRIDGE
            r = distances[salt_mask]
            # Include actual charge magnitudes in calculation
            charge_products = torch.abs(p_charges[salt_mask] * l_charges[salt_mask])
            energies[salt_mask] = -332.0 * charge_products / (4.0 * r * r)
            energies[salt_mask] = torch.clamp(
                energies[salt_mask],
                ENERGY_BOUNDS["salt_bridge"]["min"],
                ENERGY_BOUNDS["salt_bridge"]["max"],
            )

        # Pi-pi stacking - keeping as is (already uses Gaussian-like form)
        if pi_pi_mask.any():
            interaction_types[pi_pi_mask] = InteractionResult.PI_PI
            r = distances[pi_pi_mask]
            energies[pi_pi_mask] = -4.0 * torch.exp(-(((r - 3.8) / 1.5) ** 2))

        # Pi-cation - keeping as is (already uses Gaussian-like form)
        if pi_cation_mask.any():
            interaction_types[pi_cation_mask] = InteractionResult.PI_CATION
            r = distances[pi_cation_mask]
            energies[pi_cation_mask] = -3.0 * torch.exp(-(((r - 3.5) / 1.5) ** 2))

        # VDW for remaining close contacts
        vdw_mask = (distances <= self.cutoffs["vdw"]) & (energies == 0)
        if vdw_mask.any():
            interaction_types[vdw_mask] = InteractionResult.VDW
            sigma = 3.4
            epsilon = 0.238
            r = distances[vdw_mask]
            sigma_over_r = sigma / r
            r6 = (sigma_over_r) ** 6
            vdw_energies = 4 * epsilon * (r6**2 - r6)
            energies[vdw_mask] = torch.clamp(
                vdw_energies, ENERGY_BOUNDS["vdw"]["min"], ENERGY_BOUNDS["vdw"]["max"]
            )

        # Filter out non-interacting pairs
        active_mask = energies != 0
        if not active_mask.any():
            return None

        # Calculate interaction vectors (ligand -> protein direction)
        inter_vectors = (
            ligand_coords_gpu[l_idx[active_mask]]
            - self.protein_properties["coords"][p_idx[active_mask]]
        )

        return InteractionResult(
            protein_indices=p_idx[active_mask],
            residue_ids=self.protein_properties["residue_ids"][p_idx[active_mask]],
            inter_vectors=inter_vectors,
            energies=energies[active_mask],
            interaction_types=interaction_types[active_mask],
        )

    def process_trajectory_batch(
        self,
        trajectory: np.ndarray,
        ligand_base_coords: np.ndarray,
        n_rotations: int = 36,
        ligand_atoms_df: pd.DataFrame = None,
    ) -> List[InteractionResult]:
        """Process a trajectory batch, returning raw GPU interaction results."""
        trajectory_gpu = torch.tensor(trajectory, device=self.device, dtype=torch.float32)
        ligand_base_gpu = torch.tensor(ligand_base_coords, device=self.device, dtype=torch.float32)
        ligand_center = ligand_base_gpu.mean(dim=0)

        # Pre-generate rotation matrices
        angles = torch.linspace(0, 2 * np.pi, n_rotations, device=self.device)
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
        if ligand_atoms_df is None:
            # Fallback if no dataframe provided (shouldn't happen)
            print("WARNING: No ligand dataframe provided, using placeholder elements")
            ligand_atoms_df = pd.DataFrame(
                {
                    "x": ligand_base_coords[:, 0],
                    "y": ligand_base_coords[:, 1],
                    "z": ligand_base_coords[:, 2],
                    "element": ["C"] * len(ligand_base_coords),  # Placeholder
                    "name": [f"L{i}" for i in range(len(ligand_base_coords))],
                }
            )
        base_ligand_properties = self.precompute_ligand_properties(ligand_atoms_df)

        all_results = []

        for position in trajectory_gpu:
            # Apply all rotations at once
            rotated_ligands = torch.matmul(
                ligand_centered.unsqueeze(0), rotation_matrices.transpose(1, 2)
            )
            transformed_ligands = rotated_ligands + position.unsqueeze(0).unsqueeze(0)

            # Find best rotation
            best_energy = float("inf")
            best_result = None

            for rot_idx in range(n_rotations):
                # Update ligand coordinates
                current_ligand_properties = base_ligand_properties.copy()
                current_ligand_properties["coords"] = transformed_ligands[rot_idx]

                # Calculate interactions
                result = self.calculate_interactions_for_frame(
                    transformed_ligands[rot_idx], current_ligand_properties
                )

                if result is not None and len(result.energies) > 0:
                    total_energy = result.energies.sum().item()
                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_result = result

            if best_result is not None:
                all_results.append(best_result)

        return all_results
