"""
Protonation-Aware Non-Covalent Interaction Detection for FluxMD
Handles donor/acceptor role swapping and charge-dependent interactions
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from scipy.spatial.distance import cdist

from fluxmd.core.energy_config import ENERGY_BOUNDS
from fluxmd.core.ref15_energy import AtomContext, get_ref15_calculator
from fluxmd.core.rosetta_atom_types import get_atom_typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProtonationAwareAtom:
    """Atom with protonation-dependent properties"""

    index: int
    element: str
    coords: np.ndarray
    residue_name: str
    atom_name: str
    chain: str
    residue_id: int

    # Protonation state
    formal_charge: float = 0.0
    protonation_state: str = "neutral"

    # H-bond capabilities (these flip with pH!)
    can_donate_hbond: bool = False
    can_accept_hbond: bool = False

    # Electronic properties
    is_aromatic: bool = False
    aromatic_electron_density: float = 1.0

    # Hydration effects
    effective_vdw_radius: float = 1.7


class ProtonationAwareInteractionDetector:
    """
    Detects non-covalent interactions with full protonation awareness.
    Critical: H-bond donors/acceptors swap roles based on pH!
    """

    def __init__(self, pH: float = 7.4):
        self.pH = pH

        # Initialize REF15 calculator
        self.ref15_calculator = get_ref15_calculator(pH)
        self.atom_typer = get_atom_typer()

        # Protonation rules with donor/acceptor swapping
        self.residue_protonation_rules = {
            "ASP": {
                "pKa": 3.9,
                "type": "acid",
                "atoms": {
                    "OD1": {"protonated": "can_donate", "deprotonated": "can_accept"},
                    "OD2": {"protonated": "can_donate", "deprotonated": "can_accept"},
                },
            },
            "GLU": {
                "pKa": 4.2,
                "type": "acid",
                "atoms": {
                    "OE1": {"protonated": "can_donate", "deprotonated": "can_accept"},
                    "OE2": {"protonated": "can_donate", "deprotonated": "can_accept"},
                },
            },
            "LYS": {
                "pKa": 10.5,
                "type": "base",
                "atoms": {"NZ": {"protonated": "can_donate", "deprotonated": "can_accept"}},
            },
            "ARG": {
                "pKa": 12.5,
                "type": "base",
                "atoms": {
                    "NE": {"protonated": "can_donate", "deprotonated": "can_accept"},
                    "NH1": {"protonated": "can_donate", "deprotonated": "can_accept"},
                    "NH2": {"protonated": "can_donate", "deprotonated": "can_accept"},
                },
            },
            "HIS": {  # Critical at pH 7.4!
                "pKa": 6.0,
                "type": "base",
                "atoms": {
                    "ND1": {"protonated": "can_donate", "deprotonated": "can_accept"},
                    "NE2": {"protonated": "can_donate", "deprotonated": "can_accept"},
                },
            },
            "CYS": {
                "pKa": 8.3,
                "type": "acid",
                "atoms": {"SG": {"protonated": "can_donate", "deprotonated": "can_accept"}},
            },
            "TYR": {
                "pKa": 10.1,
                "type": "acid",
                "atoms": {"OH": {"protonated": "can_donate", "deprotonated": "can_accept"}},
            },
        }

        # REF15 switching function cutoffs
        self.cutoffs = {
            "hbond": 3.3,  # REF15 H-bond cutoff
            "salt_bridge": 6.0,  # REF15 electrostatic cutoff
            "pi_pi": 4.5,  # Keep original for pi-stacking
            "pi_cation": 6.0,  # Extended for cation-pi
            "vdw": 6.0,  # REF15 LJ cutoff
        }

        # VDW radii
        self.vdw_radii = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "P": 1.80,
            "S": 1.80,
            "CL": 1.75,
            "BR": 1.85,
            "I": 1.98,
        }

    def henderson_hasselbalch(self, pKa: float, is_acid: bool) -> float:
        """Calculate fraction protonated"""
        if is_acid:
            return 1.0 / (1.0 + 10.0 ** (self.pH - pKa))
        else:
            return 1.0 / (1.0 + 10.0 ** (pKa - self.pH))

    def determine_atom_protonation(self, atom: Dict) -> ProtonationAwareAtom:
        """
        Convert atom dict to ProtonationAwareAtom with proper donor/acceptor assignment.
        This is where donor/acceptor roles are determined based on pH!
        """
        pa_atom = ProtonationAwareAtom(
            index=atom.get("atom_id", 0),
            element=atom.get("element", "C"),
            coords=np.array([atom["x"], atom["y"], atom["z"]]),
            residue_name=atom["resname"],
            atom_name=atom["name"],
            chain=atom.get("chain", "A"),
            residue_id=atom.get("resSeq", 0),
        )

        # Set VDW radius
        pa_atom.effective_vdw_radius = self.vdw_radii.get(pa_atom.element, 1.70)

        # Check ionizable residues
        if atom["resname"] in self.residue_protonation_rules:
            res_rules = self.residue_protonation_rules[atom["resname"]]

            if atom["name"] in res_rules["atoms"]:
                pKa = res_rules["pKa"]
                is_acid = res_rules["type"] == "acid"

                # Calculate protonation state
                fraction_protonated = self.henderson_hasselbalch(pKa, is_acid)
                is_protonated = fraction_protonated > 0.5

                # CRITICAL: Set donor/acceptor based on protonation state
                atom_rules = res_rules["atoms"][atom["name"]]

                if is_protonated:
                    pa_atom.protonation_state = "protonated"
                    if atom_rules["protonated"] == "can_donate":
                        pa_atom.can_donate_hbond = True
                    else:
                        pa_atom.can_accept_hbond = True

                    # Charge
                    pa_atom.formal_charge = 0 if is_acid else +1
                else:
                    pa_atom.protonation_state = "deprotonated"
                    if atom_rules["deprotonated"] == "can_donate":
                        pa_atom.can_donate_hbond = True
                    else:
                        pa_atom.can_accept_hbond = True

                    # Charge
                    pa_atom.formal_charge = -1 if is_acid else 0

                # Adjust radius for charged atoms (hydration shell)
                if pa_atom.formal_charge != 0:
                    pa_atom.effective_vdw_radius *= 1.2

        # Non-ionizable groups
        else:
            # Backbone
            if atom["name"] == "N" and atom["resname"] != "PRO":
                pa_atom.can_donate_hbond = True
            elif atom["name"] == "O":
                pa_atom.can_accept_hbond = True

            # Side chains
            elif atom["resname"] in ["SER", "THR"] and atom["name"] in ["OG", "OG1"]:
                pa_atom.can_donate_hbond = True
                pa_atom.can_accept_hbond = True
            elif atom["resname"] == "ASN" and atom["name"] == "OD1":
                pa_atom.can_accept_hbond = True
            elif atom["resname"] == "ASN" and atom["name"] == "ND2":
                pa_atom.can_donate_hbond = True
                pa_atom.can_accept_hbond = True
            elif atom["resname"] == "GLN" and atom["name"] == "OE1":
                pa_atom.can_accept_hbond = True
            elif atom["resname"] == "GLN" and atom["name"] == "NE2":
                pa_atom.can_donate_hbond = True
                pa_atom.can_accept_hbond = True

        # Aromatic properties
        aromatic_atoms = {
            "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
            "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        }

        if atom["resname"] in aromatic_atoms:
            if atom["name"] in aromatic_atoms[atom["resname"]]:
                pa_atom.is_aromatic = True
                # His+ is electron poor
                if atom["resname"] == "HIS" and pa_atom.formal_charge > 0:
                    pa_atom.aromatic_electron_density = 0.7

        return pa_atom

    def process_ligand_atom(self, atom: Dict) -> ProtonationAwareAtom:
        """Process ligand atoms with element-based rules"""
        pa_atom = ProtonationAwareAtom(
            index=atom.get("atom_id", 0),
            element=atom.get("element", "C").upper(),
            coords=np.array([atom["x"], atom["y"], atom["z"]]),
            residue_name="LIG",
            atom_name=atom["name"],
            chain="L",
            residue_id=1,
        )

        # Simple element-based rules for ligands
        if pa_atom.element == "N":
            # Check if it's likely an aromatic N (e.g., in heterocycles)
            # For now, assume aromatic N are acceptors, not donors
            # This is a simplification - ideally we'd check connectivity
            atom_name = atom.get("name", "").upper()
            if "AR" in atom_name or atom.get("is_aromatic", False):
                # Aromatic N (like in thiadiazole) are typically acceptors
                pa_atom.can_accept_hbond = True
                pa_atom.formal_charge = 0
                pa_atom.protonation_state = "neutral"
            else:
                # Aliphatic N are often protonated at pH 7.4
                pa_atom.formal_charge = +1
                pa_atom.can_donate_hbond = True
                pa_atom.protonation_state = "protonated"

            # FIXED: Also make N acceptors regardless (they can be both)
            pa_atom.can_accept_hbond = True
        elif pa_atom.element == "O":
            pa_atom.can_accept_hbond = True
            # Check if carboxylate (simplified)
            if "COO" in atom.get("name", ""):
                pa_atom.formal_charge = -0.5
                pa_atom.protonation_state = "deprotonated"
            # FIXED: O can also be donors if part of OH group
            # This is simplified - ideally we'd check for attached H
            if "OH" in atom.get("name", "") or "HO" in atom.get("name", ""):
                pa_atom.can_donate_hbond = True
        elif pa_atom.element == "S":
            pa_atom.can_accept_hbond = True
        elif pa_atom.element == "H":
            pa_atom.can_donate_hbond = True

        # VDW radius
        pa_atom.effective_vdw_radius = self.vdw_radii.get(pa_atom.element, 1.70)
        if pa_atom.formal_charge != 0:
            pa_atom.effective_vdw_radius *= 1.2

        # Check if aromatic C
        if pa_atom.element == "C":
            pa_atom.is_aromatic = True  # Simplified

        return pa_atom

    def detect_all_interactions(
        self, protein_atom: Dict, ligand_atom: Dict, distance: float
    ) -> List[Dict]:
        """
        Detect all interactions using REF15 energy function.
        """
        # Skip if too far
        if distance > 6.0:
            return []

        # Convert to protonation-aware atoms
        p_atom = self.determine_atom_protonation(protein_atom)
        l_atom = self.process_ligand_atom(ligand_atom)

        # Create REF15 atom contexts
        protein_context = self._create_ref15_context(protein_atom, p_atom)
        ligand_context = self._create_ref15_context(ligand_atom, l_atom)

        # Calculate REF15 energy
        total_energy = self.ref15_calculator.calculate_interaction_energy(
            protein_context, ligand_context, distance, detailed=True
        )

        # Get energy components
        components = self.ref15_calculator.last_energy_components

        interactions = []

        # Map REF15 components to FluxMD interaction types
        if "hbond" in components and components["hbond"] < -0.5:
            # Determine H-bond direction
            if p_atom.can_donate_hbond and l_atom.can_accept_hbond:
                donor, acceptor = "protein", "ligand"
            elif l_atom.can_donate_hbond and p_atom.can_accept_hbond:
                donor, acceptor = "ligand", "protein"
            else:
                donor, acceptor = "unknown", "unknown"

            interactions.append(
                {
                    "type": "HBond",
                    "energy": components["hbond"],
                    "donor": donor,
                    "acceptor": acceptor,
                }
            )

        # Electrostatics (salt bridges)
        if "fa_elec" in components and abs(components["fa_elec"]) > 1.0:
            if p_atom.formal_charge * l_atom.formal_charge < 0:
                interactions.append({"type": "Salt Bridge", "energy": components["fa_elec"]})

        # Pi-cation (special case not in standard REF15)
        if distance <= self.cutoffs["pi_cation"]:
            if (p_atom.is_aromatic and l_atom.formal_charge > 0) or (
                l_atom.is_aromatic and p_atom.formal_charge > 0
            ):
                # Use simplified pi-cation energy
                energy = -3.0 * np.exp(-(((distance - 3.5) / 1.5) ** 2))
                interactions.append({"type": "Pi-Cation", "energy": energy})

        # Pi-stacking (special case)
        if distance <= self.cutoffs["pi_pi"]:
            if p_atom.is_aromatic and l_atom.is_aromatic:
                # Use simplified pi-stacking energy
                energy = -3.5 * np.exp(-(((distance - 3.8) / 1.5) ** 2))
                density_diff = abs(
                    p_atom.aromatic_electron_density - l_atom.aromatic_electron_density
                )
                energy *= 1.0 + 0.5 * density_diff
                interactions.append({"type": "Pi-Stacking", "energy": energy})

        # Van der Waals (from LJ terms)
        lj_energy = components.get("fa_atr", 0.0) + components.get("fa_rep", 0.0)
        if abs(lj_energy) > 0.1 or not interactions:
            interactions.append(
                {
                    "type": "Van der Waals",
                    "energy": np.clip(
                        lj_energy, ENERGY_BOUNDS["vdw"]["min"], ENERGY_BOUNDS["vdw"]["max"]
                    ),
                }
            )

        # Apply energy bounds to all interactions
        for interaction in interactions:
            interaction["energy"] = np.clip(
                interaction["energy"],
                ENERGY_BOUNDS["default"]["min"],
                ENERGY_BOUNDS["default"]["max"],
            )

        return interactions

    def _create_ref15_context(self, atom_dict: Dict, pa_atom: ProtonationAwareAtom) -> AtomContext:
        """Create REF15 AtomContext from atom data"""
        # Enhance atom_dict with connectivity info for typing
        enhanced_dict = atom_dict.copy()
        enhanced_dict["formal_charge"] = pa_atom.formal_charge
        enhanced_dict["is_aromatic"] = pa_atom.is_aromatic

        context = self.ref15_calculator.create_atom_context(enhanced_dict)

        # Override with protonation-aware properties
        context.formal_charge = pa_atom.formal_charge
        context.is_donor = pa_atom.can_donate_hbond
        context.is_acceptor = pa_atom.can_accept_hbond
        context.is_aromatic = pa_atom.is_aromatic

        return context

    def _calculate_salt_bridge_energy(
        self, charge1: float, charge2: float, distance: float
    ) -> float:
        """Legacy method - kept for compatibility. REF15 handles this internally."""
        # Create dummy contexts for REF15 calculation
        atom1_context = AtomContext(
            atom_type="Nlys",  # Dummy charged type
            coords=np.array([0, 0, 0]),
            formal_charge=charge1,
        )
        atom2_context = AtomContext(
            atom_type="OOC",  # Dummy charged type
            coords=np.array([distance, 0, 0]),
            formal_charge=charge2,
        )

        # Use REF15 electrostatic calculation
        energy = self.ref15_calculator._calculate_electrostatic_energy(
            atom1_context, atom2_context, distance
        )

        return np.clip(
            energy, ENERGY_BOUNDS["salt_bridge"]["min"], ENERGY_BOUNDS["salt_bridge"]["max"]
        )

    def _calculate_vdw_energy(self, res1: Residue, res2: Residue) -> float:
        # Placeholder for VdW calculation if needed within this class,
        # otherwise, it relies on intra_protein_interactions.py
        return 0.0


def enhance_fluxmd_with_protonation(
    protein_atoms: pd.DataFrame, ligand_atoms: pd.DataFrame, pH: float = 7.4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add protonation information to atom dataframes for FluxMD.
    """
    detector = ProtonationAwareInteractionDetector(pH=pH)

    # Process protein atoms
    protein_enhanced = protein_atoms.copy()
    for idx, atom in protein_atoms.iterrows():
        pa_atom = detector.determine_atom_protonation(atom.to_dict())
        protein_enhanced.loc[idx, "formal_charge"] = pa_atom.formal_charge
        protein_enhanced.loc[idx, "can_donate_hbond"] = pa_atom.can_donate_hbond
        protein_enhanced.loc[idx, "can_accept_hbond"] = pa_atom.can_accept_hbond
        protein_enhanced.loc[idx, "protonation_state"] = pa_atom.protonation_state

    # Process ligand atoms
    ligand_enhanced = ligand_atoms.copy()
    for idx, atom in ligand_atoms.iterrows():
        pa_atom = detector.process_ligand_atom(atom.to_dict())
        ligand_enhanced.loc[idx, "formal_charge"] = pa_atom.formal_charge
        ligand_enhanced.loc[idx, "can_donate_hbond"] = pa_atom.can_donate_hbond
        ligand_enhanced.loc[idx, "can_accept_hbond"] = pa_atom.can_accept_hbond
        ligand_enhanced.loc[idx, "protonation_state"] = pa_atom.protonation_state

    # Print summary
    print(f"\nProtonation Summary at pH {pH}:")
    print(f"Protein charged atoms: {(protein_enhanced['formal_charge'] != 0).sum()}")
    print(f"Ligand charged atoms: {(ligand_enhanced['formal_charge'] != 0).sum()}")

    # Add ligand donor/acceptor summary
    ligand_donors = ligand_enhanced["can_donate_hbond"].sum()
    ligand_acceptors = ligand_enhanced["can_accept_hbond"].sum()
    print(f"Ligand H-bond donors: {ligand_donors}")
    print(f"Ligand H-bond acceptors: {ligand_acceptors}")

    # Key residues
    for res in ["HIS", "ASP", "GLU", "LYS", "ARG", "CYS"]:
        mask = protein_enhanced["resname"] == res
        if mask.any():
            charged = (protein_enhanced.loc[mask, "formal_charge"] != 0).sum()
            total = mask.sum()
            print(f"  {res}: {charged}/{total} atoms charged")

    return protein_enhanced, ligand_enhanced


def calculate_interactions_with_protonation(
    protein_atoms: pd.DataFrame, ligand_atoms: pd.DataFrame, pH: float = 7.4, iteration_num: int = 0
) -> pd.DataFrame:
    """
    Calculate interactions using protonation-aware detection.
    Drop-in replacement for the original calculate_interactions method.
    """
    detector = ProtonationAwareInteractionDetector(pH=pH)
    interactions = []

    # Get coordinates
    protein_coords = protein_atoms[["x", "y", "z"]].values
    ligand_coords = ligand_atoms[["x", "y", "z"]].values

    # Distance matrix
    dist_matrix = cdist(protein_coords, ligand_coords)

    # Find close contacts
    close_contacts = np.where(dist_matrix < 6.0)

    # Process each contact
    for p_idx, l_idx in zip(close_contacts[0], close_contacts[1]):
        distance = dist_matrix[p_idx, l_idx]

        p_atom = protein_atoms.iloc[p_idx].to_dict()
        l_atom = ligand_atoms.iloc[l_idx].to_dict()

        # Detect ALL interactions (not just strongest)
        detected = detector.detect_all_interactions(p_atom, l_atom, distance)

        for interaction in detected:
            # Calculate vectors
            vector = ligand_coords[l_idx] - protein_coords[p_idx]

            record = {
                "frame": iteration_num,
                "protein_chain": p_atom.get("chain", "A"),
                "protein_residue": p_atom.get("resSeq", 0),
                "protein_resname": p_atom["resname"],
                "protein_atom": p_atom["name"],
                "protein_atom_id": p_idx,
                "ligand_atom": l_atom["name"],
                "ligand_atom_id": l_idx,
                "distance": distance,
                "bond_type": interaction["type"],
                "bond_energy": interaction["energy"],
                "vector_x": vector[0],
                "vector_y": vector[1],
                "vector_z": vector[2],
                "pH": pH,
            }

            # Add H-bond direction if applicable
            if interaction["type"] == "HBond":
                record["hbond_donor"] = interaction.get("donor", "")
                record["hbond_acceptor"] = interaction.get("acceptor", "")

            interactions.append(record)

    return pd.DataFrame(interactions)


# Example usage
if __name__ == "__main__":
    # Demo pH effects on HIS-ASP interaction
    print("pH-Dependent Interaction Example: HIS-ASP")
    print("=" * 50)

    his_atom = {
        "resname": "HIS",
        "name": "NE2",
        "element": "N",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "atom_id": 1,
        "chain": "A",
        "resSeq": 50,
    }

    asp_atom = {
        "resname": "ASP",
        "name": "OD1",
        "element": "O",
        "x": 3.0,
        "y": 0.0,
        "z": 0.0,
        "atom_id": 2,
        "chain": "A",
        "resSeq": 75,
    }

    distance = 3.0

    for pH in [5.0, 6.0, 7.0, 7.4, 8.0]:
        detector = ProtonationAwareInteractionDetector(pH=pH)
        interactions = detector.detect_all_interactions(his_atom, asp_atom, distance)

        print(f"\npH {pH}:")
        his_pa = detector.determine_atom_protonation(his_atom)
        asp_pa = detector.determine_atom_protonation(asp_atom)

        print(f"  HIS-NE2: {his_pa.protonation_state}, charge={his_pa.formal_charge:+.1f}")
        print(f"  ASP-OD1: {asp_pa.protonation_state}, charge={asp_pa.formal_charge:+.1f}")

        if interactions:
            for i in interactions:
                print(f"  -> {i['type']}: {i['energy']:.2f} kcal/mol")
                if "donor" in i:
                    print(f"     Direction: {i['donor']} -> {i['acceptor']}")
        else:
            print("  -> No interactions")
