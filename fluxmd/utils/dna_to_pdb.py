#!/usr/bin/env python3
"""
DNA to PDB Converter - Enhanced version with dinucleotide-specific parameters
Generates accurate B-DNA double helix structures from DNA sequences.
Includes CONECT records for proper connectivity and optimized base pair alignment.
"""

import argparse
import math
from typing import Dict, Tuple


import numpy as np

# --- Constants -------------------------------------------------------------
GROOVE_ASYMMETRY_DEG = 160.0
MINOR_OFFSET_DEG = 80.0
MINOR_BASE_RADIUS = 6.0
MAJOR_BASE_RADIUS = 11.0
SEQ_RADIUS_COEF = 0.50


# --- Public helpers --------------------------------------------------------
def groove_vectors(twist_rad: float, roll_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit vectors toward minor and major grooves."""
    dphi = math.radians(MINOR_OFFSET_DEG + 0.5 * roll_deg)
    v_minor = np.array([math.cos(twist_rad + dphi), math.sin(twist_rad + dphi), 0.0])
    v_major = np.array(
        [
            math.cos(twist_rad + dphi + math.radians(GROOVE_ASYMMETRY_DEG)),
            math.sin(twist_rad + dphi + math.radians(GROOVE_ASYMMETRY_DEG)),
            0.0,
        ]
    )
    return v_minor / np.linalg.norm(v_minor), v_major / np.linalg.norm(v_major)


def minor_radius(sequence: str, idx: int, window: int = 5) -> float:
    """Sequence-aware minor-groove radius."""
    half = window // 2
    seg = sequence[max(0, idx - half): idx + half + 1]
    gc = (seg.count("G") + seg.count("C")) / max(len(seg), 1.0)
    return MINOR_BASE_RADIUS + SEQ_RADIUS_COEF * (gc - 0.5)


class DNABuilder:
    """Builds B-DNA structures with dinucleotide-specific parameters."""

    # Standard B-DNA parameters (updated based on crystallographic data)
    RISE = 3.38  # Å per base pair (can vary by dinucleotide)
    RADIUS = 10.0  # Å from helix axis to phosphate
    BASE_PAIR_DISTANCE = 10.85  # Å between C1' atoms in a base pair

    # Watson-Crick pairing
    COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

    # Dinucleotide-specific twist angles (degrees)
    # Based on Olson et al. (1998) PNAS 95:11163-11168
    DINUCLEOTIDE_TWIST = {
        "AA": 35.6,
        "AT": 31.5,
        "AG": 31.9,
        "AC": 34.4,
        "TA": 36.0,
        "TT": 35.6,
        "TG": 34.0,
        "TC": 33.6,
        "GA": 36.9,
        "GT": 34.4,
        "GG": 32.9,
        "GC": 40.0,
        "CA": 34.5,
        "CT": 31.9,
        "CG": 29.8,
        "CC": 32.9,
    }

    # Dinucleotide-specific rise values (Å)
    DINUCLEOTIDE_RISE = {
        "AA": 3.32,
        "AT": 3.38,
        "AG": 3.38,
        "AC": 3.38,
        "TA": 3.36,
        "TT": 3.32,
        "TG": 3.44,
        "TC": 3.30,
        "GA": 3.39,
        "GT": 3.38,
        "GG": 3.38,
        "GC": 3.38,
        "CA": 3.45,
        "CT": 3.38,
        "CG": 3.32,
        "CC": 3.38,
    }

    # Propeller twist angles for base pairs (degrees)
    BASE_PAIR_PROPELLER = {
        "AT": -11.0,
        "TA": -11.0,
        "GC": -10.5,
        "CG": -10.5,
    }

    # Atom templates are defined in a local coordinate system where C1' is the
    # point of attachment, but not necessarily the origin.
    # The _process_atom_templates method will correct this upon initialization.
    SUGAR_ATOMS = {
        "C1'": np.array([0.0, 0.0, 0.0]),
        "C2'": np.array([0.862, -1.373, 0.206]),
        "C3'": np.array([2.211, -1.124, -0.528]),
        "C4'": np.array([2.086, 0.403, -0.768]),
        "O4'": np.array([0.677, 0.748, -0.759]),
        "C5'": np.array([2.810, 1.286, 0.277]),
        "O5'": np.array([2.697, 2.666, -0.074]),
        "O3'": np.array([3.335, -1.538, 0.250]),
    }

    # Base atoms for purines (A, G) and pyrimidines (C, T)
    BASE_ATOMS = {
        "A": {  # Adenine (purine)
            "N9": np.array([1.214, 0.523, 0.000]),
            "C8": np.array([1.527, 1.852, 0.000]),
            "N7": np.array([2.840, 2.087, 0.000]),
            "C5": np.array([3.408, 0.867, 0.000]),
            "C4": np.array([2.424, -0.136, 0.000]),
            "N3": np.array([2.602, -1.472, 0.000]),
            "C2": np.array([3.868, -1.893, 0.000]),
            "N1": np.array([4.910, -1.014, 0.000]),
            "C6": np.array([4.727, 0.337, 0.000]),
            "N6": np.array([5.783, 1.193, 0.000]),
        },
        "T": {  # Thymine (pyrimidine)
            "N1": np.array([1.214, 0.523, 0.000]),
            "C2": np.array([1.496, 1.878, 0.000]),
            "O2": np.array([0.612, 2.710, 0.000]),
            "N3": np.array([2.791, 2.227, 0.000]),
            "C4": np.array([3.843, 1.348, 0.000]),
            "O4": np.array([5.025, 1.691, 0.000]),
            "C5": np.array([3.475, -0.030, 0.000]),
            "C5M": np.array([4.497, -1.096, 0.000]),
            "C6": np.array([2.201, -0.380, 0.000]),
        },
        "G": {  # Guanine (purine)
            "N9": np.array([1.214, 0.523, 0.000]),
            "C8": np.array([1.527, 1.852, 0.000]),
            "N7": np.array([2.840, 2.087, 0.000]),
            "C5": np.array([3.408, 0.867, 0.000]),
            "C4": np.array([2.424, -0.136, 0.000]),
            "N3": np.array([2.602, -1.472, 0.000]),
            "C2": np.array([3.868, -1.893, 0.000]),
            "N2": np.array([4.090, -3.217, 0.000]),
            "N1": np.array([4.910, -1.014, 0.000]),
            "C6": np.array([4.727, 0.337, 0.000]),
            "O6": np.array([5.712, 1.103, 0.000]),
        },
        "C": {  # Cytosine (pyrimidine)
            "N1": np.array([1.214, 0.523, 0.000]),
            "C2": np.array([1.496, 1.878, 0.000]),
            "O2": np.array([0.612, 2.710, 0.000]),
            "N3": np.array([2.791, 2.227, 0.000]),
            "C4": np.array([3.771, 1.348, 0.000]),
            "N4": np.array([5.027, 1.771, 0.000]),
            "C5": np.array([3.431, -0.030, 0.000]),
            "C6": np.array([2.183, -0.380, 0.000]),
        },
    }

    # H-bond donor/acceptor pairs for Watson-Crick base pairs
    HBOND_PAIRS = {
        ("A", "T"): [("N1", "N3"), ("N6", "O4")],  # A-T has 2 H-bonds
        ("T", "A"): [("N3", "N1"), ("O4", "N6")],
        ("G", "C"): [("N1", "N3"), ("N2", "O2"), ("O6", "N4")],  # G-C has 3 H-bonds
        ("C", "G"): [("N3", "N1"), ("O2", "N2"), ("N4", "O6")],
    }

    def __init__(self):
        self.atoms = []
        self.atom_index_map = {}  # Maps (chain, res_id, atom_name) to atom index
        self.connectivity = []  # List of atom index pairs for CONECT records
        self._groove_pts = []
        self._process_atom_templates()

    def _process_atom_templates(self):
        """
        Re-centers the base templates so the glycosidic bond attachment point is at the origin,
        and flips the base to extend into the negative X direction. This ensures that the
        sugar and base are on opposite sides, which is essential for correct geometry.
        The sugar template is assumed to be in the positive X direction.
        """
        self.PROCESSED_BASE_ATOMS = {}
        for base_type, atoms in self.BASE_ATOMS.items():
            processed_atoms = {}
            # Determine anchor atom (N9 for purines, N1 for pyrimidines)
            anchor_atom = "N9" if base_type in ["A", "G"] else "N1"

            # Ensure the anchor atom exists before proceeding
            if anchor_atom not in atoms:
                raise ValueError(f"Anchor atom {anchor_atom} not found in base {base_type}")

            offset = atoms[anchor_atom]

            for atom_name, pos in atoms.items():
                # Step 1: Recenter the coordinates around the anchor atom.
                new_pos = pos - offset
                # Step 2: Flip the base by negating the X-coordinate.
                # This places the base on the opposite side of the sugar.
                new_pos[0] = -new_pos[0]
                processed_atoms[atom_name] = new_pos
            self.PROCESSED_BASE_ATOMS[base_type] = processed_atoms

    def rotation_z(self, angle: float) -> np.ndarray:
        """Rotation matrix around Z axis (angle in radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rotation_x(self, angle: float) -> np.ndarray:
        """Rotation matrix around X axis (angle in radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rotation_y(self, angle: float) -> np.ndarray:
        """Rotation matrix around Y axis (angle in radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def get_dinucleotide_params(self, base1: str, base2: str) -> Tuple[float, float]:
        """Get twist and rise for a dinucleotide step."""
        dinuc = base1 + base2
        twist = self.DINUCLEOTIDE_TWIST.get(dinuc, 36.0)
        rise = self.DINUCLEOTIDE_RISE.get(dinuc, 3.38)
        return twist, rise

    def get_propeller_twist(self, base1: str, base2: str) -> float:
        """Get propeller twist angle for a base pair."""
        bp = base1 + base2
        if bp in self.BASE_PAIR_PROPELLER:
            return self.BASE_PAIR_PROPELLER[bp]
        # Try reverse
        bp_rev = base2 + base1
        return self.BASE_PAIR_PROPELLER.get(bp_rev, -11.0)

    def add_nucleotide(
        self,
        base_type: str,
        chain: str,
        res_id: int,
        position: np.ndarray,
        rotation: np.ndarray,
        propeller: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Add a nucleotide (sugar + base) at the specified position."""
        atom_positions = {}
        atom_indices = {}

        # Apply propeller twist if specified
        if propeller != 0.0:
            if chain == "B":
                # For chain B, apply negative propeller
                prop_rot = self.rotation_x(-math.radians(propeller / 2))
            else:
                # For chain A, apply positive propeller
                prop_rot = self.rotation_x(math.radians(propeller / 2))
            rotation = rotation @ prop_rot

        # Add sugar atoms
        for atom_name, local_pos in self.SUGAR_ATOMS.items():
            global_pos = position + rotation @ local_pos
            atom_positions[atom_name] = global_pos

            element = atom_name[0]
            atom_index = len(self.atoms) + 1
            atom_indices[atom_name] = atom_index
            self.atom_index_map[(chain, res_id, atom_name)] = atom_index

            self.atoms.append(
                {
                    "index": atom_index,
                    "name": atom_name,
                    "element": element,
                    "coord": global_pos,
                    "chain": chain,
                    "res_id": res_id,
                    "res_name": f"D{base_type}",
                }
            )

        # Add base atoms
        for atom_name, local_pos in self.PROCESSED_BASE_ATOMS[base_type].items():
            global_pos = position + rotation @ local_pos

            element = "N" if atom_name[0] == "N" else "O" if atom_name[0] == "O" else "C"

            atom_index = len(self.atoms) + 1
            atom_indices[atom_name] = atom_index
            self.atom_index_map[(chain, res_id, atom_name)] = atom_index

            self.atoms.append(
                {
                    "index": atom_index,
                    "name": atom_name,
                    "element": element,
                    "coord": global_pos,
                    "chain": chain,
                    "res_id": res_id,
                    "res_name": f"D{base_type}",
                }
            )

        # Add intra-residue connectivity
        self._add_nucleotide_connectivity(chain, res_id, atom_indices, base_type)

        return atom_positions

    def _add_nucleotide_connectivity(
        self, chain: str, res_id: int, atom_indices: Dict[str, int], base_type: str
    ):
        """Add CONECT records for intra-nucleotide bonds."""
        # Sugar ring connectivity
        sugar_bonds = [
            ("C1'", "C2'"),
            ("C2'", "C3'"),
            ("C3'", "C4'"),
            ("C4'", "O4'"),
            ("O4'", "C1'"),
            ("C4'", "C5'"),
            ("C5'", "O5'"),
            ("C3'", "O3'"),
        ]

        # Base-sugar connection
        if base_type in ["A", "G"]:  # Purines connect via N9
            sugar_bonds.append(("C1'", "N9"))
        else:  # Pyrimidines connect via N1
            sugar_bonds.append(("C1'", "N1"))

        # Add sugar bonds
        for atom1, atom2 in sugar_bonds:
            if atom1 in atom_indices and atom2 in atom_indices:
                self.connectivity.append((atom_indices[atom1], atom_indices[atom2]))

        # Base ring connectivity (simplified - just major bonds)
        if base_type == "A":
            base_bonds = [
                ("N9", "C8"),
                ("C8", "N7"),
                ("N7", "C5"),
                ("C5", "C4"),
                ("C4", "N9"),
                ("C4", "N3"),
                ("N3", "C2"),
                ("C2", "N1"),
                ("N1", "C6"),
                ("C6", "C5"),
                ("C6", "N6"),
            ]
        elif base_type == "T":
            base_bonds = [
                ("N1", "C2"),
                ("C2", "N3"),
                ("N3", "C4"),
                ("C4", "C5"),
                ("C5", "C6"),
                ("C6", "N1"),
                ("C2", "O2"),
                ("C4", "O4"),
                ("C5", "C5M"),
            ]
        elif base_type == "G":
            base_bonds = [
                ("N9", "C8"),
                ("C8", "N7"),
                ("N7", "C5"),
                ("C5", "C4"),
                ("C4", "N9"),
                ("C4", "N3"),
                ("N3", "C2"),
                ("C2", "N1"),
                ("N1", "C6"),
                ("C6", "C5"),
                ("C6", "O6"),
                ("C2", "N2"),
            ]
        elif base_type == "C":
            base_bonds = [
                ("N1", "C2"),
                ("C2", "N3"),
                ("N3", "C4"),
                ("C4", "C5"),
                ("C5", "C6"),
                ("C6", "N1"),
                ("C2", "O2"),
                ("C4", "N4"),
            ]

        # Add base bonds
        for atom1, atom2 in base_bonds:
            if atom1 in atom_indices and atom2 in atom_indices:
                self.connectivity.append((atom_indices[atom1], atom_indices[atom2]))

    def add_phosphate(self, chain: str, res_id: int, o3_prev: np.ndarray, o5_curr: np.ndarray):
        """Add phosphate group between O3' of previous and O5' of current residue."""
        # Calculate phosphate position
        # Place P at 1.59 Å from O3' along the O3'-O5' vector
        vec = o5_curr - o3_prev
        dist = np.linalg.norm(vec)

        if dist > 0:
            unit_vec = vec / dist
            p_pos = o3_prev + unit_vec * 1.59

            # Add phosphorus
            p_index = len(self.atoms) + 1
            self.atom_index_map[(chain, res_id, "P")] = p_index

            self.atoms.append(
                {
                    "index": p_index,
                    "name": "P",
                    "element": "P",
                    "coord": p_pos,
                    "chain": chain,
                    "res_id": res_id,
                    "res_name": f"D{self.get_base_type(chain, res_id)}",
                }
            )

            # Add phosphate oxygens
            # Calculate perpendicular vectors for OP1 and OP2
            # Use cross product with a reference vector
            ref = np.array([1, 0, 0]) if abs(unit_vec[0]) < 0.9 else np.array([0, 1, 0])
            perp1 = np.cross(unit_vec, ref)
            perp1 = (
                perp1 / np.linalg.norm(perp1) if np.linalg.norm(perp1) > 0 else np.array([0, 0, 1])
            )
            # Position OP1 and OP2
            op1_pos = p_pos + 1.48 * (0.7 * perp1 + 0.3 * unit_vec)
            op2_pos = p_pos + 1.48 * (-0.7 * perp1 + 0.3 * unit_vec)

            op1_index = len(self.atoms) + 1
            self.atom_index_map[(chain, res_id, "OP1")] = op1_index
            self.atoms.append(
                {
                    "index": op1_index,
                    "name": "OP1",
                    "element": "O",
                    "coord": op1_pos,
                    "chain": chain,
                    "res_id": res_id,
                    "res_name": f"D{self.get_base_type(chain, res_id)}",
                }
            )

            op2_index = len(self.atoms) + 1
            self.atom_index_map[(chain, res_id, "OP2")] = op2_index
            self.atoms.append(
                {
                    "index": op2_index,
                    "name": "OP2",
                    "element": "O",
                    "coord": op2_pos,
                    "chain": chain,
                    "res_id": res_id,
                    "res_name": f"D{self.get_base_type(chain, res_id)}",
                }
            )

            # Add connectivity for phosphate group
            self.connectivity.append((p_index, op1_index))
            self.connectivity.append((p_index, op2_index))

            # Connect to O3' of previous residue
            prev_res_id = res_id - 1 if chain == "A" else res_id + 1
            o3_key = (chain, prev_res_id, "O3'")
            if o3_key in self.atom_index_map:
                self.connectivity.append((self.atom_index_map[o3_key], p_index))

            # Connect to O5' of current residue
            o5_key = (chain, res_id, "O5'")
            if o5_key in self.atom_index_map:
                self.connectivity.append((p_index, self.atom_index_map[o5_key]))

    def get_base_type(self, chain: str, res_id: int) -> str:
        """Get base type for a residue from the atoms list."""
        for atom in self.atoms:
            if atom["chain"] == chain and atom["res_id"] == res_id:
                return atom["res_name"][-1]  # Last character of res_name
        return "N"  # Default

    def build_dna(self, sequence: str, groove_mode: str = "static"):
        """Build B-DNA double helix from sequence."""
        if not sequence:
            raise ValueError("sequence must not be empty")
        illegal = set(sequence) - set("ATGC")
        if illegal:
            raise ValueError(f"illegal characters in sequence: {illegal}")
        self.atoms = []
        self.atom_index_map = {}
        self.connectivity = []
        self._groove_pts = []
        n = len(sequence)

        # Store O3' and O5' positions for backbone connectivity
        o3_positions = {"A": {}, "B": {}}
        o5_positions = {"A": {}, "B": {}}

        # Build base pairs with dinucleotide-specific parameters
        cumulative_twist = 0.0
        z_position = 0.0

        for i in range(n):
            # Get current base and its complement
            base_a = sequence[i]
            base_b = self.COMPLEMENT[base_a]

            # Get propeller twist for this base pair
            propeller = self.get_propeller_twist(base_a, base_b)

            # Calculate positions to ensure proper base pair distance
            c1_distance = self.BASE_PAIR_DISTANCE / 2

            # Common rotation around the helix axis
            rot_z_twist = self.rotation_z(cumulative_twist)

            r_minor = minor_radius(sequence, i) if groove_mode == "seq" else MINOR_BASE_RADIUS
            r_major = MAJOR_BASE_RADIUS
            v_minor, v_major = groove_vectors(cumulative_twist)
            axis = np.array([0.0, 0.0, z_position])
            self._groove_pts.append((axis + r_minor * v_minor, axis + r_major * v_major))

            # With the processed templates, the geometry is now intrinsically correct.
            # The sugar will point away from the helix axis, and the base will point toward it.

            # Chain A (5' to 3')
            # The nucleotide template now has the sugar pointing outward and base inward.
            # We just need to apply the helical twist.
            rot_a = rot_z_twist

            # Position C1' atom at the correct radius
            pos_a = rot_z_twist @ np.array([c1_distance, 0, 0])
            pos_a = np.array([pos_a[0], pos_a[1], z_position])

            atom_pos_a = self.add_nucleotide(base_a, "A", i + 1, pos_a, rot_a, propeller)
            o3_positions["A"][i + 1] = atom_pos_a["O3'"]
            o5_positions["A"][i + 1] = atom_pos_a["O5'"]

            # Chain B (3' to 5', antiparallel)
            # To make the strand antiparallel and oriented correctly, we need two rotations:
            # 1. A 180° rotation around the X-axis to flip the strand direction.
            # 2. A 180° rotation around the Y-axis to ensure the sugar points outward
            #    and the base points inward from the other side of the helix.
            antiparallel_rot = self.rotation_x(math.pi)
            reorient_rot = self.rotation_y(math.pi)
            rot_b = rot_z_twist @ reorient_rot @ antiparallel_rot

            # Position on opposite side of helix
            pos_b = rot_z_twist @ np.array([-c1_distance, 0, 0])
            pos_b = np.array([pos_b[0], pos_b[1], z_position])

            atom_pos_b = self.add_nucleotide(base_b, "B", n - i, pos_b, rot_b, propeller)
            o3_positions["B"][n - i] = atom_pos_b["O3'"]
            o5_positions["B"][n - i] = atom_pos_b["O5'"]

            # Update position for next base pair
            if i < n - 1:
                # Get dinucleotide-specific parameters
                twist_deg, rise = self.get_dinucleotide_params(sequence[i], sequence[i + 1])
                cumulative_twist += math.radians(twist_deg)
                z_position += rise

        # Add phosphate groups for backbone connectivity
        # Chain A: connect 5' to 3' (residues 1->2, 2->3, etc.)
        for i in range(1, n):
            if i in o3_positions["A"] and (i + 1) in o5_positions["A"]:
                self.add_phosphate("A", i + 1, o3_positions["A"][i], o5_positions["A"][i + 1])

        # Chain B: connect 5' to 3' (residues n->n-1, n-1->n-2, etc.)
        for i in range(n, 1, -1):
            if i in o3_positions["B"] and (i - 1) in o5_positions["B"]:
                self.add_phosphate("B", i - 1, o3_positions["B"][i], o5_positions["B"][i - 1])

    def write_pdb(self, filename: str):
        """Write structure to PDB file with CONECT records."""
        with open(filename, "w") as f:
            # Write header
            f.write("REMARK   Generated by FluxMD DNA Builder\n")
            f.write("REMARK   B-DNA with dinucleotide-specific parameters\n")
            f.write("REMARK   Based on Olson et al. (1998) PNAS 95:11163-11168\n")
            f.write("REMARK   Structure includes full atomic connectivity\n")

            # Write atoms
            for atom in self.atoms:
                name = atom["name"]
                if len(name) < 4:
                    name = f" {name:<3}"
                else:
                    name = f"{name:<4}"

                f.write(
                    f"ATOM  {atom['index']:>5} {name} {atom['res_name']:>3} "
                    f"{atom['chain']}{atom['res_id']:>4}    "
                    f"{atom['coord'][0]:>8.3f}{atom['coord'][1]:>8.3f}"
                    f"{atom['coord'][2]:>8.3f}  1.00  0.00          "
                    f"{atom['element']:>2}\n"
                )

            # Write TER records for each chain
            chain_atoms = {}
            for atom in self.atoms:
                chain = atom["chain"]
                if chain not in chain_atoms:
                    chain_atoms[chain] = []
                chain_atoms[chain].append(atom)

            # Sort chains
            for chain in sorted(chain_atoms.keys()):
                last_atom = chain_atoms[chain][-1]
                f.write(
                    f"TER   {last_atom['index']+1:>5}      {last_atom['res_name']:>3} "
                    f"{chain}{last_atom['res_id']:>4}\n"
                )

            # Write CONECT records
            # Group connections by atom
            conect_dict = {}
            for atom1, atom2 in self.connectivity:
                if atom1 not in conect_dict:
                    conect_dict[atom1] = []
                if atom2 not in conect_dict:
                    conect_dict[atom2] = []
                if atom2 not in conect_dict[atom1]:
                    conect_dict[atom1].append(atom2)
                if atom1 not in conect_dict[atom2]:
                    conect_dict[atom2].append(atom1)

            # Write CONECT records
            for atom_idx in sorted(conect_dict.keys()):
                connected = sorted(conect_dict[atom_idx])
                # Write in groups of 4 connections per line
                for i in range(0, len(connected), 4):
                    conect_line = f"CONECT{atom_idx:>5}"
                    for j in range(4):
                        if i + j < len(connected):
                            conect_line += f"{connected[i+j]:>5}"
                    f.write(conect_line + "\n")

            f.write("END\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate B-DNA structure from sequence (5' to 3').",
        epilog="Example: %(prog)s ATCGATCG -o my_dna.pdb",

    )
    parser.add_argument("sequence", help="DNA sequence (5' to 3'), using A, T, G, C")
    parser.add_argument(
        "-o",
        "--output",
        default="dna_structure.pdb",
        help="Output PDB file (default: dna_structure.pdb)",
    )

    parser.add_argument("--no-conect", action="store_true", help="Skip writing CONECT records")
    parser.add_argument(
        "--groove-mode",
        default="static",
        choices=["static", "seq"],
        help="Minor groove radius mode",
    )

    args = parser.parse_args()

    # Validate sequence
    sequence = args.sequence.upper()
    if not all(base in "ATGC" for base in sequence):
        print("Error: Sequence must only contain A, T, G, or C.")
        return

    print(f"Building B-DNA structure for: 5'-{sequence}-3'")
    print(f"Complementary strand: 3'-{''.join(DNABuilder.COMPLEMENT[b] for b in sequence)}-5'")
    print("Using dinucleotide-specific parameters from Olson et al. (1998)...")

    # Build and save
    builder = DNABuilder()
    builder.build_dna(sequence, groove_mode=args.groove_mode)

    # Modify write method if no CONECT requested
    if args.no_conect:
        builder.connectivity = []

    builder.write_pdb(args.output)

    print(f"\nGenerated: {args.output}")
    print(f"Total atoms: {len(builder.atoms)}")
    print(f"Total bonds: {len(builder.connectivity)}")

    # Report dinucleotide steps used
    print("\nDinucleotide steps:")
    for i in range(len(sequence) - 1):
        dinuc = sequence[i: i + 2]

        twist, rise = builder.get_dinucleotide_params(sequence[i], sequence[i + 1])
        print(f"  {dinuc}: twist={twist}°, rise={rise} Å")

    # Report base pairs
    print(f"\nBase pairs: {len(sequence)}")
    for i in range(len(sequence)):
        base_a = sequence[i]
        base_b = DNABuilder.COMPLEMENT[base_a]
        print(f"  {i+1}: {base_a}-{base_b}")

def dna_to_pdb_structure(sequence: str):
    """Return atoms and connectivity for the given DNA sequence."""
    builder = DNABuilder()
    builder.build_dna(sequence)
    return builder.atoms, builder.connectivity
if __name__ == "__main__":
    main()
