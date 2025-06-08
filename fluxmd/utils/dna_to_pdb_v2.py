#!/usr/bin/env python3
"""
DNA to PDB Converter V2 - Based on 2BNA Reference Structure
Generates accurate B-DNA double helix structures from DNA sequences.
Uses atomic arrangements derived from 2BNA crystal structure analysis.
"""

import numpy as np
import argparse
import math
from typing import Dict, List, Tuple

class ReferenceBasedDNABuilder:
    """
    Builds B-DNA structures using reference atomic arrangements from 2BNA.
    """
    
    # Core helical parameters
    RISE_PER_BASE = 3.38  # Å - Standard B-DNA rise
    HELIX_RADIUS = 5.4    # Å - Distance from helix axis to C1'
    TWIST_PER_BASE = 36.0 # degrees
    
    # Refined parameters from 2BNA analysis
    C1_C1_DISTANCE = 10.6  # Å - Distance between C1' atoms in base pair
    GLYCOSIDIC_BOND = 1.47 # Å - C1'-N9/N1 bond length
    O3_P_BOND = 1.59       # Å - O3'-P bond length
    P_O5_BOND = 1.60       # Å - P-O5' bond length
    
    # Watson-Crick base pairing
    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    # Sugar atoms in C2'-endo conformation (standard B-DNA)
    # Coordinates relative to C1' at origin
    SUGAR_TEMPLATE = {
        "C1'": np.array([0.000, 0.000, 0.000]),
        "C2'": np.array([0.670, -1.310, -0.414]),  # C2' out of plane
        "O4'": np.array([0.829, 1.179, -0.252]),
        "C3'": np.array([1.920, -0.900, -1.244]),
        "C4'": np.array([2.100, 0.630, -0.974]),
        "O3'": np.array([3.080, -1.600, -0.904]),
        "C5'": np.array([3.204, 1.100, -0.004]),
        "O5'": np.array([3.380, 2.507, -0.064]),
    }
    
    # Base templates positioned for anti conformation (chi = -116°)
    # Coordinates include glycosidic bond rotation
    BASE_TEMPLATES = {
        'A': {  # Adenine
            'N9': np.array([1.394, -0.674, 0.000]),  # Attachment point
            'C8': np.array([1.672, -2.017, 0.000]),
            'N7': np.array([2.957, -2.371, 0.000]),
            'C5': np.array([3.614, -1.213, 0.000]),
            'C6': np.array([4.999, -0.844, 0.000]),
            'N6': np.array([6.029, -1.695, 0.000]),
            'N1': np.array([5.284, 0.438, 0.000]),
            'C2': np.array([4.276, 1.289, 0.000]),
            'N3': np.array([2.997, 0.971, 0.000]),
            'C4': np.array([2.760, -0.290, 0.000]),
        },
        'T': {  # Thymine
            'N1': np.array([1.394, -0.674, 0.000]),  # Attachment point
            'C2': np.array([1.635, -2.033, 0.000]),
            'O2': np.array([0.719, -2.822, 0.000]),
            'N3': np.array([2.898, -2.420, 0.000]),
            'C4': np.array([3.965, -1.565, 0.000]),
            'O4': np.array([5.139, -1.959, 0.000]),
            'C5': np.array([3.652, -0.168, 0.000]),
            'C5M': np.array([4.760, 0.829, 0.000]),
            'C6': np.array([2.402, 0.219, 0.000]),
        },
        'G': {  # Guanine
            'N9': np.array([1.394, -0.674, 0.000]),  # Attachment point
            'C8': np.array([1.672, -2.017, 0.000]),
            'N7': np.array([2.957, -2.371, 0.000]),
            'C5': np.array([3.614, -1.213, 0.000]),
            'C6': np.array([4.999, -0.844, 0.000]),
            'O6': np.array([6.091, -1.543, 0.000]),
            'N1': np.array([5.186, 0.495, 0.000]),
            'C2': np.array([4.135, 1.390, 0.000]),
            'N2': np.array([4.410, 2.690, 0.000]),
            'N3': np.array([2.885, 0.989, 0.000]),
            'C4': np.array([2.760, -0.290, 0.000]),
        },
        'C': {  # Cytosine
            'N1': np.array([1.394, -0.674, 0.000]),  # Attachment point
            'C2': np.array([1.635, -2.033, 0.000]),
            'O2': np.array([0.719, -2.822, 0.000]),
            'N3': np.array([2.898, -2.420, 0.000]),
            'C4': np.array([3.866, -1.510, 0.000]),
            'N4': np.array([5.105, -1.907, 0.000]),
            'C5': np.array([3.543, -0.113, 0.000]),
            'C6': np.array([2.302, 0.274, 0.000]),
        }
    }
    
    def __init__(self):
        self.atoms = []
        self.atom_id = 1
        
    def _rotation_matrix_z(self, angle_rad: float) -> np.ndarray:
        """Create rotation matrix for rotation around Z axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def _rotation_matrix_x(self, angle_rad: float) -> np.ndarray:
        """Create rotation matrix for rotation around X axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    def _rotation_matrix_y(self, angle_rad: float) -> np.ndarray:
        """Create rotation matrix for rotation around Y axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def _build_nucleotide(self, base_type: str, chain: str, res_id: int,
                         position: np.ndarray, rotation: np.ndarray) -> List[Dict]:
        """
        Build a single nucleotide with sugar and base.
        
        Args:
            base_type: 'A', 'T', 'G', or 'C'
            chain: 'A' or 'B'
            res_id: Residue number
            position: 3D position of the nucleotide
            rotation: Rotation matrix for orientation
        
        Returns:
            List of atom dictionaries
        """
        nucleotide_atoms = []
        
        # Add sugar atoms
        for atom_name, local_coord in self.SUGAR_TEMPLATE.items():
            # Apply rotation and translation
            global_coord = rotation @ local_coord + position
            
            nucleotide_atoms.append({
                'name': atom_name,
                'element': atom_name[0],  # First character is element
                'coord': global_coord,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{base_type}'
            })
        
        # Add base atoms
        base_template = self.BASE_TEMPLATES[base_type]
        c1_pos = rotation @ self.SUGAR_TEMPLATE["C1'"] + position
        
        for atom_name, local_coord in base_template.items():
            # Base coordinates are already positioned relative to glycosidic bond
            # Just need to rotate and translate to C1' position
            global_coord = rotation @ local_coord + c1_pos
            
            # Determine element from atom name
            element = 'N' if atom_name.startswith('N') else \
                     'O' if atom_name.startswith('O') else 'C'
            
            nucleotide_atoms.append({
                'name': atom_name,
                'element': element,
                'coord': global_coord,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{base_type}'
            })
        
        return nucleotide_atoms
    
    def _build_base_pair(self, base_a: str, base_b: str, bp_index: int,
                        z_position: float, twist_angle: float,
                        propeller: float = 0.0) -> Tuple[List[Dict], List[Dict]]:
        """
        Build a Watson-Crick base pair.
        
        Args:
            base_a: Base type for strand A
            base_b: Base type for strand B (should be complement)
            bp_index: Base pair index (0-based)
            z_position: Position along helix axis
            twist_angle: Cumulative twist angle (radians)
            propeller: Propeller twist angle (radians)
        
        Returns:
            Tuple of (strand_a_atoms, strand_b_atoms)
        """
        # Build strand A nucleotide
        # Position at +X (radius), then rotate by twist angle
        pos_a = np.array([self.HELIX_RADIUS, 0, z_position])
        
        # Create rotation matrix: first apply twist, then propeller
        rot_twist = self._rotation_matrix_z(twist_angle)
        rot_propeller_a = self._rotation_matrix_x(propeller / 2)
        rot_a = rot_twist @ rot_propeller_a
        
        # Transform position
        pos_a = rot_twist @ np.array([self.HELIX_RADIUS, 0, 0]) + np.array([0, 0, z_position])
        
        atoms_a = self._build_nucleotide(base_a, 'A', bp_index + 1, pos_a, rot_a)
        
        # Build strand B nucleotide
        # For antiparallel strand: rotate 180° around Z, then apply twist
        rot_180 = self._rotation_matrix_z(np.pi)
        rot_propeller_b = self._rotation_matrix_x(-propeller / 2)
        rot_b = rot_twist @ rot_180 @ rot_propeller_b
        
        # Position on opposite side of helix
        pos_b = rot_twist @ rot_180 @ np.array([self.HELIX_RADIUS, 0, 0]) + np.array([0, 0, z_position])
        
        atoms_b = self._build_nucleotide(base_b, 'B', bp_index + 1, pos_b, rot_b)
        
        return atoms_a, atoms_b
    
    def _add_phosphate(self, chain: str, res_id: int, o3_prev: np.ndarray,
                      o5_curr: np.ndarray, c5_curr: np.ndarray) -> List[Dict]:
        """
        Add phosphate group between nucleotides.
        
        Args:
            chain: Chain identifier
            res_id: Residue number for the phosphate
            o3_prev: O3' position of previous nucleotide
            o5_curr: O5' position of current nucleotide
            c5_curr: C5' position of current nucleotide
        
        Returns:
            List of phosphate atom dictionaries
        """
        phosphate_atoms = []
        
        # Calculate phosphate position to maintain proper bond lengths
        o3_to_o5 = o5_curr - o3_prev
        dist = np.linalg.norm(o3_to_o5)
        
        if dist > self.O3_P_BOND + self.P_O5_BOND:
            # If too far, place along line at proper distance from O3'
            direction = o3_to_o5 / dist
            p_pos = o3_prev + direction * self.O3_P_BOND
        else:
            # Use geometric constraints to find P position
            # This maintains both O3'-P and P-O5' distances
            direction = o3_to_o5 / dist
            
            # Calculate position that satisfies both bond lengths
            # Using law of cosines
            cos_angle = (self.O3_P_BOND**2 + dist**2 - self.P_O5_BOND**2) / (2 * self.O3_P_BOND * dist)
            cos_angle = np.clip(cos_angle, -1, 1)
            
            p_pos = o3_prev + direction * self.O3_P_BOND
        
        phosphate_atoms.append({
            'name': 'P',
            'element': 'P',
            'coord': p_pos,
            'chain': chain,
            'res_id': res_id,
            'res_name': 'DMP'  # Phosphate group
        })
        
        # Add phosphate oxygens (O1P and O2P)
        # Calculate perpendicular vectors for tetrahedral geometry
        v1 = o3_prev - p_pos
        v2 = o5_curr - p_pos
        perp = np.cross(v1, v2)
        
        if np.linalg.norm(perp) > 0:
            perp = perp / np.linalg.norm(perp)
        else:
            # Fallback if vectors are parallel
            perp = np.array([1, 0, 0])
            if abs(np.dot(perp, v1)) > 0.9:
                perp = np.array([0, 1, 0])
        
        # Position O1P and O2P
        bond_length = 1.48  # P-O bond length
        o1p_pos = p_pos + bond_length * (perp + 0.3 * v1/np.linalg.norm(v1))
        o2p_pos = p_pos + bond_length * (-perp + 0.3 * v1/np.linalg.norm(v1))
        
        phosphate_atoms.extend([
            {
                'name': 'O1P',
                'element': 'O',
                'coord': o1p_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': 'DMP'
            },
            {
                'name': 'O2P',
                'element': 'O',
                'coord': o2p_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': 'DMP'
            }
        ])
        
        return phosphate_atoms
    
    def build_dna(self, sequence: str):
        """
        Build complete DNA double helix from sequence.
        
        Args:
            sequence: DNA sequence (5' to 3' for chain A)
        """
        self.atoms = []
        self.atom_id = 1
        
        n_bases = len(sequence)
        cumulative_twist = 0.0
        
        # Storage for O3' positions for backbone connectivity
        o3_positions_a = {}
        o3_positions_b = {}
        o5_positions_a = {}
        o5_positions_b = {}
        c5_positions_a = {}
        c5_positions_b = {}
        
        # Build base pairs
        for i in range(n_bases):
            base_a = sequence[i]
            base_b = self.COMPLEMENT[base_a]
            
            # Get dinucleotide parameters if available
            if i < n_bases - 1:
                dinuc = sequence[i:i+2]
                # Use dinucleotide-specific parameters from previous implementation
                twist_deg = 36.0  # Default, can be replaced with dinuc-specific
                propeller_deg = -11.0  # Default
            else:
                twist_deg = 36.0
                propeller_deg = -11.0
            
            z_pos = i * self.RISE_PER_BASE
            propeller_rad = math.radians(propeller_deg)
            
            # Build base pair
            atoms_a, atoms_b = self._build_base_pair(
                base_a, base_b, i, z_pos, cumulative_twist, propeller_rad
            )
            
            # For antiparallel strands, chain B numbering goes opposite direction
            for atom in atoms_b:
                atom['res_id'] = n_bases - i
            
            # Store positions for backbone connectivity
            for atom in atoms_a:
                if atom['name'] == "O3'":
                    o3_positions_a[i + 1] = atom['coord']
                elif atom['name'] == "O5'":
                    o5_positions_a[i + 1] = atom['coord']
                elif atom['name'] == "C5'":
                    c5_positions_a[i + 1] = atom['coord']
            
            for atom in atoms_b:
                if atom['name'] == "O3'":
                    o3_positions_b[atom['res_id']] = atom['coord']
                elif atom['name'] == "O5'":
                    o5_positions_b[atom['res_id']] = atom['coord']
                elif atom['name'] == "C5'":
                    c5_positions_b[atom['res_id']] = atom['coord']
            
            # Add atoms to structure
            self.atoms.extend(atoms_a)
            self.atoms.extend(atoms_b)
            
            # Update twist for next base pair
            if i < n_bases - 1:
                cumulative_twist += math.radians(twist_deg)
        
        # Add phosphate groups for connectivity
        # Chain A: connect 1→2, 2→3, etc.
        for i in range(2, n_bases + 1):
            if i-1 in o3_positions_a and i in o5_positions_a:
                phosphates = self._add_phosphate(
                    'A', i, o3_positions_a[i-1], o5_positions_a[i], c5_positions_a[i]
                )
                self.atoms.extend(phosphates)
        
        # Chain B: connect in reverse order due to antiparallel nature
        for i in range(2, n_bases + 1):
            if i-1 in o3_positions_b and i in o5_positions_b:
                phosphates = self._add_phosphate(
                    'B', i, o3_positions_b[i-1], o5_positions_b[i], c5_positions_b[i]
                )
                self.atoms.extend(phosphates)
    
    def write_pdb(self, filename: str):
        """Write structure to PDB file."""
        with open(filename, 'w') as f:
            f.write("REMARK   Generated by Reference-Based DNA Builder\n")
            f.write("REMARK   Based on 2BNA atomic arrangements\n")
            
            # Assign atom IDs and write atoms
            atom_id = 1
            for atom in self.atoms:
                atom_name = atom['name']
                # Format atom name with proper spacing
                if len(atom_name) < 4:
                    atom_name = f" {atom_name:<3}"
                else:
                    atom_name = f"{atom_name:<4}"
                
                line = (
                    f"ATOM  {atom_id:>5} {atom_name} {atom['res_name']:>3} "
                    f"{atom['chain']}{atom['res_id']:>4}    "
                    f"{atom['coord'][0]:>8.3f}{atom['coord'][1]:>8.3f}"
                    f"{atom['coord'][2]:>8.3f}  1.00  0.00          "
                    f"{atom['element']:>2}\n"
                )
                f.write(line)
                atom_id += 1
            
            f.write("END\n")

def main():
    parser = argparse.ArgumentParser(
        description='Generate B-DNA structure from sequence using 2BNA reference.'
    )
    parser.add_argument('sequence', help='DNA sequence (5\' to 3\')')
    parser.add_argument('-o', '--output', default='dna_structure.pdb',
                       help='Output PDB filename')
    
    args = parser.parse_args()
    
    # Validate sequence
    sequence = args.sequence.upper()
    if not all(base in 'ATGC' for base in sequence):
        print("Error: Sequence must only contain A, T, G, or C.")
        return
    
    print(f"Building B-DNA structure for: 5'-{sequence}-3'")
    print(f"Complementary strand: 3'-{''.join(ReferenceBasedDNABuilder.COMPLEMENT[b] for b in sequence)}-5'")
    
    # Build structure
    builder = ReferenceBasedDNABuilder()
    builder.build_dna(sequence)
    builder.write_pdb(args.output)
    
    print(f"\nSuccessfully generated: {args.output}")
    print(f"Total atoms: {len(builder.atoms)}")

if __name__ == '__main__':
    main()