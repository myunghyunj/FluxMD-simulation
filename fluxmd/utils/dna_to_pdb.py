#!/usr/bin/env python3
"""
DNA to PDB Converter - Enhanced version with dinucleotide-specific parameters
Generates accurate B-DNA double helix structures from DNA sequences.
"""

import numpy as np
import argparse
import math
from typing import Dict, List, Tuple

class DNABuilder:
    """Builds B-DNA structures with dinucleotide-specific parameters."""
    
    # Standard B-DNA parameters
    RISE = 3.38  # Å per base pair (can vary by dinucleotide)
    RADIUS = 10.0  # Å from helix axis to phosphate
    
    # Watson-Crick pairing
    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    # Dinucleotide-specific twist angles (degrees)
    # Based on Olson et al. (1998) PNAS 95:11163-11168
    DINUCLEOTIDE_TWIST = {
        'AA': 35.6, 'AT': 31.5, 'AG': 31.9, 'AC': 34.4,
        'TA': 36.0, 'TT': 35.6, 'TG': 34.0, 'TC': 33.6,
        'GA': 36.9, 'GT': 34.4, 'GG': 32.9, 'GC': 40.0,
        'CA': 34.5, 'CT': 31.9, 'CG': 29.8, 'CC': 32.9,
    }
    
    # Dinucleotide-specific rise values (Å)
    DINUCLEOTIDE_RISE = {
        'AA': 3.32, 'AT': 3.38, 'AG': 3.38, 'AC': 3.38,
        'TA': 3.36, 'TT': 3.32, 'TG': 3.44, 'TC': 3.30,
        'GA': 3.39, 'GT': 3.38, 'GG': 3.38, 'GC': 3.38,
        'CA': 3.45, 'CT': 3.38, 'CG': 3.32, 'CC': 3.38,
    }
    
    # Propeller twist angles for base pairs (degrees)
    BASE_PAIR_PROPELLER = {
        'AT': -11.0, 'TA': -11.0,
        'GC': -10.5, 'CG': -10.5,
    }
    
    # Atom templates (simplified but chemically accurate)
    # Sugar atoms relative to base attachment point
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
        'A': {  # Adenine (purine)
            'N9': np.array([1.214, 0.523, 0.000]),
            'C8': np.array([1.527, 1.852, 0.000]),
            'N7': np.array([2.840, 2.087, 0.000]),
            'C5': np.array([3.408, 0.867, 0.000]),
            'C4': np.array([2.424, -0.136, 0.000]),
            'N3': np.array([2.602, -1.472, 0.000]),
            'C2': np.array([3.868, -1.893, 0.000]),
            'N1': np.array([4.910, -1.014, 0.000]),
            'C6': np.array([4.727, 0.337, 0.000]),
            'N6': np.array([5.783, 1.193, 0.000]),
        },
        'T': {  # Thymine (pyrimidine)
            'N1': np.array([1.214, 0.523, 0.000]),
            'C2': np.array([1.496, 1.878, 0.000]),
            'O2': np.array([0.612, 2.710, 0.000]),
            'N3': np.array([2.791, 2.227, 0.000]),
            'C4': np.array([3.843, 1.348, 0.000]),
            'O4': np.array([5.025, 1.691, 0.000]),
            'C5': np.array([3.475, -0.030, 0.000]),
            'C5M': np.array([4.497, -1.096, 0.000]),
            'C6': np.array([2.201, -0.380, 0.000]),
        },
        'G': {  # Guanine (purine)
            'N9': np.array([1.214, 0.523, 0.000]),
            'C8': np.array([1.527, 1.852, 0.000]),
            'N7': np.array([2.840, 2.087, 0.000]),
            'C5': np.array([3.408, 0.867, 0.000]),
            'C4': np.array([2.424, -0.136, 0.000]),
            'N3': np.array([2.602, -1.472, 0.000]),
            'C2': np.array([3.868, -1.893, 0.000]),
            'N2': np.array([4.090, -3.217, 0.000]),
            'N1': np.array([4.910, -1.014, 0.000]),
            'C6': np.array([4.727, 0.337, 0.000]),
            'O6': np.array([5.712, 1.103, 0.000]),
        },
        'C': {  # Cytosine (pyrimidine)
            'N1': np.array([1.214, 0.523, 0.000]),
            'C2': np.array([1.496, 1.878, 0.000]),
            'O2': np.array([0.612, 2.710, 0.000]),
            'N3': np.array([2.791, 2.227, 0.000]),
            'C4': np.array([3.771, 1.348, 0.000]),
            'N4': np.array([5.027, 1.771, 0.000]),
            'C5': np.array([3.431, -0.030, 0.000]),
            'C6': np.array([2.183, -0.380, 0.000]),
        }
    }
    
    def __init__(self):
        self.atoms = []
        
    def rotation_z(self, angle: float) -> np.ndarray:
        """Rotation matrix around Z axis (angle in radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def rotation_x(self, angle: float) -> np.ndarray:
        """Rotation matrix around X axis (angle in radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
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
    
    def add_nucleotide(self, base_type: str, chain: str, res_id: int,
                      position: np.ndarray, rotation: np.ndarray,
                      propeller: float = 0.0) -> Dict[str, np.ndarray]:
        """Add a nucleotide (sugar + base) at the specified position."""
        atom_positions = {}
        
        # Apply propeller twist if specified
        if propeller != 0.0 and chain == 'B':
            # For chain B, apply negative propeller
            prop_rot = self.rotation_x(-math.radians(propeller))
            rotation = rotation @ prop_rot
        elif propeller != 0.0:
            # For chain A, apply positive propeller
            prop_rot = self.rotation_x(math.radians(propeller))
            rotation = rotation @ prop_rot
        
        # Add sugar atoms
        for atom_name, local_pos in self.SUGAR_ATOMS.items():
            global_pos = position + rotation @ local_pos
            atom_positions[atom_name] = global_pos
            
            element = atom_name[0]
            self.atoms.append({
                'name': atom_name,
                'element': element,
                'coord': global_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{base_type}'
            })
        
        # Add base atoms
        for atom_name, local_pos in self.BASE_ATOMS[base_type].items():
            global_pos = position + rotation @ local_pos
            
            element = 'N' if atom_name[0] == 'N' else \
                     'O' if atom_name[0] == 'O' else 'C'
            
            self.atoms.append({
                'name': atom_name,
                'element': element,
                'coord': global_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{base_type}'
            })
        
        return atom_positions
    
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
            self.atoms.append({
                'name': 'P',
                'element': 'P',
                'coord': p_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{self.get_base_type(chain, res_id)}'
            })
            
            # Add phosphate oxygens
            # Calculate perpendicular vectors for OP1 and OP2
            # Use cross product with a reference vector
            ref = np.array([1, 0, 0]) if abs(unit_vec[0]) < 0.9 else np.array([0, 1, 0])
            perp1 = np.cross(unit_vec, ref)
            perp1 = perp1 / np.linalg.norm(perp1) if np.linalg.norm(perp1) > 0 else np.array([0, 0, 1])
            perp2 = np.cross(unit_vec, perp1)
            
            # Position OP1 and OP2
            op1_pos = p_pos + 1.48 * (0.7 * perp1 + 0.3 * unit_vec)
            op2_pos = p_pos + 1.48 * (-0.7 * perp1 + 0.3 * unit_vec)
            
            self.atoms.append({
                'name': 'OP1',
                'element': 'O',
                'coord': op1_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{self.get_base_type(chain, res_id)}'
            })
            
            self.atoms.append({
                'name': 'OP2',
                'element': 'O',
                'coord': op2_pos,
                'chain': chain,
                'res_id': res_id,
                'res_name': f'D{self.get_base_type(chain, res_id)}'
            })
    
    def get_base_type(self, chain: str, res_id: int) -> str:
        """Get base type for a residue from the atoms list."""
        for atom in self.atoms:
            if atom['chain'] == chain and atom['res_id'] == res_id:
                return atom['res_name'][-1]  # Last character of res_name
        return 'N'  # Default
    
    def build_dna(self, sequence: str):
        """Build B-DNA double helix from sequence with dinucleotide-specific parameters."""
        self.atoms = []
        n = len(sequence)
        
        # Store O3' and O5' positions for backbone connectivity
        o3_positions = {'A': {}, 'B': {}}
        o5_positions = {'A': {}, 'B': {}}
        
        # Build base pairs with dinucleotide-specific parameters
        cumulative_twist = 0.0
        z_position = 0.0
        
        for i in range(n):
            # Get current base and its complement
            base_a = sequence[i]
            base_b = self.COMPLEMENT[base_a]
            
            # Get propeller twist for this base pair
            propeller = self.get_propeller_twist(base_a, base_b)
            
            # Chain A (5' to 3')
            rot_a = self.rotation_z(cumulative_twist)
            pos_a = np.array([self.RADIUS / 2, 0, z_position])
            pos_a[:2] = rot_a[:2, :2] @ np.array([self.RADIUS / 2, 0])
            
            atom_pos_a = self.add_nucleotide(base_a, 'A', i + 1, pos_a, rot_a, propeller/2)
            o3_positions['A'][i + 1] = atom_pos_a["O3'"]
            o5_positions['A'][i + 1] = atom_pos_a["O5'"]
            
            # Chain B (3' to 5', antiparallel)
            rot_b = self.rotation_z(cumulative_twist + math.pi)  # 180° rotation for opposite strand
            pos_b = np.array([-self.RADIUS / 2, 0, z_position])
            pos_b[:2] = rot_b[:2, :2] @ np.array([self.RADIUS / 2, 0])
            
            atom_pos_b = self.add_nucleotide(base_b, 'B', n - i, pos_b, rot_b, propeller/2)
            o3_positions['B'][n - i] = atom_pos_b["O3'"]
            o5_positions['B'][n - i] = atom_pos_b["O5'"]
            
            # Update position for next base pair
            if i < n - 1:
                # Get dinucleotide-specific parameters
                twist_deg, rise = self.get_dinucleotide_params(base_a, sequence[i + 1])
                cumulative_twist += math.radians(twist_deg)
                z_position += rise
        
        # Add phosphate groups for backbone connectivity
        # Chain A: connect 5' to 3' (residues 1->2, 2->3, etc.)
        for i in range(1, n):
            if i in o3_positions['A'] and (i + 1) in o5_positions['A']:
                self.add_phosphate('A', i + 1, o3_positions['A'][i], o5_positions['A'][i + 1])
        
        # Chain B: connect 5' to 3' (residues n->n-1, n-1->n-2, etc.)
        for i in range(n, 1, -1):
            if i in o3_positions['B'] and (i - 1) in o5_positions['B']:
                self.add_phosphate('B', i - 1, o3_positions['B'][i], o5_positions['B'][i - 1])
    
    def write_pdb(self, filename: str):
        """Write structure to PDB file."""
        with open(filename, 'w') as f:
            f.write("REMARK   Generated by FluxMD DNA Builder\n")
            f.write("REMARK   B-DNA with dinucleotide-specific parameters\n")
            f.write("REMARK   Based on Olson et al. (1998) PNAS 95:11163-11168\n")
            
            atom_id = 1
            for atom in self.atoms:
                name = atom['name']
                if len(name) < 4:
                    name = f" {name:<3}"
                else:
                    name = f"{name:<4}"
                
                f.write(
                    f"ATOM  {atom_id:>5} {name} {atom['res_name']:>3} "
                    f"{atom['chain']}{atom['res_id']:>4}    "
                    f"{atom['coord'][0]:>8.3f}{atom['coord'][1]:>8.3f}"
                    f"{atom['coord'][2]:>8.3f}  1.00  0.00          "
                    f"{atom['element']:>2}\n"
                )
                atom_id += 1
            
            f.write("END\n")

def main():
    parser = argparse.ArgumentParser(description='Generate B-DNA structure from sequence.')
    parser.add_argument('sequence', help='DNA sequence (5\' to 3\')')
    parser.add_argument('-o', '--output', default='dna_structure.pdb', help='Output PDB file')
    
    args = parser.parse_args()
    
    # Validate sequence
    sequence = args.sequence.upper()
    if not all(base in 'ATGC' for base in sequence):
        print("Error: Sequence must only contain A, T, G, or C.")
        return
    
    print(f"Building B-DNA structure for: 5'-{sequence}-3'")
    print(f"Complementary strand: 3'-{''.join(DNABuilder.COMPLEMENT[b] for b in sequence)}-5'")
    print("Using dinucleotide-specific parameters...")
    
    # Build and save
    builder = DNABuilder()
    builder.build_dna(sequence)
    builder.write_pdb(args.output)
    
    print(f"\nGenerated: {args.output}")
    print(f"Total atoms: {len(builder.atoms)}")
    
    # Report dinucleotide steps used
    print("\nDinucleotide steps:")
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        twist, rise = builder.get_dinucleotide_params(sequence[i], sequence[i+1])
        print(f"  {dinuc}: twist={twist}°, rise={rise} Å")

if __name__ == '__main__':
    main()