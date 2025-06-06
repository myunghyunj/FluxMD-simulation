#!/usr/bin/env python3
"""
DNA Sequence to PDB Converter for FluxMD
Generates 3D B-DNA double helix structures from sequences
Automatically creates complementary strand with Watson-Crick base pairing
Based on reference implementation with proper backbone connectivity
"""

import numpy as np
import os
import argparse
from typing import List, Tuple, Dict
import math

class DNAStructureGenerator:
    """Generate 3D B-DNA structures from sequences with proper atomic detail"""
    
    # B-DNA helical parameters (from reference)
    DELTA_X = 3.38  # Rise per base pair in Angstroms
    DELTA_X_REV_OFFSET = 0.78  # Offset for reverse strand
    BASES_PER_TURN = 10.5  # Standard B-DNA
    TWIST_PER_BASE = 360.0 / BASES_PER_TURN  # 34.3 degrees
    RADIUS = 10.175  # Approximate radius of each base in Angstroms
    THETA_REV_OFFSET = 0.07476  # Reverse strand angular offset
    
    # Base atoms with proper coordinates for Watson-Crick pairing
    # Coordinates from reference PDB files, adjusted for proper orientation
    BASE_ATOMS = {
        'A': {  # Adenine
            'atoms': [
                ('N9', 'N', np.array([1.660, 4.388, 1.478])),
                ('C8', 'C', np.array([1.580, 4.837, 0.184])),
                ('N7', 'N', np.array([1.650, 3.888, -0.699])),
                ('C5', 'C', np.array([1.800, 2.740, 0.058])),
                ('C6', 'C', np.array([1.930, 1.379, -0.294])),
                ('N6', 'N', np.array([1.940, 0.938, -1.548])),
                ('N1', 'N', np.array([2.050, 0.492, 0.705])),
                ('C2', 'C', np.array([2.040, 0.932, 1.960])),
                ('N3', 'N', np.array([1.920, 2.160, 2.415])),
                ('C4', 'C', np.array([1.800, 3.025, 1.391])),
            ],
            'bonds': [(1,2), (2,3), (3,4), (4,5), (5,6), (5,7), (7,8), (8,9), (9,10), (10,1), (10,4)]
        },
        'T': {  # Thymine
            'atoms': [
                ('N1', 'N', np.array([1.660, 4.387, 1.478])),
                ('C2', 'C', np.array([1.810, 3.022, 1.600])),
                ('O2', 'O', np.array([1.900, 2.465, 2.679])),
                ('N3', 'N', np.array([1.850, 2.324, 0.409])),
                ('C4', 'C', np.array([1.760, 2.855, -0.855])),
                ('O4', 'O', np.array([1.810, 2.126, -1.853])),
                ('C5', 'C', np.array([1.610, 4.289, -0.887])),
                ('C5M', 'C', np.array([1.500, 4.911, -2.246])),  # Methyl group
                ('C6', 'C', np.array([1.560, 5.003, 0.255])),
            ],
            'bonds': [(1,2), (2,3), (2,4), (4,5), (5,6), (5,7), (7,8), (7,9), (9,1)]
        },
        'G': {  # Guanine
            'atoms': [
                ('N9', 'N', np.array([1.660, 4.388, 1.478])),
                ('C8', 'C', np.array([1.580, 4.816, 0.169])),
                ('N7', 'N', np.array([1.660, 3.855, -0.713])),
                ('C5', 'C', np.array([1.800, 2.699, 0.057])),
                ('C6', 'C', np.array([1.930, 1.348, -0.339])),
                ('O6', 'O', np.array([1.950, 0.870, -1.471])),
                ('N1', 'N', np.array([2.050, 0.496, 0.775])),
                ('C2', 'C', np.array([2.050, 0.908, 2.091])),
                ('N2', 'N', np.array([2.180, -0.052, 3.010])),
                ('N3', 'N', np.array([1.920, 2.179, 2.465])),
                ('C4', 'C', np.array([1.800, 3.020, 1.403])),
            ],
            'bonds': [(1,2), (2,3), (3,4), (4,5), (5,6), (5,7), (7,8), (8,9), (8,10), (10,11), (11,1), (11,4)]
        },
        'C': {  # Cytosine
            'atoms': [
                ('N1', 'N', np.array([1.660, 4.387, 1.478])),
                ('C2', 'C', np.array([1.810, 3.008, 1.585])),
                ('O2', 'O', np.array([1.900, 2.501, 2.713])),
                ('N3', 'N', np.array([1.860, 2.297, 0.254])),
                ('C4', 'C', np.array([1.760, 2.843, -0.750])),
                ('N4', 'N', np.array([1.810, 2.070, -1.826])),
                ('C5', 'C', np.array([1.610, 4.258, -0.890])),
                ('C6', 'C', np.array([1.560, 4.983, 0.259])),
            ],
            'bonds': [(1,2), (2,3), (2,4), (4,5), (5,6), (5,7), (7,8), (8,1)]
        }
    }
    
    # Sugar-phosphate backbone atoms (deoxyribose)
    # Coordinates from reference PDB
    SUGAR_ATOMS = [
        ("C5'", 'C', np.array([-0.690, 7.424, 2.047])),
        ("C4'", 'C', np.array([0.040, 6.861, 3.247])),
        ("O4'", 'O', np.array([0.250, 5.429, 3.037])),
        ("C3'", 'C', np.array([1.440, 7.413, 3.508])),
        ("O3'", 'O', np.array([1.830, 7.271, 4.868])),
        ("C2'", 'C', np.array([2.320, 6.527, 2.637])),
        ("C1'", 'C', np.array([1.610, 5.184, 2.732])),
    ]
    
    # 5' end atoms (includes OH)
    FIVE_PRIME_END = [
        ('HTER', 'H', np.array([-1.547, 9.377, -1.216])),
        ('OXT', 'O', np.array([-1.440, 8.896, -0.401])),
    ]
    
    # Phosphate group atoms
    PHOSPHATE_ATOMS = [
        ('P', 'P', np.array([0.000, 8.910, 0.000])),
        ('O1P', 'O', np.array([0.220, 10.175, 0.734])),
        ('O2P', 'O', np.array([0.790, 8.733, -1.240])),
        ("O5'", 'O', np.array([0.250, 7.669, 0.971])),
    ]
    
    # 3' end cap (H)
    THREE_PRIME_CAP = ('HCAP', 'H', np.array([1.699, 6.361, 5.144]))
    
    # Watson-Crick base pairs
    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.atom_id = 1
        
    def rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Rotation matrix around X axis (angle in radians)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    def rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Rotation matrix around Y axis (angle in radians)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Rotation matrix around Z axis (angle in radians)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def apply_transform(self, coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply 4x4 transformation matrix to coordinates"""
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        # Add homogeneous coordinate
        rows = coords.shape[0]
        homo_coords = np.hstack((coords, np.ones((rows, 1))))
        
        # Apply transformation (use only 3x4 part of matrix)
        transformed = np.dot(homo_coords, matrix[:3, :].T)
        
        return transformed
    
    def make_translation(self, x: float, y: float, z: float) -> np.ndarray:
        """Create 4x4 translation matrix"""
        m = np.eye(4)
        m[:3, 3] = [x, y, z]
        return m
    
    def build_nucleotide(self, base: str, position: int, strand: int,
                        is_5_prime: bool, is_3_prime: bool) -> List[Dict]:
        """Build a complete nucleotide with proper geometry"""
        
        nucleotide_atoms = []
        residue_name = f' D{base}'
        chain = 'A' if strand == 1 else 'B'
        
        # Starting atom ID for this residue
        start_atom_id = self.atom_id
        
        # Add 5' end atoms if this is the first residue
        if is_5_prime:
            for name, element, coord in self.FIVE_PRIME_END:
                nucleotide_atoms.append({
                    'id': self.atom_id,
                    'name': name,
                    'element': element,
                    'coord': coord.copy(),
                    'residue_name': residue_name,
                    'chain': chain,
                    'residue_id': position
                })
                self.atom_id += 1
        
        # Add phosphate group (except for 5' end)
        if not is_5_prime:
            for name, element, coord in self.PHOSPHATE_ATOMS:
                nucleotide_atoms.append({
                    'id': self.atom_id,
                    'name': name,
                    'element': element,
                    'coord': coord.copy(),
                    'residue_name': residue_name,
                    'chain': chain,
                    'residue_id': position
                })
                self.atom_id += 1
        else:
            # For 5' end, only add O5'
            name, element, coord = self.PHOSPHATE_ATOMS[3]  # O5'
            nucleotide_atoms.append({
                'id': self.atom_id,
                'name': name,
                'element': element,
                'coord': coord.copy(),
                'residue_name': residue_name,
                'chain': chain,
                'residue_id': position
            })
            self.atom_id += 1
        
        # Add sugar atoms
        for name, element, coord in self.SUGAR_ATOMS:
            nucleotide_atoms.append({
                'id': self.atom_id,
                'name': name,
                'element': element,
                'coord': coord.copy(),
                'residue_name': residue_name,
                'chain': chain,
                'residue_id': position
            })
            self.atom_id += 1
        
        # Add base atoms
        base_data = self.BASE_ATOMS[base]
        for name, element, coord in base_data['atoms']:
            nucleotide_atoms.append({
                'id': self.atom_id,
                'name': name,
                'element': element,
                'coord': coord.copy(),
                'residue_name': residue_name,
                'chain': chain,
                'residue_id': position
            })
            self.atom_id += 1
        
        # Add 3' cap if this is the last residue
        if is_3_prime:
            name, element, coord = self.THREE_PRIME_CAP
            nucleotide_atoms.append({
                'id': self.atom_id,
                'name': name,
                'element': element,
                'coord': coord.copy(),
                'residue_name': residue_name,
                'chain': chain,
                'residue_id': position
            })
            self.atom_id += 1
        
        return nucleotide_atoms
    
    def generate_dna(self, sequence: str) -> None:
        """Generate B-DNA double helix from sequence with automatic complementary strand"""
        sequence = sequence.upper()
        n_bases = len(sequence)
        
        # Build strand 1 (5' to 3')
        print(f"Building strand 1 (5' to 3'): {sequence}")
        
        for i, base in enumerate(sequence):
            # Build nucleotide
            is_5_prime = (i == 0)
            is_3_prime = (i == n_bases - 1)
            
            nucleotide = self.build_nucleotide(base, i+1, 1, is_5_prime, is_3_prime)
            
            # Calculate position and orientation
            z_offset = i * self.DELTA_X
            twist_angle = i * np.radians(self.TWIST_PER_BASE)
            
            # Create transformation matrix
            trans_matrix = self.make_translation(0, 0, z_offset)
            rot_matrix = np.eye(4)
            rot_matrix[:3, :3] = self.rotation_matrix_z(twist_angle)
            
            # Combined transformation
            transform = np.dot(trans_matrix, rot_matrix)
            
            # Apply transformation to all atoms
            for atom in nucleotide:
                atom['coord'] = self.apply_transform(atom['coord'], transform)[0]
            
            self.atoms.extend(nucleotide)
        
        # Build strand 2 (3' to 5', complementary)
        # Generate complementary sequence
        complement_seq = ''.join([self.COMPLEMENT[base] for base in sequence[::-1]])
        print(f"Building strand 2 (3' to 5'): {complement_seq}")
        
        for i in range(n_bases):
            # Get complementary base
            original_base = sequence[n_bases - 1 - i]
            base = self.COMPLEMENT[original_base]
            
            # Build nucleotide
            is_5_prime = (i == 0)
            is_3_prime = (i == n_bases - 1)
            
            nucleotide = self.build_nucleotide(base, i+1, 2, is_5_prime, is_3_prime)
            
            # Calculate position for antiparallel strand
            # This base pairs with sequence[n_bases-1-i]
            z_offset = (n_bases - 1 - i) * self.DELTA_X + self.DELTA_X_REV_OFFSET
            # Add 180 degrees to position on opposite side of helix, plus offset
            twist_angle = (n_bases - 1 - i) * np.radians(self.TWIST_PER_BASE) + np.pi + self.THETA_REV_OFFSET
            
            # Create transformation matrices
            trans_matrix = self.make_translation(0, 0, z_offset)
            rot_z_matrix = np.eye(4)
            rot_z_matrix[:3, :3] = self.rotation_matrix_z(twist_angle)
            
            # Combined transformation
            transform = np.dot(trans_matrix, rot_z_matrix)
            
            # Flip around local Y axis to face bases inward
            flip_matrix = np.eye(4)
            flip_matrix[:3, :3] = self.rotation_matrix_y(np.pi)
            transform = np.dot(transform, flip_matrix)
            
            # Apply transformation
            for atom in nucleotide:
                atom['coord'] = self.apply_transform(atom['coord'], transform)[0]
            
            self.atoms.extend(nucleotide)
    
    def write_pdb(self, filename: str) -> None:
        """Write structure to PDB file"""
        with open(filename, 'w') as f:
            f.write("REMARK   Generated by FluxMD DNA Structure Generator\n")
            f.write("REMARK   B-DNA double helix\n")
            
            # Write atoms
            for atom in self.atoms:
                atom_name = atom['name']
                if len(atom_name) < 4:
                    atom_name = ' ' + atom_name.ljust(3)
                else:
                    atom_name = atom_name.ljust(4)
                
                # Format: ATOM or HETATM
                record_type = "ATOM  "
                
                f.write(f"{record_type}{atom['id']:>5d} {atom_name} "
                       f"{atom['residue_name']} {atom['chain']}{atom['residue_id']:>4d}    "
                       f"{atom['coord'][0]:>8.3f}{atom['coord'][1]:>8.3f}{atom['coord'][2]:>8.3f}"
                       f"  1.00  0.00          {atom['element']:>2s}  \n")
            
            # Add CONECT records for backbone connectivity
            self._write_conect_records(f)
            
            f.write("END\n")
    
    def _write_conect_records(self, f):
        """Write CONECT records for proper connectivity"""
        # Group atoms by residue
        residue_atoms = {}
        for atom in self.atoms:
            key = (atom['chain'], atom['residue_id'])
            if key not in residue_atoms:
                residue_atoms[key] = {}
            residue_atoms[key][atom['name']] = atom['id']
        
        # Write backbone connectivity
        for (chain, res_id), atoms in sorted(residue_atoms.items()):
            # Phosphate bonds
            if 'P' in atoms:
                if "O5'" in atoms:
                    f.write(f"CONECT{atoms['P']:>5d}{atoms["O5'"]:>5d}\n")
                if 'O1P' in atoms:
                    f.write(f"CONECT{atoms['P']:>5d}{atoms['O1P']:>5d}\n")
                if 'O2P' in atoms:
                    f.write(f"CONECT{atoms['P']:>5d}{atoms['O2P']:>5d}\n")
            
            # Sugar bonds
            sugar_bonds = [
                ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"),
                ("C3'", "O3'"), ("C3'", "C2'"), ("C2'", "C1'"),
                ("C1'", "O4'"), ("O4'", "C4'")
            ]
            
            for a1, a2 in sugar_bonds:
                if a1 in atoms and a2 in atoms:
                    f.write(f"CONECT{atoms[a1]:>5d}{atoms[a2]:>5d}\n")
            
            # Glycosidic bond
            if "C1'" in atoms:
                c1_id = atoms["C1'"]
                if 'N9' in atoms:  # Purine
                    n9_id = atoms['N9']
                    f.write(f"CONECT{c1_id:>5d}{n9_id:>5d}\n")
                elif 'N1' in atoms:  # Pyrimidine
                    n1_id = atoms['N1']
                    f.write(f"CONECT{c1_id:>5d}{n1_id:>5d}\n")
            
            # Connect O3' to next residue's P
            if "O3'" in atoms:
                # For chain A: connect to res_id + 1
                # For chain B: connect to res_id + 1 (both chains go 5' to 3' in structure)
                next_key = (chain, res_id + 1)
                if next_key in residue_atoms and 'P' in residue_atoms[next_key]:
                    f.write(f"CONECT{atoms["O3'"]:>5d}{residue_atoms[next_key]['P']:>5d}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Generate B-DNA double helix PDB files from sequences for FluxMD')
    parser.add_argument('sequence', help='DNA sequence (e.g., ATCGATCG)')
    parser.add_argument('-o', '--output', default='dna_structure.pdb',
                       help='Output PDB filename')
    
    args = parser.parse_args()
    
    # Validate sequence
    valid_bases = set('ATGC')
    if not all(base.upper() in valid_bases for base in args.sequence):
        print("Error: Sequence must contain only A, T, G, C")
        return
    
    # Generate structure
    print(f"Generating B-DNA double helix for: {args.sequence}")
    generator = DNAStructureGenerator()
    generator.generate_dna(args.sequence)
    generator.write_pdb(args.output)
    
    # Print statistics
    print(f"Total atoms: {len(generator.atoms)}")
    print(f"Base pairs: {len(args.sequence)}")
    print(f"Helix length: {len(args.sequence) * generator.DELTA_X:.1f} Å")
    print(f"Structure written to: {args.output}")

if __name__ == '__main__':
    main()

#   p53 consensus binding site [Note] ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACAGGCAAGTTTGATCTGGGGCATGCTTGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
