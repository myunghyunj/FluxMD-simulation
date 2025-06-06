#!/usr/bin/env python3
"""
DNA Sequence to PDB Converter
Generates a 3D B-DNA double helix structure from a given DNA sequence.
It automatically creates the complementary strand with Watson-Crick base pairing
and constructs a PDB file with correct atomic coordinates.

NOTE: CONECT records are omitted to ensure compatibility with visualization
software like PyMOL for very large structures. Viewers will infer bonds
based on standard residue templates and atom proximity.
"""

import numpy as np
import argparse
import math

class DNAStructureGenerator:
    """
    Generates 3D B-DNA structures from DNA sequences.
    The class holds atomic coordinate data and applies geometric transformations
    to build a double helix structure.
    """
    
    # B-DNA helical parameters
    DELTA_Z = 3.38  # Rise per base pair in Angstroms
    TWIST_PER_BASE = 34.3  # Degrees of twist per base pair
    HELIX_RADIUS = 10.0  # Radius of the helix in Angstroms
    
    # Atom coordinates for bases positioned at standard orientation
    # These will be transformed to create the helix
    BASE_ATOMS = {
        'A': {
            'atoms': [
                ('N9', 'N', np.array([1.660, 0.000, 0.000])),
                ('C8', 'C', np.array([1.580, 0.000, 1.390])),
                ('N7', 'N', np.array([1.650, 0.000, 2.490])),
                ('C5', 'C', np.array([1.800, 0.000, 3.760])),
                ('C6', 'C', np.array([1.930, 0.000, 5.130])),
                ('N6', 'N', np.array([1.940, 0.000, 6.420])),
                ('N1', 'N', np.array([2.050, 0.000, 5.950])),
                ('C2', 'C', np.array([2.040, 0.000, 4.660])),
                ('N3', 'N', np.array([1.920, 0.000, 3.540])),
                ('C4', 'C', np.array([1.800, 0.000, 2.730])),
            ]
        },
        'T': {
            'atoms': [
                ('N1', 'N', np.array([1.660, 0.000, 0.000])),
                ('C2', 'C', np.array([1.810, 0.000, 1.370])),
                ('O2', 'O', np.array([1.900, 0.000, 2.420])),
                ('N3', 'N', np.array([1.850, 0.000, 1.280])),
                ('C4', 'C', np.array([1.760, 0.000, 2.570])),
                ('O4', 'O', np.array([1.810, 0.000, 3.640])),
                ('C5', 'C', np.array([1.610, 0.000, 2.460])),
                ('C5M','C', np.array([1.500, 0.000, 3.830])),
                ('C6', 'C', np.array([1.560, 0.000, 1.170])),
            ]
        },
        'G': {
            'atoms': [
                ('N9', 'N', np.array([1.660, 0.000, 0.000])),
                ('C8', 'C', np.array([1.580, 0.000, 1.390])),
                ('N7', 'N', np.array([1.660, 0.000, 2.490])),
                ('C5', 'C', np.array([1.800, 0.000, 3.760])),
                ('C6', 'C', np.array([1.930, 0.000, 5.130])),
                ('O6', 'O', np.array([1.950, 0.000, 6.330])),
                ('N1', 'N', np.array([2.050, 0.000, 5.950])),
                ('C2', 'C', np.array([2.050, 0.000, 4.660])),
                ('N2', 'N', np.array([2.180, 0.000, 4.450])),
                ('N3', 'N', np.array([1.920, 0.000, 3.540])),
                ('C4', 'C', np.array([1.800, 0.000, 2.730])),
            ]
        },
        'C': {
            'atoms': [
                ('N1', 'N', np.array([1.660, 0.000, 0.000])),
                ('C2', 'C', np.array([1.810, 0.000, 1.370])),
                ('O2', 'O', np.array([1.900, 0.000, 2.420])),
                ('N3', 'N', np.array([1.860, 0.000, 1.280])),
                ('C4', 'C', np.array([1.760, 0.000, 2.570])),
                ('N4', 'N', np.array([1.810, 0.000, 3.640])),
                ('C5', 'C', np.array([1.610, 0.000, 2.460])),
                ('C6', 'C', np.array([1.560, 0.000, 1.170])),
            ]
        }
    }
    
    # Sugar atoms positioned relative to base
    SUGAR_ATOMS = [
        ("C5'", 'C', np.array([-2.350, 0.000, -1.550])),
        ("C4'", 'C', np.array([-1.620, 0.000, -2.850])),
        ("O4'", 'O', np.array([-0.360, 0.000, -2.640])),
        ("C3'", 'C', np.array([-2.320, 0.000, -4.110])),
        ("O3'", 'O', np.array([-2.130, 0.000, -5.420])),
        ("C2'", 'C', np.array([-1.490, 0.000, -3.790])),
        ("C1'", 'C', np.array([-0.140, 0.000, -1.290])),
    ]
    
    # Phosphate atoms
    PHOSPHATE_ATOMS = [
        ('P',   'P', np.array([-2.270, 0.000, -6.910])),
        ('O1P', 'O', np.array([-2.450, -1.200, -7.640])),
        ('O2P', 'O', np.array([-2.490, 1.200, -7.640])),
        ("O5'", 'O', np.array([-1.700, 0.000, -0.370])),
    ]

    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    def __init__(self):
        self.atoms = []
        self.atom_id_counter = 1
        
    def _rotate_z(self, coord, angle):
        """Rotate coordinate around Z axis"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y, z = coord
        return np.array([x*cos_a - y*sin_a, x*sin_a + y*cos_a, z])
    
    def _rotate_x(self, coord, angle):
        """Rotate coordinate around X axis"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y, z = coord
        return np.array([x, y*cos_a - z*sin_a, y*sin_a + z*cos_a])
    
    def _build_nucleotide(self, base, res_id, chain, position, twist_angle, is_complement=False):
        """Constructs a single nucleotide at the specified position and orientation."""
        nucleotide_atoms = []
        
        # Determine if this is the first residue (no 5' phosphate for first residue)
        is_first = (res_id == 1)
        
        # Add atoms (skip phosphate for first residue)
        if not is_first:
            for name, element, coord in self.PHOSPHATE_ATOMS:
                nucleotide_atoms.append({
                    'name': name,
                    'element': element,
                    'coord': coord.copy()
                })
        
        for name, element, coord in self.SUGAR_ATOMS:
            nucleotide_atoms.append({
                'name': name,
                'element': element,
                'coord': coord.copy()
            })
        
        for name, element, coord in self.BASE_ATOMS[base]['atoms']:
            nucleotide_atoms.append({
                'name': name,
                'element': element,
                'coord': coord.copy()
            })
        
        # Transform all atoms
        for atom in nucleotide_atoms:
            # For complementary strand, rotate 180° around X axis (flip upside down)
            if is_complement:
                atom['coord'] = self._rotate_x(atom['coord'], math.pi)
            
            # Position at helix radius
            atom['coord'][0] += self.HELIX_RADIUS
            
            # Apply helical twist
            atom['coord'] = self._rotate_z(atom['coord'], twist_angle)
            
            # Translate along Z axis
            atom['coord'][2] += position
            
            # Add metadata
            atom['residue_id'] = res_id
            atom['residue_name'] = f'D{base}'
            atom['chain'] = chain
            atom['atom_id'] = self.atom_id_counter
            self.atom_id_counter += 1
        
        return nucleotide_atoms

    def generate_dna(self, sequence):
        """Generates the full double helix structure."""
        n_bases = len(sequence)
        
        # Strand 1 (5' to 3', Chain A)
        for i, base in enumerate(sequence):
            res_id = i + 1
            z_position = i * self.DELTA_Z
            twist_angle = math.radians(i * self.TWIST_PER_BASE)
            
            nucleotide = self._build_nucleotide(base, res_id, 'A', z_position, twist_angle, False)
            self.atoms.extend(nucleotide)
        
        # Strand 2 (3' to 5', Chain B) - antiparallel and complementary
        complement_seq = [self.COMPLEMENT[b] for b in reversed(sequence)]
        
        for i, base in enumerate(complement_seq):
            res_id = i + 1
            # Position this base pair-wise with its partner on strand 1
            partner_index = n_bases - 1 - i
            z_position = partner_index * self.DELTA_Z
            # Same twist as partner, but add 180° offset for opposite side of helix
            twist_angle = math.radians(partner_index * self.TWIST_PER_BASE + 180)
            
            nucleotide = self._build_nucleotide(base, res_id, 'B', z_position, twist_angle, True)
            self.atoms.extend(nucleotide)

    def write_pdb(self, filename):
        """Writes the generated structure to a PDB file."""
        with open(filename, 'w') as f:
            f.write("REMARK   Generated by DNA to PDB Python Script\n")
            f.write("REMARK   B-DNA double helix structure\n")
            f.write(f"REMARK   Total base pairs: {len([a for a in self.atoms if a['chain'] == 'A' and a['name'] == 'N1' or a['name'] == 'N9'])}\n")
            
            for atom in self.atoms:
                atom_name = atom['name']
                # PDB format requires specific spacing for atom names
                if len(atom_name) < 4:
                    if atom['element'] in ['C', 'N', 'O', 'P', 'S']:
                        atom_name = ' ' + atom_name.ljust(3)
                    else:
                        atom_name = atom_name.ljust(4)
                
                line = (
                    f"ATOM  {atom['atom_id']:>5} {atom_name:<4} {atom['residue_name']:>3} {atom['chain']}"
                    f"{atom['residue_id']:>4}    {atom['coord'][0]:>8.3f}{atom['coord'][1]:>8.3f}"
                    f"{atom['coord'][2]:>8.3f}  1.00  0.00          {atom['element']:>2}\n"
                )
                f.write(line)
            
            f.write("END\n")

def main():
    parser = argparse.ArgumentParser(description='Generate a B-DNA double helix PDB file from a DNA sequence.')
    parser.add_argument('sequence', help='The DNA sequence (e.g., "ATGC").')
    parser.add_argument('-o', '--output', default='dna_structure.pdb', help='Output PDB filename.')
    
    args = parser.parse_args()
    
    sequence_upper = args.sequence.upper()
    if not all(base in 'ATGC' for base in sequence_upper):
        print("Error: Sequence must only contain A, T, G, or C.")
        return
    
    print(f"Generating B-DNA structure for sequence: {sequence_upper}")
    print(f"Complementary strand: {''.join(DNAStructureGenerator.COMPLEMENT[b] for b in reversed(sequence_upper))}")
    
    generator = DNAStructureGenerator()
    generator.generate_dna(sequence_upper)
    generator.write_pdb(args.output)
    
    print(f"Successfully generated {args.output}")
    print(f"Total atoms: {len(generator.atoms)}")
    print(f"Structure contains {len(sequence_upper)} base pairs")

if __name__ == '__main__':
    main()
