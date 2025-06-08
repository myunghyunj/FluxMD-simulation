#!/usr/bin/env python3
"""
DNA Sequence → PDB  (fixed B-DNA builder · 2025-06-08)
-----------------------------------------------------
• Canonical C1′ radius = 5.4 Å
• Bases now oriented toward helix axis
• Antiparallel strands, phosphates & CONECT remain as in v2
"""

import numpy as np
import argparse
import math

class DNAStructureGenerator:
    """
    Generates 3D B-DNA structures from DNA sequences with proper base pairing.
    """
    
    # B-DNA helical parameters
    RISE_PER_BASE = 3.38  # Rise per base pair in Angstroms
    TWIST_PER_BASE = 36.0  # Degrees of twist per base pair
    HELIX_RADIUS = 5.4    # Å (C1′ distance from axis)
    
    # Base pair geometry
    BASE_PAIR_WIDTH = 10.8  # Distance between C1' atoms in a base pair
    BASE_PAIR_PROPELLER = -11.0  # Propeller twist angle in degrees
    
    # Standard bond lengths
    C1_N_BOND = 1.48  # C1'-N9/N1 bond length
    O3_P_BOND = 1.61  # O3'-P phosphodiester bond
    P_O5_BOND = 1.61  # P-O5' phosphodiester bond
    
    # Watson-Crick base pairing
    COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    # Base atoms (relative to glycosidic bond at origin)
    BASE_ATOMS = {
        'A': [  # Adenine
            ('N9', 'N', np.array([0.000, 0.000, 0.000])),
            ('C8', 'C', np.array([0.000, -1.374, 0.000])),
            ('N7', 'N', np.array([1.155, -2.024, 0.000])),
            ('C5', 'C', np.array([2.062, -1.045, 0.000])),
            ('C6', 'C', np.array([3.447, -0.907, 0.000])),
            ('N6', 'N', np.array([4.331, -1.873, 0.000])),
            ('N1', 'N', np.array([3.909, 0.302, 0.000])),
            ('C2', 'C', np.array([3.025, 1.268, 0.000])),
            ('N3', 'N', np.array([1.704, 1.156, 0.000])),
            ('C4', 'C', np.array([1.265, 0.005, 0.000])),
        ],
        'T': [  # Thymine
            ('N1', 'N', np.array([0.000, 0.000, 0.000])),
            ('C2', 'C', np.array([0.000, 1.390, 0.000])),
            ('O2', 'O', np.array([0.999, 2.076, 0.000])),
            ('N3', 'N', np.array([-1.229, 1.903, 0.000])),
            ('C4', 'C', np.array([-2.380, 1.170, 0.000])),
            ('O4', 'O', np.array([-3.487, 1.678, 0.000])),
            ('C5', 'C', np.array([-2.217, -0.245, 0.000])),
            ('C5M','C', np.array([-3.453, -1.108, 0.000])),
            ('C6', 'C', np.array([-0.975, -0.757, 0.000])),
        ],
        'G': [  # Guanine
            ('N9', 'N', np.array([0.000, 0.000, 0.000])),
            ('C8', 'C', np.array([0.000, -1.374, 0.000])),
            ('N7', 'N', np.array([1.155, -2.024, 0.000])),
            ('C5', 'C', np.array([2.062, -1.045, 0.000])),
            ('C6', 'C', np.array([3.447, -0.907, 0.000])),
            ('O6', 'O', np.array([4.412, -1.831, 0.000])),
            ('N1', 'N', np.array([3.827, 0.394, 0.000])),
            ('C2', 'C', np.array([2.903, 1.441, 0.000])),
            ('N2', 'N', np.array([3.352, 2.690, 0.000])),
            ('N3', 'N', np.array([1.604, 1.249, 0.000])),
            ('C4', 'C', np.array([1.265, 0.005, 0.000])),
        ],
        'C': [  # Cytosine
            ('N1', 'N', np.array([0.000, 0.000, 0.000])),
            ('C2', 'C', np.array([0.000, 1.390, 0.000])),
            ('O2', 'O', np.array([0.999, 2.076, 0.000])),
            ('N3', 'N', np.array([-1.229, 1.903, 0.000])),
            ('C4', 'C', np.array([-2.281, 1.115, 0.000])),
            ('N4', 'N', np.array([-3.475, 1.642, 0.000])),
            ('C5', 'C', np.array([-2.108, -0.298, 0.000])),
            ('C6', 'C', np.array([-0.866, -0.810, 0.000])),
        ]
    }
    
    # Sugar-phosphate backbone atoms
    SUGAR_ATOMS = [
        ("C1'", 'C', np.array([0.000, 0.000, -1.480])),
        ("C2'", 'C', np.array([0.755, -1.217, -2.023])),
        ("O4'", 'O', np.array([0.829, 1.179, -1.728])),
        ("C3'", 'C', np.array([1.784, -0.564, -2.963])),
        ("C4'", 'C', np.array([2.173, 0.751, -2.279])),
        ("O3'", 'O', np.array([2.880, -1.433, -3.201])),
        ("C5'", 'C', np.array([3.204, 0.706, -1.154])),
        ("O5'", 'O', np.array([3.455, 2.007, -0.636])),
    ]
    
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

    def _rotate_y(self, coord, angle):
        """Rotate `coord` around the Y-axis by `angle` radians."""
        rot = np.array([[math.cos(angle), 0.0, math.sin(angle)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(angle), 0.0, math.cos(angle)]])
        return rot @ coord
    
    def _build_base_pair(self, base1, base2, bp_index, z_position, twist_angle):
        """
        Build a Watson-Crick base pair at the specified position.
        base1: base type for strand 1 (A, T, G, or C)
        base2: base type for strand 2 (should be complement of base1)
        bp_index: base pair index (0-based)
        z_position: position along helix axis
        twist_angle: cumulative twist angle
        """
        # Strand 1 (chain A)
        strand1_atoms = []
        
        # Find C1' position from sugar atoms
        c1_pos = None
        
        # Add sugar atoms for strand 1
        for name, element, coord in self.SUGAR_ATOMS:
            atom = {
                'name': name,
                'element': element,
                'coord': coord.copy(),
                'residue_id': bp_index + 1,
                'residue_name': f'D{base1}',
                'chain': 'A',
                'atom_id': self.atom_id_counter
            }
            if name == "C1'":
                c1_pos = coord.copy()
            self.atom_id_counter += 1
            strand1_atoms.append(atom)
        
        # Add base atoms for strand 1 - flipped 180° about Y so bases face axis
        for name, element, coord in self.BASE_ATOMS[base1]:
            coord = self._rotate_y(coord, math.pi)
            attached_coord = coord + c1_pos
            attached_coord[2] += self.C1_N_BOND
            
            atom = {
                'name': name,
                'element': element,
                'coord': attached_coord,
                'residue_id': bp_index + 1,
                'residue_name': f'D{base1}',
                'chain': 'A',
                'atom_id': self.atom_id_counter
            }
            self.atom_id_counter += 1
            strand1_atoms.append(atom)
        
        # Position strand 1
        for atom in strand1_atoms:
            # Move to helix radius
            atom['coord'][0] += self.HELIX_RADIUS
            # Apply twist
            atom['coord'] = self._rotate_z(atom['coord'], twist_angle)
            # Move to z position
            atom['coord'][2] += z_position
        
        # Strand 2 (chain B) - antiparallel
        strand2_atoms = []
        
        # Find C1' position for strand 2
        c1_pos_2 = None
        
        # Add sugar atoms for strand 2
        for name, element, coord in self.SUGAR_ATOMS:
            atom = {
                'name': name,
                'element': element,
                'coord': coord.copy(),
                'residue_id': bp_index + 1,
                'residue_name': f'D{base2}',
                'chain': 'B',
                'atom_id': self.atom_id_counter
            }
            if name == "C1'":
                c1_pos_2 = coord.copy()
            self.atom_id_counter += 1
            strand2_atoms.append(atom)
        
        # Add base atoms for strand 2 - properly attached to C1'
        for name, element, coord in self.BASE_ATOMS[base2]:
            # Translate base to be attached to C1'
            attached_coord = coord + c1_pos_2
            # Add glycosidic bond vector to position base properly
            attached_coord[2] += self.C1_N_BOND
            
            atom = {
                'name': name,
                'element': element,
                'coord': attached_coord,
                'residue_id': bp_index + 1,
                'residue_name': f'D{base2}',
                'chain': 'B',
                'atom_id': self.atom_id_counter
            }
            self.atom_id_counter += 1
            strand2_atoms.append(atom)
        
        # Position strand 2 (opposite strand)
        for atom in strand2_atoms:
            # Move to helix radius on opposite side
            atom['coord'][0] -= self.HELIX_RADIUS
            # Apply twist (same angle as strand 1)
            atom['coord'] = self._rotate_z(atom['coord'], twist_angle)
            # Move to z position
            atom['coord'][2] += z_position
        
        return strand1_atoms, strand2_atoms
    
    def generate_dna(self, sequence):
        """Generate the full double helix structure with antiparallel strands."""
        n_bases = len(sequence)
        
        # Create the full chain B sequence (5' to 3')
        # This is the reverse complement of chain A
        chain_b_seq = ''.join(reversed([self.COMPLEMENT[base] for base in sequence]))
        
        # Storage for all atoms
        all_atoms_a = []
        all_atoms_b = []
        o3_positions_a = {}
        o3_positions_b = {}
        
        # Build base pairs
        for i in range(n_bases):
            z_pos = i * self.RISE_PER_BASE
            twist = math.radians(i * self.TWIST_PER_BASE)
            
            # Chain A base at this position
            base_a = sequence[i]
            res_id_a = i + 1
            
            # Chain B base at this position (antiparallel)
            # At z-position i, chain B has residue (n_bases - i)
            res_id_b = n_bases - i
            base_b = chain_b_seq[res_id_b - 1]  # -1 because residues are 1-based
            
            # Verify Watson-Crick pairing
            assert self.COMPLEMENT[base_a] == base_b, f"Base pair mismatch at position {i}: {base_a}-{base_b}"
            
            # Build the base pair with correct residue IDs
            atoms_a, atoms_b = self._build_base_pair(base_a, base_b, i, z_pos, twist)
            
            # Update residue ID for chain B
            for atom in atoms_b:
                atom['residue_id'] = res_id_b
            
            # Store O3' positions for connectivity
            for atom in atoms_a:
                if atom['name'] == "O3'":
                    o3_positions_a[res_id_a] = atom['coord']
            for atom in atoms_b:
                if atom['name'] == "O3'":
                    o3_positions_b[res_id_b] = atom['coord']
            
            all_atoms_a.extend(atoms_a)
            all_atoms_b.extend(atoms_b)
        
        # Add phosphate groups for connectivity
        # Chain A: connect residues 1→2→3→4
        for i in range(2, n_bases + 1):
            if i-1 in o3_positions_a:
                self._add_phosphate_to_residue(all_atoms_a, 'A', i, o3_positions_a[i-1])
        
        # Chain B: connect residues 1→2→3→4 (in their 5' to 3' order)
        for i in range(2, n_bases + 1):
            if i-1 in o3_positions_b:
                self._add_phosphate_to_residue(all_atoms_b, 'B', i, o3_positions_b[i-1])
        
        # Combine all atoms
        self.atoms = all_atoms_a + all_atoms_b
    
    def _add_phosphate_to_residue(self, atoms_list, chain, res_id, prev_o3_pos):
        """Add phosphate group to connect to previous residue with improved geometry."""
        # Find O5' and C5' of current residue
        o5_pos = None
        c5_pos = None
        for atom in atoms_list:
            if atom['residue_id'] == res_id:
                if atom['name'] == "O5'":
                    o5_pos = atom['coord']
                elif atom['name'] == "C5'":
                    c5_pos = atom['coord']
        
        if o5_pos is not None and c5_pos is not None:
            # Calculate phosphate position with proper tetrahedral geometry
            p_pos = self._calculate_phosphate_position(prev_o3_pos, o5_pos, c5_pos)
            
            # Calculate positions for O1P and O2P with tetrahedral geometry
            o1p_pos, o2p_pos = self._calculate_phosphate_oxygens(p_pos, prev_o3_pos, o5_pos)
            
            # Add phosphate atoms
            phosphate_atoms = [
                {'name': 'P', 'element': 'P', 'coord': p_pos},
                {'name': 'O1P', 'element': 'O', 'coord': o1p_pos},
                {'name': 'O2P', 'element': 'O', 'coord': o2p_pos},
            ]
            
            for atom_data in phosphate_atoms:
                atom_data.update({
                    'residue_id': res_id,
                    'residue_name': next(a['residue_name'] for a in atoms_list if a['residue_id'] == res_id),
                    'chain': chain,
                    'atom_id': self.atom_id_counter
                })
                self.atom_id_counter += 1
                atoms_list.append(atom_data)
    
    def _calculate_phosphate_position(self, o3_prev, o5_curr, c5_curr):
        """Calculate phosphate position with proper geometry."""
        # Calculate midpoint between O3' and O5'
        midpoint = (o3_prev + o5_curr) / 2.0
        
        # Offset slightly towards O5' to avoid overlaps
        direction = o5_curr - o3_prev
        direction = direction / np.linalg.norm(direction)
        
        # Position phosphate closer to O5' to avoid overlaps with previous residue
        p_pos = midpoint + direction * 0.5
        
        return p_pos
    
    def _calculate_phosphate_oxygens(self, p_pos, o3_pos, o5_pos):
        """Calculate O1P and O2P positions with tetrahedral geometry."""
        # Vector along P-O3' bond
        po3_vec = o3_pos - p_pos
        po3_vec = po3_vec / np.linalg.norm(po3_vec)
        
        # Vector along P-O5' bond
        po5_vec = o5_pos - p_pos
        po5_vec = po5_vec / np.linalg.norm(po5_vec)
        
        # Create perpendicular vector using cross product
        perp = np.cross(po3_vec, po5_vec)
        if np.linalg.norm(perp) > 0:
            perp = perp / np.linalg.norm(perp)
        else:
            # If vectors are parallel, use arbitrary perpendicular
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(perp, po3_vec)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
        
        # Create another perpendicular vector
        perp2 = np.cross(po3_vec, perp)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Position O1P and O2P at tetrahedral angles
        bond_length = 1.48  # P-O bond length
        angle = math.radians(109.5)  # Tetrahedral angle
        
        o1p_pos = p_pos + bond_length * (
            -0.5 * (po3_vec + po5_vec) + 
            math.sin(angle) * perp
        )
        
        o2p_pos = p_pos + bond_length * (
            -0.5 * (po3_vec + po5_vec) - 
            math.sin(angle) * perp
        )
        
        return o1p_pos, o2p_pos

    def write_pdb(self, filename, include_conect=False):
        """Write the generated structure to a PDB file with optional CONECT records."""
        with open(filename, 'w') as f:
            f.write("REMARK   Generated by DNA to PDB Converter V2\n")
            f.write("REMARK   B-DNA double helix with proper backbone connectivity\n")
            f.write(f"REMARK   Total base pairs: {max(a['residue_id'] for a in self.atoms if a['chain'] == 'A')}\n")
            
            if not include_conect:
                f.write("REMARK   No CONECT records - use viewer's automatic bond detection\n")
                f.write("REMARK   For DNA visualization:\n")
                f.write("REMARK     PyMOL: set connect_mode, 1\n")
                f.write("REMARK     VMD: mol representation CPK\n")
                f.write("REMARK     ChimeraX: automatic\n")
            else:
                f.write("REMARK   CONECT records included for backbone connectivity\n")
                f.write("REMARK   Warning: Large DNA may have performance issues with CONECT\n")
            
            f.write("REMARK   Color scheme: P=red, pyrimidines=blue, purines=green, backbone=yellow\n")
            
            # Sort atoms by chain, then residue, then atom name for proper ordering
            sorted_atoms = sorted(self.atoms, key=lambda a: (a['chain'], a['residue_id'], 
                                                            0 if a['name'] == 'P' else 
                                                            1 if 'P' in a['name'] else 
                                                            2 if "'" in a['name'] else 3))
            
            # Create atom index mapping for CONECT records
            atom_indices = {}
            atom_counter = 1
            
            # Track last residue for each chain for TER records
            last_res_a = max((a['residue_id'] for a in self.atoms if a['chain'] == 'A'), default=0)
            last_res_b = max((a['residue_id'] for a in self.atoms if a['chain'] == 'B'), default=0)
            
            # Write ATOM records
            current_chain = None
            for atom in sorted_atoms:
                # Add TER record when switching chains
                if current_chain and current_chain != atom['chain']:
                    f.write(f"TER   {atom_counter:>5}      {sorted_atoms[atom_counter-2]['residue_name']:>3} {current_chain}{sorted_atoms[atom_counter-2]['residue_id']:>4}\n")
                    atom_counter += 1
                
                current_chain = atom['chain']
                
                # Store atom index for CONECT records
                atom['index'] = atom_counter
                atom_indices[(atom['chain'], atom['residue_id'], atom['name'])] = atom_counter
                
                atom_name = atom['name']
                # PDB format spacing
                if len(atom_name) < 4:
                    if atom['element'] in ['C', 'N', 'O', 'P', 'S']:
                        atom_name = ' ' + atom_name.ljust(3)
                    else:
                        atom_name = atom_name.ljust(4)
                
                line = (
                    f"ATOM  {atom_counter:>5} {atom_name:<4} {atom['residue_name']:>3} {atom['chain']}"
                    f"{atom['residue_id']:>4}    {atom['coord'][0]:>8.3f}{atom['coord'][1]:>8.3f}"
                    f"{atom['coord'][2]:>8.3f}  1.00  0.00          {atom['element']:>2}\n"
                )
                f.write(line)
                atom_counter += 1
            
            # Add final TER record
            if sorted_atoms:
                last_atom = sorted_atoms[-1]
                f.write(f"TER   {atom_counter:>5}      {last_atom['residue_name']:>3} {last_atom['chain']}{last_atom['residue_id']:>4}\n")
            
            # Write CONECT records if requested
            if include_conect and len(self.atoms) < 1000:  # Only for small structures
                self._write_minimal_conect_records(f, sorted_atoms, atom_indices)
            
            f.write("END\n")
    
    def _write_minimal_conect_records(self, f, sorted_atoms, atom_indices):
        """Write minimal CONECT records for backbone connectivity only."""
        f.write("REMARK   Minimal CONECT records for backbone connectivity\n")
        
        # Only write backbone connections
        for atom in sorted_atoms:
            chain = atom['chain']
            res_id = atom['residue_id']
            atom_name = atom['name']
            atom_idx = atom['index']
            
            connections = []
            
            # Only phosphodiester backbone connections
            if atom_name == "O3'" and (chain, res_id + 1, 'P') in atom_indices:
                connections.append(atom_indices[(chain, res_id + 1, 'P')])
            elif atom_name == 'P':
                if (chain, res_id - 1, "O3'") in atom_indices:
                    connections.append(atom_indices[(chain, res_id - 1, "O3'")])
                if (chain, res_id, "O5'") in atom_indices:
                    connections.append(atom_indices[(chain, res_id, "O5'")])
            elif atom_name == "O5'" and (chain, res_id, 'P') in atom_indices:
                connections.append(atom_indices[(chain, res_id, 'P')])
            
            if connections:
                conect_line = f"CONECT{atom_idx:>5}"
                for conn_idx in connections:
                    conect_line += f"{conn_idx:>5}"
                f.write(conect_line + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Generate a B-DNA double helix PDB file from a DNA sequence.'
    )
    parser.add_argument('sequence', help='DNA sequence (5\' to 3\', e.g., "ATGC")')
    parser.add_argument('-o', '--output', default='dna_structure.pdb', 
                       help='Output PDB filename (default: dna_structure.pdb)')
    parser.add_argument('--conect', action='store_true',
                       help='Include CONECT records (not recommended for large DNA)')
    
    args = parser.parse_args()
    
    # Validate sequence
    sequence_upper = args.sequence.upper()
    if not all(base in 'ATGC' for base in sequence_upper):
        print("Error: Sequence must only contain A, T, G, or C.")
        return
    
    print(f"Generating B-DNA structure for sequence: 5'-{sequence_upper}-3'")
    complement = ''.join(DNAStructureGenerator.COMPLEMENT[b] for b in sequence_upper)
    print(f"Complementary strand: 3'-{complement}-5'")
    
    # Generate structure
    generator = DNAStructureGenerator()
    generator.generate_dna(sequence_upper)
    generator.write_pdb(args.output, include_conect=args.conect)
    
    print(f"\nSuccessfully generated {args.output}")
    print(f"Chain A: 5'-{sequence_upper}-3' (residues 1-{len(sequence_upper)})")
    print(f"Chain B: 3'-{complement}-5' (residues 1-{len(sequence_upper)})")
    print(f"Total atoms: {len(generator.atoms)}")
    
    if not args.conect:
        print("\nViewing tips:")
        print("- PyMOL will automatically detect bonds")
        print("- For large DNA, use: set connect_mode, 1")
        print("- Color by element: color atomic")

if __name__ == '__main__':
    main()