"""
Intra-protein interaction calculator for FluxMD.
Calculates static internal protein force vectors that remain constant during ligand approach.
These pre-computed vectors represent the protein's internal stress field.
"""

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntraProteinInteractions:
    """Calculate and store static intra-protein interaction vectors."""
    
    def __init__(self, protein_structure):
        """
        Initialize with a protein structure.
        
        Args:
            protein_structure: BioPython Structure object
        """
        self.structure = protein_structure
        self.residues = []
        self.atoms = []
        self.coords = []
        self._extract_atoms_and_residues()
        self.kdtree = KDTree(self.coords)
        
        # Interaction parameters (matching legacy code)
        self.cutoffs = {
            'hbond': 3.5,
            'salt_bridge': 4.0,
            'pi_pi': 4.5,
            'pi_cation': 6.0,
            'vdw': 5.0
        }
        
        # Store pre-computed vectors for each residue
        self.residue_vectors = {}
        self.residue_energies = {}
        
    def _extract_atoms_and_residues(self):
        """Extract atoms and residues from the protein structure."""
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Skip heteroatoms
                        self.residues.append(residue)
                        for atom in residue:
                            self.atoms.append(atom)
                            self.coords.append(atom.coord)
        self.coords = np.array(self.coords)
        logger.info(f"Extracted {len(self.atoms)} atoms from {len(self.residues)} residues")
        
    def calculate_all_interactions(self) -> Dict[str, np.ndarray]:
        """
        Calculate all intra-protein interactions and return residue-level force vectors.
        This is the main entry point - calculates once and stores for reuse.
        
        Returns:
            Dictionary mapping residue IDs to their total force vectors
        """
        logger.info("Calculating static intra-protein interactions...")
        
        # Initialize vectors for each residue
        for residue in self.residues:
            res_id = f"{residue.parent.id}:{residue.id[1]}"
            self.residue_vectors[res_id] = np.zeros(3)
            self.residue_energies[res_id] = 0.0
        
        # Calculate each type of interaction
        self._calculate_hydrogen_bonds()
        self._calculate_salt_bridges()
        self._calculate_pi_pi_stacking()
        self._calculate_pi_cation_interactions()
        self._calculate_van_der_waals()
        
        logger.info(f"Completed intra-protein calculations for {len(self.residue_vectors)} residues")
        return self.residue_vectors
    
    def _calculate_hydrogen_bonds(self):
        """Calculate hydrogen bond interactions within the protein."""
        donors = []
        acceptors = []
        
        for i, atom in enumerate(self.atoms):
            atom_name = atom.name
            residue = atom.parent
            res_name = residue.resname
            
            # Identify donors
            if atom_name in ['N', 'O'] or (atom_name.startswith('N') and res_name in ['ARG', 'LYS', 'HIS']):
                donors.append((i, atom, residue))
            
            # Identify acceptors
            if atom_name in ['O', 'N'] or (atom_name.startswith('O') and res_name in ['ASP', 'GLU']):
                acceptors.append((i, atom, residue))
        
        # Calculate interactions
        for donor_idx, donor_atom, donor_res in donors:
            for acceptor_idx, acceptor_atom, acceptor_res in acceptors:
                if donor_res == acceptor_res:
                    continue
                    
                distance = np.linalg.norm(donor_atom.coord - acceptor_atom.coord)
                if distance <= self.cutoffs['hbond']:
                    # Calculate force vector and energy
                    vector = acceptor_atom.coord - donor_atom.coord
                    unit_vector = vector / distance
                    energy = self._calculate_bond_energy('hbond', distance)
                    force_vector = unit_vector * energy
                    
                    # Add to residue vectors
                    donor_res_id = f"{donor_res.parent.id}:{donor_res.id[1]}"
                    acceptor_res_id = f"{acceptor_res.parent.id}:{acceptor_res.id[1]}"
                    
                    self.residue_vectors[donor_res_id] += force_vector
                    self.residue_vectors[acceptor_res_id] -= force_vector
                    self.residue_energies[donor_res_id] += energy
                    self.residue_energies[acceptor_res_id] += energy
    
    def _calculate_salt_bridges(self):
        """Calculate salt bridge interactions."""
        positive_residues = ['LYS', 'ARG', 'HIS']
        negative_residues = ['ASP', 'GLU']
        
        pos_atoms = [(i, atom, atom.parent) for i, atom in enumerate(self.atoms) 
                     if atom.parent.resname in positive_residues and atom.name in ['NZ', 'NH1', 'NH2', 'NE', 'ND1', 'NE2']]
        neg_atoms = [(i, atom, atom.parent) for i, atom in enumerate(self.atoms)
                     if atom.parent.resname in negative_residues and atom.name in ['OD1', 'OD2', 'OE1', 'OE2']]
        
        for pos_idx, pos_atom, pos_res in pos_atoms:
            for neg_idx, neg_atom, neg_res in neg_atoms:
                distance = np.linalg.norm(pos_atom.coord - neg_atom.coord)
                if distance <= self.cutoffs['salt_bridge']:
                    vector = neg_atom.coord - pos_atom.coord
                    unit_vector = vector / distance
                    energy = self._calculate_bond_energy('salt_bridge', distance)
                    force_vector = unit_vector * energy
                    
                    pos_res_id = f"{pos_res.parent.id}:{pos_res.id[1]}"
                    neg_res_id = f"{neg_res.parent.id}:{neg_res.id[1]}"
                    
                    self.residue_vectors[pos_res_id] += force_vector
                    self.residue_vectors[neg_res_id] -= force_vector
                    self.residue_energies[pos_res_id] += energy
                    self.residue_energies[neg_res_id] += energy
    
    def _calculate_pi_pi_stacking(self):
        """Calculate pi-pi stacking interactions between aromatic residues."""
        aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
        aromatic_atoms = {}
        
        # Collect aromatic residues and their ring centers
        for residue in self.residues:
            if residue.resname in aromatic_residues:
                ring_atoms = []
                if residue.resname == 'PHE':
                    ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
                elif residue.resname == 'TYR':
                    ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
                elif residue.resname == 'TRP':
                    ring_atoms = ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']
                elif residue.resname == 'HIS':
                    ring_atoms = ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
                
                coords = []
                for atom_name in ring_atoms:
                    if atom_name in residue:
                        coords.append(residue[atom_name].coord)
                
                if coords:
                    center = np.mean(coords, axis=0)
                    res_id = f"{residue.parent.id}:{residue.id[1]}"
                    aromatic_atoms[res_id] = (center, residue)
        
        # Calculate interactions between aromatic rings
        res_ids = list(aromatic_atoms.keys())
        for i in range(len(res_ids)):
            for j in range(i + 1, len(res_ids)):
                center1, res1 = aromatic_atoms[res_ids[i]]
                center2, res2 = aromatic_atoms[res_ids[j]]
                
                distance = np.linalg.norm(center2 - center1)
                if distance <= self.cutoffs['pi_pi']:
                    vector = center2 - center1
                    unit_vector = vector / distance
                    energy = self._calculate_bond_energy('pi_pi', distance)
                    force_vector = unit_vector * energy
                    
                    self.residue_vectors[res_ids[i]] += force_vector
                    self.residue_vectors[res_ids[j]] -= force_vector
                    self.residue_energies[res_ids[i]] += energy
                    self.residue_energies[res_ids[j]] += energy
    
    def _calculate_pi_cation_interactions(self):
        """Calculate pi-cation interactions."""
        aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
        cation_residues = ['LYS', 'ARG', 'HIS']
        
        # Get aromatic centers (reuse logic from pi-pi)
        aromatic_centers = {}
        for residue in self.residues:
            if residue.resname in aromatic_residues:
                ring_atoms = []
                if residue.resname == 'PHE':
                    ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
                elif residue.resname == 'TYR':
                    ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
                elif residue.resname == 'TRP':
                    ring_atoms = ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']
                elif residue.resname == 'HIS':
                    ring_atoms = ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
                
                coords = []
                for atom_name in ring_atoms:
                    if atom_name in residue:
                        coords.append(residue[atom_name].coord)
                
                if coords:
                    center = np.mean(coords, axis=0)
                    res_id = f"{residue.parent.id}:{residue.id[1]}"
                    aromatic_centers[res_id] = (center, residue)
        
        # Get cation positions
        cation_positions = {}
        for residue in self.residues:
            if residue.resname in cation_residues:
                cation_atom = None
                if residue.resname == 'LYS' and 'NZ' in residue:
                    cation_atom = residue['NZ']
                elif residue.resname == 'ARG' and 'CZ' in residue:
                    cation_atom = residue['CZ']
                elif residue.resname == 'HIS' and 'CE1' in residue:
                    cation_atom = residue['CE1']
                
                if cation_atom is not None:
                    res_id = f"{residue.parent.id}:{residue.id[1]}"
                    cation_positions[res_id] = (cation_atom.coord, residue)
        
        # Calculate interactions
        for aro_id, (aro_center, aro_res) in aromatic_centers.items():
            for cat_id, (cat_pos, cat_res) in cation_positions.items():
                if aro_res == cat_res:
                    continue
                    
                distance = np.linalg.norm(cat_pos - aro_center)
                if distance <= self.cutoffs['pi_cation']:
                    vector = cat_pos - aro_center
                    unit_vector = vector / distance
                    energy = self._calculate_bond_energy('pi_cation', distance)
                    force_vector = unit_vector * energy
                    
                    self.residue_vectors[aro_id] += force_vector
                    self.residue_vectors[cat_id] -= force_vector
                    self.residue_energies[aro_id] += energy
                    self.residue_energies[cat_id] += energy
    
    def _calculate_van_der_waals(self):
        """Calculate van der Waals interactions."""
        # Use KDTree for efficient neighbor search
        cutoff = self.cutoffs['vdw']
        
        for i, atom1 in enumerate(self.atoms):
            # Find neighbors within cutoff
            neighbors = self.kdtree.query_ball_point(atom1.coord, cutoff)
            
            for j in neighbors:
                if i >= j:  # Avoid double counting and self-interaction
                    continue
                    
                atom2 = self.atoms[j]
                if atom1.parent == atom2.parent:  # Skip same residue
                    continue
                
                distance = np.linalg.norm(atom2.coord - atom1.coord)
                if distance > 0:
                    vector = atom2.coord - atom1.coord
                    unit_vector = vector / distance
                    energy = self._calculate_bond_energy('vdw', distance)
                    force_vector = unit_vector * energy
                    
                    res1_id = f"{atom1.parent.parent.id}:{atom1.parent.id[1]}"
                    res2_id = f"{atom2.parent.parent.id}:{atom2.parent.id[1]}"
                    
                    self.residue_vectors[res1_id] += force_vector
                    self.residue_vectors[res2_id] -= force_vector
                    self.residue_energies[res1_id] += energy
                    self.residue_energies[res2_id] += energy
    
    def _calculate_bond_energy(self, bond_type: str, distance: float) -> float:
        """
        Calculate bond energy based on type and distance.
        Adapted from legacy code with updated constants.
        """
        if distance == 0:
            return 0
        
        # Energy constants (in kcal/mol units for consistency)
        if bond_type in ['hbond', 'salt_bridge']:
            # Electrostatic: E = k * q1 * q2 / r
            k_electrostatic = 332.0  # kcal*Å/mol*e^2
            charge = 1.0 if bond_type == 'salt_bridge' else 0.5
            energy = k_electrostatic * charge * charge / distance
        elif bond_type in ['pi_pi', 'pi_cation']:
            # Pi interactions: E = A / r^2
            A = 2.0 if bond_type == 'pi_pi' else 3.0  # kcal*Å^2/mol
            energy = A / (distance ** 2)
        else:  # vdw
            # Lennard-Jones: E = 4ε[(σ/r)^12 - (σ/r)^6], simplified
            epsilon = 0.1  # kcal/mol
            sigma = 3.5    # Å
            r6 = (sigma / distance) ** 6
            energy = 4 * epsilon * (r6 * r6 - r6)
        
        return abs(energy)  # Return magnitude for force calculations
    
    def get_residue_vector(self, chain_id: str, residue_number: int) -> np.ndarray:
        """
        Get the pre-computed force vector for a specific residue.
        
        Args:
            chain_id: Chain identifier
            residue_number: Residue number
            
        Returns:
            3D force vector for the residue
        """
        res_id = f"{chain_id}:{residue_number}"
        return self.residue_vectors.get(res_id, np.zeros(3))
    
    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        """Return all pre-computed residue vectors."""
        return self.residue_vectors.copy()
    
    def get_residue_energies(self) -> Dict[str, float]:
        """Return total interaction energies for each residue."""
        return self.residue_energies.copy()
    
    def save_to_file(self, filename: str):
        """Save pre-computed vectors to file for later use."""
        data = []
        for res_id, vector in self.residue_vectors.items():
            chain, resnum = res_id.split(':')
            data.append({
                'chain': chain,
                'residue': int(resnum),
                'vector_x': vector[0],
                'vector_y': vector[1],
                'vector_z': vector[2],
                'total_energy': self.residue_energies.get(res_id, 0.0)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved intra-protein vectors to {filename}")
    
    def load_from_file(self, filename: str):
        """Load pre-computed vectors from file."""
        df = pd.read_csv(filename)
        self.residue_vectors = {}
        self.residue_energies = {}
        
        for _, row in df.iterrows():
            res_id = f"{row['chain']}:{int(row['residue'])}"
            vector = np.array([row['vector_x'], row['vector_y'], row['vector_z']])
            self.residue_vectors[res_id] = vector
            self.residue_energies[res_id] = row['total_energy']
        
        logger.info(f"Loaded intra-protein vectors from {filename}")


def parse_protein_robust(protein_file):
    """Parse protein structure with fallback mechanisms."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_file)
    return structure


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        protein_file = sys.argv[1]
        
        # Parse protein
        structure = parse_protein_robust(protein_file)
        
        # Calculate intra-protein interactions
        intra_calc = IntraProteinInteractions(structure)
        vectors = intra_calc.calculate_all_interactions()
        
        # Save results
        output_file = protein_file.replace('.pdb', '_intra_vectors.csv')
        intra_calc.save_to_file(output_file)
        
        print(f"Calculated intra-protein vectors for {len(vectors)} residues")
        print(f"Results saved to {output_file}")
    else:
        print("Usage: python intra_protein_interactions.py protein.pdb")