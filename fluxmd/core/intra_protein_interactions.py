"""
Intra-protein interaction calculator for FluxMD
Calculates complete n×n residue-residue interactions for internal protein force field
"""

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import logging
from itertools import combinations
from .protonation_aware_interactions import ProtonationAwareInteractionDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntraProteinInteractions:
    """Calculate complete n×n residue-residue interaction matrix"""
    
    def __init__(self, structure, physiological_pH=7.4):
        """
        Initialize with a protein structure.
        
        Args:
            structure: BioPython Structure object
            physiological_pH: pH for protonation state calculations (default 7.4)
        """
        self.structure = structure
        self.physiological_pH = physiological_pH
        self.protonation_detector = ProtonationAwareInteractionDetector(pH=self.physiological_pH)
        self.residues = []
        self.residue_atoms = {}  # residue_id -> list of atoms
        self.residue_coords = {}  # residue_id -> array of coordinates
        
        self._extract_residues()
        
        # Interaction parameters
        self.cutoffs = {
            'hbond': 3.5,
            'salt_bridge': 4.0,
            'pi_pi': 4.5,  # Proper pi-stacking cutoff (was 7.0)
            'pi_cation': 6.0,
            'vdw': 5.0
        }
        
        # Atom type classifications
        self.donor_atoms = {'N', 'O', 'S'}  # Potential H-bond donors
        self.acceptor_atoms = {'O', 'N', 'S'}  # Potential H-bond acceptors
        self.positive_atoms = {'N'}  # Potentially positive
        self.negative_atoms = {'O'}  # Potentially negative
        self.aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        
    def _extract_residues(self):
        """Extract all residues and their atoms"""
        residue_count = 0
        
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Skip heterogens
                        res_id = f"{chain.id}:{residue.id[1]}"
                        self.residues.append({
                            'id': res_id,
                            'residue': residue,
                            'resname': residue.resname,
                            'chain': chain.id,
                            'resnum': residue.id[1]
                        })
                        
                        # Extract atoms and coordinates
                        atoms = []
                        coords = []
                        for atom in residue:
                            atoms.append(atom)
                            coords.append(atom.coord)
                        
                        self.residue_atoms[res_id] = atoms
                        self.residue_coords[res_id] = np.array(coords)
                        residue_count += 1
        
        logger.info(f"Extracted {residue_count} residues from protein")
    
    def calculate_all_interactions(self) -> Dict[str, np.ndarray]:
        """
        Calculate complete n×n residue interaction matrix.
        Returns residue-level force vectors.
        """
        n_residues = len(self.residues)
        logger.info(f"Calculating {n_residues}×{n_residues} residue interaction matrix...")
        
        # Initialize result storage
        residue_vectors = {}
        for res in self.residues:
            residue_vectors[res['id']] = np.zeros(3)
        
        # Calculate all unique residue pairs
        total_pairs = 0
        for i, res1 in enumerate(self.residues):
            for j, res2 in enumerate(self.residues):
                if i >= j:  # Skip self and avoid double counting
                    continue
                
                # Calculate all interactions between these two residues
                force_vector = self._calculate_residue_pair_interaction(res1, res2)
                
                # Add force to res1, subtract from res2 (Newton's third law)
                residue_vectors[res1['id']] += force_vector
                residue_vectors[res2['id']] -= force_vector
                
                total_pairs += 1
        
        logger.info(f"Calculated {total_pairs} unique residue pairs")
        
        # Calculate average force magnitudes for logging
        magnitudes = [np.linalg.norm(v) for v in residue_vectors.values()]
        logger.info(f"Average force magnitude: {np.mean(magnitudes):.3f}")
        logger.info(f"Max force magnitude: {np.max(magnitudes):.3f}")
        
        return residue_vectors
    
    def _calculate_residue_pair_interaction(self, res1: dict, res2: dict) -> np.ndarray:
        """
        Calculate total interaction force between two residues.
        Considers all atom pairs and all interaction types.
        """
        total_force = np.zeros(3)
        
        # Get atoms and coordinates
        atoms1 = self.residue_atoms[res1['id']]
        atoms2 = self.residue_atoms[res2['id']]
        coords1 = self.residue_coords[res1['id']]
        coords2 = self.residue_coords[res2['id']]
        
        # OPTIMIZATION: Check CB-CB distance first (or CA for GLY)
        cb1_coord = None
        cb2_coord = None
        
        # Find CB (or CA for glycine) in residue 1
        for atom in atoms1:
            if atom.name == 'CB' or (res1['resname'] == 'GLY' and atom.name == 'CA'):
                cb1_coord = atom.coord
                break
        
        # Find CB (or CA for glycine) in residue 2
        for atom in atoms2:
            if atom.name == 'CB' or (res2['resname'] == 'GLY' and atom.name == 'CA'):
                cb2_coord = atom.coord
                break
        
        # If CB not found (unusual), fall back to CA
        if cb1_coord is None:
            for atom in atoms1:
                if atom.name == 'CA':
                    cb1_coord = atom.coord
                    break
        
        if cb2_coord is None:
            for atom in atoms2:
                if atom.name == 'CA':
                    cb2_coord = atom.coord
                    break
        
        # Calculate CB-CB distance
        if cb1_coord is not None and cb2_coord is not None:
            cb_dist = np.linalg.norm(cb2_coord - cb1_coord)
            
            # Skip if CB atoms are too far apart
            # 8.0 Å is a standard cutoff for CB-CB contacts in protein folding
            if cb_dist > 12.0:  # Slightly relaxed for long-range interactions
                return np.zeros(3)
        
        # Calculate distance matrix between all atom pairs
        dist_matrix = cdist(coords1, coords2)
        
        # Find all atom pairs within max cutoff
        max_cutoff = max(self.cutoffs.values())
        close_pairs = np.where(dist_matrix <= max_cutoff)
        
        # Process each close atom pair
        for idx1, idx2 in zip(close_pairs[0], close_pairs[1]):
            atom1 = atoms1[idx1]
            atom2 = atoms2[idx2]
            distance = dist_matrix[idx1, idx2]
            
            # Vector from atom1 to atom2
            vector = coords2[idx2] - coords1[idx1]
            if distance > 0:
                unit_vector = vector / distance
            else:
                continue
            
            # Calculate all applicable interaction energies
            energy = 0.0
            
            # 1. Van der Waals (always present)
            if distance <= self.cutoffs['vdw']:
                vdw_energy = self._calculate_vdw_energy(distance)
                energy += vdw_energy
            
            # 2. Hydrogen bonds
            if distance <= self.cutoffs['hbond']:
                if self._can_form_hbond(atom1, atom2):
                    hbond_energy = self._calculate_hbond_energy(distance)
                    energy += hbond_energy
            
            # 3. Salt bridges
            if distance <= self.cutoffs['salt_bridge']:
                if self._can_form_salt_bridge(atom1, atom2, res1['resname'], res2['resname']):
                    salt_energy = self._calculate_salt_bridge_energy(distance)
                    energy += salt_energy
            
            # Add force contribution
            force = unit_vector * energy
            total_force += force
        
        # 4. Pi-pi stacking (residue level)
        if res1['resname'] in self.aromatic_residues and res2['resname'] in self.aromatic_residues:
            pi_force = self._calculate_pi_pi_force(res1, res2)
            total_force += pi_force
        
        # 5. Pi-cation (residue level)
        if self._can_form_pi_cation(res1['resname'], res2['resname']):
            pi_cation_force = self._calculate_pi_cation_force(res1, res2)
            total_force += pi_cation_force
        
        return total_force
    
    def _can_form_hbond(self, atom1, atom2) -> bool:
        """Check if two atoms can form hydrogen bond using protonation awareness"""
        # Create atom dictionaries for protonation detector
        atom1_dict = {
            'resname': atom1.get_parent().get_resname(),
            'name': atom1.name,
            'element': atom1.element.upper() if atom1.element else atom1.name[0].upper(),
            'x': atom1.coord[0],
            'y': atom1.coord[1],
            'z': atom1.coord[2],
            'chain': atom1.get_parent().get_parent().get_id(),
            'resSeq': atom1.get_parent().get_id()[1]
        }
        
        atom2_dict = {
            'resname': atom2.get_parent().get_resname(),
            'name': atom2.name,
            'element': atom2.element.upper() if atom2.element else atom2.name[0].upper(),
            'x': atom2.coord[0],
            'y': atom2.coord[1],
            'z': atom2.coord[2],
            'chain': atom2.get_parent().get_parent().get_id(),
            'resSeq': atom2.get_parent().get_id()[1]
        }
        
        # Get protonation-aware properties
        pa_atom1 = self.protonation_detector.determine_atom_protonation(atom1_dict)
        pa_atom2 = self.protonation_detector.determine_atom_protonation(atom2_dict)
        
        # Check both directions for H-bond capability
        return ((pa_atom1.can_donate_hbond and pa_atom2.can_accept_hbond) or
                (pa_atom2.can_donate_hbond and pa_atom1.can_accept_hbond))
    
    def _can_form_salt_bridge(self, atom1, atom2, resname1: str, resname2: str) -> bool:
        """Check if atoms can form salt bridge using protonation-aware charge states"""
        # Create atom dictionaries for protonation detector
        atom1_dict = {
            'resname': resname1,
            'name': atom1.name,
            'element': atom1.element.upper() if atom1.element else atom1.name[0].upper(),
            'x': atom1.coord[0],
            'y': atom1.coord[1],
            'z': atom1.coord[2],
            'chain': atom1.get_parent().get_parent().get_id(),
            'resSeq': atom1.get_parent().get_id()[1]
        }
        
        atom2_dict = {
            'resname': resname2,
            'name': atom2.name,
            'element': atom2.element.upper() if atom2.element else atom2.name[0].upper(),
            'x': atom2.coord[0],
            'y': atom2.coord[1],
            'z': atom2.coord[2],
            'chain': atom2.get_parent().get_parent().get_id(),
            'resSeq': atom2.get_parent().get_id()[1]
        }
        
        # Get protonation-aware properties
        pa_atom1 = self.protonation_detector.determine_atom_protonation(atom1_dict)
        pa_atom2 = self.protonation_detector.determine_atom_protonation(atom2_dict)
        
        # Check for opposite charges
        return pa_atom1.formal_charge * pa_atom2.formal_charge < 0
    
    def _can_form_pi_cation(self, resname1: str, resname2: str) -> bool:
        """Check if residues can form pi-cation interaction"""
        cation_residues = {'ARG', 'LYS', 'HIS'}
        
        return ((resname1 in self.aromatic_residues and resname2 in cation_residues) or
                (resname2 in self.aromatic_residues and resname1 in cation_residues))
    
    def _calculate_vdw_energy(self, distance: float) -> float:
        """Lennard-Jones potential"""
        if distance == 0:
            return 0
        sigma = 3.5  # Å
        epsilon = 0.1  # kcal/mol
        r6 = (sigma / distance) ** 6
        energy = 4 * epsilon * (r6 * r6 - r6)
        return max(min(energy, 10.0), -10.0)  # Cap to reasonable range
    
    def _calculate_hbond_energy(self, distance: float) -> float:
        """Hydrogen bond energy"""
        if distance == 0:
            return 0
        optimal = 2.8  # Å
        strength = -5.0  # kcal/mol
        return strength * np.exp(-((distance - optimal) / 0.5) ** 2)
    
    def _calculate_salt_bridge_energy(self, distance: float) -> float:
        """Electrostatic energy for salt bridge"""
        if distance == 0:
            return 0
        k_electrostatic = 332.0  # kcal*Å/mol*e^2
        return -k_electrostatic / distance  # Attractive
    
    def _calculate_pi_pi_force(self, res1: dict, res2: dict) -> np.ndarray:
        """Calculate pi-pi stacking force between aromatic residues"""
        # Get aromatic ring centers
        aromatic_atoms = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
        }
        
        # Get ring atoms for each residue
        ring1_coords = []
        ring2_coords = []
        
        for atom in self.residue_atoms[res1['id']]:
            if atom.name in aromatic_atoms.get(res1['resname'], []):
                ring1_coords.append(atom.coord)
        
        for atom in self.residue_atoms[res2['id']]:
            if atom.name in aromatic_atoms.get(res2['resname'], []):
                ring2_coords.append(atom.coord)
        
        if len(ring1_coords) < 3 or len(ring2_coords) < 3:
            return np.zeros(3)
        
        # Calculate ring centers
        center1 = np.mean(ring1_coords, axis=0)
        center2 = np.mean(ring2_coords, axis=0)
        
        # Distance and vector
        vector = center2 - center1
        distance = np.linalg.norm(vector)
        
        if distance > self.cutoffs['pi_pi'] or distance == 0:
            return np.zeros(3)
        
        unit_vector = vector / distance
        
        # Pi-pi energy (simplified)
        optimal_distance = 3.8  # Å
        strength = -4.0  # kcal/mol
        energy = strength * np.exp(-((distance - optimal_distance) / 1.5) ** 2)
        
        return unit_vector * energy
    
    def _calculate_pi_cation_force(self, res1: dict, res2: dict) -> np.ndarray:
        """Calculate pi-cation interaction force"""
        # Determine which is aromatic and which is cationic
        if res1['resname'] in self.aromatic_residues:
            aro_res = res1
            cat_res = res2
        else:
            aro_res = res2
            cat_res = res1
        
        # Get aromatic center (same as pi-pi)
        aromatic_atoms = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
        }
        
        ring_coords = []
        for atom in self.residue_atoms[aro_res['id']]:
            if atom.name in aromatic_atoms.get(aro_res['resname'], []):
                ring_coords.append(atom.coord)
        
        if len(ring_coords) < 3:
            return np.zeros(3)
        
        aromatic_center = np.mean(ring_coords, axis=0)
        
        # Get cation position
        cation_atoms = {
            'ARG': 'CZ',
            'LYS': 'NZ',
            'HIS': 'CE1'
        }
        
        cation_pos = None
        for atom in self.residue_atoms[cat_res['id']]:
            if atom.name == cation_atoms.get(cat_res['resname'], ''):
                cation_pos = atom.coord
                break
        
        if cation_pos is None:
            return np.zeros(3)
        
        # Calculate force
        vector = cation_pos - aromatic_center
        distance = np.linalg.norm(vector)
        
        if distance > self.cutoffs['pi_cation'] or distance == 0:
            return np.zeros(3)
        
        unit_vector = vector / distance
        
        # Pi-cation energy
        optimal_distance = 3.5  # Å
        strength = -3.0  # kcal/mol
        energy = strength * np.exp(-((distance - optimal_distance) / 1.5) ** 2)
        
        # Return force on aromatic residue
        if res1['resname'] in self.aromatic_residues:
            return unit_vector * energy
        else:
            return -unit_vector * energy
    
    def save_to_file(self, filename: str, residue_vectors: Dict[str, np.ndarray] = None):
        """Save pre-computed vectors to file for later use"""
        if residue_vectors is None:
            residue_vectors = getattr(self, 'residue_vectors', {})
            
        data = []
        for res in self.residues:
            res_id = res['id']
            vector = residue_vectors.get(res_id, np.zeros(3))
            chain, resnum = res_id.split(':')
            
            data.append({
                'chain': chain,
                'residue': int(resnum),
                'resname': res['resname'],
                'vector_x': vector[0],
                'vector_y': vector[1],
                'vector_z': vector[2],
                'magnitude': np.linalg.norm(vector)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values(['chain', 'residue'])
        df.to_csv(filename, index=False)
        logger.info(f"Saved intra-protein vectors to {filename}")


def parse_protein_robust(protein_file):
    """Parse protein structure with fallback mechanisms"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_file)
    return structure


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        protein_file = sys.argv[1]
        
        # Parse protein
        structure = parse_protein_robust(protein_file)
        
        # Calculate intra-protein interactions
        intra_calc = IntraProteinInteractions(structure)
        residue_vectors = intra_calc.calculate_all_interactions()
        
        # Save results
        output_file = protein_file.replace('.pdb', '_intra_vectors.csv')
        intra_calc.save_to_file(output_file, residue_vectors)
        
        print(f"Calculated intra-protein vectors for {len(intra_calc.residues)} residues")
        print(f"Results saved to {output_file}")