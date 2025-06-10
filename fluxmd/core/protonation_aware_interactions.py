"""
Protonation-Aware Non-Covalent Interaction Detection for FluxMD
Handles donor/acceptor role swapping and charge-dependent interactions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import logging
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from fluxmd.core.energy_config import ENERGY_BOUNDS

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
        
        # Protonation rules with donor/acceptor swapping
        self.residue_protonation_rules = {
            'ASP': {
                'pKa': 3.9,
                'type': 'acid',
                'atoms': {
                    'OD1': {'protonated': 'can_donate', 'deprotonated': 'can_accept'},
                    'OD2': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            },
            'GLU': {
                'pKa': 4.2,
                'type': 'acid',
                'atoms': {
                    'OE1': {'protonated': 'can_donate', 'deprotonated': 'can_accept'},
                    'OE2': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            },
            'LYS': {
                'pKa': 10.5,
                'type': 'base',
                'atoms': {
                    'NZ': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            },
            'ARG': {
                'pKa': 12.5,
                'type': 'base',
                'atoms': {
                    'NE': {'protonated': 'can_donate', 'deprotonated': 'can_accept'},
                    'NH1': {'protonated': 'can_donate', 'deprotonated': 'can_accept'},
                    'NH2': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            },
            'HIS': {  # Critical at pH 7.4!
                'pKa': 6.0,
                'type': 'base',
                'atoms': {
                    'ND1': {'protonated': 'can_donate', 'deprotonated': 'can_accept'},
                    'NE2': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            },
            'CYS': {
                'pKa': 8.3,
                'type': 'acid',
                'atoms': {
                    'SG': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            },
            'TYR': {
                'pKa': 10.1,
                'type': 'acid',
                'atoms': {
                    'OH': {'protonated': 'can_donate', 'deprotonated': 'can_accept'}
                }
            }
        }
        
        # Cutoffs
        self.cutoffs = {
            'hbond': 3.5,
            'salt_bridge': 5.0,
            'pi_pi': 4.5,
            'pi_cation': 6.0,
            'vdw': 5.0
        }
        
        # VDW radii
        self.vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
            'F': 1.47, 'P': 1.80, 'S': 1.80, 'CL': 1.75,
            'BR': 1.85, 'I': 1.98
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
            index=atom.get('atom_id', 0),
            element=atom.get('element', 'C'),
            coords=np.array([atom['x'], atom['y'], atom['z']]),
            residue_name=atom['resname'],
            atom_name=atom['name'],
            chain=atom.get('chain', 'A'),
            residue_id=atom.get('resSeq', 0)
        )
        
        # Set VDW radius
        pa_atom.effective_vdw_radius = self.vdw_radii.get(pa_atom.element, 1.70)
        
        # Check ionizable residues
        if atom['resname'] in self.residue_protonation_rules:
            res_rules = self.residue_protonation_rules[atom['resname']]
            
            if atom['name'] in res_rules['atoms']:
                pKa = res_rules['pKa']
                is_acid = res_rules['type'] == 'acid'
                
                # Calculate protonation state
                fraction_protonated = self.henderson_hasselbalch(pKa, is_acid)
                is_protonated = fraction_protonated > 0.5
                
                # CRITICAL: Set donor/acceptor based on protonation state
                atom_rules = res_rules['atoms'][atom['name']]
                
                if is_protonated:
                    pa_atom.protonation_state = "protonated"
                    if atom_rules['protonated'] == 'can_donate':
                        pa_atom.can_donate_hbond = True
                    else:
                        pa_atom.can_accept_hbond = True
                    
                    # Charge
                    pa_atom.formal_charge = 0 if is_acid else +1
                else:
                    pa_atom.protonation_state = "deprotonated"
                    if atom_rules['deprotonated'] == 'can_donate':
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
            if atom['name'] == 'N' and atom['resname'] != 'PRO':
                pa_atom.can_donate_hbond = True
            elif atom['name'] == 'O':
                pa_atom.can_accept_hbond = True
            
            # Side chains
            elif atom['resname'] in ['SER', 'THR'] and atom['name'] in ['OG', 'OG1']:
                pa_atom.can_donate_hbond = True
                pa_atom.can_accept_hbond = True
            elif atom['resname'] == 'ASN' and atom['name'] == 'OD1':
                pa_atom.can_accept_hbond = True
            elif atom['resname'] == 'ASN' and atom['name'] == 'ND2':
                pa_atom.can_donate_hbond = True
                pa_atom.can_accept_hbond = True
            elif atom['resname'] == 'GLN' and atom['name'] == 'OE1':
                pa_atom.can_accept_hbond = True
            elif atom['resname'] == 'GLN' and atom['name'] == 'NE2':
                pa_atom.can_donate_hbond = True
                pa_atom.can_accept_hbond = True
        
        # Aromatic properties
        aromatic_atoms = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
        }
        
        if atom['resname'] in aromatic_atoms:
            if atom['name'] in aromatic_atoms[atom['resname']]:
                pa_atom.is_aromatic = True
                # His+ is electron poor
                if atom['resname'] == 'HIS' and pa_atom.formal_charge > 0:
                    pa_atom.aromatic_electron_density = 0.7
        
        return pa_atom
    
    def process_ligand_atom(self, atom: Dict) -> ProtonationAwareAtom:
        """Process ligand atoms with element-based rules"""
        pa_atom = ProtonationAwareAtom(
            index=atom.get('atom_id', 0),
            element=atom.get('element', 'C').upper(),
            coords=np.array([atom['x'], atom['y'], atom['z']]),
            residue_name='LIG',
            atom_name=atom['name'],
            chain='L',
            residue_id=1
        )
        
        # Simple element-based rules for ligands
        if pa_atom.element == 'N':
            # Check if it's likely an aromatic N (e.g., in heterocycles)
            # For now, assume aromatic N are acceptors, not donors
            # This is a simplification - ideally we'd check connectivity
            atom_name = atom.get('name', '').upper()
            if 'AR' in atom_name or atom.get('is_aromatic', False):
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
        elif pa_atom.element == 'O':
            pa_atom.can_accept_hbond = True
            # Check if carboxylate (simplified)
            if 'COO' in atom.get('name', ''):
                pa_atom.formal_charge = -0.5
                pa_atom.protonation_state = "deprotonated"
            # FIXED: O can also be donors if part of OH group
            # This is simplified - ideally we'd check for attached H
            if 'OH' in atom.get('name', '') or 'HO' in atom.get('name', ''):
                pa_atom.can_donate_hbond = True
        elif pa_atom.element == 'S':
            pa_atom.can_accept_hbond = True
        elif pa_atom.element == 'H':
            pa_atom.can_donate_hbond = True
        
        # VDW radius
        pa_atom.effective_vdw_radius = self.vdw_radii.get(pa_atom.element, 1.70)
        if pa_atom.formal_charge != 0:
            pa_atom.effective_vdw_radius *= 1.2
        
        # Check if aromatic C
        if pa_atom.element == 'C':
            pa_atom.is_aromatic = True  # Simplified
        
        return pa_atom
    
    def detect_all_interactions(self, protein_atom: Dict, ligand_atom: Dict,
                              distance: float) -> List[Dict]:
        """
        Detect all interactions with proper protonation awareness.
        Key: Check H-bonds in BOTH directions!
        """
        # Convert to protonation-aware atoms
        p_atom = self.determine_atom_protonation(protein_atom)
        l_atom = self.process_ligand_atom(ligand_atom)
        
        interactions = []
        
        # 1. H-bonds - CHECK BOTH DIRECTIONS!
        if distance <= self.cutoffs['hbond']:
            # Protein donor -> Ligand acceptor
            if p_atom.can_donate_hbond and l_atom.can_accept_hbond:
                energy = -5.0 * np.exp(-((distance - 2.8) / 0.5) ** 2)
                if p_atom.formal_charge != 0 or l_atom.formal_charge != 0:
                    energy *= 1.5  # Stronger for charged groups
                interactions.append({
                    'type': 'HBond',
                    'energy': energy,
                    'donor': 'protein',
                    'acceptor': 'ligand'
                })
            
            # Ligand donor -> Protein acceptor (REVERSED!)
            if l_atom.can_donate_hbond and p_atom.can_accept_hbond:
                energy = -5.0 * np.exp(-((distance - 2.8) / 0.5) ** 2)
                if p_atom.formal_charge != 0 or l_atom.formal_charge != 0:
                    energy *= 1.5
                interactions.append({
                    'type': 'HBond',
                    'energy': energy,
                    'donor': 'ligand',
                    'acceptor': 'protein'
                })
        
        # 2. Salt bridges - requires opposite charges
        if distance <= self.cutoffs['salt_bridge']:
            if p_atom.formal_charge * l_atom.formal_charge < 0:
                energy = self._calculate_salt_bridge_energy(p_atom.formal_charge, l_atom.formal_charge, distance)
                interactions.append({
                    'type': 'Salt Bridge',
                    'energy': energy
                })
        
        # 3. π-cation
        if distance <= self.cutoffs['pi_cation']:
            if p_atom.is_aromatic and l_atom.formal_charge > 0:
                energy = -3.0 * np.exp(-((distance - 3.5) / 1.5) ** 2)
                interactions.append({'type': 'Pi-Cation', 'energy': energy})
            elif l_atom.is_aromatic and p_atom.formal_charge > 0:
                energy = -3.0 * np.exp(-((distance - 3.5) / 1.5) ** 2)
                interactions.append({'type': 'Pi-Cation', 'energy': energy})
        
        # 4. π-π stacking
        if distance <= self.cutoffs['pi_pi']:
            if p_atom.is_aromatic and l_atom.is_aromatic:
                # Simplified - would need ring geometry for accurate calculation
                energy = -3.5 * np.exp(-((distance - 3.8) / 1.5) ** 2)
                # Adjust for electron density differences
                density_diff = abs(p_atom.aromatic_electron_density - l_atom.aromatic_electron_density)
                energy *= (1.0 + 0.5 * density_diff)
                interactions.append({'type': 'Pi-Stacking', 'energy': energy})
        
        # 5. Van der Waals (default for close contacts)
        if distance <= self.cutoffs['vdw'] and not interactions:
            sigma = (p_atom.effective_vdw_radius + l_atom.effective_vdw_radius) / 2
            r_ratio = sigma / distance
            energy = 0.4 * (r_ratio**12 - r_ratio**6)
            energy = np.clip(energy, -1.0, 10.0)
            interactions.append({'type': 'Van der Waals', 'energy': energy})
        
        return interactions

    def _calculate_salt_bridge_energy(self, charge1: float, charge2: float, distance: float, epsilon_r: float = 80.0) -> float:
        """Calculates salt bridge energy using a simplified Coulomb's model."""
        # Simplified Coulomb's law, capped for stability
        # Using a dielectric constant for water
        # Note: This is a simplified model. A more accurate model would use distance-dependent dielectric.
        if distance == 0:
            return 0
        
        # Conversion factor to get kcal/mol
        # q1*q2 / (epsilon_r * r) is in elementary charge units.
        # 1.60218e-19 C/e, 1 cal = 4.184 J, Avogadro = 6.022e23
        # (1.60218e-19)^2 * (6.022e23 / 4.184) / (4 * pi * 8.854e-12 * 1e-10) -> conversion factor
        conversion_factor = 332.0637 
        
        energy = (conversion_factor * charge1 * charge2) / (epsilon_r * distance)
        
        # Cap the energy to prevent singularities at very close distances
        energy = max(energy, ENERGY_BOUNDS['salt_bridge']['min'])
        return energy

    def _calculate_vdw_energy(self, res1: Residue, res2: Residue) -> float:
        # Placeholder for VdW calculation if needed within this class,
        # otherwise, it relies on intra_protein_interactions.py
        return 0.0


def enhance_fluxmd_with_protonation(protein_atoms: pd.DataFrame,
                                   ligand_atoms: pd.DataFrame,
                                   pH: float = 7.4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add protonation information to atom dataframes for FluxMD.
    """
    detector = ProtonationAwareInteractionDetector(pH=pH)
    
    # Process protein atoms
    protein_enhanced = protein_atoms.copy()
    for idx, atom in protein_atoms.iterrows():
        pa_atom = detector.determine_atom_protonation(atom.to_dict())
        protein_enhanced.loc[idx, 'formal_charge'] = pa_atom.formal_charge
        protein_enhanced.loc[idx, 'can_donate_hbond'] = pa_atom.can_donate_hbond
        protein_enhanced.loc[idx, 'can_accept_hbond'] = pa_atom.can_accept_hbond
        protein_enhanced.loc[idx, 'protonation_state'] = pa_atom.protonation_state
    
    # Process ligand atoms
    ligand_enhanced = ligand_atoms.copy()
    for idx, atom in ligand_atoms.iterrows():
        pa_atom = detector.process_ligand_atom(atom.to_dict())
        ligand_enhanced.loc[idx, 'formal_charge'] = pa_atom.formal_charge
        ligand_enhanced.loc[idx, 'can_donate_hbond'] = pa_atom.can_donate_hbond
        ligand_enhanced.loc[idx, 'can_accept_hbond'] = pa_atom.can_accept_hbond
        ligand_enhanced.loc[idx, 'protonation_state'] = pa_atom.protonation_state
    
    # Print summary
    print(f"\nProtonation Summary at pH {pH}:")
    print(f"Protein charged atoms: {(protein_enhanced['formal_charge'] != 0).sum()}")
    print(f"Ligand charged atoms: {(ligand_enhanced['formal_charge'] != 0).sum()}")
    
    # Add ligand donor/acceptor summary
    ligand_donors = ligand_enhanced['can_donate_hbond'].sum()
    ligand_acceptors = ligand_enhanced['can_accept_hbond'].sum()
    print(f"Ligand H-bond donors: {ligand_donors}")
    print(f"Ligand H-bond acceptors: {ligand_acceptors}")
    
    # Key residues
    for res in ['HIS', 'ASP', 'GLU', 'LYS', 'ARG', 'CYS']:
        mask = protein_enhanced['resname'] == res
        if mask.any():
            charged = (protein_enhanced.loc[mask, 'formal_charge'] != 0).sum()
            total = mask.sum()
            print(f"  {res}: {charged}/{total} atoms charged")
    
    return protein_enhanced, ligand_enhanced


def calculate_interactions_with_protonation(protein_atoms: pd.DataFrame,
                                          ligand_atoms: pd.DataFrame,
                                          pH: float = 7.4,
                                          iteration_num: int = 0) -> pd.DataFrame:
    """
    Calculate interactions using protonation-aware detection.
    Drop-in replacement for the original calculate_interactions method.
    """
    detector = ProtonationAwareInteractionDetector(pH=pH)
    interactions = []
    
    # Get coordinates
    protein_coords = protein_atoms[['x', 'y', 'z']].values
    ligand_coords = ligand_atoms[['x', 'y', 'z']].values
    
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
                'frame': iteration_num,
                'protein_chain': p_atom.get('chain', 'A'),
                'protein_residue': p_atom.get('resSeq', 0),
                'protein_resname': p_atom['resname'],
                'protein_atom': p_atom['name'],
                'protein_atom_id': p_idx,
                'ligand_atom': l_atom['name'],
                'ligand_atom_id': l_idx,
                'distance': distance,
                'bond_type': interaction['type'],
                'bond_energy': interaction['energy'],
                'vector_x': vector[0],
                'vector_y': vector[1],
                'vector_z': vector[2],
                'pH': pH
            }
            
            # Add H-bond direction if applicable
            if interaction['type'] == 'HBond':
                record['hbond_donor'] = interaction.get('donor', '')
                record['hbond_acceptor'] = interaction.get('acceptor', '')
            
            interactions.append(record)
    
    return pd.DataFrame(interactions)


# Example usage
if __name__ == "__main__":
    # Demo pH effects on HIS-ASP interaction
    print("pH-Dependent Interaction Example: HIS-ASP")
    print("="*50)
    
    his_atom = {
        'resname': 'HIS', 'name': 'NE2', 'element': 'N',
        'x': 0.0, 'y': 0.0, 'z': 0.0, 'atom_id': 1,
        'chain': 'A', 'resSeq': 50
    }
    
    asp_atom = {
        'resname': 'ASP', 'name': 'OD1', 'element': 'O',
        'x': 3.0, 'y': 0.0, 'z': 0.0, 'atom_id': 2,
        'chain': 'A', 'resSeq': 75
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
                if 'donor' in i:
                    print(f"     Direction: {i['donor']} -> {i['acceptor']}")
        else:
            print("  -> No interactions")
