"""
Protein-Ligand Trajectory Generator
Advanced molecular dynamics simulation with GPU acceleration
Now with integrated intra-protein force field calculations
"""

# Set matplotlib backend before any other imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for blackbox operation

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Import intra-protein interaction calculator
from intra_protein_interactions import IntraProteinInteractions


class CollisionDetector:
    """Efficient collision detection for protein-ligand systems"""
    
    def __init__(self, clash_tolerance=0.8):
        self.clash_tolerance = clash_tolerance
        
        # VDW radii in Angstroms
        self.vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
            'F': 1.47, 'P': 1.80, 'S': 1.80, 'CL': 1.75,
            'BR': 1.85, 'I': 1.98
        }
        self.default_radius = 1.70
        
    def build_protein_tree(self, protein_coords, protein_atoms):
        """Build KD-tree for fast collision detection"""
        self.protein_tree = cKDTree(protein_coords)
        
        # Store VDW radii for protein atoms
        self.protein_radii = []
        for _, atom in protein_atoms.iterrows():
            element = atom.get('element', atom['name'][0]).upper()
            radius = self.vdw_radii.get(element, self.default_radius)
            self.protein_radii.append(radius)
        
        self.protein_radii = np.array(self.protein_radii)
        
    def check_collision(self, ligand_coords, ligand_atoms):
        """Check if ligand configuration has collision with protein"""
        for i, (coord, atom) in enumerate(zip(ligand_coords, ligand_atoms.itertuples())):
            # Get ligand atom radius
            element = getattr(atom, 'element', getattr(atom, 'name', 'C')[0]).upper()
            lig_radius = self.vdw_radii.get(element, self.default_radius)
            
            # Find nearby protein atoms
            search_radius = lig_radius + max(self.protein_radii) + 1.0
            nearby_indices = self.protein_tree.query_ball_point(coord, search_radius)
            
            # Check detailed distances
            for prot_idx in nearby_indices:
                prot_radius = self.protein_radii[prot_idx]
                distance = np.linalg.norm(coord - self.protein_tree.data[prot_idx])
                
                # Minimum allowed distance with tolerance
                min_allowed = (lig_radius + prot_radius) * self.clash_tolerance
                
                if distance < min_allowed:
                    return True  # Collision detected
        
        return False  # No collision
    
    def find_collision_free_path(self, start_pos, end_pos, ligand_coords, ligand_atoms,
                                n_steps=10, max_attempts=5):
        """Find collision-free path from start to end position"""
        path = []
        current_pos = start_pos.copy()
        
        for step in range(n_steps):
            # Linear interpolation
            t = (step + 1) / n_steps
            target_pos = (1 - t) * start_pos + t * end_pos
            
            # Try direct movement first
            test_coords = ligand_coords + (target_pos - ligand_coords.mean(axis=0))
            
            if not self.check_collision(test_coords, ligand_atoms):
                path.append(target_pos)
                current_pos = target_pos
            else:
                # Try alternative paths
                found_alternative = False
                
                for attempt in range(max_attempts):
                    # Add random perturbation
                    perturbation = np.random.randn(3) * 2.0
                    alt_pos = target_pos + perturbation
                    
                    test_coords = ligand_coords + (alt_pos - ligand_coords.mean(axis=0))
                    
                    if not self.check_collision(test_coords, ligand_atoms):
                        path.append(alt_pos)
                        current_pos = alt_pos
                        found_alternative = True
                        break
                
                if not found_alternative:
                    # Stay at current position
                    path.append(current_pos)
        
        return np.array(path)


class ProteinLigandFluxAnalyzer:
    """Main analyzer class for trajectory generation with integrated force fields"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.collision_detector = CollisionDetector()
        
        # Residue properties for interaction detection
        self.init_residue_properties()
        
        # Intra-protein force field calculator
        self.intra_protein_calc = None
        self.intra_protein_vectors = None
        
    def init_residue_properties(self):
        """Initialize residue property definitions"""
        # Hydrogen bond donors
        self.DONORS = {
            'SER': ['OG'], 'THR': ['OG1'], 'CYS': ['SG'],
            'TYR': ['OH'], 'ASN': ['ND2'], 'GLN': ['NE2'],
            'LYS': ['NZ'], 'ARG': ['NE', 'NH1', 'NH2'],
            'HIS': ['ND1', 'NE2'], 'TRP': ['NE1']
        }
        
        # Hydrogen bond acceptors
        self.ACCEPTORS = {
            'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2'],
            'SER': ['OG'], 'THR': ['OG1'], 'CYS': ['SG'],
            'TYR': ['OH'], 'ASN': ['OD1'], 'GLN': ['OE1'],
            'HIS': ['ND1', 'NE2'], 'MET': ['SD']
        }
        
        # Charged groups
        self.POSITIVE = {'ARG': ['CZ'], 'LYS': ['NZ'], 'HIS': ['CE1']}
        self.NEGATIVE = {'ASP': ['CG'], 'GLU': ['CD']}
        
        # Hydrophobic residues
        self.HYDROPHOBIC = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR']
        
        # Aromatic residues
        self.AROMATIC = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']
        }
    
    def get_protein_surface_points(self, protein_coords, n_points=100, buffer_distance=5.0):
        """
        Generate starting points on protein surface using alpha shapes
        This ensures ligand starts OUTSIDE the protein
        """
        print("\n   Calculating protein surface points...")
        
        # Method 1: Convex hull with buffer
        try:
            hull = ConvexHull(protein_coords)
            
            # Sample points on hull surface
            surface_points = []
            
            for simplex in hull.simplices[:n_points]:
                # Get face vertices
                face_points = protein_coords[simplex]
                
                # Calculate face center
                face_center = np.mean(face_points, axis=0)
                
                # Calculate outward normal
                v1 = face_points[1] - face_points[0]
                v2 = face_points[2] - face_points[0]
                normal = np.cross(v1, v2)
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                
                # Ensure normal points outward
                center_to_face = face_center - protein_coords.mean(axis=0)
                if np.dot(normal, center_to_face) < 0:
                    normal = -normal
                
                # Place point at buffer distance
                surface_point = face_center + normal * buffer_distance
                surface_points.append(surface_point)
            
            # If we need more points, interpolate
            while len(surface_points) < n_points:
                idx1, idx2 = np.random.choice(len(surface_points), 2, replace=False)
                interp_point = (surface_points[idx1] + surface_points[idx2]) / 2
                
                # Project back to surface + buffer
                direction = interp_point - protein_coords.mean(axis=0)
                direction = direction / np.linalg.norm(direction)
                
                # Find intersection with hull and add buffer
                surface_points.append(protein_coords.mean(axis=0) + direction *
                                    (np.max(cdist([protein_coords.mean(axis=0)], protein_coords)[0]) + buffer_distance))
            
            print(f"   ✓ Generated {len(surface_points)} surface points using convex hull")
            return np.array(surface_points[:n_points])
            
        except Exception as e:
            print(f"   ⚠️  Convex hull failed: {e}, using spherical sampling")
        
        # Method 2: Spherical sampling (fallback)
        center = protein_coords.mean(axis=0)
        max_radius = np.max(cdist([center], protein_coords)[0])
        
        # Generate points on sphere
        surface_points = []
        
        # Use golden spiral for even distribution
        golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
        
        for i in range(n_points):
            theta = golden_angle * i
            phi = np.arccos(1 - 2 * i / n_points)
            
            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            # Scale and translate
            point = center + np.array([x, y, z]) * (max_radius + buffer_distance)
            surface_points.append(point)
        
        print(f"   ✓ Generated {n_points} surface points using spherical sampling")
        return np.array(surface_points)
    
    def calculate_molecular_weight(self, ligand_atoms):
        """Calculate molecular weight from ligand atoms"""
        # Simple atomic mass table
        masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                 'S': 32.065, 'P': 30.974, 'F': 18.998, 'CL': 35.453,
                 'BR': 79.904, 'I': 126.904, 'FE': 55.845, 'ZN': 65.38}
        
        total_mass = 0
        for _, atom in ligand_atoms.iterrows():
            element = atom['element'].upper()
            mass = masses.get(element, 12.0)  # Default to carbon if unknown
            total_mass += mass
        
        return total_mass

    def calculate_diffusion_coefficient(self, molecular_weight, temperature=36.5):
        """Calculate diffusion coefficient based on molecular weight (legacy method)"""
        # Physical constants
        k_B = 1.381e-23  # Boltzmann constant (J/K)
        
        temp_kelvin = temperature + 273.15  # Convert to Kelvin
        
        # Approximate radius from molecular weight (Å)
        # Using empirical relation: r ≈ 0.066 * MW^(1/3)
        radius_angstrom = 0.066 * (molecular_weight ** (1/3))
        radius_meter = radius_angstrom * 1e-10
        
        # Water viscosity at 36.5°C (Pa·s)
        viscosity = 0.00069
        
        # Diffusion coefficient (m²/s)
        D = (k_B * temp_kelvin) / (6 * np.pi * viscosity * radius_meter)
        
        # Convert to Å²/ps
        D_angstrom_ps = D * 1e20 / 1e12
        
        return D_angstrom_ps
    
    def generate_brownian_trajectory_collision_free(self, start_pos, end_pos, n_steps,
                                                  ligand_coords, ligand_atoms,
                                                  molecular_weight=300.0, dt=40):
        """
        Generate physics-based Brownian trajectory (following legacy implementation)
        
        Args:
            start_pos: Starting position
            end_pos: Target end position 
            n_steps: Number of trajectory steps
            ligand_coords: Ligand atom coordinates
            ligand_atoms: Ligand atom data
            molecular_weight: Molecular weight for diffusion calculation
            dt: Time step (ps)
        """
        # Calculate diffusion coefficient
        D = self.calculate_diffusion_coefficient(molecular_weight)
        
        # Calculate step size using Brownian motion theory
        # For Brownian motion: <x²> = 2Dt
        step_size = np.sqrt(2 * D * dt)
        
        trajectory = [start_pos]
        times = [0]
        current_pos = start_pos.copy()
        
        # Calculate target distance (distance from start to protein center)
        # Note: protein center should be passed as parameter, using origin as approximation
        protein_center = np.array([0.0, 0.0, 0.0])  # Protein is centered at origin
        target_distance = np.linalg.norm(start_pos - protein_center)
        
        # Gradually decrease target distance toward end position
        end_distance = np.linalg.norm(end_pos - protein_center)
        
        for i in range(1, n_steps):
            # Generate random displacement (true Brownian motion)
            displacement = np.random.randn(3) * step_size
            
            # Propose new position
            proposed_pos = current_pos + displacement
            
            # Calculate target distance for this step (linear interpolation)
            t = i / n_steps
            current_target_distance = (1 - t) * target_distance + t * end_distance
            
            # Adjust position to maintain approximate target distance
            direction_to_proposed = proposed_pos - protein_center
            distance_to_proposed = np.linalg.norm(direction_to_proposed)
            
            if distance_to_proposed > 0:
                # Blend physics-based position with distance constraint
                direction_unit = direction_to_proposed / distance_to_proposed
                distance_constrained_pos = protein_center + direction_unit * current_target_distance
                
                # Weighted combination (favor physics, but respect constraints)
                final_pos = 0.7 * proposed_pos + 0.3 * distance_constrained_pos
            else:
                final_pos = proposed_pos
            
            # Check collision
            test_coords = ligand_coords + (final_pos - ligand_coords.mean(axis=0))
            
            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                current_pos = final_pos
            else:
                # Try smaller steps if collision detected
                for scale in [0.5, 0.25, 0.1]:
                    smaller_displacement = displacement * scale
                    smaller_pos = current_pos + smaller_displacement
                    test_coords = ligand_coords + (smaller_pos - ligand_coords.mean(axis=0))
                    
                    if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                        current_pos = smaller_pos
                        break
                # If all collision attempts fail, stay at current position
            
            trajectory.append(current_pos.copy())
            times.append(i * dt)
        
        return np.array(trajectory)
    
    def parse_structure(self, pdb_file, parse_heterogens=True):
        """Parse PDB structure and extract atom information
        
        Args:
            pdb_file: Path to PDB file
            parse_heterogens: If True, include HETATM records (needed for ligands)
        """
        structure = self.parser.get_structure('structure', pdb_file)
        
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Handle both standard residues and heterogens
                    residue_id = residue.get_id()
                    hetatm_flag = residue_id[0]
                    
                    # Skip water molecules
                    if residue.get_resname() in ['HOH', 'WAT']:
                        continue
                    
                    # For proteins: only standard residues (hetatm_flag == ' ')
                    # For ligands: only heterogens (hetatm_flag != ' ')
                    if not parse_heterogens and hetatm_flag != ' ':
                        continue
                    
                    for atom in residue:
                        # Get element - BioPython should parse this from PDB
                        element = atom.element if hasattr(atom, 'element') else None
                        
                        # Fallback element detection
                        if not element or element.strip() == '':
                            atom_name = atom.get_name().strip()
                            # Common patterns
                            if atom_name[:2] in ['CL', 'BR']:
                                element = atom_name[:2]
                            elif atom_name and atom_name[0] in ['C', 'N', 'O', 'S', 'P', 'H', 'F']:
                                element = atom_name[0]
                            else:
                                # Try harder to extract element
                                for e in ['C', 'N', 'O', 'S', 'P', 'H', 'F']:
                                    if atom_name.startswith(e):
                                        element = e
                                        break
                                if not element:
                                    element = 'C'  # Default to carbon
                                    print(f"Warning: Could not determine element for atom {atom_name}, defaulting to C")
                        
                        atom_info = {
                            'chain': chain.get_id(),
                            'resname': residue.get_resname(),
                            'resSeq': residue.get_id()[1],
                            'name': atom.get_name(),
                            'element': element.strip().upper() if element else 'C',
                            'x': atom.get_coord()[0],
                            'y': atom.get_coord()[1],
                            'z': atom.get_coord()[2],
                            'atom_id': atom.get_serial_number(),
                            'is_hetatm': hetatm_flag != ' ',
                            'residue_id': residue.get_id()[1]  # Add this for consistency
                        }
                        atoms.append(atom_info)
        
        return pd.DataFrame(atoms)
    
    def parse_structure_robust(self, pdb_file, parse_heterogens=True):
        """
        Parse PDB structure with fallback to manual parsing for problematic files
        
        Args:
            pdb_file: Path to PDB file
            parse_heterogens: If True, include HETATM records (needed for ligands)
        
        Returns:
            pd.DataFrame: Parsed atom information
        """
        # First try BioPython parsing
        try:
            bio_atoms = self.parse_structure(pdb_file, parse_heterogens)
            
            # Check if we got reasonable number of atoms
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
            
            expected_atoms = sum(1 for line in lines if line.startswith(('ATOM', 'HETATM')))
            
            if len(bio_atoms) >= expected_atoms * 0.9:  # Got at least 90% of atoms
                return bio_atoms
            else:
                print(f"  ⚠️  BioPython only parsed {len(bio_atoms)}/{expected_atoms} atoms")
                print(f"  Switching to manual parsing...")
                
        except Exception as e:
            print(f"  BioPython parsing failed: {e}")
            print(f"  Switching to manual parsing...")
        
        # Manual parsing fallback
        manual_atoms = []
        
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    record_type = line[0:6].strip()
                    atom_serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    res_seq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # Element detection
                    element = ''
                    if len(line) >= 78:
                        element = line[76:78].strip()
                    
                    if not element:
                        # Guess from atom name
                        if atom_name.startswith('CL'):
                            element = 'Cl'
                        elif atom_name.startswith('BR'):
                            element = 'Br'
                        elif atom_name and atom_name[0] in ['C', 'N', 'O', 'S', 'P', 'H', 'F']:
                            element = atom_name[0].upper()
                        else:
                            element = 'C'  # Default
                    
                    # Skip water
                    if res_name in ['HOH', 'WAT']:
                        continue
                    
                    # Apply parse_heterogens filter
                    is_hetatm = record_type == 'HETATM'
                    if not parse_heterogens and is_hetatm:
                        continue
                    
                    manual_atoms.append({
                        'chain': chain_id if chain_id else 'A',
                        'resname': res_name,
                        'resSeq': res_seq,
                        'name': atom_name,
                        'element': element,
                        'x': x,
                        'y': y,
                        'z': z,
                        'atom_id': atom_serial,
                        'is_hetatm': is_hetatm,
                        'residue_id': res_seq
                    })
                    
                except Exception as e:
                    continue
        
        df = pd.DataFrame(manual_atoms)
        print(f"  ✓ Manual parsing succeeded: {len(df)} atoms")
        
        return df
    
    def calculate_pi_stacking(self, aromatic_atoms1, aromatic_atoms2):
        """Calculate pi-stacking interactions between aromatic rings"""
        if len(aromatic_atoms1) < 3 or len(aromatic_atoms2) < 3:
            return None
        
        # Calculate ring centers
        center1 = aromatic_atoms1[['x', 'y', 'z']].mean()
        center2 = aromatic_atoms2[['x', 'y', 'z']].mean()
        
        # Distance between centers
        distance = np.linalg.norm(center2 - center1)
        
        # Check if within pi-stacking range (3.4-4.5 Å typically)
        if distance > 7.0:
            return None
        
        # Calculate ring normals using SVD
        coords1 = aromatic_atoms1[['x', 'y', 'z']].values - center1.values
        coords2 = aromatic_atoms2[['x', 'y', 'z']].values - center2.values
        
        # SVD to find plane normal
        _, _, vh1 = np.linalg.svd(coords1)
        _, _, vh2 = np.linalg.svd(coords2)
        
        normal1 = vh1[2]
        normal2 = vh2[2]
        
        # Calculate angle between normals
        cos_angle = abs(np.dot(normal1, normal2))
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        
        # Calculate offset
        center_vector = center2.values - center1.values
        offset = np.linalg.norm(center_vector - np.dot(center_vector, normal1) * normal1)
        
        # Determine stacking type
        if angle < 30:  # Parallel
            if offset < 2.0:
                stack_type = "Pi-Stacking-Parallel"
                energy = -5.0  # Stronger
            else:
                stack_type = "Pi-Stacking-Offset"
                energy = -3.5
        elif 60 < angle < 120:  # T-shaped
            stack_type = "Pi-Stacking-T-Shaped"
            energy = -4.0
        else:  # Angled
            stack_type = "Pi-Stacking-Angled"
            energy = -2.5
        
        # Apply distance penalty
        optimal_distance = 3.8
        distance_penalty = np.exp(-((distance - optimal_distance) / 1.5) ** 2)
        energy *= distance_penalty
        
        return {
            'type': stack_type,
            'energy': energy,
            'distance': distance,
            'angle': angle,
            'offset': offset,
            'center1': center1.values,
            'center2': center2.values,
            'normal1': normal1,
            'normal2': normal2
        }
    
    def calculate_interactions(self, protein_atoms, ligand_atoms, iteration_num):
        """Calculate all non-covalent interactions including pi-stacking and intra-protein forces"""
        interactions = []
        
        # Extract coordinates
        protein_coords = protein_atoms[['x', 'y', 'z']].values
        ligand_coords = ligand_atoms[['x', 'y', 'z']].values
        
        # Calculate distance matrix
        dist_matrix = cdist(protein_coords, ligand_coords)
        
        # Find close contacts
        close_contacts = np.where(dist_matrix < 5.0)
        
        # Track residue-level interactions for combined vector calculation
        residue_interactions = {}
        
        # Process each close contact
        for p_idx, l_idx in zip(close_contacts[0], close_contacts[1]):
            distance = dist_matrix[p_idx, l_idx]
            
            p_atom = protein_atoms.iloc[p_idx]
            l_atom = ligand_atoms.iloc[l_idx]
            
            # Determine interaction type
            interaction_type, energy = self.determine_interaction_type(
                p_atom, l_atom, distance
            )
            
            if interaction_type:
                # Calculate inter-protein (protein-ligand) vector
                inter_vector = ligand_coords[l_idx] - protein_coords[p_idx]
                
                # Get residue ID
                chain = p_atom['chain']
                residue_num = p_atom['resSeq']
                res_id = f"{chain}:{residue_num}"
                
                # Get intra-protein vector for this residue
                intra_vector = np.zeros(3)
                if self.intra_protein_vectors and res_id in self.intra_protein_vectors:
                    intra_vector = self.intra_protein_vectors[res_id]
                
                # Calculate 합벡터 (combined vector)
                combined_vector = inter_vector + intra_vector
                
                # Store residue-level vectors for aggregation
                if res_id not in residue_interactions:
                    residue_interactions[res_id] = {
                        'inter_vectors': [],
                        'intra_vector': intra_vector,
                        'energies': []
                    }
                residue_interactions[res_id]['inter_vectors'].append(inter_vector)
                residue_interactions[res_id]['energies'].append(energy)
                
                interaction = {
                    'frame': iteration_num,
                    'protein_chain': chain,
                    'protein_residue': residue_num,
                    'protein_resname': p_atom['resname'],
                    'protein_atom': p_atom['name'],
                    'protein_atom_id': p_atom['atom_id'],
                    'ligand_atom': l_atom['name'],
                    'distance': distance,
                    'bond_type': interaction_type,
                    'bond_energy': energy,
                    # Store both inter and combined vectors
                    'inter_vector_x': inter_vector[0],
                    'inter_vector_y': inter_vector[1],
                    'inter_vector_z': inter_vector[2],
                    'intra_vector_x': intra_vector[0],
                    'intra_vector_y': intra_vector[1],
                    'intra_vector_z': intra_vector[2],
                    'vector_x': combined_vector[0],  # 합벡터
                    'vector_y': combined_vector[1],
                    'vector_z': combined_vector[2],
                    'combined_magnitude': np.linalg.norm(combined_vector)
                }
                interactions.append(interaction)
        
        # Check for pi-stacking interactions
        pi_stacking_interactions = self.detect_pi_stacking(protein_atoms, ligand_atoms, iteration_num)
        
        # Add intra-protein vectors to pi-stacking interactions
        for pi_interaction in pi_stacking_interactions:
            chain = pi_interaction.get('protein_chain', 'A')
            residue_num = pi_interaction.get('protein_residue', pi_interaction.get('protein_residue_id', 0))
            res_id = f"{chain}:{residue_num}"
            
            # Get intra-protein vector
            intra_vector = np.zeros(3)
            if self.intra_protein_vectors and res_id in self.intra_protein_vectors:
                intra_vector = self.intra_protein_vectors[res_id]
            
            # Update pi-stacking interaction with vectors
            inter_vector = np.array([
                pi_interaction.get('vector_x', 0),
                pi_interaction.get('vector_y', 0),
                pi_interaction.get('vector_z', 0)
            ])
            combined_vector = inter_vector + intra_vector
            
            pi_interaction.update({
                'inter_vector_x': inter_vector[0],
                'inter_vector_y': inter_vector[1],
                'inter_vector_z': inter_vector[2],
                'intra_vector_x': intra_vector[0],
                'intra_vector_y': intra_vector[1],
                'intra_vector_z': intra_vector[2],
                'vector_x': combined_vector[0],  # 합벡터
                'vector_y': combined_vector[1],
                'vector_z': combined_vector[2],
                'combined_magnitude': np.linalg.norm(combined_vector)
            })
        
        interactions.extend(pi_stacking_interactions)
        
        return pd.DataFrame(interactions)
    
    def detect_pi_stacking(self, protein_atoms, ligand_atoms, iteration_num):
        """Detect pi-stacking interactions between aromatic systems"""
        pi_interactions = []
        
        # Find aromatic residues in protein
        for res_name in self.AROMATIC:
            # Get aromatic residues of this type
            aromatic_residues = protein_atoms[protein_atoms['resname'] == res_name]['resSeq'].unique()
            
            for res_id in aromatic_residues:
                # Get atoms for this aromatic residue
                res_atoms = protein_atoms[
                    (protein_atoms['resSeq'] == res_id) &
                    (protein_atoms['resname'] == res_name) &
                    (protein_atoms['name'].isin(self.AROMATIC[res_name]))
                ]
                
                if len(res_atoms) >= 3:
                    # Check against ligand aromatic atoms
                    # Simple heuristic: C atoms in specific patterns
                    ligand_aromatic = ligand_atoms[
                        (ligand_atoms['element'] == 'C') |
                        (ligand_atoms['element'] == 'N')
                    ]
                    
                    if len(ligand_aromatic) >= 3:
                        # Calculate pi-stacking
                        pi_result = self.calculate_pi_stacking(res_atoms, ligand_aromatic)
                        
                        if pi_result:
                            interaction = {
                                'frame': iteration_num,
                                'protein_chain': res_atoms.iloc[0]['chain'],
                                'protein_residue': res_id,
                                'protein_resname': res_name,
                                'protein_atom': 'RING',
                                'protein_atom_id': -1,  # Special marker for pi-stacking
                                'protein_residue_id': res_id,  # CRITICAL: Add residue mapping!
                                'ligand_atom': 'RING',
                                'distance': pi_result['distance'],
                                'bond_type': pi_result['type'],
                                'bond_energy': pi_result['energy'],
                                'angle': pi_result['angle'],
                                'offset_distance': pi_result['offset'],
                                'centroid1_x': pi_result['center1'][0],
                                'centroid1_y': pi_result['center1'][1],
                                'centroid1_z': pi_result['center1'][2],
                                'centroid2_x': pi_result['center2'][0],
                                'centroid2_y': pi_result['center2'][1],
                                'centroid2_z': pi_result['center2'][2],
                                'vector_x': pi_result['center2'][0] - pi_result['center1'][0],
                                'vector_y': pi_result['center2'][1] - pi_result['center1'][1],
                                'vector_z': pi_result['center2'][2] - pi_result['center1'][2]
                            }
                            pi_interactions.append(interaction)
        
        return pi_interactions
    
    def determine_interaction_type(self, protein_atom, ligand_atom, distance):
        """Determine the type of non-covalent interaction"""
        p_resname = protein_atom['resname']
        p_atom_name = protein_atom['name']
        l_atom_name = str(ligand_atom.get('name', '')).strip()
        l_element = str(ligand_atom.get('element', '')).strip().upper()
        
        # Ensure element is valid
        if not l_element:
            # Try to infer from atom name
            l_atom_name_upper = l_atom_name.upper()
            if l_atom_name_upper[:2] in ['CL', 'BR']:
                l_element = l_atom_name_upper[:2]
            elif l_atom_name_upper and l_atom_name_upper[0] in ['C', 'N', 'O', 'S', 'P', 'H', 'F']:
                l_element = l_atom_name_upper[0]
            else:
                l_element = 'C'  # Default
                print(f"Warning: Unknown element for ligand atom {l_atom_name}, defaulting to C")
        
        # Hydrogen bond detection
        if distance < 3.5:
            # Check if protein is donor and ligand is acceptor
            if p_resname in self.DONORS and p_atom_name in self.DONORS[p_resname]:
                if l_element in ['O', 'N', 'S', 'F', 'CL', 'BR']:  # Common acceptors
                    return 'HBond', -2.5
            
            # Check if protein is acceptor and ligand is donor
            if p_resname in self.ACCEPTORS and p_atom_name in self.ACCEPTORS[p_resname]:
                # Ligand donor: H atom or polar H (simplified detection)
                if l_element == 'H':
                    return 'HBond', -2.5
                # Also check if it's a polar group that likely has H
                elif l_element in ['N', 'O'] and ('H' in l_atom_name.upper() or l_element == 'N'):
                    return 'HBond', -2.5
        
        # Salt bridge detection
        if distance < 5.0:
            # Positive protein to negative ligand
            if p_resname in self.POSITIVE and p_atom_name in self.POSITIVE[p_resname]:
                # Check for negative ligand groups
                if l_element in ['O', 'S']:
                    # Look for carboxylate, phosphate, sulfate patterns or any O
                    return 'Salt Bridge', -5.0
            
            # Negative protein to positive ligand
            if p_resname in self.NEGATIVE and p_atom_name in self.NEGATIVE[p_resname]:
                if l_element == 'N':
                    # Nitrogen atoms are often positive
                    return 'Salt Bridge', -5.0
        
        # Pi-cation interactions
        if distance < 6.0:
            # Aromatic protein to charged ligand
            if p_resname in self.AROMATIC and p_atom_name in self.AROMATIC[p_resname]:
                if l_element == 'N':  # Nitrogen often positive
                    return 'Pi-Cation', -3.0
            
            # Charged protein to aromatic ligand
            if p_resname in self.POSITIVE and p_atom_name in self.POSITIVE[p_resname]:
                if l_element == 'C':  # Carbon could be aromatic
                    return 'Pi-Cation', -3.0
        
        # Van der Waals - default for all close contacts
        if distance < 5.0:
            return 'Van der Waals', -0.5
        
        return None, 0
    
    def extract_ca_backbone(self, protein_atoms):
        """Extract CA (alpha-carbon) coordinates for backbone visualization"""
        # Filter for CA atoms
        ca_atoms = protein_atoms[protein_atoms['name'] == 'CA'].copy()
        
        if len(ca_atoms) == 0:
            # Fallback: use every 8th atom (approximate CA spacing)
            ca_coords = protein_atoms.iloc[::8][['x', 'y', 'z']].values
        else:
            # Sort by residue sequence if available
            if 'resSeq' in ca_atoms.columns:
                ca_atoms = ca_atoms.sort_values('resSeq')
            elif 'residue_id' in ca_atoms.columns:
                ca_atoms = ca_atoms.sort_values('residue_id')
            
            ca_coords = ca_atoms[['x', 'y', 'z']].values
        
        return ca_coords
    
    def smooth_backbone_trace(self, coords, smoothing_factor=3):
        """
        Smooth the backbone trace for professional visualization
        Similar to academic structure visualization
        """
        from scipy.interpolate import UnivariateSpline
        
        if len(coords) < 4:
            return coords
        
        # Parametric smoothing
        t = np.arange(len(coords))
        t_smooth = np.linspace(0, len(coords)-1, len(coords) * smoothing_factor)
        
        # Smooth coordinates
        smooth_coords = []
        for i in range(3):  # x, y, z
            spl = UnivariateSpline(t, coords[:, i], s=len(coords)*0.1)
            smooth_coords.append(spl(t_smooth))
        smooth_coords = np.array(smooth_coords).T
        
        return smooth_coords

    def visualize_trajectory_cocoon(self, protein_atoms, trajectory, iteration_num, approach_idx, output_dir):
        """Visualize the Brownian trajectory around protein with professional backbone representation"""
        fig = plt.figure(figsize=(15, 12))
        
        # Extract CA backbone
        ca_coords = self.extract_ca_backbone(protein_atoms)
        
        # Smooth backbone for professional appearance
        smooth_backbone = self.smooth_backbone_trace(ca_coords, smoothing_factor=4)
        
        # 3D trajectory plot (main visualization)
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot protein backbone as professional black smoothed line
        ax1.plot(smooth_backbone[:, 0], smooth_backbone[:, 1], smooth_backbone[:, 2],
                'k-', linewidth=3, alpha=0.4, label='Protein backbone', solid_capstyle='round')
        
        # Plot trajectory with gradient coloring
        trajectory = np.array(trajectory)
        n_points = len(trajectory)
        colors = plt.cm.plasma(np.linspace(0, 1, n_points))
        
        for i in range(len(trajectory) - 1):
            ax1.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                    color=colors[i], linewidth=2, alpha=0.8, solid_capstyle='round')
        
        # Start and end points with circles
        ax1.scatter(*trajectory[0], s=200, c='green', marker='o',
                   edgecolors='darkgreen', linewidth=3, label='Start', zorder=5)
        ax1.scatter(*trajectory[-1], s=200, c='red', marker='o',
                   edgecolors='darkred', linewidth=3, label='End', zorder=5)
        
        ax1.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax1.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Ligand Trajectory - Iteration {iteration_num}, Approach {approach_idx + 1}',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.view_init(elev=20, azim=45)
        
        # Remove grid for cleaner look
        ax1.grid(False)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        
        # Distance from protein backbone over trajectory
        ax2 = fig.add_subplot(222)
        min_distances = []
        for pos in trajectory:
            distances = cdist([pos], ca_coords)[0]
            min_distances.append(np.min(distances))
        
        ax2.plot(range(len(trajectory)), min_distances, 'k-', linewidth=3, solid_capstyle='round')
        ax2.fill_between(range(len(trajectory)), min_distances, alpha=0.3, color='lightgray')
        ax2.set_xlabel('Trajectory Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Distance to Backbone (Å)', fontsize=12, fontweight='bold')
        ax2.set_title('Distance from Protein Backbone', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.2, color='gray', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # XY projection (top view) with smoothed backbone
        ax3 = fig.add_subplot(223)
        # Protein backbone projection (smoothed)
        ax3.plot(smooth_backbone[:, 0], smooth_backbone[:, 1], 'k-', linewidth=2,
                alpha=0.4, label='Backbone', solid_capstyle='round')
        
        # Trajectory path
        for i in range(len(trajectory) - 1):
            ax3.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                    color=colors[i], linewidth=2, alpha=0.7, solid_capstyle='round')
        
        ax3.scatter(*trajectory[0, :2], s=150, c='green', marker='o',
                   edgecolors='darkgreen', linewidth=3, zorder=5)
        ax3.scatter(*trajectory[-1, :2], s=150, c='red', marker='o',
                   edgecolors='darkred', linewidth=3, zorder=5)
        ax3.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax3.set_title('XY Projection', fontsize=14, fontweight='bold')
        ax3.axis('equal')
        ax3.grid(True, alpha=0.2, color='gray', linewidth=0.5)
        ax3.legend(fontsize=11)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # XZ projection (side view) with smoothed backbone
        ax4 = fig.add_subplot(224)
        # Protein backbone projection (smoothed)
        ax4.plot(smooth_backbone[:, 0], smooth_backbone[:, 2], 'k-', linewidth=2,
                alpha=0.4, label='Backbone', solid_capstyle='round')
        
        # Trajectory path
        for i in range(len(trajectory) - 1):
            ax4.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 2],
                    color=colors[i], linewidth=2, alpha=0.7, solid_capstyle='round')
        
        ax4.scatter(trajectory[0, 0], trajectory[0, 2], s=150, c='green', marker='o',
                   edgecolors='darkgreen', linewidth=3, zorder=5)
        ax4.scatter(trajectory[-1, 0], trajectory[-1, 2], s=150, c='red', marker='o',
                   edgecolors='darkred', linewidth=3, zorder=5)
        ax4.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Z (Å)', fontsize=12, fontweight='bold')
        ax4.set_title('XZ Projection', fontsize=14, fontweight='bold')
        ax4.axis('equal')
        ax4.grid(True, alpha=0.2, color='gray', linewidth=0.5)
        ax4.legend(fontsize=11)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        vis_filename = os.path.join(output_dir, f'trajectory_iteration_{iteration_num}_approach_{approach_idx + 1}.png')
        plt.savefig(vis_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved trajectory visualization to {vis_filename}")
        
        # Close without displaying (blackbox record)
        plt.close()
        
        return fig
    
    def run_single_iteration(self, protein_file, ligand_file, output_dir,
                           n_steps, n_approaches, approach_distance,
                           starting_distance, iteration_num, use_gpu=False):
        """Run a single iteration of the flux analysis"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration_num}")
        print(f"{'='*60}")
        
        # Parse structures
        print("Parsing structures...")
        # Parse protein (exclude heterogens)
        protein_atoms = self.parse_structure(protein_file, parse_heterogens=False)
        # Parse ligand (include heterogens) - use robust parser for ligands
        ligand_atoms = self.parse_structure_robust(ligand_file, parse_heterogens=True)
        
        # Initialize intra-protein force field (only on first iteration)
        if iteration_num == 1 and self.intra_protein_calc is None:
            print("\n📊 Calculating intra-protein force field (one-time computation)...")
            structure = self.parser.get_structure('protein', protein_file)
            self.intra_protein_calc = IntraProteinInteractions(structure)
            self.intra_protein_vectors = self.intra_protein_calc.calculate_all_interactions()
            print(f"  ✓ Calculated static force field for {len(self.intra_protein_vectors)} residues")
        
        # For ligands, filter to only HETATM records if mixed file
        if 'is_hetatm' in ligand_atoms.columns:
            hetatm_count = ligand_atoms['is_hetatm'].sum()
            non_hetatm_count = (~ligand_atoms['is_hetatm']).sum()
            
            if hetatm_count > 0 and non_hetatm_count > 0:
                print(f"  Ligand file contains both ATOM ({non_hetatm_count}) and HETATM ({hetatm_count}) records")
                print(f"  Using HETATM records for ligand")
                ligand_atoms = ligand_atoms[ligand_atoms['is_hetatm']]
            elif hetatm_count == 0:
                print(f"  Note: No HETATM records found in ligand file")
                # If no HETATM, use all atoms (might be a non-standard ligand file)
        
        print(f"  Parsed {len(protein_atoms)} protein atoms")
        print(f"  Parsed {len(ligand_atoms)} ligand atoms")
        
        if len(ligand_atoms) == 0:
            print("  ERROR: No ligand atoms found! Check if ligand file contains HETATM records.")
            return pd.DataFrame()
        
        # Verify element assignments
        print(f"  Protein elements: {protein_atoms['element'].value_counts().to_dict()}")
        print(f"  Ligand elements: {ligand_atoms['element'].value_counts().to_dict()}")
        
        # Continue with rest of the method...
        # Get coordinates
        protein_coords = protein_atoms[['x', 'y', 'z']].values
        ligand_coords = ligand_atoms[['x', 'y', 'z']].values
        
        # Build collision detection tree
        self.collision_detector.build_protein_tree(protein_coords, protein_atoms)
        
        # Get protein surface starting points
        surface_points = self.get_protein_surface_points(
            protein_coords,
            n_points=n_approaches * 10,  # Generate extra points
            buffer_distance=starting_distance
        )
        
        # Create iteration directory
        iter_dir = os.path.join(output_dir, f'iteration_{iteration_num}')
        os.makedirs(iter_dir, exist_ok=True)
        
        all_interactions = []
        all_trajectories = []
        
        # Use GPU acceleration if available
        if use_gpu:
            try:
                from gpu_accelerated_flux import GPUAcceleratedInteractionCalculator
                
                gpu_calc = GPUAcceleratedInteractionCalculator()
                gpu_calc.precompute_protein_properties_gpu(protein_atoms)
                gpu_calc.precompute_ligand_properties_gpu(ligand_atoms)
                
                # Pass intra-protein vectors to GPU if available
                if self.intra_protein_vectors:
                    gpu_calc.set_intra_protein_vectors(self.intra_protein_vectors)
                
                print("   ✓ GPU acceleration enabled!")
                
                # Process trajectory on GPU
                # Select random surface points for approaches
                approach_indices = np.random.choice(len(surface_points), n_approaches, replace=False)
                selected_points = surface_points[approach_indices]
                
                # Generate full trajectory
                trajectory = []
                for i, start_point in enumerate(selected_points):
                    # End point closer to protein
                    direction = protein_coords.mean(axis=0) - start_point
                    direction = direction / np.linalg.norm(direction)
                    end_point = start_point + direction * (starting_distance - approach_distance * (n_approaches - 1))
                    
                    # Generate sub-trajectory with molecular weight
                    ligand_mw = self.calculate_molecular_weight(ligand_atoms)
                    sub_trajectory = self.generate_brownian_trajectory_collision_free(
                        start_point, end_point, n_steps // n_approaches,
                        ligand_coords, ligand_atoms, molecular_weight=ligand_mw
                    )
                    trajectory.extend(sub_trajectory)
                
                trajectory = np.array(trajectory)
                
                # Process on GPU
                gpu_results = gpu_calc.process_trajectory_batch_gpu(
                    trajectory, ligand_coords, n_rotations=36
                )
                
                # Convert GPU results to interaction dataframe
                for frame_idx, frame_result in enumerate(gpu_results):
                    if 'best_interactions' in frame_result and frame_result['best_interactions'] is not None:
                        # Extract InteractionResult from GPU
                        gpu_interaction = frame_result['best_interactions']
                        
                        # Convert to pandas DataFrame with proper format
                        interaction_data = []
                        
                        # Process each interaction from GPU result
                        for i in range(len(gpu_interaction.indices)):
                            protein_idx = gpu_interaction.indices[i, 0].item()
                            ligand_idx = gpu_interaction.indices[i, 1].item()
                            
                            # Map interaction type to string
                            interaction_type_map = {
                                0: 'Van der Waals',
                                1: 'Hydrogen Bond',
                                2: 'Salt Bridge',
                                3: 'Pi-Pi Parallel',
                                4: 'Pi-Cation',
                                5: 'Pi-Pi T-Shaped',
                                6: 'Pi-Pi Offset'
                            }
                            
                            # Get protein atom info
                            p_atom = protein_atoms.iloc[protein_idx]
                            l_atom = ligand_atoms.iloc[ligand_idx]
                            
                            # Calculate vector
                            vector = frame_result['position'] + ligand_coords[ligand_idx] - protein_coords[protein_idx]
                            
                            interaction_dict = {
                                'frame': frame_idx,
                                'protein_chain': p_atom.get('chain', 'A'),
                                'protein_residue': gpu_interaction.residue_ids[i].item(),
                                'protein_resname': p_atom['resname'],
                                'protein_atom': p_atom['name'],
                                'protein_atom_id': p_atom.get('atom_id', protein_idx),
                                'ligand_atom': l_atom['name'],
                                'ligand_atom_id': ligand_idx,
                                'distance': gpu_interaction.distances[i].item(),
                                'bond_type': interaction_type_map.get(
                                    gpu_interaction.types[i].item(),
                                    'Unknown'
                                ),
                                'bond_energy': gpu_interaction.energies[i].item(),
                                'vector_x': vector[0],
                                'vector_y': vector[1],
                                'vector_z': vector[2],
                                'protein_residue_id': gpu_interaction.residue_ids[i].item()
                            }
                            interaction_data.append(interaction_dict)
                        
                        if interaction_data:
                            interactions_df = pd.DataFrame(interaction_data)
                            all_interactions.append(interactions_df)
                
                # Store trajectories for visualization
                all_trajectories = [trajectory[i:i+n_steps//n_approaches]
                                  for i in range(0, len(trajectory), n_steps//n_approaches)]
                
                # Visualize trajectories for each approach
                for approach_idx, approach_trajectory in enumerate(all_trajectories):
                    self.visualize_trajectory_cocoon(protein_atoms, approach_trajectory, iteration_num, approach_idx, iter_dir)
                
                print(f"   ✓ GPU processing complete!")
                
            except Exception as e:
                print(f"   ⚠️  GPU acceleration failed: {e}")
                print("   Falling back to CPU processing...")
                use_gpu = False
        
        if not use_gpu:
            # CPU processing
            for approach_idx in range(n_approaches):
                print(f"\nApproach {approach_idx + 1}/{n_approaches}")
                
                # Select random starting point from surface
                start_idx = np.random.randint(0, len(surface_points))
                start_point = surface_points[start_idx]
                
                # Calculate end point (closer to protein)
                direction = protein_coords.mean(axis=0) - start_point
                direction = direction / np.linalg.norm(direction)
                
                current_distance = starting_distance - approach_idx * approach_distance
                end_point = start_point + direction * approach_distance
                
                print(f"  Starting distance: {current_distance:.1f} Å")
                
                # Generate collision-free Brownian trajectory with molecular weight
                ligand_mw = self.calculate_molecular_weight(ligand_atoms)
                trajectory = self.generate_brownian_trajectory_collision_free(
                    start_point, end_point, n_steps, ligand_coords, ligand_atoms,
                    molecular_weight=ligand_mw
                )
                
                all_trajectories.append(trajectory)
                
                # Visualize trajectory (cocoon)
                self.visualize_trajectory_cocoon(protein_atoms, trajectory, iteration_num, approach_idx, iter_dir)
                
                # Calculate interactions at each position
                print("  Calculating interactions...")
                approach_interactions = []
                
                for step, position in enumerate(trajectory):
                    if step % 10 == 0:
                        print(f"\r    Progress: {step}/{n_steps}", end="")
                    
                    # Try multiple rotations
                    best_interactions = None
                    best_energy = float('inf')
                    
                    for rotation in range(0, 360, 10):  # 36 rotations
                        # Rotate ligand
                        angle = np.radians(rotation)
                        rotation_matrix = np.array([
                            [np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]
                        ])
                        
                        # Apply rotation around ligand center
                        ligand_center = ligand_coords.mean(axis=0)
                        rotated_coords = ligand_coords - ligand_center
                        rotated_coords = rotated_coords @ rotation_matrix.T
                        rotated_coords = rotated_coords + position
                        
                        # Update ligand atoms with new coordinates
                        ligand_atoms_rot = ligand_atoms.copy()
                        ligand_atoms_rot[['x', 'y', 'z']] = rotated_coords
                        
                        # Check collision
                        if not self.collision_detector.check_collision(rotated_coords, ligand_atoms_rot):
                            # Calculate interactions
                            interactions = self.calculate_interactions(
                                protein_atoms, ligand_atoms_rot,
                                approach_idx * n_steps + step
                            )
                            
                            if len(interactions) > 0:
                                total_energy = interactions['bond_energy'].sum()
                                if total_energy < best_energy:
                                    best_energy = total_energy
                                    best_interactions = interactions
                    
                    if best_interactions is not None:
                        approach_interactions.append(best_interactions)
                
                print(f"\n  Found {len(approach_interactions)} valid conformations")
                
                if approach_interactions:
                    combined_interactions = pd.concat(approach_interactions, ignore_index=True)
                    all_interactions.append(combined_interactions)
        
        # Combine all interactions
        if all_interactions:
            final_interactions = pd.concat(all_interactions, ignore_index=True)
            
            # Add residue mapping for all interactions
            final_interactions['protein_residue_id'] = final_interactions['protein_residue']
            
            # Save results
            output_file = os.path.join(iter_dir, f'flux_iteration_{iteration_num}_output_vectors.csv')
            final_interactions.to_csv(output_file, index=False)
            print(f"\nSaved {len(final_interactions)} interactions to {output_file}")
            
            # Print summary
            print("\nInteraction summary:")
            print(final_interactions['bond_type'].value_counts())
            
            # Check for pi-stacking
            pi_stacking_count = final_interactions['bond_type'].str.contains('Pi-Stacking').sum()
            if pi_stacking_count > 0:
                print(f"\n✓ Found {pi_stacking_count} pi-stacking interactions!")
                print("  Pi-stacking properly mapped to residues")
        else:
            print("\n⚠️  No interactions found in this iteration")
        
        # Visualize trajectory
        if all_trajectories:
            self.visualize_trajectory(
                protein_atoms, all_trajectories, ligand_coords,
                os.path.join(iter_dir, f'trajectory_iteration_{iteration_num}.png')
            )
        
        return final_interactions if all_interactions else pd.DataFrame()
    
    def visualize_trajectory(self, protein_atoms, trajectories, ligand_coords, output_file):
        """Visualize the combined Brownian trajectories with professional backbone"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract CA backbone for professional visualization
        ca_coords = self.extract_ca_backbone(protein_atoms)
        
        # Smooth backbone for professional appearance
        smooth_backbone = self.smooth_backbone_trace(ca_coords, smoothing_factor=4)
        
        # Plot protein backbone as professional black smoothed line
        ax.plot(smooth_backbone[:, 0], smooth_backbone[:, 1], smooth_backbone[:, 2],
               'k-', linewidth=4, alpha=0.4, label='Protein backbone', solid_capstyle='round')
        
        # Plot trajectories with distinct colors
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
        
        for i, (trajectory, color) in enumerate(zip(trajectories, colors)):
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color=color, alpha=0.9, linewidth=3, label=f'Approach {i+1}', solid_capstyle='round')
            
            # Mark start and end with circles
            ax.scatter(*trajectory[0], color=color, s=180, marker='o',
                      edgecolor='black', linewidth=3, zorder=5)
            ax.scatter(*trajectory[-1], color=color, s=180, marker='o',
                      edgecolor='white', linewidth=3, zorder=5)
        
        # Set labels with professional styling
        ax.set_xlabel('X (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z (Å)', fontsize=14, fontweight='bold')
        ax.set_title('Ligand Trajectory Exploration', fontsize=16, fontweight='bold')
        
        # Remove grid and background for cleaner academic look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Set equal aspect ratio based on backbone
        max_range = np.array([
            ca_coords[:, 0].max() - ca_coords[:, 0].min(),
            ca_coords[:, 1].max() - ca_coords[:, 1].min(),
            ca_coords[:, 2].max() - ca_coords[:, 2].min()
        ]).max() / 1.5
        
        mid_x = (ca_coords[:, 0].max() + ca_coords[:, 0].min()) * 0.5
        mid_y = (ca_coords[:, 1].max() + ca_coords[:, 1].min()) * 0.5
        mid_z = (ca_coords[:, 2].max() + ca_coords[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Better viewing angle
        ax.view_init(elev=15, azim=45)
        ax.legend(fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved trajectory visualization to {output_file}")
    
    def run_complete_analysis(self, protein_file, ligand_file, output_dir,
                            n_steps=100, n_iterations=3, n_approaches=5,
                            approach_distance=2.5, starting_distance=35,
                            n_jobs=-1, use_gpu=False):
        """Run complete flux analysis with multiple iterations"""
        print("\n" + "="*80)
        print("PROTEIN-LIGAND FLUX ANALYSIS")
        print("="*80)
        print(f"Protein: {protein_file}")
        print(f"Ligand: {ligand_file}")
        print(f"Output directory: {output_dir}")
        print(f"Iterations: {n_iterations}")
        print(f"Steps per approach: {n_steps}")
        print(f"Number of approaches: {n_approaches}")
        print(f"Starting distance: {starting_distance} Å")
        print(f"GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        response = input("\nContinue with analysis? (y/n): ").strip().lower()
        if response != 'y':
            print("Analysis cancelled.")
            return None
        
        # Run iterations
        iteration_data = []
        
        for iteration in range(n_iterations):
            print(f"\n{'#'*80}")
            print(f"STARTING ITERATION {iteration + 1} OF {n_iterations}")
            print(f"{'#'*80}")
            
            iteration_start = time.time()
            
            # Run single iteration
            interactions = self.run_single_iteration(
                protein_file, ligand_file, output_dir,
                n_steps, n_approaches, approach_distance,
                starting_distance, iteration + 1, use_gpu
            )
            
            iteration_time = time.time() - iteration_start
            
            if not interactions.empty:
                iteration_data.append({
                    'iteration': iteration + 1,
                    'interactions': interactions,
                    'n_interactions': len(interactions),
                    'time': iteration_time
                })
                
                print(f"\nIteration {iteration + 1} complete!")
                print(f"Time: {iteration_time:.1f} seconds")
                print(f"Total interactions: {len(interactions)}")
            else:
                print(f"\n⚠️  Iteration {iteration + 1} produced no interactions")
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        if iteration_data:
            total_interactions = sum(d['n_interactions'] for d in iteration_data)
            total_time = sum(d['time'] for d in iteration_data)
            
            print(f"Total iterations completed: {len(iteration_data)}")
            print(f"Total interactions found: {total_interactions}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average time per iteration: {total_time/len(iteration_data):.1f} seconds")
            
            # Check for pi-stacking
            pi_stacking_total = 0
            for iter_data in iteration_data:
                pi_count = iter_data['interactions']['bond_type'].str.contains('Pi-Stacking').sum()
                pi_stacking_total += pi_count
            
            if pi_stacking_total > 0:
                print(f"\n✓ Total pi-stacking interactions: {pi_stacking_total}")
        else:
            print("No valid iterations completed.")
        
        return iteration_data


def main():
    """Main entry point"""
    analyzer = ProteinLigandFluxAnalyzer()
    
    # Example usage
    print("PROTEIN-LIGAND TRAJECTORY GENERATOR")
    print("-" * 50)
    
    protein_file = input("Enter protein PDB file: ").strip()
    ligand_file = input("Enter ligand PDB/PDBQT file: ").strip()
    output_dir = input("Output directory (default: flux_output): ").strip() or "flux_output"
    
    # Run analysis
    analyzer.run_complete_analysis(
        protein_file, ligand_file, output_dir,
        n_steps=100, n_iterations=3, n_approaches=5,
        approach_distance=2.5, starting_distance=35,
        use_gpu=True  # Try to use GPU if available
    )


if __name__ == "__main__":
    main()
