"""
Flux Differential Integration for Trajectory Analysis
Advanced statistical analysis of molecular dynamics trajectories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from Bio.PDB import PDBParser
from scipy.interpolate import UnivariateSpline, splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy import stats
from joblib import Parallel, delayed
import os
import glob
import warnings
from scipy.stats import norm
# from ..visualization.visualize_flux import visualize_trajectory_flux_heatmap  # Unused import
from datetime import datetime, timedelta

class TrajectoryFluxAnalyzer:
    """Flux analyzer for trajectory-based analysis"""
    
    def __init__(self, output_dir, protein_name, target_is_dna=False):
        self.output_dir = output_dir
        self.protein_name = protein_name
        self.target_is_dna = target_is_dna
        self.parser = PDBParser(QUIET=True)
        self.atom_to_residue_map = None
        self.residue_to_atom_map = None
        self.bootstrap_validator = BootstrapFluxValidator()
        self.gpu_results = None  # Store GPU trajectory results if available
        self.all_interactions = []
        self.flux_data = None
        
    def create_atom_to_residue_mapping(self, structure):
        """Create accurate atom-to-residue mapping from PDB structure"""
        atom_to_residue = {}
        residue_to_atoms = {}
        atom_index = 0
        
        print("   Creating accurate atom-to-residue mapping...")
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Skip heterogens
                        res_id = residue.get_id()[1]
                        res_name = residue.get_resname()
                        
                        if res_id not in residue_to_atoms:
                            residue_to_atoms[res_id] = {
                                'indices': [],
                                'name': res_name,
                                'atoms': []
                            }
                        
                        for atom in residue:
                            atom_to_residue[atom_index] = res_id
                            residue_to_atoms[res_id]['indices'].append(atom_index)
                            residue_to_atoms[res_id]['atoms'].append(atom.get_name())
                            atom_index += 1
        
        # Validate mapping
        print(f"   ✓ Mapped {atom_index} atoms to {len(residue_to_atoms)} residues")
        
        # Show examples of atom counts per residue type
        residue_types = {}
        for res_data in residue_to_atoms.values():
            res_name = res_data['name']
            atom_count = len(res_data['indices'])
            if res_name not in residue_types:
                residue_types[res_name] = []
            residue_types[res_name].append(atom_count)
        
        print("   Atoms per residue type:")
        for res_name, counts in sorted(residue_types.items()):
            avg_count = np.mean(counts)
            print(f"     {res_name}: {avg_count:.1f} atoms")
        
        return atom_to_residue, residue_to_atoms
    
    def parse_pdb_for_ribbon(self, filepath):
        """Extract CA atoms for ribbon representation with proper mapping"""
        structure = self.parser.get_structure('protein', filepath)
        
        # Create atom mapping FIRST
        self.atom_to_residue_map, self.residue_to_atom_map = \
            self.create_atom_to_residue_mapping(structure)
        
        ca_atoms = []
        ca_coords = []
        residue_indices = []
        residue_names = []
        
        for model in structure:
            if model.get_id() == 0:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':
                            if 'CA' in residue:
                                ca_atoms.append(residue['CA'])
                                ca_coords.append(residue['CA'].coord)
                                residue_indices.append(residue.get_id()[1])
                                residue_names.append(residue.get_resname())
                break
        
        ca_coords = np.array(ca_coords)
        return ca_atoms, ca_coords, residue_indices, residue_names
    
    def calculate_pi_stacking_vector(self, centroid1, centroid2, angle, offset, energy):
        """Calculate energy vector for pi-stacking interaction"""
        # Vector from aromatic center to ligand center
        vector = centroid2 - centroid1
        distance = np.linalg.norm(vector)
        
        if distance < 1e-10:
            return np.zeros(3)
        
        vector_norm = vector / distance
        
        # Weight by interaction geometry
        # Parallel stacking (angle ~ 0°) is favorable
        parallel_weight = np.cos(np.radians(angle)) ** 2
        
        # Optimal offset is around 3.5 Å for pi-stacking
        offset_weight = np.exp(-((offset - 3.5) ** 2) / 2.0)
        
        # Distance weight (optimal 3.4-3.8 Å)
        distance_weight = np.exp(-((distance - 3.6) ** 2) / 2.0)
        
        # Combined weight
        geometric_weight = parallel_weight * offset_weight * distance_weight
        
        # Final energy vector
        return vector_norm * energy * geometric_weight
    
    def add_residue_mapping_to_pi_stacking(self, df, structure_file):
        """Add residue mapping information to pi-stacking data"""
        # This would need to be implemented based on how pi-stacking is detected
        # For now, we'll use the centroid positions to find nearest residues
        
        if 'protein_residue_id' not in df.columns:
            print("   Adding residue mapping to pi-stacking interactions...")
            
            # Parse structure to get residue positions
            structure = self.parser.get_structure('protein', structure_file)
            residue_positions = {}
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':
                            # Get aromatic atoms
                            aromatic_atoms = []
                            res_name = residue.get_resname()
                            
                            if res_name in ['PHE', 'TYR', 'TRP', 'HIS']:
                                for atom in residue:
                                    if atom.get_name() in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']:
                                        aromatic_atoms.append(atom.coord)
                            
                            if aromatic_atoms:
                                centroid = np.mean(aromatic_atoms, axis=0)
                                residue_positions[residue.get_id()[1]] = {
                                    'centroid': centroid,
                                    'name': res_name
                                }
            
            # Map pi-stacking interactions to residues
            residue_ids = []
            for _, row in df.iterrows():
                if 'centroid1_x' in row:
                    centroid = np.array([row['centroid1_x'], row['centroid1_y'], row['centroid1_z']])
                    
                    # Find nearest aromatic residue
                    min_dist = float('inf')
                    best_res_id = -1
                    
                    for res_id, res_data in residue_positions.items():
                        dist = np.linalg.norm(centroid - res_data['centroid'])
                        if dist < min_dist:
                            min_dist = dist
                            best_res_id = res_id
                    
                    residue_ids.append(best_res_id)
                else:
                    residue_ids.append(-1)
            
            df['protein_residue_id'] = residue_ids
        
        return df
    
    def process_iteration_files(self, directory, file_pattern, structure_file=None):
        """Process all iteration files INCLUDING pi-stacking and intra-protein contributions"""
        residue_energy_tensors = {}
        pi_stacking_contributions = {}
        interaction_type_counts = {}
        intra_protein_contributions = {}
        inter_protein_contributions = {}
        combined_vector_magnitudes = {}
        
        # Find all matching files
        files = sorted(glob.glob(os.path.join(directory, file_pattern)))
        print(f"   Found {len(files)} files matching pattern: {file_pattern}")
        
        if len(files) == 0:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        
        # Process each iteration file
        for iteration, file_path in enumerate(files):
            try:
                df = pd.read_csv(file_path)
                
                # Check for pi-stacking data and add residue mapping if needed
                if 'bond_type' in df.columns and structure_file:
                    if any('Pi-Stacking' in str(bt) for bt in df['bond_type'].unique()):
                        df = self.add_residue_mapping_to_pi_stacking(df, structure_file)
                
                # Track interaction types
                if 'bond_type' in df.columns:
                    for bond_type in df['bond_type'].unique():
                        if bond_type not in interaction_type_counts:
                            interaction_type_counts[bond_type] = 0
                        interaction_type_counts[bond_type] += len(df[df['bond_type'] == bond_type])
                
                # Process interactions
                for _, row in df.iterrows():
                    # Check if this is a pi-stacking interaction
                    if 'Pi-Stacking' in str(row.get('bond_type', '')):
                        # Handle pi-stacking properly!
                        if 'protein_residue_id' in row and row['protein_residue_id'] != -1:
                            residue_id = int(row['protein_residue_id'])
                            
                            # Calculate pi-stacking energy vector
                            if all(col in row for col in ['centroid1_x', 'centroid1_y', 'centroid1_z',
                                                          'centroid2_x', 'centroid2_y', 'centroid2_z']):
                                centroid1 = np.array([row['centroid1_x'], row['centroid1_y'], row['centroid1_z']])
                                centroid2 = np.array([row['centroid2_x'], row['centroid2_y'], row['centroid2_z']])
                                angle = row.get('angle', 0)
                                offset = row.get('offset_distance', np.linalg.norm(centroid2 - centroid1))
                                energy = row['bond_energy']
                                
                                energy_vector = self.calculate_pi_stacking_vector(
                                    centroid1, centroid2, angle, offset, energy
                                )
                                
                                # Add to pi-stacking contributions
                                if residue_id not in pi_stacking_contributions:
                                    pi_stacking_contributions[residue_id] = []
                                pi_stacking_contributions[residue_id].append(energy_vector)
                            else:
                                # Fallback: use vector if centroid not available
                                vector = np.array([row['vector_x'], row['vector_y'], row['vector_z']])
                                energy_vector = vector * row['bond_energy']
                                
                                if residue_id not in pi_stacking_contributions:
                                    pi_stacking_contributions[residue_id] = []
                                pi_stacking_contributions[residue_id].append(energy_vector)
                    
                    # Normal atom-based interactions
                    elif row.get('protein_atom_id', -1) != -1:
                        atom_id = int(row['protein_atom_id'])
                        
                        # Use proper atom-to-residue mapping
                        if self.atom_to_residue_map and atom_id in self.atom_to_residue_map:
                            residue_id = self.atom_to_residue_map[atom_id]
                        else:
                            # Fallback if mapping not available (should not happen)
                            warnings.warn(f"Atom {atom_id} not in mapping, skipping")
                            continue
                        
                        # Calculate energy-weighted vector
                        vector = np.array([row['vector_x'], row['vector_y'], row['vector_z']])
                        energy = row['bond_energy']
                        energy_vector = vector * energy
                        
                        # Store vector for this iteration
                        if residue_id not in residue_energy_tensors:
                            residue_energy_tensors[residue_id] = []
                        residue_energy_tensors[residue_id].append(energy_vector)
                        
                        # If we have separate inter/intra vectors, track them
                        if 'inter_vector_x' in row and 'intra_vector_x' in row:
                            inter_vec = np.array([row['inter_vector_x'], row['inter_vector_y'], row['inter_vector_z']])
                            intra_vec = np.array([row['intra_vector_x'], row['intra_vector_y'], row['intra_vector_z']])
                            
                            if residue_id not in inter_protein_contributions:
                                inter_protein_contributions[residue_id] = []
                                intra_protein_contributions[residue_id] = []
                            
                            inter_protein_contributions[residue_id].append(inter_vec * energy)
                            intra_protein_contributions[residue_id].append(intra_vec * energy)
                        
                        # Track combined vector magnitudes if available
                        if 'combined_magnitude' in row:
                            if residue_id not in combined_vector_magnitudes:
                                combined_vector_magnitudes[residue_id] = []
                            combined_vector_magnitudes[residue_id].append(row['combined_magnitude'])
                    
            except Exception as e:
                print(f"   ⚠️  Error processing {file_path}: {e}")
        
        # Merge pi-stacking contributions with main tensors
        print(f"\n   Merging {len(pi_stacking_contributions)} pi-stacking contributions...")
        pi_stacking_energy_total = 0
        
        for res_id, pi_vectors in pi_stacking_contributions.items():
            if res_id in residue_energy_tensors:
                residue_energy_tensors[res_id].extend(pi_vectors)
            else:
                residue_energy_tensors[res_id] = pi_vectors
            
            # Calculate total pi-stacking energy
            pi_energies = [np.linalg.norm(v) for v in pi_vectors]
            pi_stacking_energy_total += sum(pi_energies)
        
        # Print interaction type summary
        if interaction_type_counts:
            print("\n   Interaction types found:")
            total_interactions = sum(interaction_type_counts.values())
            for itype, count in sorted(interaction_type_counts.items()):
                percentage = (count / total_interactions) * 100
                print(f"     - {itype}: {count} ({percentage:.1f}%)")
            
            # Calculate pi-stacking contribution
            pi_stack_count = sum(count for itype, count in interaction_type_counts.items()
                               if 'Pi-Stacking' in str(itype))
            if pi_stack_count > 0:
                pi_percentage = (pi_stack_count / total_interactions) * 100
                print(f"\n   ✓ Pi-stacking contribution: {pi_percentage:.1f}% of all interactions")
                print(f"   ✓ Pi-stacking total energy: {pi_stacking_energy_total:.2e}")
            else:
                print("\n   ⚠️  No pi-stacking interactions found in this dataset")
        
        # Convert to numpy arrays and filter
        filtered_tensors = {}
        for res_id in residue_energy_tensors:
            tensor = np.array(residue_energy_tensors[res_id])
            if len(tensor) >= 5:
                filtered_tensors[res_id] = tensor
        
        # Store inter/intra contributions for later analysis
        self.inter_contributions = {}
        self.intra_contributions = {}
        
        for res_id in inter_protein_contributions:
            self.inter_contributions[res_id] = np.array(inter_protein_contributions[res_id])
            self.intra_contributions[res_id] = np.array(intra_protein_contributions[res_id])
        
        print(f"\n   Processed {len(filtered_tensors)} residues with sufficient data")
        
        if len(self.inter_contributions) > 0:
            print(f"   ✓ Tracked inter/intra contributions for {len(self.inter_contributions)} residues")
        
        return filtered_tensors
    
    def process_gpu_trajectory_results(self, gpu_trajectory_results, residue_indices):
        """
        Process GPU trajectory results directly without CSV parsing.
        Converts InteractionResult objects to residue energy tensors.
        """
        residue_energy_tensors = {}
        
        print(f"\n   Processing {len(gpu_trajectory_results)} GPU trajectory frames...")
        
        for frame_idx, frame_result in enumerate(gpu_trajectory_results):
            if 'best_interactions' not in frame_result:
                continue
                
            interactions = frame_result['best_interactions']
            
            # Get vectors - prefer combined vectors if available
            if hasattr(interactions, 'combined_vectors') and interactions.combined_vectors is not None:
                vectors = interactions.combined_vectors.cpu().numpy()
            elif hasattr(interactions, 'vectors') and interactions.vectors is not None:
                vectors = interactions.vectors.cpu().numpy()
            else:
                # Fallback: create vectors from energies
                energies = interactions.energies.cpu().numpy()
                vectors = np.column_stack([energies, np.zeros_like(energies), np.zeros_like(energies)])
            
            # Get residue IDs
            residue_ids = interactions.residue_ids.cpu().numpy()
            
            # Accumulate vectors by residue
            for i, res_id in enumerate(residue_ids):
                if res_id not in residue_energy_tensors:
                    residue_energy_tensors[res_id] = []
                residue_energy_tensors[res_id].append(vectors[i])
        
        # Convert to numpy arrays and filter
        filtered_tensors = {}
        for res_id in residue_energy_tensors:
            tensor = np.array(residue_energy_tensors[res_id])
            if len(tensor) >= 5:  # Minimum data points for analysis
                filtered_tensors[res_id] = tensor
        
        print(f"   ✓ Processed {len(filtered_tensors)} residues with sufficient GPU data")
        
        # Store GPU results for later use
        self.gpu_results = gpu_trajectory_results
        
        return filtered_tensors
    
    def calculate_tensor_flux_differentials(self, residue_tensors, ca_coords, residue_indices):
        """Calculate flux differentials using tensor analysis with proper mapping"""
        n_residues = len(residue_indices)
        flux_differentials = np.zeros(n_residues)
        flux_derivatives = np.zeros(n_residues)
        
        print(f"\n   Calculating flux for {n_residues} residues...")
        
        mapped_count = 0
        unmapped_residues = []
        
        # For each PDB residue, find matching tensor data
        for i, res_id in enumerate(residue_indices):
            if res_id in residue_tensors:
                tensor = residue_tensors[res_id]
                mapped_count += 1
                
                if len(tensor) > 3:
                    try:
                        # Apply temporal smoothing
                        t = np.linspace(0, 1, len(tensor))
                        smoothed_tensor = []
                        
                        for dim in range(3):
                            if len(tensor) >= 4:
                                k = min(3, len(tensor) - 1)
                                spline = UnivariateSpline(t, tensor[:, dim], s=0.1, k=k)
                                smoothed_tensor.append(spline(t))
                            else:
                                smoothed_tensor.append(tensor[:, dim])
                        
                        smoothed_tensor = np.array(smoothed_tensor).T
                        
                        # Calculate magnitudes
                        magnitudes = np.linalg.norm(smoothed_tensor, axis=1)
                        
                        # Calculate rate of change
                        if len(magnitudes) > 1:
                            if len(magnitudes) >= 5:
                                window_length = min(5, len(magnitudes))
                                if window_length % 2 == 0:
                                    window_length -= 1
                                if window_length >= 3:
                                    mag_derivative = savgol_filter(magnitudes,
                                                                 window_length=window_length,
                                                                 polyorder=min(2, window_length-1),
                                                                 deriv=1)
                                else:
                                    mag_derivative = np.gradient(magnitudes)
                            else:
                                mag_derivative = np.gradient(magnitudes)
                            
                            rate_of_change = np.sqrt(np.mean(mag_derivative**2))
                        else:
                            rate_of_change = 0
                        
                        # Calculate directional consistency
                        if len(smoothed_tensor) > 1:
                            norms = np.linalg.norm(smoothed_tensor, axis=1, keepdims=True)
                            normalized = smoothed_tensor / (norms + 1e-10)
                            
                            mean_direction = np.mean(normalized, axis=0)
                            mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-10)
                            
                            consistencies = [np.dot(v, mean_direction) for v in normalized]
                            directional_consistency = np.mean(consistencies)
                            directional_consistency = (directional_consistency + 1) / 2
                        else:
                            directional_consistency = 0.5
                        
                        # Calculate flux
                        mean_magnitude = np.mean(magnitudes)
                        flux_differentials[i] = mean_magnitude * directional_consistency * (1 + rate_of_change)
                        flux_derivatives[i] = rate_of_change
                        
                    except Exception as e:
                        print(f"   ⚠️  Error processing residue {res_id}: {e}")
                        flux_differentials[i] = 0
                        flux_derivatives[i] = 0
            else:
                unmapped_residues.append(res_id)
                flux_differentials[i] = 0
                flux_derivatives[i] = 0
        
        print(f"   ✓ Successfully mapped {mapped_count}/{n_residues} residues ({100*mapped_count/n_residues:.1f}%)")
        
        if unmapped_residues:
            print(f"   ℹ️  {len(unmapped_residues)} residues had no interactions: {unmapped_residues[:5]}...")
        
        # Apply spatial smoothing
        if np.any(flux_differentials > 0):
            flux_differentials = gaussian_filter1d(flux_differentials, sigma=2.0, mode='reflect')
            flux_derivatives = gaussian_filter1d(flux_derivatives, sigma=2.0, mode='reflect')
        
        # Normalize
        if flux_differentials.max() > flux_differentials.min():
            flux_differentials = (flux_differentials - flux_differentials.min()) / \
                               (flux_differentials.max() - flux_differentials.min())
        
        return flux_differentials, flux_derivatives
    
    def create_ultra_smooth_backbone(self, ca_coords, flux_values, smoothing_factor=10):
        """Create ultra-smooth backbone using B-spline interpolation"""
        if len(ca_coords) < 4:
            return ca_coords, flux_values
        
        # Arc length parameterization
        distances = np.sqrt(np.sum(np.diff(ca_coords, axis=0)**2, axis=1))
        arc_length = np.concatenate(([0], np.cumsum(distances)))
        arc_length = arc_length / arc_length[-1]
        
        # Fit B-spline
        k = min(5, len(ca_coords) - 1)
        tck, u = splprep([ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2]],
                         u=arc_length, s=len(ca_coords)*0.05, k=k)
        
        # Evaluate at high resolution
        u_new = np.linspace(0, 1, len(ca_coords) * smoothing_factor)
        smooth_coords = np.array(splev(u_new, tck)).T
        
        # Interpolate flux values
        from scipy.interpolate import interp1d
        flux_interp = interp1d(arc_length, flux_values, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
        smooth_flux = flux_interp(u_new)
        
        # Additional smoothing
        smooth_flux = gaussian_filter1d(smooth_flux, sigma=smoothing_factor/2)
        smooth_flux = np.clip(smooth_flux, 0, 1)
        
        return smooth_coords, smooth_flux
    
    def _style_3d_axis(self, ax):
        """Apply 3D axis styling"""
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=135)
        ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
    
    def process_trajectory_iterations(self, results_dir, protein_pdb, gpu_trajectory_results=None):
        """
        Process trajectory iteration data for flux analysis with validation.
        Can handle both CSV files and direct GPU trajectory results.
        """
        print("\n" + "="*80)
        print("TRAJECTORY FLUX DIFFERENTIAL ANALYSIS")
        print("="*80)
        
        # Parse protein structure
        print("\n1. Loading protein structure...")
        ca_atoms, ca_coords, res_indices, res_names = self.parse_pdb_for_ribbon(protein_pdb)
        print(f"   ✓ Loaded {len(ca_coords)} residues")
        print(f"   ✓ Created proper atom-to-residue mapping")
        
        # Check if we have GPU results
        if gpu_trajectory_results is not None:
            print(f"\n2. Using GPU trajectory results directly (bypassing CSV parsing)")
            print(f"   Found {len(gpu_trajectory_results)} trajectory frames")
            
            # Process GPU results into flux data
            all_flux_data = []
            all_derivatives = []
            
            # Group GPU results by iteration (if structured that way)
            # Otherwise treat as single iteration
            if isinstance(gpu_trajectory_results, list) and len(gpu_trajectory_results) > 0:
                # Process as single iteration for now
                residue_tensors = self.process_gpu_trajectory_results(gpu_trajectory_results, res_indices)
                
                # Calculate flux differentials
                flux, derivatives = self.calculate_tensor_flux_differentials(
                    residue_tensors, ca_coords, res_indices)
                
                all_flux_data.append(flux)
                all_derivatives.append(derivatives)
            
        else:
            # Original CSV-based processing
            # Find all iteration directories
            iter_dirs = sorted(glob.glob(os.path.join(results_dir, "iteration_*")))
            print(f"\n2. Found {len(iter_dirs)} iterations to process")
            
            if len(iter_dirs) == 0:
                print("\n   ❌ ERROR: No iteration directories found!")
                raise ValueError("No iteration data found to process")
            
            # Initialize for CSV processing
            pass  # Will be handled below
        
        # Continue with appropriate processing
        if gpu_trajectory_results is None:
            # CSV-based processing
            all_flux_data = []
            all_derivatives = []
            
            import time
            
            total_iterations = len(iter_dirs)
            start_time = time.time()
            
            for idx, iter_dir in enumerate(iter_dirs):
                iter_start = time.time()
                
                # Progress tracking
                progress = (idx / total_iterations) * 100
            
                if idx > 0:
                    elapsed = time.time() - start_time
                    avg_time_per_iter = elapsed / idx
                    remaining_iters = total_iterations - idx
                    eta_seconds = avg_time_per_iter * remaining_iters
                    eta = datetime.now() + timedelta(seconds=eta_seconds)
                    eta_str = eta.strftime("%H:%M:%S")
                else:
                    eta_str = "Calculating..."
                
                print(f"\n   [{idx+1}/{total_iterations}] Processing {os.path.basename(iter_dir)}... "
                      f"({progress:.1f}% complete, ETA: {eta_str})")
                
                # Process CSV files in this iteration
                try:
                    residue_tensors = self.process_iteration_files(
                        iter_dir,
                        "flux_iteration_*_output_vectors.csv",
                        structure_file=protein_pdb
                    )
                except ValueError as e:
                    print(f"   ⚠️  Skipping {os.path.basename(iter_dir)}: {e}")
                    continue
                
                # Calculate flux differentials
                flux, derivatives = self.calculate_tensor_flux_differentials(
                    residue_tensors, ca_coords, res_indices)
                
                all_flux_data.append(flux)
                all_derivatives.append(derivatives)
                
                iter_time = time.time() - iter_start
                print(f"   ✓ Calculated flux for {len(flux)} residues in {iter_time:.2f}s")
        
        # Average flux across iterations
        print("\n3. Averaging flux across iterations...")
        
        if len(all_flux_data) == 0:
            raise ValueError("No flux data to process")
        
        avg_flux = np.mean(all_flux_data, axis=0)
        std_flux = np.std(all_flux_data, axis=0)
        avg_derivatives = np.mean(all_derivatives, axis=0)
        
        print(f"   ✓ Average flux range: {avg_flux.min():.3f} - {avg_flux.max():.3f}")
        print(f"   ✓ Standard deviation range: {std_flux.min():.3f} - {std_flux.max():.3f}")
        
        # Bootstrap validation
        print("\n4. Performing bootstrap statistical validation...")
        bootstrap_stats = self.bootstrap_validator.bootstrap_flux_analysis(
            all_flux_data, res_indices, n_bootstrap=100
        )
        
        # Extract relevant columns
        if self.target_is_dna:
            id_col = 'dna_nucleotide_id'
            res_col = 'dna_resname'
            chain_col = 'dna_chain'
        else:
            id_col = 'protein_residue_id'
            res_col = 'protein_resname'
            chain_col = 'protein_chain'

        df_data = {
            id_col: res_indices,
            res_col: res_names,
            chain_col: ['A'] * len(res_indices),  # Default to chain A for now
            'average_flux': avg_flux,
            'std_flux': std_flux,
            'avg_derivatives': avg_derivatives,
            'ci_lower_95': avg_flux - std_flux,
            'ci_upper_95': avg_flux + std_flux,
            'smoothed_flux': gaussian_filter1d(avg_flux, sigma=2.0, mode='reflect')
        }
        
        # Add bootstrap statistics
        for res_id, stats in bootstrap_stats.items():
            df_data[f'bootstrap_mean_{res_id}'] = stats['mean']
            df_data[f'bootstrap_std_{res_id}'] = stats['std']
            df_data[f'bootstrap_ci_lower_{res_id}'] = stats['ci_lower']
            df_data[f'bootstrap_ci_upper_{res_id}'] = stats['ci_upper']
            df_data[f'bootstrap_p_value_{res_id}'] = stats['p_value']
            df_data[f'bootstrap_effect_size_{res_id}'] = stats['effect_size']
            df_data[f'bootstrap_is_significant_{res_id}'] = stats['is_significant']
        
        # Create DataFrame
        self.flux_data = pd.DataFrame(df_data)
        
        # Store all_flux_data for visualization
        self.all_flux_data = all_flux_data
        
        # Bootstrap validation
        n_significant = sum(1 for stats in bootstrap_stats.values() if stats['is_significant'])
        print(f"   ✓ Bootstrap complete: {n_significant}/{len(res_indices)} residues significant (p<0.05)")
        
        return self.flux_data
    
    def visualize_trajectory_flux(self, flux_data, protein_name, output_dir):
        """Create comprehensive flux visualization with both normalized and absolute flux"""
        fig = plt.figure(figsize=(20, 16))
        
        res_indices = np.array(flux_data['protein_residue_id'])
        avg_flux = flux_data['average_flux']
        
        # Get bootstrap confidence intervals
        if 'bootstrap_stats' in flux_data:
            ci_lower = np.array([flux_data['bootstrap_stats'][res_id]['ci_lower']
                                for res_id in res_indices])
            ci_upper = np.array([flux_data['bootstrap_stats'][res_id]['ci_upper']
                                for res_id in res_indices])
        else:
            # Fallback to standard deviation
            std_flux = flux_data['std_flux']
            ci_lower = avg_flux - std_flux
            ci_upper = avg_flux + std_flux
        
        # Calculate absolute flux (non-normalized)
        flux_matrix = np.array(self.all_flux_data)  # Shape: [n_iterations, n_residues]
        
        # Get max flux for normalization
        max_flux = np.max(avg_flux)
        absolute_flux = avg_flux * max_flux  # Convert back to absolute values
        absolute_ci_lower = ci_lower * max_flux
        absolute_ci_upper = ci_upper * max_flux
        
        # === Panel 1: Normalized Flux profile ===
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Plot normalized flux with confidence intervals
        ax1.plot(res_indices, avg_flux, 'b-', linewidth=2, label='Normalized Flux')
        ax1.fill_between(res_indices, ci_lower, ci_upper,
                        alpha=0.3, color='blue', label='95% CI')
        
        # Mark high-flux residues (>75th percentile)
        threshold = np.percentile(avg_flux[avg_flux > 0], 75)
        high_flux_mask = avg_flux > threshold
        high_flux_indices = res_indices[high_flux_mask]
        high_flux_values = avg_flux[high_flux_mask]
        
        ax1.scatter(high_flux_indices, high_flux_values,
                   color='red', s=50, zorder=5, label='High Flux')
        
        ax1.set_xlabel('Residue Index')
        ax1.set_ylabel('Normalized Flux (0-1)')
        ax1.set_title(f'{protein_name} - Normalized Energy Flux')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # === Panel 2: Absolute Flux profile ===
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Plot absolute flux with confidence intervals
        ax2.plot(res_indices, absolute_flux, 'g-', linewidth=2, label='Absolute Flux')
        ax2.fill_between(res_indices, absolute_ci_lower, absolute_ci_upper,
                        alpha=0.3, color='green', label='95% CI')
        
        # Mark high-flux residues
        high_flux_absolute = absolute_flux[high_flux_mask]
        ax2.scatter(high_flux_indices, high_flux_absolute,
                   color='red', s=50, zorder=5, label='High Flux')
        
        ax2.set_xlabel('Residue Index')
        ax2.set_ylabel('Absolute Flux (kcal/mol/Å)')
        ax2.set_title(f'{protein_name} - Absolute Energy Flux')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # === Panel 3: Normalized flux heatmap ===
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Normalize flux matrix to 0-1
        normalized_matrix = flux_matrix / np.max(flux_matrix)
        
        # Downsample if too many residues for clear visualization
        if normalized_matrix.shape[1] > 200:
            step = normalized_matrix.shape[1] // 200
            normalized_matrix_plot = normalized_matrix[:, ::step]
            x_labels = res_indices[::step]
        else:
            normalized_matrix_plot = normalized_matrix
            x_labels = res_indices
        
        # Create normalized heatmap
        sns.heatmap(normalized_matrix_plot, 
                    cmap='YlOrRd', 
                    cbar_kws={'label': 'Normalized Flux (0-1)'},
                    ax=ax3,
                    vmin=0, vmax=1)
        
        ax3.set_xlabel('Residue Index')
        ax3.set_ylabel('Iteration')
        ax3.set_title('Normalized Flux Across Iterations')
        
        # === Panel 4: Absolute flux heatmap ===
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Use original flux matrix for absolute values
        if flux_matrix.shape[1] > 200:
            step = flux_matrix.shape[1] // 200
            flux_matrix_plot = flux_matrix[:, ::step]
        else:
            flux_matrix_plot = flux_matrix
        
        # Create absolute heatmap
        sns.heatmap(flux_matrix_plot, 
                    cmap='YlOrRd', 
                    cbar_kws={'label': 'Absolute Flux (kcal/mol/Å)'},
                    ax=ax4)
        
        ax4.set_xlabel('Residue Index')
        ax4.set_ylabel('Iteration')
        ax4.set_title('Absolute Flux Across Iterations')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{protein_name}_trajectory_flux_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n5. Saved visualization to: {output_file}")
        
        return fig
    
    def generate_summary_report(self, flux_data, protein_name, output_dir):
        """Generate detailed analysis report with pi-stacking contribution"""
        report_file = os.path.join(output_dir, f'{protein_name}_flux_report.txt')
        
        res_indices = np.array(flux_data['protein_residue_id'])
        res_names = flux_data['protein_resname']
        avg_flux = flux_data['average_flux']
        
        with open(report_file, 'w') as f:
            f.write(f"FluxMD Analysis Report for {protein_name}\n")
            f.write(f"Unified Memory Architecture (UMA) Optimized Pipeline\n")
            f.write("=" * 60 + "\n")
            f.write(f"\nTotal residues analyzed: {len(res_indices)}\n")
            f.write(f"Average flux range: [{avg_flux.min():.4f}, {avg_flux.max():.4f}]\n")
            f.write("\nTop 10 High-Flux Residues:\n")
            f.write("-" * 30 + "\n")
            
            # Get top residues
            sorted_indices = np.argsort(avg_flux)[::-1]
            
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                if avg_flux[idx] > 0:
                    res_id = res_indices[idx]
                    res_name = res_names[idx]
                    flux_val = avg_flux[idx]
                    
                    # Get confidence intervals
                    if 'bootstrap_stats' in flux_data and res_id in flux_data['bootstrap_stats']:
                        stats = flux_data['bootstrap_stats'][res_id]
                        ci_lower = stats['ci_lower']
                        ci_upper = stats['ci_upper']
                    else:
                        std_flux = flux_data['std_flux']
                        ci_lower = flux_val - std_flux[idx]
                        ci_upper = flux_val + std_flux[idx]
                    
                    f.write(
                        f"{i+1}. Residue {res_id} ({res_name}): "
                        f"Flux = {flux_val:.4f} [95% CI: {ci_lower:.4f}-{ci_upper:.4f}]\n"
                    )
            
            # Statistical summary
            f.write("\n\nStatistical Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean flux: {np.mean(avg_flux):.4f} ± {np.std(avg_flux):.4f}\n")
            f.write(f"Median flux: {np.median(avg_flux):.4f}\n")
            f.write(f"Non-zero residues: {np.sum(avg_flux > 0)} ({np.sum(avg_flux > 0)/len(avg_flux)*100:.1f}%)\n")
        
        print(f"   ✓ Saved report to: {report_file}")
        
        return report_file
    
    def save_processed_data(self, flux_data, output_dir):
        """Save processed flux data with statistical information"""
        res_indices = np.array(flux_data['protein_residue_id'])
        res_names = flux_data['protein_resname']
        
        # Calculate smoothed flux if not present
        if 'smoothed_flux' not in flux_data:
            from scipy.ndimage import gaussian_filter1d
            smoothed_flux = gaussian_filter1d(flux_data['average_flux'], sigma=2.0, mode='reflect')
        else:
            smoothed_flux = flux_data['smoothed_flux']
        
        # Build simplified dataframe
        df_data = {
            'residue_index': res_indices,
            'residue_name': res_names,
            'average_flux': flux_data['average_flux'],
            'std_flux': flux_data['std_flux'],
        }
        
        # Add confidence intervals
        if 'bootstrap_stats' in flux_data:
            ci_lower = []
            ci_upper = []
            for res_id in res_indices:
                if res_id in flux_data['bootstrap_stats']:
                    stats = flux_data['bootstrap_stats'][res_id]
                    ci_lower.append(stats['ci_lower'])
                    ci_upper.append(stats['ci_upper'])
                else:
                    ci_lower.append(flux_data['average_flux'][list(res_indices).index(res_id)] - 
                                   flux_data['std_flux'][list(res_indices).index(res_id)])
                    ci_upper.append(flux_data['average_flux'][list(res_indices).index(res_id)] + 
                                   flux_data['std_flux'][list(res_indices).index(res_id)])
            df_data['ci_lower_95'] = ci_lower
            df_data['ci_upper_95'] = ci_upper
        else:
            # Use standard deviation as confidence interval
            df_data['ci_lower_95'] = flux_data['average_flux'] - flux_data['std_flux']
            df_data['ci_upper_95'] = flux_data['average_flux'] + flux_data['std_flux']
        
        df_data['smoothed_flux'] = smoothed_flux
        
        # Save main data
        df = pd.DataFrame(df_data)
        flux_file = os.path.join(output_dir, 'processed_flux_data.csv')
        df.to_csv(flux_file, index=False, float_format='%.6f')
        print(f"\n4. Saved final processed flux data to: {flux_file}")
        
        return flux_file
    
    def create_integrated_flux_pipeline(self, protein_file, gpu_trajectory_results, output_dir):
        """
        Integrated pipeline for GPU trajectory results to final flux analysis.
        Bypasses CSV parsing for maximum efficiency.
        """
        from gpu_accelerated_flux import GPUFluxCalculator
        
        print("\n" + "="*80)
        print("INTEGRATED GPU FLUX ANALYSIS PIPELINE")
        print("="*80)
        
        # Parse protein structure
        print("\n1. Loading protein structure...")
        ca_atoms, ca_coords, res_indices, res_names = self.parse_pdb_for_ribbon(protein_file)
        n_residues = len(res_indices)
        
        # Create GPU flux calculator
        gpu_calculator = GPUFluxCalculator()
        
        # Process trajectory to flux directly
        print("\n2. Computing flux from GPU trajectory results...")
        flux_tensor = gpu_calculator.process_trajectory_to_flux(gpu_trajectory_results, n_residues)
        
        # Convert to numpy
        flux_values = flux_tensor.cpu().numpy()
        
        # Create flux data structure compatible with visualization
        flux_data = {
            'protein_residue_id': res_indices,
            'protein_resname': res_names,
            'average_flux': flux_values,
            'std_flux': np.zeros_like(flux_values),  # Can be computed if multiple iterations
            'avg_derivatives': np.zeros_like(flux_values),
            'all_flux': [flux_values],  # Single iteration for now
            'all_derivatives': [np.zeros_like(flux_values)],
            'bootstrap_stats': {}  # Can add bootstrap if needed
        }
        
        print(f"\n3. Flux computation complete!")
        print(f"   Flux range: {flux_values.min():.3f} - {flux_values.max():.3f}")
        print(f"   Mean flux: {flux_values.mean():.3f}")
        
        return flux_data

    def write_report(self):
        """Write a detailed report of the flux analysis"""
        report_file = os.path.join(self.output_dir, f"{self.protein_name}_flux_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            if self.target_is_dna:
                f.write("FluxMD Protein-DNA Interaction Analysis Report\n".center(80) + "\n")
            else:
                f.write("FluxMD Protein-Ligand Interaction Analysis Report\n".center(80) + "\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Target Molecule: {self.protein_name}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- Top 10 High-Flux Residues ---\n")
            res_indices = np.array(self.flux_data['protein_residue_id'])
            avg_flux = self.flux_data['average_flux']
            sorted_indices = np.argsort(avg_flux)[::-1]
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                if avg_flux[idx] > 0:
                    res_id = res_indices[idx]
                    res_name = self.flux_data['protein_resname'][idx]
                    flux_val = avg_flux[idx]
                    f.write(f"{i+1}. Residue {res_id} ({res_name}): Flux = {flux_val:.4f}\n")
            
            f.write("\n\nStatistical Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean flux: {np.mean(avg_flux):.4f} ± {np.std(avg_flux):.4f}\n")
            f.write(f"Median flux: {np.median(avg_flux):.4f}\n")
            f.write(f"Non-zero residues: {np.sum(avg_flux > 0)} ({np.sum(avg_flux > 0)/len(avg_flux)*100:.1f}%)\n")
        
        print(f"   ✓ Saved analysis report to: {report_file}")

    def process_and_save_results(self):
        """Main function to run all analysis steps"""
        self.load_all_iteration_data()
        self.calculate_flux_metrics()
        self.write_report()
        
        print("\nFlux analysis complete.")


class BootstrapFluxValidator:
    """Statistical validation for flux calculations using bootstrap"""
    
    def __init__(self, n_bootstrap=1000, confidence_level=0.95, n_jobs=-1):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs
    
    def bootstrap_flux_analysis(self, all_flux_data, residue_indices, n_bootstrap=None):
        """Perform bootstrap analysis on flux data"""
        if n_bootstrap is not None:
            self.n_bootstrap = n_bootstrap
        
        n_iterations = len(all_flux_data)
        n_residues = len(residue_indices)
        
        print(f"   Running {self.n_bootstrap} bootstrap iterations...")
        
        # Parallel bootstrap
        bootstrap_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_bootstrap)(all_flux_data, n_iterations, n_residues)
            for _ in range(self.n_bootstrap)
        )
        
        # Aggregate results by residue
        residue_distributions = {res_id: [] for res_id in residue_indices}
        
        for bootstrap_flux in bootstrap_results:
            for i, res_id in enumerate(residue_indices):
                residue_distributions[res_id].append(bootstrap_flux[i])
        
        # Calculate statistics for each residue
        statistics = {}
        
        for i, res_id in enumerate(residue_indices):
            distribution = np.array(residue_distributions[res_id])
            
            # Basic statistics
            mean_flux = np.mean(distribution)
            std_flux = np.std(distribution)
            
            # Confidence intervals
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(distribution, (alpha/2) * 100)
            ci_upper = np.percentile(distribution, (1 - alpha/2) * 100)
            
            # P-value (test against null hypothesis of zero flux)
            # Using percentile method
            if mean_flux > 0:
                p_value = 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0))
            else:
                p_value = 1.0
            
            # Effect size (Cohen's d against zero)
            effect_size = mean_flux / (std_flux + 1e-10)
            
            # Store statistics
            statistics[res_id] = {
                'mean': mean_flux,
                'std': std_flux,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'effect_size': effect_size,
                'is_significant': p_value < 0.05
            }
        
        # Summary
        n_significant = sum(1 for stats in statistics.values() if stats['is_significant'])
        print(f"   ✓ Bootstrap complete: {n_significant}/{n_residues} residues significant (p<0.05)")
        
        return statistics
    
    def _single_bootstrap(self, all_flux_data, n_iterations, n_residues):
        """Single bootstrap iteration"""
        # Resample iterations with replacement
        bootstrap_indices = np.random.choice(n_iterations, size=n_iterations, replace=True)
        
        # Calculate mean flux for this bootstrap sample
        bootstrap_flux = np.zeros(n_residues)
        for idx in bootstrap_indices:
            bootstrap_flux += all_flux_data[idx]
        
        bootstrap_flux /= n_iterations
        
        return bootstrap_flux


def main():
    """Main analysis pipeline"""
    analyzer = TrajectoryFluxAnalyzer(output_dir="output_directory", protein_name="GPX4", target_is_dna=False)
    
    print("TRAJECTORY FLUX DIFFERENTIAL ANALYSIS")
    print("-" * 40)
    
    # Get input parameters
    results_dir = input("Enter trajectory results directory: ").strip()
    protein_pdb = input("Enter protein PDB file path: ").strip()
    protein_name = input("Enter protein name for labeling (e.g., GPX4): ").strip()
    
    # Check inputs
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found!")
        return
    
    if not os.path.exists(protein_pdb):
        print(f"Error: Protein PDB file {protein_pdb} not found!")
        return
    
    try:
        # Process trajectory data
        flux_data = analyzer.process_trajectory_iterations(results_dir, protein_pdb)
        
        # Create visualizations
        analyzer.visualize_trajectory_flux(flux_data, protein_name, results_dir)
        
        # Generate report
        analyzer.generate_summary_report(flux_data, protein_name, results_dir)
        
        # Save processed data
        analyzer.save_processed_data(flux_data, results_dir)
        
        # Process and save results
        analyzer.process_and_save_results()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {results_dir}")
        print("\nHigh flux regions (red) indicate potential binding sites")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
