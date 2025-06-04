"""
Flux Differential Integration for Trajectory Analysis
Advanced statistical analysis of molecular dynamics trajectories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

class TrajectoryFluxAnalyzer:
    """Flux analyzer for trajectory-based analysis"""
    
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.atom_to_residue_map = None
        self.residue_to_atom_map = None
        self.bootstrap_validator = BootstrapFluxValidator()
        
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
    
    def process_trajectory_iterations(self, results_dir, protein_pdb):
        """Process trajectory iteration data for flux analysis with validation"""
        print("\n" + "="*80)
        print("TRAJECTORY FLUX DIFFERENTIAL ANALYSIS")
        print("="*80)
        
        # Parse protein structure
        print("\n1. Loading protein structure...")
        ca_atoms, ca_coords, res_indices, res_names = self.parse_pdb_for_ribbon(protein_pdb)
        print(f"   ✓ Loaded {len(ca_coords)} residues")
        print(f"   ✓ Created proper atom-to-residue mapping")
        
        # Find all iteration directories
        iter_dirs = sorted(glob.glob(os.path.join(results_dir, "iteration_*")))
        print(f"\n2. Found {len(iter_dirs)} iterations to process")
        
        if len(iter_dirs) == 0:
            print("\n   ❌ ERROR: No iteration directories found!")
            raise ValueError("No iteration data found to process")
        
        # Process each iteration
        all_flux_data = []
        all_derivatives = []
        
        import time
        from datetime import datetime, timedelta
        
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
            residue_tensors = self.process_iteration_files(
                iter_dir,
                "flux_iteration_*_output_vectors.csv",
                structure_file=protein_pdb
            )
            
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
        
        return {
            'ca_coords': ca_coords,
            'res_indices': res_indices,
            'res_names': res_names,
            'avg_flux': avg_flux,
            'std_flux': std_flux,
            'avg_derivatives': avg_derivatives,
            'all_flux': all_flux_data,
            'all_derivatives': all_derivatives,
            'bootstrap_stats': bootstrap_stats
        }
    
    def visualize_trajectory_flux(self, flux_data, protein_name, output_dir):
        """Create comprehensive flux visualization with statistical significance"""
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle(f'{protein_name} - Trajectory-Based Flux Analysis',
                    fontsize=20, fontweight='bold')
        
        # Enhanced colormap
        colors = ['#000080', '#0000ff', '#0080ff', '#00ffff', '#00ff80',
                 '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#800000']
        cmap = LinearSegmentedColormap.from_list('flux', colors, N=256)
        
        # === Panel 1: 3D structure with average flux ===
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.set_title('Average Flux Landscape', fontsize=14, fontweight='bold')
        
        # Create smooth backbone
        smooth_coords, smooth_flux = self.create_ultra_smooth_backbone(
            flux_data['ca_coords'], flux_data['avg_flux'], smoothing_factor=10)
        
        # Plot with flux coloring
        for i in range(len(smooth_coords) - 1):
            color = cmap(smooth_flux[i])
            for offset in np.linspace(-0.3, 0.3, 3):
                start = smooth_coords[i] + np.array([offset, 0, 0])
                end = smooth_coords[i+1] + np.array([offset, 0, 0])
                ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                        color=color, linewidth=8, alpha=0.8)
        
        self._style_3d_axis(ax1)
        
        # === Panel 2: Flux profile with confidence intervals ===
        ax2 = fig.add_subplot(232)
        ax2.set_title('Flux Profile with 95% Confidence Intervals', fontsize=14, fontweight='bold')
        
        res_indices = np.array(flux_data['res_indices'])
        avg_flux = flux_data['avg_flux']
        
        # Get bootstrap confidence intervals
        if 'bootstrap_stats' in flux_data:
            ci_lower = np.array([flux_data['bootstrap_stats'][res_id]['ci_lower']
                                for res_id in res_indices])
            ci_upper = np.array([flux_data['bootstrap_stats'][res_id]['ci_upper']
                                for res_id in res_indices])
            
            # Plot with confidence intervals
            ax2.plot(res_indices, avg_flux, 'b-', linewidth=3, label='Average Flux')
            ax2.fill_between(res_indices, ci_lower, ci_upper,
                            alpha=0.3, color='blue', label='95% CI')
            
            # Mark statistically significant residues
            significant = np.array([flux_data['bootstrap_stats'][res_id]['is_significant']
                                   for res_id in res_indices])
            ax2.scatter(res_indices[significant], avg_flux[significant],
                       c='red', s=50, zorder=5, label='Significant (p<0.05)')
        else:
            # Fallback to standard deviation
            std_flux = flux_data['std_flux']
            ax2.plot(res_indices, avg_flux, 'b-', linewidth=3, label='Average Flux')
            ax2.fill_between(res_indices, avg_flux - std_flux, avg_flux + std_flux,
                            alpha=0.3, color='blue', label='±1 SD')
        
        # Mark aromatic residues
        aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
        aromatic_mask = np.array([name in aromatic_residues for name in flux_data['res_names']])
        if np.any(aromatic_mask):
            ax2.scatter(res_indices[aromatic_mask], avg_flux[aromatic_mask],
                       c='purple', s=100, marker='^', zorder=6,
                       label='Aromatic (π-stacking)', alpha=0.6)
        
        ax2.set_xlabel('Residue Number', fontsize=12)
        ax2.set_ylabel('Flux Differential', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # === Panel 3: Bootstrap p-value heatmap ===
        ax3 = fig.add_subplot(233)
        ax3.set_title('Statistical Significance (p-values)', fontsize=14, fontweight='bold')
        
        if 'bootstrap_stats' in flux_data:
            p_values = np.array([flux_data['bootstrap_stats'][res_id]['p_value']
                                for res_id in res_indices])
            
            # Create heatmap-style bar plot
            bars = ax3.bar(res_indices, -np.log10(p_values + 1e-10), width=1.0)
            
            # Color by significance
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.001:
                    bar.set_color('darkred')
                elif p_val < 0.01:
                    bar.set_color('red')
                elif p_val < 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('gray')
            
            ax3.axhline(y=-np.log10(0.05), color='black', linestyle='--',
                       label='p=0.05 threshold')
            ax3.set_xlabel('Residue Number', fontsize=12)
            ax3.set_ylabel('-log10(p-value)', fontsize=12)
            ax3.legend()
        
        # === Panel 4: Top binding sites with pi-stacking contribution ===
        ax4 = fig.add_subplot(234)
        ax4.set_title('Top 20 Binding Sites (Energy Sinkholes)', fontsize=14, fontweight='bold')
        
        # Find top 20 residues
        top_indices = np.argsort(avg_flux)[-20:][::-1]
        top_residues = res_indices[top_indices]
        top_flux = avg_flux[top_indices]
        top_names = [flux_data['res_names'][i] for i in top_indices]
        
        # Get confidence intervals for top residues
        if 'bootstrap_stats' in flux_data:
            top_ci_lower = np.array([flux_data['bootstrap_stats'][res_id]['ci_lower']
                                    for res_id in top_residues])
            top_ci_upper = np.array([flux_data['bootstrap_stats'][res_id]['ci_upper']
                                    for res_id in top_residues])
            error_bars = [top_flux - top_ci_lower, top_ci_upper - top_flux]
        else:
            top_std = flux_data['std_flux'][top_indices]
            error_bars = top_std
        
        # Color bars by residue type
        bar_colors = []
        for name in top_names:
            if name in ['PHE', 'TYR', 'TRP', 'HIS']:
                bar_colors.append('purple')  # Aromatic
            elif name in ['ARG', 'LYS']:
                bar_colors.append('blue')    # Positive
            elif name in ['ASP', 'GLU']:
                bar_colors.append('red')     # Negative
            else:
                bar_colors.append('gray')    # Other
        
        # Bar plot with error bars
        y_pos = np.arange(len(top_residues))
        bars = ax4.barh(y_pos, top_flux, xerr=error_bars, align='center',
                       color=bar_colors, alpha=0.8, capsize=3)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'{top_names[i]} {top_residues[i]}' for i in range(len(top_residues))])
        ax4.invert_yaxis()
        ax4.set_xlabel('Flux Differential', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='purple', label='Aromatic (π-stacking capable)'),
            Patch(facecolor='blue', label='Positive charged'),
            Patch(facecolor='red', label='Negative charged'),
            Patch(facecolor='gray', label='Other')
        ]
        ax4.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        # === Panel 5: Iteration consistency heatmap ===
        ax5 = fig.add_subplot(235)
        ax5.set_title('Flux Consistency Across Iterations', fontsize=14, fontweight='bold')
        
        # Create heatmap of flux values across iterations
        flux_matrix = np.array(flux_data['all_flux'])
        im = ax5.imshow(flux_matrix, aspect='auto', cmap='viridis',
                       extent=[res_indices[0], res_indices[-1],
                              len(flux_matrix), 0])
        
        ax5.set_xlabel('Residue Number', fontsize=12)
        ax5.set_ylabel('Iteration', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Flux Differential', fontsize=10)
        
        # === Panel 6: Effect size plot ===
        ax6 = fig.add_subplot(236)
        ax6.set_title("Effect Size (Cohen's d)", fontsize=14, fontweight='bold')
        
        if 'bootstrap_stats' in flux_data:
            effect_sizes = np.array([flux_data['bootstrap_stats'][res_id]['effect_size']
                                    for res_id in res_indices])
            
            ax6.plot(res_indices, effect_sizes, 'g-', linewidth=2)
            ax6.axhline(y=0.8, color='red', linestyle='--', label='Large effect (d=0.8)')
            ax6.axhline(y=0.5, color='orange', linestyle='--', label='Medium effect (d=0.5)')
            ax6.axhline(y=0.2, color='yellow', linestyle='--', label='Small effect (d=0.2)')
            
            ax6.set_xlabel('Residue Number', fontsize=12)
            ax6.set_ylabel("Cohen's d", fontsize=12)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{protein_name}_trajectory_flux_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n5. Saved visualization to: {output_file}")
        
        return fig
    
    def generate_summary_report(self, flux_data, protein_name, output_dir):
        """Generate detailed analysis report with pi-stacking contribution"""
        report_file = os.path.join(output_dir, f'{protein_name}_flux_report.txt')
        
        res_indices = np.array(flux_data['res_indices'])
        res_names = flux_data['res_names']
        avg_flux = flux_data['avg_flux']
        std_flux = flux_data['std_flux']
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"TRAJECTORY FLUX ANALYSIS REPORT\n")
            f.write(f"Protein: {protein_name}\n")
            if hasattr(self, 'physiological_pH'):
                f.write(f"Analysis pH: {self.physiological_pH}\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total residues analyzed: {len(res_indices)}\n")
            f.write(f"Number of iterations: {len(flux_data['all_flux'])}\n")
            f.write(f"Average flux range: {avg_flux.min():.4f} - {avg_flux.max():.4f}\n")
            f.write(f"Mean flux: {avg_flux.mean():.4f} ± {avg_flux.std():.4f}\n")
            
            # Count aromatic residues
            aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
            aromatic_count = sum(1 for name in res_names if name in aromatic_residues)
            f.write(f"Aromatic residues (π-stacking capable): {aromatic_count}\n")
            
            # Statistical summary
            if 'bootstrap_stats' in flux_data:
                significant_count = sum(1 for res_id in res_indices
                                      if flux_data['bootstrap_stats'][res_id]['is_significant'])
                f.write(f"Statistically significant residues (p<0.05): {significant_count}\n")
            f.write("\n")
            
            # Top binding sites with statistics
            f.write("TOP 20 BINDING SITES (ENERGY SINKHOLES) WITH STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write("Rank | Residue | Type    | Avg Flux | 95% CI        | p-value | Signif\n")
            f.write("-"*70 + "\n")
            
            top_indices = np.argsort(avg_flux)[-20:][::-1]
            
            for rank, idx in enumerate(top_indices, 1):
                res_num = res_indices[idx]
                res_name = res_names[idx] if idx < len(res_names) else 'UNK'
                avg_flux_val = avg_flux[idx]
                
                # Get statistics
                if 'bootstrap_stats' in flux_data and res_num in flux_data['bootstrap_stats']:
                    stats = flux_data['bootstrap_stats'][res_num]
                    ci_lower = stats['ci_lower']
                    ci_upper = stats['ci_upper']
                    p_value = stats['p_value']
                    is_sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                else:
                    ci_lower = avg_flux_val - std_flux[idx]
                    ci_upper = avg_flux_val + std_flux[idx]
                    p_value = 1.0
                    is_sig = ""
                
                # Determine residue type
                if res_name in aromatic_residues:
                    res_type = "Aromatic"
                elif res_name in ['ARG', 'LYS']:
                    res_type = "Positive"
                elif res_name in ['ASP', 'GLU']:
                    res_type = "Negative"
                else:
                    res_type = "Other   "
                
                f.write(f"{rank:4d} | {res_name:3s}{res_num:4d} | {res_type} | "
                       f"{avg_flux_val:8.4f} | [{ci_lower:5.3f},{ci_upper:5.3f}] | "
                       f"{p_value:7.4f} | {is_sig:3s}\n")
            
            f.write("\n*** p<0.001, ** p<0.01, * p<0.05\n\n")
            
            # Pi-stacking analysis
            f.write("PI-STACKING ANALYSIS\n")
            f.write("-"*40 + "\n")
            
            # Find aromatic residues in top binding sites
            aromatic_in_top = []
            for idx in top_indices[:10]:  # Top 10
                res_name = res_names[idx]
                if res_name in aromatic_residues:
                    res_num = res_indices[idx]
                    flux_val = avg_flux[idx]
                    aromatic_in_top.append((res_name, res_num, flux_val))
            
            if aromatic_in_top:
                f.write(f"Aromatic residues in top 10 binding sites:\n")
                for res_name, res_num, flux_val in aromatic_in_top:
                    f.write(f"  {res_name:3s}{res_num:4d}: flux = {flux_val:.4f}\n")
                f.write(f"\nThese residues are capable of π-π stacking interactions.\n")
            else:
                f.write("No aromatic residues found in top 10 binding sites.\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("Analysis complete\n")
        
        print(f"   ✓ Saved report to: {report_file}")
        
        return report_file
    
    def save_processed_data(self, flux_data, output_dir):
        """Save processed flux data with statistical information"""
        res_indices = np.array(flux_data['res_indices'])
        res_names = flux_data['res_names']
        
        # Mark aromatic residues
        aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
        is_aromatic = [1 if name in aromatic_residues else 0 for name in res_names]
        
        # Build main dataframe
        data = {
            'residue_index': res_indices,
            'residue_name': res_names,
            'average_flux': flux_data['avg_flux'],
            'std_flux': flux_data['std_flux'],
            'average_derivative': flux_data['avg_derivatives'],
            'is_aromatic': is_aromatic
        }
        
        # Add pH information if available
        if hasattr(self, 'physiological_pH'):
            data['analysis_pH'] = [self.physiological_pH] * len(res_indices)
        
        # Add bootstrap statistics if available
        if 'bootstrap_stats' in flux_data:
            ci_lower = []
            ci_upper = []
            p_values = []
            effect_sizes = []
            is_significant = []
            
            for res_id in res_indices:
                if res_id in flux_data['bootstrap_stats']:
                    stats = flux_data['bootstrap_stats'][res_id]
                    ci_lower.append(stats['ci_lower'])
                    ci_upper.append(stats['ci_upper'])
                    p_values.append(stats['p_value'])
                    effect_sizes.append(stats['effect_size'])
                    is_significant.append(1 if stats['is_significant'] else 0)
                else:
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)
                    p_values.append(np.nan)
                    effect_sizes.append(np.nan)
                    is_significant.append(0)
            
            data.update({
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper,
                'p_value': p_values,
                'effect_size': effect_sizes,
                'is_significant': is_significant
            })
        
        # Add vector contribution analysis if available
        if hasattr(self, 'inter_contributions') and hasattr(self, 'intra_contributions'):
            inter_magnitudes = []
            intra_magnitudes = []
            combined_ratios = []
            
            for res_id in res_indices:
                if res_id in self.inter_contributions:
                    inter_mag = np.mean([np.linalg.norm(v) for v in self.inter_contributions[res_id]])
                    intra_mag = np.mean([np.linalg.norm(v) for v in self.intra_contributions[res_id]])
                    inter_magnitudes.append(inter_mag)
                    intra_magnitudes.append(intra_mag)
                    combined_ratios.append(inter_mag / (intra_mag + 1e-10))
                else:
                    inter_magnitudes.append(0.0)
                    intra_magnitudes.append(0.0)
                    combined_ratios.append(1.0)
            
            data.update({
                'inter_protein_flux': inter_magnitudes,
                'intra_protein_flux': intra_magnitudes,
                'inter_intra_ratio': combined_ratios
            })
        
        # Save main data
        flux_df = pd.DataFrame(data)
        flux_file = os.path.join(output_dir, 'processed_flux_data.csv')
        flux_df.to_csv(flux_file, index=False)
        print(f"   ✓ Saved processed data to: {flux_file}")
        
        # Save all iterations
        all_flux_df = pd.DataFrame(flux_data['all_flux']).T
        all_flux_df.columns = [f'iteration_{i}' for i in range(len(flux_data['all_flux']))]
        all_flux_df['residue_index'] = res_indices
        all_flux_df['residue_name'] = res_names
        all_flux_df['is_aromatic'] = is_aromatic
        
        all_flux_file = os.path.join(output_dir, 'all_iterations_flux.csv')
        all_flux_df.to_csv(all_flux_file, index=False)
        
        return flux_file, all_flux_file


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
    analyzer = TrajectoryFluxAnalyzer()
    
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
