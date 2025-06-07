#!/usr/bin/env python3
"""
Modified trajectory_generator.py section for UMA-optimized workflow.
This shows the key changes needed to integrate the in-memory GPU pipeline.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Add to imports at the top of trajectory_generator.py:
from ..gpu.gpu_accelerated_flux_uma import GPUAcceleratedInteractionCalculator, InteractionResult, get_device
from ..analysis.flux_analyzer_uma import TrajectoryFluxAnalyzer

# Replace the run_single_iteration method with this version:
def run_single_iteration_uma(self, iteration_num, protein_atoms_df, ligand_atoms_df, 
                             protein_com, ligand_com, ligand_radius, 
                             starting_distance, n_steps, n_approaches, n_rotations,
                             output_dir, gpu_calc=None, approach_angles=None,
                             ca_coords=None, ligand_mw=None, approach_distance=2.5,
                             save_trajectories=False):
    """
    Run a single iteration with UMA-optimized GPU processing.
    Returns raw GPU InteractionResult objects instead of writing files.
    
    Args:
        save_trajectories: If True, generate and save trajectory visualizations
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num + 1}")
    print(f"{'='*60}")
    
    iteration_results = []
    
    for approach_num in range(n_approaches):
        print(f"\n--- Approach {approach_num + 1}/{n_approaches} ---")
        
        # Calculate initial distance for this approach
        initial_distance = starting_distance - approach_num * approach_distance
        print(f"   Initial distance: {initial_distance:.1f} Å")
        
        # Generate trajectory
        ligand_coords = ligand_atoms_df[['x', 'y', 'z']].values
        trajectory, times = self.generate_cocoon_trajectory(
            ca_coords, ligand_coords, ligand_atoms_df,
            ligand_mw, n_steps=n_steps, dt=40,
            target_distance=initial_distance
        )
        
        if len(trajectory) < 10:
            print("   ⚠️  Trajectory too short, skipping...")
            continue
        
        # Generate visualization if requested
        if save_trajectories:
            print("   📸 Generating trajectory visualization...")
            iter_dir = os.path.join(output_dir, f'iteration_{iteration_num}')
            os.makedirs(iter_dir, exist_ok=True)
            
            # Use the parent class visualization method if available
            if hasattr(self, 'visualize_trajectory_cocoon'):
                self.visualize_trajectory_cocoon(
                    protein_atoms_df, trajectory, iteration_num, 
                    approach_num, iter_dir
                )
        
        print(f"   Generated {len(trajectory)} trajectory points")
        
        # Process trajectory on GPU
        if gpu_calc is not None:
            print("   Processing trajectory on GPU (UMA-optimized)...")
            
            # Convert ligand coordinates to numpy array
            ligand_coords = ligand_atoms_df[['x', 'y', 'z']].values
            
            # Process entire trajectory batch on GPU
            approach_results = gpu_calc.process_trajectory_batch(
                trajectory, ligand_coords, n_rotations
            )
            
            # Extend iteration results
            iteration_results.extend(approach_results)
            
            print(f"   ✓ Found {len(approach_results)} frames with interactions")
            
            # Track interaction types if available
            if approach_results:
                total_interactions = sum(len(r.energies) for r in approach_results if r is not None)
                print(f"   📊 Total interactions in this approach: {total_interactions}")
        else:
            print("   ⚠️  GPU calculator not provided, skipping GPU processing")
    
    # Summary for iteration
    if iteration_results:
        total_frames = len(iteration_results)
        total_interactions = sum(len(r.energies) for r in iteration_results if r is not None)
        print(f"\n📈 Iteration {iteration_num + 1} summary:")
        print(f"   - Frames with interactions: {total_frames}")
        print(f"   - Total interactions detected: {total_interactions}")
        if total_frames > 0:
            print(f"   - Average interactions per frame: {total_interactions / total_frames:.1f}")
        
        # Save iteration data log
        iter_dir = os.path.join(output_dir, f'iteration_{iteration_num}')
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save iteration summary
        summary_file = os.path.join(iter_dir, 'iteration_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"ITERATION {iteration_num + 1} SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Frames with interactions: {total_frames}\n")
            f.write(f"Total interactions detected: {total_interactions}\n")
            if total_frames > 0:
                f.write(f"Average interactions per frame: {total_interactions / total_frames:.1f}\n")
            f.write("\n")
            
            # Energy statistics
            all_energies = []
            for result in iteration_results:
                if result is not None and len(result.energies) > 0:
                    all_energies.extend(result.energies.cpu().numpy())
            
            if all_energies:
                import numpy as np
                energies_array = np.array(all_energies)
                f.write("ENERGY STATISTICS:\n")
                f.write(f"  Mean: {np.mean(energies_array):.2f} kcal/mol\n")
                f.write(f"  Std: {np.std(energies_array):.2f} kcal/mol\n")
                f.write(f"  Min: {np.min(energies_array):.2f} kcal/mol\n")
                f.write(f"  Max: {np.max(energies_array):.2f} kcal/mol\n")
                
                
                # Calculate and save interaction type breakdown
                interaction_type_counts_file = {
                    InteractionResult.HBOND: 0,
                    InteractionResult.SALT_BRIDGE: 0,
                    InteractionResult.PI_PI: 0,
                    InteractionResult.PI_CATION: 0,
                    InteractionResult.VDW: 0
                }
                
                for result in iteration_results:
                    if result is not None and hasattr(result, 'interaction_types') and len(result.interaction_types) > 0:
                        types = result.interaction_types.cpu().numpy()
                        for itype in range(5):
                            interaction_type_counts_file[itype] += (types == itype).sum()
                
                total_typed_file = sum(interaction_type_counts_file.values())
                if total_typed_file > 0:
                    f.write("\nINTERACTION TYPE BREAKDOWN:\n")
                    for itype, count in interaction_type_counts_file.items():
                        percentage = (count / total_typed_file * 100)
                        f.write(f"  {InteractionResult.get_interaction_name(itype)}: {count} ({percentage:.1f}%)\n")
        
        # Save trajectory data as CSV for this iteration
        trajectory_data_file = os.path.join(iter_dir, 'trajectory_data.csv')
        trajectory_data = []
        for frame_idx, result in enumerate(iteration_results):
            if result is not None and len(result.energies) > 0:
                residue_ids = result.residue_ids.cpu().numpy()
                energies = result.energies.cpu().numpy()
                
                # Check if interaction types are available
                if hasattr(result, 'interaction_types') and len(result.interaction_types) > 0:
                    interaction_types = result.interaction_types.cpu().numpy()
                    for res_id, energy, itype in zip(residue_ids, energies, interaction_types):
                        trajectory_data.append({
                            'frame': frame_idx,
                            'residue_id': res_id,
                            'energy': energy,
                            'interaction_type': InteractionResult.get_interaction_name(itype)
                        })
                else:
                    # Fallback for backward compatibility
                    for res_id, energy in zip(residue_ids, energies):
                        trajectory_data.append({
                            'frame': frame_idx,
                            'residue_id': res_id,
                            'energy': energy
                        })
        
        if trajectory_data:
            import pandas as pd
            df = pd.DataFrame(trajectory_data)
            df.to_csv(trajectory_data_file, index=False)
            print(f"   💾 Saved trajectory data: {trajectory_data_file}")
        
        print(f"   📄 Saved iteration summary: {summary_file}")
    
    return iteration_results


# Add this new method to run complete UMA-optimized analysis:
def run_complete_analysis_uma(self, protein_file, ligand_file, output_dir,
                             n_steps=200, n_iterations=10, n_approaches=10,
                             starting_distance=20.0, n_rotations=36,
                             use_gpu=True, physiological_pH=7.4,
                             save_trajectories=False, approach_distance=2.5):
    """
    Orchestrates the entire analysis with UMA-optimized GPU workflow.
    Keeps everything in GPU memory from start to finish.
    
    Args:
        save_trajectories: If True, generate and save trajectory visualizations
        approach_distance: Distance step between approaches in Angstroms (default: 2.5)
    """
    print("\n" + "="*80)
    print("FLUXMD ANALYSIS - UNIFIED MEMORY ARCHITECTURE (UMA) OPTIMIZED")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse structures
    print("\n📁 Loading structures...")
    protein_atoms_df = self.parse_structure_robust(protein_file, parse_heterogens=False)
    ligand_atoms_df = self.parse_structure_robust(ligand_file, parse_heterogens=True)
    
    # Import required modules
    import numpy as np
    import torch
    import pandas as pd
    
    # Get device
    device = get_device() if use_gpu else torch.device('cpu')
    
    # Calculate centers of mass
    protein_com = protein_atoms_df[['x', 'y', 'z']].mean().values
    ligand_com = ligand_atoms_df[['x', 'y', 'z']].mean().values
    ligand_coords = ligand_atoms_df[['x', 'y', 'z']].values
    ligand_radius = np.max(np.linalg.norm(ligand_coords - ligand_com, axis=1))
    
    # Pre-compute intra-protein forces
    print("\n📊 Calculating intra-protein force field (one-time)...")
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('protein', protein_file)
    
    # Initialize intra-protein calculator
    from .intra_protein_interactions import IntraProteinInteractions
    self.intra_protein_calc = IntraProteinInteractions(structure, physiological_pH=physiological_pH)
    intra_protein_vectors = self.intra_protein_calc.calculate_all_interactions()
    print("   ✓ Static force field computed")
    
    # Initialize GPU calculator
    print("\n🚀 Initializing GPU calculator...")
    gpu_calc = GPUAcceleratedInteractionCalculator(device=device, physiological_pH=physiological_pH)
    
    # Pre-compute protein properties on GPU
    gpu_calc.precompute_protein_properties(protein_atoms_df)
    
    # Set intra-protein vectors
    gpu_calc.set_intra_protein_vectors(intra_protein_vectors)
    
    # Extract CA coordinates for cocoon trajectory
    ca_coords = self.extract_ca_backbone(protein_atoms_df)
    print(f"   Extracted {len(ca_coords)} CA atoms")
    
    # Build collision detection tree
    protein_coords = protein_atoms_df[['x', 'y', 'z']].values
    self.collision_detector.build_protein_tree(protein_coords, protein_atoms_df)
    print(f"   ✓ Built collision detection tree with {len(protein_coords)} atoms")
    
    # Calculate molecular weight
    ligand_mw = self.calculate_molecular_weight(ligand_atoms_df)
    print(f"   Ligand molecular weight: {ligand_mw:.1f} Da")
    
    # Generate approach angles
    approach_angles = np.linspace(0, 2*np.pi, n_approaches, endpoint=False)
    
    # Store all iteration results in memory
    all_iteration_results = []
    
    # Main iteration loop
    for i in range(n_iterations):
        # Run single iteration with GPU processing
        iteration_results = run_single_iteration_uma(
            self, i, protein_atoms_df, ligand_atoms_df,
            protein_com, ligand_com, ligand_radius,
            starting_distance, n_steps, n_approaches, n_rotations,
            output_dir, gpu_calc, approach_angles,
            ca_coords=ca_coords, ligand_mw=ligand_mw,
            save_trajectories=save_trajectories,
            approach_distance=approach_distance
        )
        
        all_iteration_results.append(iteration_results)
        
        print(f"\n✓ Iteration {i+1} complete: {len(iteration_results)} trajectory frames processed")
        total_interactions = sum(len(r.energies) for r in iteration_results if r is not None)
        print(f"  Total interactions: {total_interactions}")
        
        # Track interaction statistics
        if iteration_results and total_interactions > 0:
            # Collect energy statistics
            all_energies = []
            for result in iteration_results:
                if result is not None and len(result.energies) > 0:
                    all_energies.extend(result.energies.cpu().numpy())
            
            if all_energies:
                energies_array = np.array(all_energies)
                print(f"  Energy statistics:")
                print(f"    - Mean: {np.mean(energies_array):.2f} kcal/mol")
                print(f"    - Std: {np.std(energies_array):.2f} kcal/mol")
                print(f"    - Min: {np.min(energies_array):.2f} kcal/mol")
                print(f"    - Max: {np.max(energies_array):.2f} kcal/mol")
                
            # Add detailed interaction type analysis
            if iteration_results and total_interactions > 0:
                # Collect interaction types
                interaction_type_counts = {
                    InteractionResult.HBOND: 0,
                    InteractionResult.SALT_BRIDGE: 0,
                    InteractionResult.PI_PI: 0,
                    InteractionResult.PI_CATION: 0,
                    InteractionResult.VDW: 0
                }
                
                for result in iteration_results:
                    if result is not None and hasattr(result, 'interaction_types') and len(result.interaction_types) > 0:
                        types = result.interaction_types.cpu().numpy()
                        for itype in range(5):  # 5 interaction types
                            interaction_type_counts[itype] += (types == itype).sum()
                
                print(f"  Interaction type breakdown:")
                total_typed = sum(interaction_type_counts.values())
                if total_typed > 0:
                    for itype, count in interaction_type_counts.items():
                        percentage = (count / total_typed * 100)
                        print(f"    - {InteractionResult.get_interaction_name(itype)}: {count} ({percentage:.1f}%)")
                
            # Generate per-iteration flux visualization
            if save_trajectories:
                print(f"  📊 Generating iteration {i+1} flux visualization...")
                iter_dir = os.path.join(output_dir, f'iteration_{i}')
                
                # Calculate flux for this iteration only
                # TrajectoryFluxAnalyzer is already imported at the top
                iter_flux_analyzer = TrajectoryFluxAnalyzer(device=device)
                
                # Process single iteration data
                single_iter_results = [iteration_results]
                
                # Run simplified flux calculation for this iteration
                try:
                    iter_flux_analyzer.parse_protein_for_analysis(protein_file)
                    iter_flux_data = iter_flux_analyzer.process_iterations_and_calculate_flux(
                        single_iter_results,
                        gpu_calc.intra_protein_vectors_gpu
                    )
                    
                    # Generate flux heatmap for this iteration
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 6))
                    
                    residue_indices = iter_flux_data['res_indices']
                    avg_flux = iter_flux_data['avg_flux']
                    
                    plt.bar(residue_indices, avg_flux, color='steelblue', alpha=0.8)
                    plt.xlabel('Residue Index')
                    plt.ylabel('Flux (normalized)')
                    plt.title(f'Flux Analysis - Iteration {i+1}')
                    plt.grid(True, alpha=0.3)
                    
                    # Save figure
                    flux_fig_path = os.path.join(iter_dir, f'iteration_{i+1}_flux.png')
                    plt.savefig(flux_fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  📸 Saved flux visualization: {flux_fig_path}")
                    
                    # Save flux data for this iteration
                    flux_csv_path = os.path.join(iter_dir, f'iteration_{i+1}_flux_data.csv')
                    flux_df = pd.DataFrame({
                        'residue_index': residue_indices,
                        'residue_name': iter_flux_data['res_names'],
                        'avg_flux': avg_flux,
                        'std_flux': iter_flux_data['std_flux']
                    })
                    flux_df.to_csv(flux_csv_path, index=False)
                    print(f"  💾 Saved flux data: {flux_csv_path}")
                    
                except Exception as e:
                    print(f"  ⚠️  Could not generate iteration flux visualization: {e}")
    
    # Run flux analysis directly on GPU results
    print("\n" + "="*80)
    print("FLUX ANALYSIS - ZERO-COPY GPU PROCESSING")
    print("="*80)
    
    # Import here to ensure it is available in the monkey-patched context
    from ..analysis.flux_analyzer_uma import TrajectoryFluxAnalyzer
    flux_analyzer = TrajectoryFluxAnalyzer(device=device)
    
    # Extract protein name from filename
    protein_name = os.path.splitext(os.path.basename(protein_file))[0]
    
    # Analyze interactions before flux analysis
    print("\n" + "="*80)
    print("INTERACTION ANALYSIS")
    print("="*80)
    
    # Aggregate interaction data
    total_frames = sum(len(iter_results) for iter_results in all_iteration_results)
    total_interactions = sum(
        sum(len(r.energies) for r in iter_results if r is not None) 
        for iter_results in all_iteration_results
    )
    
    print(f"\nOverall statistics:")
    print(f"  - Total iterations: {n_iterations}")
    print(f"  - Total frames analyzed: {total_frames}")
    print(f"  - Total interactions detected: {total_interactions}")
    if total_frames > 0:
        print(f"  - Average interactions per frame: {total_interactions / total_frames:.1f}")
    
    # Categorize interactions by energy ranges
    interaction_categories = {
        'Strong (< -5 kcal/mol)': 0,
        'Medium (-5 to -1 kcal/mol)': 0,
        'Weak (-1 to 0 kcal/mol)': 0,
        'Repulsive (> 0 kcal/mol)': 0
    }
    
    for iter_results in all_iteration_results:
        for result in iter_results:
            if result is not None and len(result.energies) > 0:
                energies = result.energies.cpu().numpy()
                for energy in energies:
                    if energy < -5:
                        interaction_categories['Strong (< -5 kcal/mol)'] += 1
                    elif energy < -1:
                        interaction_categories['Medium (-5 to -1 kcal/mol)'] += 1
                    elif energy < 0:
                        interaction_categories['Weak (-1 to 0 kcal/mol)'] += 1
                    else:
                        interaction_categories['Repulsive (> 0 kcal/mol)'] += 1
    
    print("\nInteraction strength distribution:")
    for category, count in interaction_categories.items():
        percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
        print(f"  - {category}: {count} ({percentage:.1f}%)")
    
    # Add detailed interaction type analysis for all iterations
    print("\nDetailed interaction type breakdown:")
    overall_interaction_types = {
        InteractionResult.HBOND: 0,
        InteractionResult.SALT_BRIDGE: 0,
        InteractionResult.PI_PI: 0,
        InteractionResult.PI_CATION: 0,
        InteractionResult.VDW: 0
    }
    
    # Collect per-residue interaction data
    residue_interaction_types = {}  # residue_id -> {type: count}
    
    for iter_results in all_iteration_results:
        for result in iter_results:
            if result is not None and hasattr(result, 'interaction_types') and len(result.interaction_types) > 0:
                types = result.interaction_types.cpu().numpy()
                residue_ids = result.residue_ids.cpu().numpy()
                
                # Count overall types
                for itype in range(5):
                    overall_interaction_types[itype] += (types == itype).sum()
                
                # Track per-residue types
                for res_id, itype in zip(residue_ids, types):
                    if res_id not in residue_interaction_types:
                        residue_interaction_types[res_id] = {i: 0 for i in range(5)}
                    residue_interaction_types[res_id][itype] += 1
    
    # Display overall interaction types
    total_typed = sum(overall_interaction_types.values())
    if total_typed > 0:
        for itype, count in overall_interaction_types.items():
            percentage = (count / total_typed * 100)
            print(f"  - {InteractionResult.get_interaction_name(itype)}: {count} ({percentage:.1f}%)")
    
    # Find top residues by interaction type
    print("\nTop residues by interaction type:")
    
    # For each interaction type, find top 3 residues
    for itype in range(5):
        type_name = InteractionResult.get_interaction_name(itype)
        print(f"\n  {type_name}:")
        
        # Get residues with this interaction type
        residue_counts = [(res_id, counts[itype]) 
                         for res_id, counts in residue_interaction_types.items() 
                         if counts[itype] > 0]
        residue_counts.sort(key=lambda x: x[1], reverse=True)
        
        if residue_counts:
            for i, (res_id, count) in enumerate(residue_counts[:3]):
                print(f"    {i+1}. Residue {res_id}: {count} interactions")
        else:
            print(f"    No {type_name} interactions detected")
    
    # Run analysis pipeline with GPU data
    flux_data = flux_analyzer.run_analysis_pipeline(
        all_iteration_results,
        gpu_calc.intra_protein_vectors_gpu,  # Pass GPU tensor directly
        protein_file,
        protein_name,
        output_dir
    )
    
    # Save parameters
    self._save_parameters_uma(
        output_dir, protein_file, ligand_file,
        n_steps, n_iterations, n_approaches,
        starting_distance, n_rotations, physiological_pH,
        device.type, save_trajectories, approach_distance
    )
    
    print("\n" + "="*80)
    print("✅ UMA-OPTIMIZED ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)
    
    return flux_data


def _save_parameters_uma(self, output_dir, protein_file, ligand_file,
                        n_steps, n_iterations, n_approaches,
                        starting_distance, n_rotations, physiological_pH,
                        device_type, save_trajectories=False, approach_distance=2.5):
    """Save simulation parameters for UMA run in standard format."""
    import pandas as pd
    import os
    from datetime import datetime
    
    # Get protein name from filename
    protein_name = os.path.splitext(os.path.basename(protein_file))[0]
    
    param_file = os.path.join(output_dir, 'simulation_parameters.txt')
    with open(param_file, 'w') as f:
        f.write("FLUXMD SIMULATION PARAMETERS\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("INPUT FILES\n")
        f.write("-"*40 + "\n")
        f.write(f"Protein: {os.path.abspath(protein_file)}\n")
        f.write(f"Ligand: {os.path.abspath(ligand_file)}\n")
        f.write(f"Protein name: {protein_name}\n")
        f.write("\n")
        f.write("TRAJECTORY PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write(f"Mode: UMA-OPTIMIZED WINDING TRAJECTORY\n")
        f.write(f"Steps per approach: {n_steps}\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Number of approaches: {n_approaches}\n")
        f.write(f"Approach distance: {approach_distance} Angstroms\n")
        f.write(f"Starting distance: {starting_distance} Angstroms\n")
        f.write(f"Distance range: ~5-{starting_distance * 2.5:.0f} Angstroms (free variation)\n")
        f.write(f"Rotations per position: {n_rotations}\n")
        f.write(f"Total steps per iteration: {n_steps * n_approaches}\n")
        f.write(f"Total rotations sampled: {n_steps * n_approaches * n_rotations}\n")
        f.write("\n")
        f.write("CALCULATION PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write(f"pH: {physiological_pH}\n")
        f.write(f"GPU acceleration: {'ENABLED' if device_type != 'cpu' else 'DISABLED'}\n")
        if device_type != 'cpu':
            f.write(f"GPU device: {device_type}\n")
        f.write(f"Optimization: Zero-copy GPU processing (UMA)\n")
        f.write(f"Trajectory visualization: {'ENABLED' if save_trajectories else 'DISABLED'}\n")
        f.write("\n")
        f.write("OUTPUT DIRECTORY\n")
        f.write("-"*40 + "\n")
        f.write(f"{os.path.abspath(output_dir)}\n")
    
    print(f"   Saved parameters to: {param_file}")
