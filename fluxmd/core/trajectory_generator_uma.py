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
    
    return iteration_results


# Add this new method to run complete UMA-optimized analysis:
def run_complete_analysis_uma(self, protein_file, ligand_file, output_dir,
                             n_steps=200, n_iterations=10, n_approaches=10,
                             starting_distance=20.0, n_rotations=36,
                             use_gpu=True, physiological_pH=7.4,
                             save_trajectories=False):
    """
    Orchestrates the entire analysis with UMA-optimized GPU workflow.
    Keeps everything in GPU memory from start to finish.
    
    Args:
        save_trajectories: If True, generate and save trajectory visualizations
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
            save_trajectories=save_trajectories
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
    
    # Run flux analysis directly on GPU results
    print("\n" + "="*80)
    print("FLUX ANALYSIS - ZERO-COPY GPU PROCESSING")
    print("="*80)
    
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
        device.type
    )
    
    print("\n" + "="*80)
    print("✅ UMA-OPTIMIZED ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)
    
    return flux_data


def _save_parameters_uma(self, output_dir, protein_file, ligand_file,
                        n_steps, n_iterations, n_approaches,
                        starting_distance, n_rotations, physiological_pH,
                        device_type, save_trajectories=False):
    """Save simulation parameters for UMA run."""
    import pandas as pd
    import os
    
    params = {
        'workflow': 'UMA-optimized (Unified Memory Architecture)',
        'protein_file': protein_file,
        'ligand_file': ligand_file,
        'n_steps': n_steps,
        'n_iterations': n_iterations,
        'n_approaches': n_approaches,
        'starting_distance': starting_distance,
        'n_rotations': n_rotations,
        'physiological_pH': physiological_pH,
        'device': device_type,
        'optimization': 'Zero-copy GPU processing with scatter operations',
        'trajectory_visualization': 'Enabled' if save_trajectories else 'Disabled'
    }
    
    param_file = os.path.join(output_dir, 'simulation_parameters_uma.txt')
    with open(param_file, 'w') as f:
        f.write("FluxMD UMA-Optimized Simulation Parameters\n")
        f.write("="*50 + "\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTimestamp: {pd.Timestamp.now()}\n")
    
    print(f"   Saved parameters to: {param_file}")
