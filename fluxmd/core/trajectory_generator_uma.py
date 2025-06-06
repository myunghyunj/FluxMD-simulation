#!/usr/bin/env python3
"""
Modified trajectory_generator.py section for UMA-optimized workflow.
This shows the key changes needed to integrate the in-memory GPU pipeline.
"""
import os
import numpy as np

# Add to imports at the top of trajectory_generator.py:
from ..gpu.gpu_accelerated_flux_uma import GPUAcceleratedInteractionCalculator, InteractionResult, get_device
from ..analysis.flux_analyzer_uma import TrajectoryFluxAnalyzer

# Replace the run_single_iteration method with this version:
def run_single_iteration_uma(self, iteration_num, protein_atoms_df, ligand_atoms_df, 
                             protein_com, ligand_com, ligand_radius, 
                             starting_distance, n_steps, n_approaches, n_rotations,
                             output_dir, gpu_calc=None):
    """
    Run a single iteration with UMA-optimized GPU processing.
    Returns raw GPU InteractionResult objects instead of writing files.
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num + 1}")
    print(f"{'='*60}")
    
    iteration_results = []
    
    for approach_num in range(n_approaches):
        print(f"\n--- Approach {approach_num + 1}/{n_approaches} ---")
        
        # Generate trajectory
        trajectory = self.generate_winding_trajectory(
            protein_com, ligand_radius, starting_distance,
            n_steps, self.config['approach_angles'][approach_num]
        )
        
        if len(trajectory) < 10:
            print("   ⚠️  Trajectory too short, skipping...")
            continue
        
        # Skip visualization in UMA mode for performance
        # (can be re-enabled if needed)
        
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
        else:
            print("   ⚠️  GPU calculator not provided, skipping GPU processing")
    
    return iteration_results


# Add this new method to run complete UMA-optimized analysis:
def run_complete_analysis_uma(self, protein_file, ligand_file, output_dir,
                             n_steps=200, n_iterations=10, n_approaches=10,
                             starting_distance=20.0, n_rotations=36,
                             use_gpu=True, physiological_pH=7.4):
    """
    Orchestrates the entire analysis with UMA-optimized GPU workflow.
    Keeps everything in GPU memory from start to finish.
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
    
    # Generate approach angles
    self.config['approach_angles'] = np.linspace(0, 2*np.pi, n_approaches, endpoint=False)
    
    # Store all iteration results in memory
    all_iteration_results = []
    
    # Main iteration loop
    for i in range(n_iterations):
        # Run single iteration with GPU processing
        iteration_results = self.run_single_iteration_uma(
            i, protein_atoms_df, ligand_atoms_df,
            protein_com, ligand_com, ligand_radius,
            starting_distance, n_steps, n_approaches, n_rotations,
            output_dir, gpu_calc
        )
        
        all_iteration_results.append(iteration_results)
        
        print(f"\n✓ Iteration {i+1} complete: {len(iteration_results)} trajectory frames processed")
        print(f"  Total interactions: {sum(len(r.energies) for r in iteration_results if r is not None)}")
    
    # Run flux analysis directly on GPU results
    print("\n" + "="*80)
    print("FLUX ANALYSIS - ZERO-COPY GPU PROCESSING")
    print("="*80)
    
    flux_analyzer = TrajectoryFluxAnalyzer(device=device)
    
    # Extract protein name from filename
    protein_name = os.path.splitext(os.path.basename(protein_file))[0]
    
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
                        device_type):
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
        'optimization': 'Zero-copy GPU processing with scatter operations'
    }
    
    param_file = os.path.join(output_dir, 'simulation_parameters_uma.txt')
    with open(param_file, 'w') as f:
        f.write("FluxMD UMA-Optimized Simulation Parameters\n")
        f.write("="*50 + "\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTimestamp: {pd.Timestamp.now()}\n")
    
    print(f"   Saved parameters to: {param_file}")
