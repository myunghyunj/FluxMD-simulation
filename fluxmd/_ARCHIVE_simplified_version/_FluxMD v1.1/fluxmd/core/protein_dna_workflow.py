"""
Core workflow for Protein-DNA interaction analysis using UMA pipeline.
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

# Import UMA-specific components
from ..gpu.gpu_accelerated_flux_uma import GPUAcceleratedInteractionCalculator, get_device, InteractionResult
from ..analysis.flux_analyzer_uma import TrajectoryFluxAnalyzer
from ..utils.pdb_parser import PDBParser
from .trajectory_generator import ProteinLigandFluxAnalyzer
from .trajectory_generator_uma import save_iteration_data_uma


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_protein_dna_workflow(dna_file: str, protein_file: str, output_dir: str, **kwargs):
    """
    Main function to run the Protein-DNA interaction workflow using UMA pipeline.
    - DNA is the static target (fixed at origin)
    - Protein is the mobile molecule with approach trajectories
    """
    print_banner("PROTEIN-DNA INTERACTION ANALYSIS (UMA)")
    
    # Set random seed if provided for reproducibility
    random_seed = kwargs.get('random_seed', None)
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")
    
    # 1. Parse input files
    parser = PDBParser()
    dna_atoms = parser.parse(dna_file, is_dna=True)
    protein_atoms = parser.parse(protein_file)
    
    if dna_atoms is None or protein_atoms is None:
        print("Error parsing input files. Aborting.")
        return
    
    print(f"Target DNA: {os.path.basename(dna_file)} ({len(dna_atoms)} atoms)")
    print(f"Mobile Protein: {os.path.basename(protein_file)} ({len(protein_atoms)} atoms)")
    
    # 2. Get device and check GPU status
    use_gpu = not kwargs.get('force_cpu', False)
    device = get_device() if use_gpu else torch.device('cpu')
    
    if device.type == 'mps':
        print("✓ Apple Silicon GPU detected - optimal for UMA processing")
    elif device.type == 'cuda':
        print("✓ NVIDIA GPU detected")
    else:
        print("! Running on CPU (slower)")
    
    # 3. Initialize GPU calculator with DNA support
    physiological_pH = kwargs.get('physiological_pH', 7.4)
    print(f"\nInitializing GPU calculator (pH={physiological_pH})...")
    
    gpu_calc = GPUAcceleratedInteractionCalculator(
        device=device,
        physiological_pH=physiological_pH,
        target_is_dna=True
    )
    
    # 4. Pre-compute properties with proper naming
    # DNA is the target, protein is mobile
    print("Pre-computing molecular properties...")
    gpu_calc.precompute_target_properties_gpu(dna_atoms)
    ligand_properties = gpu_calc.precompute_mobile_properties_gpu(protein_atoms)
    
    # 5. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Save parameters
    params_file = os.path.join(output_dir, 'simulation_parameters.txt')
    _save_dna_parameters(params_file, dna_file, protein_file, output_dir, **kwargs)
    
    # 7. Initialize trajectory generator for cocoon-style trajectories
    # For DNA workflow, we swap the roles: DNA is "protein" (target), protein is "ligand" (mobile)
    trajectory_generator = ProteinLigandFluxAnalyzer(
        protein_file=dna_file,      # DNA as target
        ligand_file=protein_file,   # Protein as mobile molecule
        output_dir=output_dir,
        target_is_dna=True
    )
    
    # Initialize collision detector with DNA structure
    dna_coords = dna_atoms[['x', 'y', 'z']].values
    trajectory_generator.collision_detector.build_protein_tree(dna_coords, dna_atoms)
    
    # 8. Run UMA simulation iterations
    n_iterations = kwargs.get('n_iterations', 10)
    n_approaches = kwargs.get('n_approaches', 10)
    n_steps = kwargs.get('n_steps', 200)
    n_rotations = kwargs.get('n_rotations', 36)
    starting_distance = kwargs.get('starting_distance', 20.0)
    approach_distance = kwargs.get('approach_distance', 2.5)
    
    all_iteration_results = []
    
    # Calculate centers of mass
    dna_com = dna_atoms[['x', 'y', 'z']].mean().values
    protein_com = protein_atoms[['x', 'y', 'z']].mean().values
    
    # Calculate protein properties for trajectory generation
    protein_coords = protein_atoms[['x', 'y', 'z']].values
    protein_radius = np.max(np.linalg.norm(protein_coords - protein_com, axis=1))
    
    # Calculate protein molecular weight (approximate)
    protein_mw = len(protein_atoms) * 110.0  # Average amino acid MW
    
    # Extract DNA backbone CA equivalent (phosphates) for trajectory generation
    dna_backbone = dna_atoms[dna_atoms['name'].isin(['P', 'C1\''])].copy()
    dna_ca_coords = dna_backbone[['x', 'y', 'z']].values if not dna_backbone.empty else dna_atoms[['x', 'y', 'z']].values[:100]
    
    print(f"\nRunning {n_iterations} iterations...")
    print(f"  Approaches per iteration: {n_approaches}")
    print(f"  Steps per approach: {n_steps}")
    print(f"  Rotations per position: {n_rotations}")
    
    for iteration_num in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration_num + 1} of {n_iterations}")
        print(f"{'='*60}")
        
        iteration_results = []
        
        # Generate approach angles for this iteration with randomness
        # Add random offset to make each iteration different
        angle_offset = np.random.uniform(0, 2*np.pi / n_approaches)
        approach_angles = np.linspace(0, 2*np.pi, n_approaches, endpoint=False) + angle_offset
        
        for approach_num in range(n_approaches):
            print(f"\n--- Approach {approach_num + 1}/{n_approaches} ---")
            
            # Calculate initial distance for this approach
            initial_distance = starting_distance - approach_num * approach_distance
            print(f"   Initial distance: {initial_distance:.1f} Å")
            
            # Generate cocoon-style trajectory using the same method as ligand simulations
            print(f"\n   Generating cocoon trajectory:")
            print(f"     Molecular weight: {protein_mw:.1f} Da")
            
            # Use trajectory generator's cocoon method
            trajectory, times = trajectory_generator.generate_cocoon_trajectory(
                dna_ca_coords,  # Target backbone (DNA)
                protein_coords,  # Mobile molecule (protein)  
                protein_atoms,   # Protein dataframe
                protein_mw,      # Molecular weight
                n_steps=n_steps,
                dt=40,           # timestep in femtoseconds
                target_distance=initial_distance
            )
            
            print(f"   Generated {len(trajectory)} trajectory points")
            
            # Save trajectory visualization using the standard cocoon visualization
            save_trajectories = kwargs.get('save_trajectories', True)
            if save_trajectories:
                print("   📸 Generating trajectory visualization...")
                iter_dir = os.path.join(output_dir, f'iteration_{iteration_num + 1}')
                os.makedirs(iter_dir, exist_ok=True)
                
                # Use the trajectory generator's visualization method
                # Pass iteration_num + 1 to match the 1-based directory naming
                trajectory_generator.visualize_trajectory_cocoon(
                    dna_atoms,  # Use DNA atoms as the "protein" for visualization
                    trajectory, 
                    iteration_num + 1,  # Convert to 1-based for filename consistency
                    approach_num, 
                    iter_dir
                )
            
            # Process trajectory on GPU
            if len(trajectory) > 0:
                print("   Processing trajectory on GPU...")
                
                # Process entire trajectory batch
                approach_results = gpu_calc.process_trajectory_batch(
                    trajectory, 
                    protein_coords,  # Mobile molecule coordinates
                    n_rotations, 
                    protein_atoms    # Pass full dataframe for property lookup
                )
                
                iteration_results.extend(approach_results)
                print(f"   ✓ Found {len(approach_results)} frames with interactions")
        
        # Save iteration data
        if iteration_results:
            save_iteration_data_uma(
                iteration_results, iteration_num, output_dir, n_approaches,
                dna_atoms, protein_atoms, gpu_calc, dna_file,
                target_is_dna=True
            )
        
        all_iteration_results.append(iteration_results)
        
        # Print iteration summary
        total_interactions = sum(len(r.energies) for r in iteration_results if r is not None)
        print(f"\nIteration {iteration_num + 1} summary: {total_interactions} total interactions")
    
    # 8. Perform final flux analysis
    print("\n" + "="*80)
    print("ANALYZING FLUX DATA")
    print("="*80)
    
    # Initialize flux analyzer
    flux_analyzer = TrajectoryFluxAnalyzer(device=device, target_is_dna=True)
    flux_analyzer.parse_target_for_analysis(dna_file)
    
    # Process all iterations to calculate flux
    if any(all_iteration_results):
        # For DNA, we don't have intra-protein vectors
        intra_vectors = torch.zeros((len(dna_atoms), 3), device=device, dtype=torch.float32)
        
        flux_data = flux_analyzer.process_iterations_and_calculate_flux(
            all_iteration_results, intra_vectors
        )
        
        # Generate visualizations
        dna_name = os.path.basename(dna_file).replace('.pdb', '')
        flux_analyzer.visualize_trajectory_flux(flux_data, dna_name, output_dir)
        
        # Save processed data
        _save_dna_flux_results(flux_data, flux_analyzer, output_dir)
        
        # Write final report
        _write_dna_report(output_dir, dna_name, flux_data, flux_analyzer)
    
    print("\n✓ Protein-DNA Interaction Workflow completed successfully!")
    print(f"Results saved to: {output_dir}")




def _save_dna_parameters(params_file, dna_file, protein_file, output_dir, **kwargs):
    """Save simulation parameters for DNA-protein analysis."""
    with open(params_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROTEIN-DNA INTERACTION ANALYSIS PARAMETERS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"DNA (target): {dna_file}\n")
        f.write(f"Protein (mobile): {protein_file}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Analysis mode: UMA-optimized (zero-copy GPU)\n\n")
        
        f.write("Simulation parameters:\n")
        f.write(f"  Number of iterations: {kwargs.get('n_iterations', 10)}\n")
        f.write(f"  Number of approaches: {kwargs.get('n_approaches', 10)}\n")
        f.write(f"  Steps per approach: {kwargs.get('n_steps', 200)}\n")
        f.write(f"  Starting distance: {kwargs.get('starting_distance', 20.0)} Å\n")
        f.write(f"  Rotations per position: {kwargs.get('n_rotations', 36)}\n")
        f.write(f"  Physiological pH: {kwargs.get('physiological_pH', 7.4)}\n")
        f.write(f"  Force CPU: {kwargs.get('force_cpu', False)}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def _save_dna_flux_results(flux_data, flux_analyzer, output_dir):
    """Save processed flux data for DNA nucleotides."""
    # Create DataFrame with DNA-specific columns
    results_df = pd.DataFrame({
        'nucleotide_position': flux_data['res_indices'],
        'base_type': [flux_analyzer.base_types[i] if i < len(flux_analyzer.base_types) else '?' 
                     for i in range(len(flux_data['res_indices']))],
        'residue_name': [flux_analyzer.residue_names[i] if i < len(flux_analyzer.residue_names) else '?' 
                        for i in range(len(flux_data['res_indices']))],
        'mean_flux': flux_data['avg_flux'],
        'std_flux': flux_data['std_flux'],
        'p_value': flux_data.get('p_values', [0.05] * len(flux_data['res_indices'])),
        'significant': flux_data['avg_flux'] > np.percentile(flux_data['avg_flux'][flux_data['avg_flux'] > 0], 75)
    })
    
    # Sort by flux value
    results_df = results_df.sort_values('mean_flux', ascending=False)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'processed_flux_data.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved flux results to: {output_file}")
    
    # Also save all iterations data
    all_flux_file = os.path.join(output_dir, 'all_iterations_flux.csv')
    np.savetxt(all_flux_file, flux_data['all_flux'], delimiter=',')


def _write_dna_report(output_dir, dna_name, flux_data, flux_analyzer):
    """Write analysis report for DNA-protein interactions."""
    report_file = os.path.join(output_dir, f'{dna_name}_flux_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROTEIN-DNA INTERACTION FLUX ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Target DNA: {dna_name}\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Find high-flux nucleotides
        high_flux_threshold = np.percentile(flux_data['avg_flux'][flux_data['avg_flux'] > 0], 75)
        high_flux_indices = np.where(flux_data['avg_flux'] > high_flux_threshold)[0]
        
        f.write("--- Top DNA Binding Sites (High-Flux Nucleotides) ---\n")
        for i, idx in enumerate(high_flux_indices[:20]):  # Top 20
            if idx < len(flux_analyzer.residue_indices):
                res_idx = flux_analyzer.residue_indices[idx]
                base = flux_analyzer.base_types[idx] if idx < len(flux_analyzer.base_types) else '?'
                flux_val = flux_data['avg_flux'][idx]
                f.write(f"{i+1}. Position {res_idx} ({base}): Flux = {flux_val:.4f}\n")
        
        # Sequence context
        f.write("\n--- Sequence Context of High-Flux Regions ---\n")
        if hasattr(flux_analyzer, 'base_types') and flux_analyzer.base_types:
            # Group consecutive high-flux positions
            high_flux_positions = sorted([flux_analyzer.residue_indices[i] for i in high_flux_indices 
                                        if i < len(flux_analyzer.residue_indices)])
            
            if high_flux_positions:
                # Find continuous regions
                regions = []
                current_region = [high_flux_positions[0]]
                
                for pos in high_flux_positions[1:]:
                    if pos == current_region[-1] + 1:
                        current_region.append(pos)
                    else:
                        regions.append(current_region)
                        current_region = [pos]
                regions.append(current_region)
                
                # Report regions
                for region in regions:
                    if len(region) >= 3:  # Report regions of 3+ nucleotides
                        start_idx = flux_analyzer.residue_indices.index(region[0])
                        end_idx = flux_analyzer.residue_indices.index(region[-1])
                        sequence = ''.join(flux_analyzer.base_types[start_idx:end_idx+1])
                        f.write(f"  Positions {region[0]}-{region[-1]}: {sequence}\n")
        
        f.write("\n--- Statistical Summary ---\n")
        f.write(f"Total nucleotides analyzed: {len(flux_data['res_indices'])}\n")
        f.write(f"Mean flux: {np.mean(flux_data['avg_flux']):.4f} ± {np.std(flux_data['avg_flux']):.4f}\n")
        f.write(f"Median flux: {np.median(flux_data['avg_flux']):.4f}\n")
        f.write(f"High-flux nucleotides (>75th percentile): {len(high_flux_indices)}\n")
        
        f.write("\n--- Analysis Method ---\n")
        f.write("UMA-optimized GPU processing (zero-copy architecture)\n")
        f.write("DNA fixed at origin, protein trajectory simulated\n")
        f.write("Flux metric: Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)\n")
    
    print(f"Saved analysis report to: {report_file}")


