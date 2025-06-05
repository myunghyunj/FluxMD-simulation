"""
FluxMD: GPU-accelerated binding site prediction using flux differential analysis
"""

import os
import sys
import multiprocessing as mp
import platform
import subprocess
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from trajectory_generator import ProteinLigandFluxAnalyzer
from flux_analyzer import TrajectoryFluxAnalyzer
from gpu_accelerated_flux import get_device
from visualize_multiflux import visualize_multiflux


def benchmark_performance(protein_atoms, ligand_atoms, n_test_frames=5, n_test_rotations=12):
    """
    Run quick benchmark to determine actual GPU vs CPU performance
    Returns: (use_gpu, reason)
    """
    import time
    from gpu_accelerated_flux import GPUAcceleratedInteractionCalculator
    
    try:
        device = get_device()
        if 'cpu' in str(device):
            return False, "no GPU available"
    except:
        return False, "GPU initialization failed"
    
    print("\n⏱️  Running performance benchmark...")
    
    # Generate test trajectory
    test_positions = np.random.randn(n_test_frames, 3) * 20
    
    # Test GPU performance
    try:
        gpu_calc = GPUAcceleratedInteractionCalculator(device=device)
        gpu_calc.precompute_protein_properties_gpu(protein_atoms)
        gpu_calc.precompute_ligand_properties_gpu(ligand_atoms)
        
        ligand_coords = ligand_atoms[['x', 'y', 'z']].values
        
        gpu_start = time.time()
        gpu_results = gpu_calc.process_trajectory_batch_gpu(
            test_positions, ligand_coords, n_rotations=n_test_rotations
        )
        gpu_time = time.time() - gpu_start
        gpu_fps = n_test_frames / gpu_time
        
        print(f"  GPU: {gpu_fps:.1f} frames/sec")
    except Exception as e:
        print(f"  GPU benchmark failed: {e}")
        return False, "GPU benchmark failed"
    
    # Test CPU performance (simplified estimation)
    import multiprocessing as mp
    n_cores = mp.cpu_count()
    cpu_time_per_frame = (n_test_rotations * 0.01) / n_cores
    cpu_time = cpu_time_per_frame * n_test_frames
    cpu_fps = n_test_frames / cpu_time
    
    print(f"  CPU: {cpu_fps:.1f} frames/sec (estimated with {n_cores} cores)")
    
    # Decision
    if gpu_fps > cpu_fps * 1.2:  # GPU needs to be 20% faster to justify overhead
        return True, f"GPU {gpu_fps/cpu_fps:.1f}x faster in benchmark"
    else:
        return False, f"CPU more efficient ({cpu_fps/gpu_fps:.1f}x) for this workload"


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def convert_cif_to_pdb(cif_file):
    """Convert CIF/mmCIF to PDB format using OpenBabel"""
    pdb_file = cif_file.rsplit('.', 1)[0] + '.pdb'
    
    print(f"Converting {cif_file} to PDB format...")
    try:
        subprocess.run(['obabel', cif_file, '-O', pdb_file],
                      check=True, capture_output=True, text=True)
        print(f"✓ Converted to: {pdb_file}")
        return pdb_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: OpenBabel not found. Please install it first.")
        print("Install with: conda install -c conda-forge openbabel")
        return None


def convert_smiles_to_pdbqt(smiles_string, output_name="ligand"):
    """Convert SMILES string to PDBQT format using OpenBabel"""
    smi_file = f"{output_name}.smi"
    mol2_file = f"{output_name}.mol2"
    pdbqt_file = f"{output_name}.pdbqt"
    
    try:
        # Write SMILES to file
        with open(smi_file, 'w') as f:
            f.write(smiles_string)
        
        print(f"Converting SMILES to 3D structure...")
        
        # Convert SMILES to MOL2 with 3D coordinates
        cmd1 = ['obabel', '-ismi', smi_file, '-omol2', '-O', mol2_file,
                '--gen3d', '-p', '7.4', '-h']
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Convert MOL2 to PDBQT
        cmd2 = ['obabel', mol2_file, '-O', pdbqt_file]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        # Clean up intermediate files
        os.remove(smi_file)
        os.remove(mol2_file)
        
        print(f"✓ Created: {pdbqt_file}")
        return pdbqt_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr}")
        for f in [smi_file, mol2_file]:
            if os.path.exists(f):
                os.remove(f)
        return None
    except FileNotFoundError:
        print("Error: OpenBabel not found.")
        return None


def parse_simulation_parameters(params_file):
    """Parse simulation parameters from existing file"""
    params = {}
    
    try:
        with open(params_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if ': ' in line:
                key, value = line.split(': ', 1)
                
                # Extract trajectory parameters
                if key == "Steps per approach":
                    params['n_steps'] = int(value)
                elif key == "Number of iterations":
                    params['n_iterations'] = int(value)
                elif key == "Number of approaches":
                    params['n_approaches'] = int(value)
                elif key == "Approach distance":
                    params['approach_distance'] = float(value.replace(' Å', ''))
                elif key == "Starting distance":
                    params['starting_distance'] = float(value.replace(' Å', ''))
                elif key == "Rotations per position":
                    params['n_rotations'] = int(value)
                elif key == "pH":
                    params['physiological_pH'] = float(value)
        
        return params
    
    except Exception as e:
        print(f"Error parsing parameters file: {e}")
        return None


def validate_results(output_dir, protein_name):
    """Validate that results contain expected improvements"""
    print_banner("🔍 VALIDATING RESULTS")
    
    validation_passed = True
    
    # Check for processed flux data
    flux_file = os.path.join(output_dir, 'processed_flux_data.csv')
    if os.path.exists(flux_file):
        df = pd.read_csv(flux_file)
        
        # Check for pi-stacking contribution
        if 'is_aromatic' in df.columns:
            aromatic_residues = df[df['is_aromatic'] == 1]
            if len(aromatic_residues) > 0:
                aromatic_flux = aromatic_residues['average_flux'].mean()
                total_flux = df['average_flux'].mean()
                
                print(f"✓ Found {len(aromatic_residues)} aromatic residues")
                print(f"  Average flux: {aromatic_flux:.3f} (vs overall {total_flux:.3f})")
                
                if aromatic_flux > total_flux:
                    print("  ✓ Aromatic residues show higher flux")
                else:
                    print("  ⚠️  Aromatic residues don't show enhanced flux")
            else:
                print("⚠️  No aromatic residues found")
        
        # Check for statistical validation
        if 'p_value' in df.columns:
            significant = df[df['p_value'] < 0.05]
            print(f"\n✓ Statistical validation present")
            print(f"  {len(significant)}/{len(df)} residues significant (p<0.05)")
        else:
            print("\n⚠️  No statistical validation found")
            validation_passed = False
    else:
        print("❌ No flux data file found!")
        validation_passed = False
    
    # Check iteration files for pi-stacking
    import glob
    csv_files = glob.glob(os.path.join(output_dir, "iteration_*/*_output_vectors.csv"))
    
    if csv_files:
        pi_stacking_found = False
        total_interactions = 0
        pi_interactions = 0
        
        for csv_file in csv_files[:3]:  # Check first 3 files
            df = pd.read_csv(csv_file)
            if 'bond_type' in df.columns:
                total_interactions += len(df)
                pi_mask = df['bond_type'].str.contains('Pi-Stacking', na=False)
                pi_count = pi_mask.sum()
                pi_interactions += pi_count
                
                if pi_count > 0:
                    pi_stacking_found = True
        
        if pi_stacking_found:
            pi_percentage = (pi_interactions / total_interactions) * 100
            print(f"\n✓ Pi-stacking interactions found!")
            print(f"  {pi_interactions}/{total_interactions} ({pi_percentage:.1f}%)")
        else:
            print("\n⚠️  No pi-stacking interactions detected")
            print("  This might be normal if ligand lacks aromatic rings")
    
    return validation_passed


def run_complete_workflow():
    """Run the complete analysis workflow"""
    print_banner("FLUXMD - COCOON TRAJECTORY ANALYSIS")
    
    print("This workflow will:")
    print("1. Calculate static intra-protein force field (one-time)")
    print("2. Generate COCOON trajectories (constant distance hovering)")
    print("3. Sample multiple ligand orientations at each position")
    print("4. Calculate non-covalent interactions with combined forces")
    print("5. Compute energy flux differentials (합벡터 analysis)")
    print("6. Identify binding sites with statistical validation")
    print("7. Create visualizations and reports\n")
    
    # Step 1: Get input files
    print("STEP 1: INPUT FILES")
    print("-" * 40)
    
    protein_file = input("Enter protein file (PDB/CIF/mmCIF): ").strip()
    if not os.path.exists(protein_file):
        print(f"Error: {protein_file} not found!")
        return
    
    # Convert CIF/mmCIF to PDB if needed
    if protein_file.lower().endswith(('.cif', '.mmcif')):
        converted_file = convert_cif_to_pdb(protein_file)
        if converted_file is None:
            return
        protein_file = converted_file
    
    ligand_file = input("Enter ligand file (PDBQT/PDB) or SMILES: ").strip()
    
    # Check if input is SMILES
    if not os.path.exists(ligand_file) and not ligand_file.endswith(('.pdbqt', '.pdb')):
        print("Detected SMILES input...")
        ligand_name = input("Enter ligand name: ").strip() or "ligand"
        converted_file = convert_smiles_to_pdbqt(ligand_file, ligand_name)
        if converted_file is None:
            return
        ligand_file = converted_file
    elif not os.path.exists(ligand_file):
        print(f"Error: {ligand_file} not found!")
        return
    
    protein_name = input("Enter protein name for labeling: ").strip()
    
    # Step 2: Set parameters
    print("\nSTEP 2: PARAMETERS")
    print("-" * 40)
    
    # Ask if user wants to use existing parameters
    use_existing = input("\nLoad parameters from existing simulation? (y/n): ").strip().lower()
    
    if use_existing == 'y':
        params_file = input("Enter path to simulation_parameters.txt: ").strip()
        if os.path.exists(params_file):
            loaded_params = parse_simulation_parameters(params_file)
            if loaded_params:
                print("\n✓ Loaded parameters from file:")
                for key, value in loaded_params.items():
                    print(f"  {key}: {value}")
                
                confirm = input("\nUse these parameters? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Use loaded parameters
                    n_steps = loaded_params.get('n_steps', 100)
                    n_iterations = loaded_params.get('n_iterations', 100)
                    n_approaches = loaded_params.get('n_approaches', 5)
                    approach_distance = loaded_params.get('approach_distance', 2.5)
                    starting_distance = loaded_params.get('starting_distance', 15)
                    n_rotations = loaded_params.get('n_rotations', 36)
                    physiological_pH = loaded_params.get('physiological_pH', 7.4)
                else:
                    use_existing = 'n'  # Fall back to manual entry
            else:
                print("Failed to parse parameters file.")
                use_existing = 'n'
        else:
            print(f"File not found: {params_file}")
            use_existing = 'n'
    
    if use_existing != 'y':
        # Manual parameter entry
        print("\nEnter parameters manually (press Enter for defaults):")
        n_steps = int(input("Steps per approach (default 100): ") or "100")
        n_iterations = int(input("Number of iterations (default 100): ") or "100")
        n_approaches = int(input("Number of approaches (default 5): ") or "5")
        approach_distance = float(input("Approach distance in Å (default 2.5): ") or "2.5")
        starting_distance = float(input("Starting distance in Å (default 15): ") or "15")
        
        # Add pH parameter
        physiological_pH = float(input("pH for protonation state calculation (default 7.4): ") or "7.4")
        print(f"  Using pH {physiological_pH} for H-bond donor/acceptor assignment")

        # Cocoon trajectory parameters
        n_rotations = int(input("Rotations per position (default 36): ") or "36")
    
    output_dir = input("Output directory (default 'flux_analysis'): ").strip() or "flux_analysis"
    
    # Automatic GPU/CPU selection based on system size and benchmarking
    use_gpu = False
    gpu_available = False
    device = None
    
    try:
        device = get_device()
        if 'mps' in str(device) or 'cuda' in str(device):
            gpu_available = True
    except:
        gpu_available = False
    
    # Calculate system complexity
    # Parse structures temporarily to get atom counts
    from trajectory_generator import ProteinLigandFluxAnalyzer
    temp_analyzer = ProteinLigandFluxAnalyzer()
    try:
        protein_atoms = temp_analyzer.parse_structure(protein_file, parse_heterogens=False)
        ligand_atoms = temp_analyzer.parse_structure_robust(ligand_file, parse_heterogens=True)
        n_protein_atoms = len(protein_atoms)
        n_ligand_atoms = len(ligand_atoms)
    except:
        n_protein_atoms = 5000  # Default estimates
        n_ligand_atoms = 50
    
    # Calculate total operations
    frames_per_iteration = n_steps * n_approaches
    total_frames = frames_per_iteration * n_iterations
    operations_per_frame = n_protein_atoms * n_ligand_atoms * n_rotations
    total_operations = total_frames * operations_per_frame
    
    # New decision logic based on empirical performance
    if gpu_available:
        # Estimate GPU performance
        # GPU processes rotations in batches of 12
        rotation_batches = (n_rotations + 11) // 12
        
        # GPU has overhead but parallel rotation processing
        gpu_time_per_frame = 0.1 + (rotation_batches * 0.05)  # seconds
        
        # CPU performance with parallel processing
        import multiprocessing as mp
        n_cores = mp.cpu_count()
        cpu_time_per_frame = (n_rotations * 0.01) / n_cores  # seconds
        
        # Choose based on estimated performance
        if gpu_time_per_frame < cpu_time_per_frame:
            use_gpu = True
            decision_reason = f"GPU faster ({gpu_time_per_frame:.2f}s vs {cpu_time_per_frame:.2f}s per frame)"
        else:
            use_gpu = False
            decision_reason = f"CPU faster ({cpu_time_per_frame:.2f}s vs {gpu_time_per_frame:.2f}s per frame)"
        
        # Override for memory constraints
        gpu_memory_needed = n_protein_atoms * n_ligand_atoms * n_rotations * 4 * 8  # bytes (float32 + indices)
        if 'cuda' in str(device):
            import torch
            gpu_memory_available = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory_needed > gpu_memory_available * 0.8:
                use_gpu = False
                decision_reason = "insufficient GPU memory"
        elif 'mps' in str(device):
            # Apple Silicon has unified memory, but limit to 32GB workloads
            if gpu_memory_needed > 32 * 1024**3:
                use_gpu = False
                decision_reason = "workload too large for GPU"
    else:
        use_gpu = False
        decision_reason = "no GPU detected"
    
    # Report decision
    print(f"\n🔍 System Analysis:")
    print(f"  Protein atoms: {n_protein_atoms:,}")
    print(f"  Ligand atoms: {n_ligand_atoms:,}")
    print(f"  Total frames: {frames_per_iteration:,} per iteration")
    print(f"  Rotations per frame: {n_rotations}")
    print(f"  Total operations: {total_operations/1e6:.1f} million")
    
    if gpu_available:
        print(f"\n🚀 GPU detected: {device}")
    
    # Initial decision
    print(f"\n📊 Performance estimation:")
    if use_gpu:
        print(f"  Initial selection: GPU ({decision_reason})")
    else:
        print(f"  Initial selection: CPU ({decision_reason})")
    
    # Offer to run benchmark
    if gpu_available:
        run_benchmark = input("\nRun performance benchmark for optimal selection? (y/n): ").strip().lower()
        if run_benchmark == 'y':
            benchmark_use_gpu, benchmark_reason = benchmark_performance(
                protein_atoms, ligand_atoms, 
                n_test_frames=min(5, n_steps),
                n_test_rotations=n_rotations
            )
            use_gpu = benchmark_use_gpu
            decision_reason = f"benchmark result - {benchmark_reason}"
    
    # Final decision
    print(f"\n✅ Final decision:")
    if use_gpu:
        print(f"  Using GPU acceleration ({decision_reason})")
    else:
        print(f"  Using CPU parallel processing ({decision_reason})")
        if gpu_available:
            print("  Note: Decision based on performance characteristics")
    
    # Set parallel processing for CPU
    n_jobs = -1 if not use_gpu else 1  # Use all cores for CPU, single thread for GPU
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Mode: COCOON TRAJECTORY (constant distance hovering)")
    print(f"  Total steps: {n_steps * n_approaches} per iteration")
    print(f"  Starting: {starting_distance} Å from surface")
    print(f"  Final: ~{starting_distance - (n_approaches-1)*approach_distance:.1f} Å")
    print(f"  Rotations: {n_rotations} per position")
    print(f"  pH: {physiological_pH} (affects H-bond donors/acceptors)")
    print(f"  Processing: {'GPU' if use_gpu else 'CPU'} {'(parallel)' if n_jobs != 1 else ''}")
    
    # Performance estimate
    if use_gpu:
        estimated_time = (total_operations / 1e6) * 0.1  # Rough estimate: 0.1 sec per million on GPU
    else:
        cores = mp.cpu_count() if n_jobs == -1 else n_jobs
        estimated_time = (total_operations / 1e6) * 0.5 / cores  # 0.5 sec per million per core
    
    print(f"\n⏱️  Estimated processing time: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)")
    
    # Provide optimization suggestions if slow
    if estimated_time > 300:  # More than 5 minutes
        print("\n⚠️  Long processing time expected. Consider:")
        if n_rotations > 24:
            print(f"  • Reduce rotations to 12-24 (currently {n_rotations})")
        if n_steps > 50:
            print(f"  • Reduce steps to 50 (currently {n_steps})")
        if n_approaches > 3:
            print(f"  • Reduce approaches to 3 (currently {n_approaches})")
    
    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return
    
    # Save parameters to file
    os.makedirs(output_dir, exist_ok=True)
    params_file = os.path.join(output_dir, "simulation_parameters.txt")
    
    with open(params_file, 'w') as f:
        f.write("FLUXMD SIMULATION PARAMETERS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("INPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Protein: {protein_file}\n")
        f.write(f"Ligand: {ligand_file}\n")
        f.write(f"Protein name: {protein_name}\n")
        f.write("\n")
        f.write("TRAJECTORY PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mode: COCOON TRAJECTORY (constant distance hovering)\n")
        f.write(f"Steps per approach: {n_steps}\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Number of approaches: {n_approaches}\n")
        f.write(f"Approach distance: {approach_distance} Å\n")
        f.write(f"Starting distance: {starting_distance} Å\n")
        f.write(f"Final distance: ~{starting_distance - (n_approaches-1)*approach_distance:.1f} Å\n")
        f.write(f"Rotations per position: {n_rotations}\n")
        f.write(f"Total steps per iteration: {n_steps * n_approaches}\n")
        f.write(f"Total rotations sampled: {n_steps * n_approaches * n_rotations}\n")
        f.write("\n")
        f.write("CALCULATION PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"pH: {physiological_pH}\n")
        f.write(f"GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED'}\n")
        if use_gpu:
            f.write(f"GPU device: {device}\n")
        f.write(f"Parallel processing: {'ENABLED' if n_jobs != 1 else 'DISABLED'}\n")
        if n_jobs != 1:
            f.write(f"CPU cores: {mp.cpu_count()}\n")
        f.write("\n")
        f.write("OUTPUT DIRECTORY\n")
        f.write("-" * 40 + "\n")
        f.write(f"{os.path.abspath(output_dir)}\n")
    
    print(f"\n✓ Parameters saved to: {params_file}")
    
    # Step 3: Run trajectory analysis
    print_banner("STEP 3: COCOON TRAJECTORY GENERATION")
    
    trajectory_analyzer = ProteinLigandFluxAnalyzer(physiological_pH=physiological_pH)
    
    start_time = datetime.now()
    
    try:
        # Run trajectory analysis with cocoon mode
        iteration_data = trajectory_analyzer.run_complete_analysis(
            protein_file, ligand_file, output_dir, n_steps, n_iterations,
            n_approaches, approach_distance, starting_distance,
            n_jobs=n_jobs, use_gpu=use_gpu, n_rotations=n_rotations
        )
        
        if iteration_data is None:
            print("\n✗ Trajectory analysis was cancelled.")
            return
        
        print("\n✓ Trajectory analysis complete!")
        
        # Verify output
        import glob
        iter_dirs = glob.glob(os.path.join(output_dir, "iteration_*"))
        if len(iter_dirs) == 0:
            print("\n✗ No iteration directories found!")
            return
        
        print(f"   Found {len(iter_dirs)} iteration directories")
        
    except Exception as e:
        print(f"\nError in trajectory analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Run flux analysis
    print_banner("STEP 4: FLUX DIFFERENTIAL ANALYSIS")
    
    flux_analyzer = TrajectoryFluxAnalyzer()
    flux_analyzer.physiological_pH = physiological_pH  # Pass pH information
    
    try:
        # Process flux differentials
        flux_data = flux_analyzer.process_trajectory_iterations(output_dir, protein_file)
        
        # Create visualizations
        flux_analyzer.visualize_trajectory_flux(flux_data, protein_name, output_dir)
        
        # Generate report
        flux_analyzer.generate_summary_report(flux_data, protein_name, output_dir)
        
        # Save processed data
        flux_analyzer.save_processed_data(flux_data, output_dir)
        
        print("\n✓ Flux analysis complete!")
        
    except Exception as e:
        print(f"\nError in flux analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Validate results
    validate_results(output_dir, protein_name)
    
    # Step 6: Summary
    print_banner("ANALYSIS COMPLETE!")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"Total analysis time: {total_time:.1f} seconds")
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey outputs:")
    print("├── simulation_parameters.txt - All simulation parameters")
    print("├── trajectory_iteration_*_approach_*.png - Cocoon trajectories")
    print("├── trajectory_iteration_*_approach_*.csv - Trajectory coordinates")
    print("├── iteration_*/ - Interaction data with rotations")
    print("├── interactions_approach_*.csv - Detailed interactions")
    print("├── *_trajectory_flux_analysis.png - Flux visualization")
    print("├── *_flux_report.txt - Statistical analysis")
    print("├── processed_flux_data.csv - Flux with p-values")
    print("└── all_iterations_flux.csv - Raw flux data")
    
    print("\n🎯 INTERPRETATION:")
    print("• Red regions = High flux = Statistically significant binding sites")
    print("• Purple markers = Aromatic residues capable of π-stacking")
    print("• Error bars = 95% confidence intervals from bootstrap")
    print("• P-values indicate statistical significance of each residue")
    print("• Flux values now include both inter & intra-protein forces (합벡터)")
    print("• Higher flux = stronger combined force convergence at binding site")
    
    # Offer comparison
    another = input("\nAnalyze another ligand for comparison? (y/n): ").strip().lower()
    if another == 'y':
        print("\nTo compare ligands:")
        print("1. Run this workflow again with the new ligand")
        print("2. Use the same protein and output to a different directory")
        print("3. Compare the flux reports to identify different binding preferences")


def main():
    """Main entry point with menu"""
    print_banner("FLUXMD - PROTEIN-LIGAND FLUX ANALYSIS")
    
    print("Welcome to FluxMD - GPU-accelerated binding site prediction")
    print("\nOptions:")
    print("1. Run complete workflow")
    print("2. Convert SMILES to PDBQT")
    print("3. Visualize multiple proteins (compare flux)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_complete_workflow()
    elif choice == "2":
        print_banner("SMILES TO PDBQT CONVERTER")
        smiles = input("Enter SMILES string: ").strip()
        if smiles:
            name = input("Enter output name: ").strip() or "ligand"
            convert_smiles_to_pdbqt(smiles, name)
    elif choice == "3":
        print_banner("MULTI-PROTEIN FLUX COMPARISON")
        n_proteins = int(input("How many proteins to compare? "))
        
        protein_flux_pairs = []
        for i in range(n_proteins):
            print(f"\nProtein {i+1}:")
            pdb_file = input("  PDB file: ").strip()
            csv_file = input("  Flux CSV file: ").strip()
            label = input("  Label: ").strip()
            
            if os.path.exists(pdb_file) and os.path.exists(csv_file):
                protein_flux_pairs.append((pdb_file, csv_file, label))
            else:
                print(f"  Error: File not found, skipping this protein")
        
        if protein_flux_pairs:
            output_file = input("\nOutput filename (default: multiflux_comparison.png): ").strip()
            if not output_file:
                output_file = "multiflux_comparison.png"
            
            print(f"\nCreating visualization for {len(protein_flux_pairs)} proteins...")
            visualize_multiflux(protein_flux_pairs, output_file)
        else:
            print("No valid protein-flux pairs provided.")
    elif choice == "4":
        print("\nThank you for using FluxMD!")
    else:
        print("Invalid choice!")
        main()


if __name__ == "__main__":
    main()
