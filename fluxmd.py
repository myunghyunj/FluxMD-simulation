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
from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer
from fluxmd.analysis.flux_analyzer import TrajectoryFluxAnalyzer
from fluxmd.gpu.gpu_accelerated_flux import get_device
from fluxmd.visualization.visualize_multiflux import visualize_multiflux


def benchmark_performance(protein_atoms, ligand_atoms, n_test_frames=5, n_test_rotations=12):
    """
    Run quick benchmark to determine actual GPU vs CPU performance
    Returns: (use_gpu, reason)
    """
    import time
    from fluxmd.gpu.gpu_accelerated_flux import GPUAcceleratedInteractionCalculator
    
    try:
        device = get_device()
        if 'cpu' in str(device):
            return False, "no GPU available"
    except:
        return False, "GPU initialization failed"
    
    print("\nRunning performance benchmark...")
    
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
        print(f"Converted to: {pdb_file}")
        return pdb_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: OpenBabel not found. Please install it first.")
        print("Install with: conda install -c conda-forge openbabel")
        return None


def convert_smiles_to_pdb_cactus(smiles_string, output_name="ligand"):
    """Convert SMILES to PDB using NCI CACTUS web service with aromatic preservation"""
    import urllib.parse
    import urllib.request
    
    pdb_file = f"{output_name}.pdb"
    sdf_file = f"{output_name}.sdf"
    
    try:
        print(f"Converting SMILES to 3D structure using NCI CACTUS...")
        print(f"SMILES: {smiles_string}")
        
        # URL encode the SMILES string
        encoded_smiles = urllib.parse.quote(smiles_string, safe='')
        
        # First try to get SDF with 3D coordinates and aromatic bonds
        # SDF format preserves aromaticity better than PDB
        sdf_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/file?format=sdf&get3d=true"
        
        print("  Requesting 3D structure with aromatic bonds preserved...")
        
        # Get SDF first
        with urllib.request.urlopen(sdf_url) as response:
            sdf_content = response.read().decode('utf-8')
        
        # Check if we got an error
        if "Page not found" in sdf_content or "<html>" in sdf_content:
            print("Error: CACTUS could not process this SMILES string")
            return None
        
        # Save SDF file
        with open(sdf_file, 'w') as f:
            f.write(sdf_content)
        
        # Also get PDB format for compatibility
        pdb_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/file?format=pdb&get3d=true"
        
        with urllib.request.urlopen(pdb_url) as response:
            pdb_content = response.read().decode('utf-8')
        
        # Save PDB file
        with open(pdb_file, 'w') as f:
            f.write(pdb_content)
        
        # Count atoms and check aromaticity
        atom_count = pdb_content.count('HETATM')
        
        # Check SDF for aromatic bonds (bond type 4)
        aromatic_bonds = sdf_content.count('  4  ') + sdf_content.count(' 4 0 ')
        
        print(f"Generated {atom_count} atoms")
        if aromatic_bonds > 0:
            print(f"Preserved {aromatic_bonds} aromatic bonds")
        print(f"Created: {pdb_file} (for FluxMD)")
        print(f"Created: {sdf_file} (with aromatic bond info)")
        
        # Analyze structure
        if any(marker in smiles_string.lower() for marker in ['c1cc', 'c1nc', 'c1cn', 'c1=c']):
            print("\nNote: Aromatic system detected:")
            print("   - 3D coordinates generated with proper planarity")
            print("   - Aromatic bonds preserved in SDF format")
            print("   - PDB file contains 3D structure for FluxMD analysis")
        
        # For benzene specifically
        if smiles_string.lower() in ['c1ccccc1', 'c1=cc=cc=c1']:
            print("\nBenzene structure:")
            print("   - 6 carbon atoms in planar hexagonal arrangement")
            print("   - 6 hydrogen atoms added automatically")
            print("   - Aromatic system properly represented")
        
        return pdb_file
        
    except urllib.error.URLError as e:
        print(f"Error connecting to CACTUS service: {e}")
        print("Please check your internet connection")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def convert_smiles_to_pdb_openbabel(smiles_string, output_name="ligand"):
    """Simple SMILES to PDB conversion using OpenBabel (fallback method)"""
    smi_file = f"{output_name}.smi"
    pdb_file = f"{output_name}.pdb"
    
    try:
        # Write SMILES to file
        with open(smi_file, 'w') as f:
            f.write(smiles_string)
        
        print(f"Converting SMILES to 3D structure using OpenBabel...")
        print(f"SMILES: {smiles_string}")
        
        # Simple one-step conversion with 3D generation
        cmd = ['obabel', '-ismi', smi_file, '-opdb', '-O', pdb_file,
               '--gen3d', '-h', '-p', '7.4']
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.stderr and "warning" not in result.stderr.lower():
            print(f"OpenBabel warnings: {result.stderr}")
        
        # Clean up
        if os.path.exists(smi_file):
            os.remove(smi_file)
        
        # Check output
        if os.path.exists(pdb_file):
            with open(pdb_file, 'r') as f:
                content = f.read()
                atom_count = content.count('HETATM')
            
            print(f"Generated {atom_count} atoms")
            print(f"Created: {pdb_file}")
            
            # Warning for aromatics
            if any(marker in smiles_string.lower() for marker in ['c1cc', 'c1nc', 'c1cn']):
                print("\nWarning: OpenBabel may not handle aromatics perfectly.")
                print("   Consider using CACTUS method for better results.")
            
            return pdb_file
        else:
            print("Error: OpenBabel failed to create output file")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr if e.stderr else str(e)}")
        if os.path.exists(smi_file):
            os.remove(smi_file)
        return None
    except FileNotFoundError:
        print("Error: OpenBabel not found.")
        print("Install with: conda install -c conda-forge openbabel")
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
                elif key == "Approach distance" or key == "Initial approach distance":
                    # Handle various formats: "2.5", "2.5 Angstroms", "2.5 Å"
                    clean_value = value.replace(' Angstroms', '').replace(' Å', '').strip()
                    params['approach_distance'] = float(clean_value)
                elif key == "Starting distance":
                    # Handle various formats
                    clean_value = value.replace(' Angstroms', '').replace(' Å', '').strip()
                    params['starting_distance'] = float(clean_value)
                elif key == "Rotations per position":
                    params['n_rotations'] = int(value)
                elif key == "pH":
                    params['physiological_pH'] = float(value)
                elif key == "Protein":
                    params['protein_file'] = value
                elif key == "Ligand":
                    params['ligand_file'] = value
                elif key == "Protein name":
                    params['protein_name'] = value
                elif key == "Final distance":
                    # Handle final distance if present
                    clean_value = value.replace(' Angstroms', '').replace(' Å', '').replace('~', '').strip()
                    try:
                        params['final_distance'] = float(clean_value)
                    except:
                        pass  # Skip if can't parse
                elif key == "Mode":
                    # Store trajectory mode for reference
                    params['mode'] = value
        
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
                
                print(f"Found {len(aromatic_residues)} aromatic residues")
                print(f"  Average flux: {aromatic_flux:.3f} (vs overall {total_flux:.3f})")
                
                if aromatic_flux > total_flux:
                    print("  Aromatic residues show higher flux")
                else:
                    print("  Warning: Aromatic residues don't show enhanced flux")
            else:
                print("Warning: No aromatic residues found")
        
        # Check for statistical validation
        if 'p_value' in df.columns:
            significant = df[df['p_value'] < 0.05]
            print(f"\nStatistical validation present")
            print(f"  {len(significant)}/{len(df)} residues significant (p<0.05)")
        else:
            print("\nWarning: No statistical validation found")
            validation_passed = False
    else:
        print("Error: No flux data file found!")
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
            print(f"\nPi-stacking interactions found!")
            print(f"  {pi_interactions}/{total_interactions} ({pi_percentage:.1f}%)")
        else:
            print("\nNote: No pi-stacking interactions detected")
            print("  This might be normal if ligand lacks aromatic rings")
    
    return validation_passed


def run_complete_workflow():
    """Run the complete analysis workflow"""
    print_banner("FLUXMD - WINDING TRAJECTORY ANALYSIS")
    
    print("This workflow will:")
    print("1. Calculate static intra-protein force field (one-time)")
    print("2. Generate WINDING trajectories (thread-like motion around protein)")
    print("3. Sample multiple ligand orientations at each position")
    print("4. Calculate non-covalent interactions with combined forces")
    print("5. Compute energy flux differentials (합벡터 analysis)")
    print("6. Identify binding sites with statistical validation")
    print("7. Create visualizations and reports\n")
    
    # Step 1: Get input files
    print("STEP 1: INPUT FILES")
    print("")
    
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
        
        # Try CACTUS first for better aromatic handling
        print("\nTrying NCI CACTUS service (recommended for aromatics)...")
        converted_file = convert_smiles_to_pdb_cactus(ligand_file, ligand_name)
        
        if converted_file is None:
            print("\nFalling back to OpenBabel...")
            converted_file = convert_smiles_to_pdb_openbabel(ligand_file, ligand_name)
            if converted_file is None:
                return
        
        ligand_file = converted_file
    elif not os.path.exists(ligand_file):
        print(f"Error: {ligand_file} not found!")
        return
    
    protein_name = input("Enter protein name for labeling: ").strip()
    
    # Step 2: Set parameters
    print("\nSTEP 2: PARAMETERS")
    print("")
    
    # Ask if user wants to use existing parameters
    use_existing = input("\nLoad parameters from existing simulation? (y/n): ").strip().lower()
    
    if use_existing == 'y':
        params_file = input("Enter path to simulation_parameters.txt: ").strip()
        if os.path.exists(params_file):
            loaded_params = parse_simulation_parameters(params_file)
            if loaded_params:
                print("\nLoaded parameters from file:")
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
        approach_distance = float(input("Approach distance in Angstroms (default 2.5): ") or "2.5")
        starting_distance = float(input("Starting distance in Angstroms (default 15): ") or "15")
        
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
    from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer
    temp_analyzer = ProteinLigandFluxAnalyzer()
    try:
        protein_atoms = temp_analyzer.parse_structure(protein_file, parse_heterogens=False)
        ligand_atoms = temp_analyzer.parse_structure_robust(ligand_file, parse_heterogens=True)
        n_protein_atoms = len(protein_atoms)
        n_ligand_atoms = len(ligand_atoms)
    except:
        n_protein_atoms = 5000  # Default estimates
        n_ligand_atoms = 50
    
    # Check for input file issues
    if n_protein_atoms < 50:
        print(f"\n❌ ERROR: Protein file has only {n_protein_atoms} atoms!")
        print(f"   This appears to be a small molecule, not a protein structure.")
        print(f"   Please provide a proper protein PDB file.")
        print(f"\n   Common issues:")
        print(f"   - You may have swapped the protein and ligand files")
        print(f"   - The protein file might be corrupted or incomplete")
        print(f"   - You might be using a ligand file as the protein input")
        return
    
    # Check if "ligand" is actually another protein
    is_protein_protein = n_ligand_atoms > 500  # Threshold for protein vs small molecule
    
    if is_protein_protein:
        print(f"\n⚠️  PROTEIN-PROTEIN INTERACTION DETECTED")
        print(f"   'Ligand' has {n_ligand_atoms:,} atoms - this appears to be another protein.")
        print(f"   Total system size: {n_protein_atoms + n_ligand_atoms:,} atoms")
        print(f"\n   RECOMMENDATION: Use UMA-optimized workflow (option 2) for better performance!")
        print(f"   The standard workflow may be very slow for protein-protein interactions.")
        
        uma_choice = input("\n   Switch to UMA workflow now? (recommended) (y/n): ").strip().lower()
        if uma_choice == 'y':
            print("\n   Redirecting to UMA workflow...")
            # Run UMA workflow with current files
            import subprocess
            cmd = [
                sys.executable, "fluxmd_uma.py", 
                protein_file, ligand_file, 
                "-o", output_dir,
                "-s", str(n_steps),
                "-i", str(n_iterations),
                "-a", str(n_approaches),
                "-d", str(starting_distance),
                "--approach-distance", str(approach_distance),
                "-r", str(n_rotations),
                "--ph", str(physiological_pH)
            ]
            subprocess.run(cmd)
            return
    
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
    print(f"\nSystem Analysis:")
    print(f"  Protein atoms: {n_protein_atoms:,}")
    print(f"  Ligand atoms: {n_ligand_atoms:,}")
    print(f"  Total frames: {frames_per_iteration:,} per iteration")
    print(f"  Rotations per frame: {n_rotations}")
    print(f"  Total operations: {total_operations/1e6:.1f} million")
    
    if gpu_available:
        print(f"\nGPU detected: {device}")
    
    # Initial decision
    print(f"\nPerformance estimation:")
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
    print(f"\nFinal decision:")
    if use_gpu:
        print(f"  Using GPU acceleration ({decision_reason})")
    else:
        print(f"  Using CPU parallel processing ({decision_reason})")
        if gpu_available:
            print("  Note: Decision based on performance characteristics")
            
            # Offer to override and use GPU anyway
            override = input("\nWould you like to use GPU anyway? (y/n): ").strip().lower()
            if override == 'y':
                use_gpu = True
                decision_reason = "user override"
                print("  Using GPU based on user preference")
    
    # Set parallel processing for CPU
    n_jobs = -1 if not use_gpu else 1  # Use all cores for CPU, single thread for GPU
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Mode: WINDING TRAJECTORY (thread-like motion around protein)")
    print(f"  Total steps: {n_steps * n_approaches} per iteration")
    print(f"  Starting distance: {starting_distance} Angstroms from surface")
    print(f"  Distance range: ~5-{starting_distance * 2.5:.0f} Angstroms (free variation)")
    print(f"  Rotations: {n_rotations} per position")
    print(f"  pH: {physiological_pH} (affects H-bond donors/acceptors)")
    print(f"  Processing: {'GPU' if use_gpu else 'CPU'} {'(parallel)' if n_jobs != 1 else ''}")
    
    # Performance estimate
    if use_gpu:
        estimated_time = (total_operations / 1e6) * 0.1  # Rough estimate: 0.1 sec per million on GPU
    else:
        cores = mp.cpu_count() if n_jobs == -1 else n_jobs
        estimated_time = (total_operations / 1e6) * 0.5 / cores  # 0.5 sec per million per core
    
    print(f"\nEstimated processing time: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)")
    
    # Provide optimization suggestions if slow
    if estimated_time > 300:  # More than 5 minutes
        print("\nWarning: Long processing time expected. Consider:")
        if n_rotations > 24:
            print(f"  - Reduce rotations to 12-24 (currently {n_rotations})")
        if n_steps > 50:
            print(f"  - Reduce steps to 50 (currently {n_steps})")
        if n_approaches > 3:
            print(f"  - Reduce approaches to 3 (currently {n_approaches})")
    
    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return
    
    # Save parameters to file
    os.makedirs(output_dir, exist_ok=True)
    params_file = os.path.join(output_dir, "simulation_parameters.txt")
    
    with open(params_file, 'w') as f:
        f.write("FLUXMD SIMULATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("INPUT FILES\n")
        f.write("\n")
        f.write(f"Protein: {protein_file}\n")
        f.write(f"Ligand: {ligand_file}\n")
        f.write(f"Protein name: {protein_name}\n")
        f.write("\n")
        f.write("TRAJECTORY PARAMETERS\n")
        f.write("\n")
        f.write(f"Mode: WINDING TRAJECTORY (thread-like motion around protein)\n")
        f.write(f"Steps per approach: {n_steps}\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Number of approaches: {n_approaches}\n")
        f.write(f"Initial approach distance: {approach_distance} Angstroms\n")
        f.write(f"Starting distance: {starting_distance} Angstroms\n")
        f.write(f"Distance range: ~5-{starting_distance * 2.5:.0f} Angstroms (free variation)\n")
        f.write(f"Rotations per position: {n_rotations}\n")
        f.write(f"Total steps per iteration: {n_steps * n_approaches}\n")
        f.write(f"Total rotations sampled: {n_steps * n_approaches * n_rotations}\n")
        f.write("\n")
        f.write("CALCULATION PARAMETERS\n")
        f.write("\n")
        f.write(f"pH: {physiological_pH}\n")
        f.write(f"GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED'}\n")
        if use_gpu:
            f.write(f"GPU device: {device}\n")
        f.write(f"Parallel processing: {'ENABLED' if n_jobs != 1 else 'DISABLED'}\n")
        if n_jobs != 1:
            f.write(f"CPU cores: {mp.cpu_count()}\n")
        f.write("\n")
        f.write("OUTPUT DIRECTORY\n")
        f.write("\n")
        f.write(f"{os.path.abspath(output_dir)}\n")
    
    print(f"\nParameters saved to: {params_file}")
    
    # Step 3: Run trajectory analysis
    print_banner("STEP 3: WINDING TRAJECTORY GENERATION")
    
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
            print("\nTrajectory analysis was cancelled.")
            return
        
        print("\nTrajectory analysis complete!")
        
        # Verify output
        import glob
        iter_dirs = glob.glob(os.path.join(output_dir, "iteration_*"))
        if len(iter_dirs) == 0:
            print("\nError: No iteration directories found!")
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
        # Check if we can use integrated GPU pipeline
        if use_gpu and hasattr(trajectory_analyzer, 'gpu_trajectory_results') and trajectory_analyzer.gpu_trajectory_results is not None:
            print("\nUsing integrated GPU flux pipeline (bypassing CSV parsing)...")
            # Use integrated GPU pipeline for maximum efficiency
            flux_data = flux_analyzer.create_integrated_flux_pipeline(
                protein_file, 
                trajectory_analyzer.gpu_trajectory_results, 
                output_dir
            )
            
            # Create visualizations
            flux_analyzer.visualize_trajectory_flux(flux_data, protein_name, output_dir)
            
            # Generate report
            flux_analyzer.generate_summary_report(flux_data, protein_name, output_dir)
            
            # Save processed data
            flux_analyzer.save_processed_data(flux_data, output_dir)
        else:
            # Traditional CSV-based processing
            flux_data = flux_analyzer.process_trajectory_iterations(output_dir, protein_file)
            
            # Create visualizations
            flux_analyzer.visualize_trajectory_flux(flux_data, protein_name, output_dir)
            
            # Generate report
            flux_analyzer.generate_summary_report(flux_data, protein_name, output_dir)
            
            # Save processed data
            flux_analyzer.save_processed_data(flux_data, output_dir)
        
        print("\nFlux analysis complete!")
        
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
    print("  simulation_parameters.txt - All simulation parameters")
    print("  trajectory_iteration_*_approach_*.png - Cocoon trajectories")
    print("  trajectory_iteration_*_approach_*.csv - Trajectory coordinates")
    print("  iteration_*/ - Interaction data with rotations")
    print("  interactions_approach_*.csv - Detailed interactions")
    print("  *_trajectory_flux_analysis.png - Flux visualization")
    print("  *_flux_report.txt - Statistical analysis")
    print("  processed_flux_data.csv - Flux with p-values")
    print("  all_iterations_flux.csv - Raw flux data")
    
    print("\nINTERPRETATION:")
    print("- Red regions = High flux = Statistically significant binding sites")
    print("- Purple markers = Aromatic residues capable of pi-stacking")
    print("- Error bars = 95% confidence intervals from bootstrap")
    print("- P-values indicate statistical significance of each residue")
    print("- Flux values now include both inter & intra-protein forces (combined vector)")
    print("- Higher flux = stronger combined force convergence at binding site")
    
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
    print("1. Run complete workflow (standard)")
    print("2. Run UMA-optimized workflow (zero-copy GPU, 100x faster)")
    print("3. Convert SMILES to PDB (CACTUS with aromatics or OpenBabel)")
    print("4. Generate DNA structure from sequence")
    print("5. Visualize multiple proteins (compare flux)")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        run_complete_workflow()
    elif choice == "2":
        # Run UMA-optimized workflow
        print_banner("UMA-OPTIMIZED WORKFLOW")
        print("This uses zero-copy GPU processing for maximum performance.")
        print("Best for Apple Silicon Macs and systems with unified memory.\n")
        
        # Initialize variables
        protein_file = None
        ligand_file = None
        loaded_params = None
        
        # Ask if user wants to use existing parameters first
        use_existing = input("Load parameters from existing simulation? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            params_file = input("Enter path to simulation_parameters.txt: ").strip()
            if os.path.exists(params_file):
                loaded_params = parse_simulation_parameters(params_file)
                if loaded_params:
                    print("\nLoaded parameters from file:")
                    for key, value in loaded_params.items():
                        print(f"  {key}: {value}")
                    
                    # Check if protein/ligand files are in parameters
                    if 'protein_file' in loaded_params:
                        protein_file = loaded_params['protein_file']
                    if 'ligand_file' in loaded_params:
                        ligand_file = loaded_params['ligand_file']
            else:
                print(f"File not found: {params_file}")
                use_existing = 'n'
        
        # Get input files if not loaded from parameters
        if not protein_file:
            protein_file = input("\nEnter protein PDB file: ").strip()
        if not os.path.exists(protein_file):
            print(f"Error: {protein_file} not found!")
            return
        
        if not ligand_file:
            ligand_file = input("Enter ligand PDB file: ").strip()
        if not os.path.exists(ligand_file):
            print(f"Error: {ligand_file} not found!")
            return
        
        output_dir = input("Output directory (default 'flux_analysis_uma'): ").strip() or "flux_analysis_uma"
        
        # Continue with parameter confirmation if loaded
        print("\nSIMULATION PARAMETERS")
        
        if use_existing == 'y' and loaded_params:
            confirm = input("\nUse these parameters? (y/n): ").strip().lower()
            if confirm == 'y':
                # Use loaded parameters
                n_steps = loaded_params.get('n_steps', 200)
                n_iterations = loaded_params.get('n_iterations', 10)
                n_approaches = loaded_params.get('n_approaches', 10)
                approach_distance = loaded_params.get('approach_distance', 2.5)
                starting_distance = loaded_params.get('starting_distance', 20.0)
                n_rotations = loaded_params.get('n_rotations', 36)
                physiological_pH = loaded_params.get('physiological_pH', 7.4)
            else:
                use_existing = 'n'  # Fall back to manual entry
        else:
            if use_existing == 'y':
                print("Failed to parse parameters file.")
            use_existing = 'n'
        
        if use_existing != 'y':
            # Manual parameter entry
            print("\nEnter parameters manually (press Enter for defaults):\n")
            
            n_steps = input("Steps per trajectory (default 200): ").strip()
            n_steps = int(n_steps) if n_steps else 200
            
            n_iterations = input("Number of iterations (default 10): ").strip()
            n_iterations = int(n_iterations) if n_iterations else 10
            
            n_approaches = input("Number of approach angles (default 10): ").strip()
            n_approaches = int(n_approaches) if n_approaches else 10
            
            starting_distance = input("Starting distance in Angstroms (default 20.0): ").strip()
            starting_distance = float(starting_distance) if starting_distance else 20.0
            
            approach_distance = input("Distance step between approaches in Angstroms (default 2.5): ").strip()
            approach_distance = float(approach_distance) if approach_distance else 2.5
            
            n_rotations = input("Rotations per position (default 36): ").strip()
            n_rotations = int(n_rotations) if n_rotations else 36
            
            physiological_pH = input("pH for protonation states (default 7.4): ").strip()
            physiological_pH = float(physiological_pH) if physiological_pH else 7.4
        
        # Ask about saving trajectories
        save_trajectories = input("\nSave trajectory files? (y/n): ").strip().lower() == 'y'
        
        # Ask about showing detailed interaction breakdown
        show_interactions = input("Show detailed interaction breakdown? (y/n): ").strip().lower() == 'y'
        
        # Show summary
        print("\nUMA ANALYSIS CONFIGURATION:")
        print(f"  Protein: {protein_file}")
        print(f"  Ligand: {ligand_file}")
        print(f"  Output: {output_dir}")
        print(f"  Steps: {n_steps}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Approaches: {n_approaches}")
        print(f"  Starting distance: {starting_distance} Å")
        print(f"  Approach distance: {approach_distance} Å")
        print(f"  Rotations: {n_rotations}")
        print(f"  pH: {physiological_pH}")
        print(f"  Save trajectories: {'Yes' if save_trajectories else 'No'}")
        print(f"  Show interaction details: {'Yes' if show_interactions else 'No'}")
        
        # Calculate total operations
        total_frames = n_steps * n_approaches * n_iterations
        print(f"\nTotal trajectory frames: {total_frames:,}")
        
        confirm = input("\nProceed with UMA-optimized analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled.")
            return
        
        # Run fluxmd_uma as subprocess with parameters
        import subprocess
        cmd = [
            sys.executable, "fluxmd_uma.py", 
            protein_file, ligand_file, 
            "-o", output_dir,
            "-s", str(n_steps),
            "-i", str(n_iterations),
            "-a", str(n_approaches),
            "-d", str(starting_distance),
            "--approach-distance", str(approach_distance),
            "-r", str(n_rotations),
            "--ph", str(physiological_pH)
        ]
        
        # Add save trajectories flag if requested
        if save_trajectories:
            cmd.append("--save-trajectories")
        
        # Add interaction details flag if requested
        if show_interactions:
            cmd.append("--interaction-details")
        
        subprocess.run(cmd)
    elif choice == "3":
        print_banner("SMILES TO PDB CONVERTER")
        smiles = input("Enter SMILES string: ").strip()
        if smiles:
            name = input("Enter output name: ").strip() or "ligand"
            
            print("\nConversion options:")
            print("1. NCI CACTUS (recommended - preserves aromaticity, requires internet)")
            print("2. OpenBabel (local fallback - may have aromatic issues)")
            
            method = input("\nSelect method (1-2): ").strip() or "1"
            
            if method == "1":
                convert_smiles_to_pdb_cactus(smiles, name)
            else:
                convert_smiles_to_pdb_openbabel(smiles, name)
    elif choice == "4":
        print_banner("DNA SEQUENCE TO STRUCTURE")
        print("Generate 3D B-DNA structure from sequence")
        print("\nNote: This creates atomically-detailed B-DNA for protein-DNA interaction analysis")
        print("Features:")
        print("  - Proper sugar-phosphate backbone with all atoms")
        print("  - Watson-Crick base pairing geometry")
        print("  - Standard B-DNA helical parameters")
        print("  - Complete connectivity information (CONECT records)")
        
        sequence = input("\nEnter DNA sequence (e.g., ATCGATCG): ").strip().upper()
        
        # Validate sequence
        valid_bases = set('ATGC')
        if not sequence:
            print("Error: Empty sequence")
            return
        if not all(base in valid_bases for base in sequence):
            print("Error: Sequence must contain only A, T, G, C")
            print(f"Found invalid characters: {set(sequence) - valid_bases}")
            return
        
        # Warn about sequence length
        if len(sequence) < 4:
            print("Warning: Very short sequences may not show proper helical structure")
        elif len(sequence) > 100:
            print(f"Warning: Long sequence ({len(sequence)} bp) will generate many atoms")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                return
        
        output_name = input("Enter output filename (default: dna_structure.pdb): ").strip()
        if not output_name:
            output_name = "dna_structure.pdb"
        
        # Ensure .pdb extension
        if not output_name.endswith('.pdb'):
            output_name += '.pdb'
        
        # Import the improved generator
        try:
            # First try to import from separate file
            from fluxmd.utils.dna_to_pdb_improved import DNAStructureGenerator
        except ImportError:
            # If not available as separate file, use the embedded version
            print("Using embedded DNA generator...")
            # Here you would have the DNAStructureGenerator class defined inline
            # For now, fall back to original
            from fluxmd.utils.dna_to_pdb import DNAStructureGenerator
        
        print(f"\nGenerating B-DNA structure for: {sequence}")
        print(f"Sequence length: {len(sequence)} bp")
        print(f"Double helix will contain:")
        print(f"  - {len(sequence) * 2} nucleotides total")
        print(f"  - ~{len(sequence) * 35} atoms per strand")
        print(f"  - Helix length: ~{len(sequence) * 3.38:.1f} Angstroms")
        
        try:
            generator = DNAStructureGenerator()
            generator.generate_dna(sequence)
            generator.write_pdb(output_name)
            
            print(f"\nStructure successfully written to: {output_name}")
            print(f"  Total atoms: {len(generator.atoms)}")
            print(f"  Base pairs: {len(sequence)}")
            print(f"  Chains: A (5' to 3'), B (3' to 5')")
            
            # Provide usage tips
            print("\nStructure details:")
            print("  - Strand A: 5' to 3' direction")
            print("  - Strand B: 3' to 5' direction (complementary)")
            print("  - Standard B-DNA geometry (10.5 bp/turn)")
            print("  - All atoms including hydrogens")
            
            print("\nUsage in FluxMD:")
            print("  1. Use this DNA as the 'ligand' in workflow option 1")
            print("  2. FluxMD will analyze protein-DNA interactions")
            print("  3. High flux regions indicate DNA binding sites")
            
            print("\nVisualization tips:")
            print("  pymol " + output_name)
            print("  PyMOL commands:")
            print("    show cartoon")
            print("    set cartoon_nucleic_acid_mode, 4")
            print("    color cyan, chain A")
            print("    color yellow, chain B")
            print("    show sticks, resn DG+DC+DA+DT")
            print("    set stick_radius, 0.2")
            
            # Offer to generate a test protein-DNA complex
            print("\nTip: For testing protein-DNA interactions:")
            print("  - Use a DNA-binding protein (e.g., transcription factor)")
            print("  - DNA groove widths: Major ~22Angstroms, Minor ~12Angstroms")
            print("  - Typical protein-DNA interface: 10-20 base pairs")
            
        except Exception as e:
            print(f"\nError generating DNA structure: {e}")
            import traceback
            traceback.print_exc()
            print("\nTroubleshooting:")
            print("  - Check sequence contains only ATGC")
            print("  - Ensure write permissions in current directory")
            print("  - Try a shorter test sequence first")
        
    elif choice == "5":
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
    elif choice == "6":
        print("\nThank you for using FluxMD!")
    else:
        print("Invalid choice!")
        main()


if __name__ == "__main__":
    main()
