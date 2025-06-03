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
    print_banner("FLUXMD - PROTEIN-LIGAND FLUX ANALYSIS")
    
    print("This workflow will:")
    print("1. Calculate static intra-protein force field (one-time)")
    print("2. Generate ligand trajectories around protein surface")
    print("3. Calculate non-covalent interactions with combined forces")
    print("4. Compute energy flux differentials (합벡터 analysis)")
    print("5. Identify binding sites with statistical validation")
    print("6. Create visualizations and reports\n")
    
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
    print("\nPRESS ENTERS FOR REPRODUCIBLE RESULTS")
    n_steps = int(input("Steps per approach (default 1000): ") or "100")
    n_iterations = int(input("Number of iterations (default 50): ") or "100")
    n_approaches = int(input("Number of approaches (default 5): ") or "5")
    approach_distance = float(input("Approach distance in Å (default 2.5): ") or "2.5")
    starting_distance = float(input("Starting distance in Å (default 15): ") or "15")
    output_dir = input("Output directory (default 'flux_analysis'): ").strip() or "flux_analysis"
    
    # Check for GPU
    use_gpu = False
    try:
        device = get_device()
        if 'mps' in str(device) or 'cuda' in str(device):
            print(f"\n🚀 GPU detected: {device}")
            gpu_choice = input("Use GPU acceleration? (Y/n): ").strip().lower()
            use_gpu = gpu_choice != 'n'
    except:
        print("\n💻 No GPU detected, using CPU")
        use_gpu = False
    
    # Check for parallel processing
    n_jobs = -1
    if not use_gpu and platform.system() == 'Darwin':
        print(f"\nDetected macOS with {mp.cpu_count()} cores")
        use_parallel = input("Use parallel processing? (Y/n): ").strip().lower()
        if use_parallel == 'n':
            n_jobs = 1
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Total steps: {n_steps * n_approaches} per iteration")
    print(f"  Starting: {starting_distance} Å from surface")
    print(f"  Final: ~{starting_distance - (n_approaches-1)*approach_distance:.1f} Å")
    print(f"  GPU: {'ENABLED' if use_gpu else 'DISABLED'}")
    print(f"  Parallel: {'ENABLED' if n_jobs != 1 else 'DISABLED'}")
    
    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return
    
    # Step 3: Run trajectory analysis
    print_banner("STEP 3: TRAJECTORY GENERATION")
    
    trajectory_analyzer = ProteinLigandFluxAnalyzer()
    
    start_time = datetime.now()
    
    try:
        # Run trajectory analysis
        iteration_data = trajectory_analyzer.run_complete_analysis(
            protein_file, ligand_file, output_dir, n_steps, n_iterations,
            n_approaches, approach_distance, starting_distance,
            n_jobs=n_jobs, use_gpu=use_gpu
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
    print("├── trajectory_visualization_*.png - Brownian trajectories")
    print("├── iteration_*/ - Interaction data with pi-stacking")
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
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_complete_workflow()
    elif choice == "2":
        print_banner("SMILES TO PDBQT CONVERTER")
        smiles = input("Enter SMILES string: ").strip()
        if smiles:
            name = input("Enter output name: ").strip() or "ligand"
            convert_smiles_to_pdbqt(smiles, name)
    elif choice == "3":
        print("\nThank you for using FluxMD!")
    else:
        print("Invalid choice!")
        main()


if __name__ == "__main__":
    main()
