"""
FluxMD: GPU-accelerated binding site prediction using flux differential analysis
"""

import os
import sys
import argparse
import multiprocessing as mp
import platform
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import our modules
from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer
from fluxmd.analysis.flux_analyzer import TrajectoryFluxAnalyzer
from fluxmd.gpu.gpu_accelerated_flux import get_device
from fluxmd.utils.config_parser import load_config, print_derived_constants, create_example_config
from fluxmd.utils.pdb_parser import PDBParser
from fluxmd.utils.cpu import parse_workers, format_workers_info


def check_gpu_availability():
    """Check if GPU is available for computation"""
    try:
        device = get_device()
        if "mps" in str(device) or "cuda" in str(device):
            return True
    except:
        pass
    return False


def benchmark_performance(protein_atoms, ligand_atoms, n_test_frames=5, n_test_rotations=12):
    """
    Run quick benchmark to determine actual GPU vs CPU performance
    Returns: (use_gpu, reason)
    """
    import time
    from fluxmd.gpu.gpu_accelerated_flux import GPUAcceleratedInteractionCalculator

    try:
        device = get_device()
        if "cpu" in str(device):
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

        ligand_coords = ligand_atoms[["x", "y", "z"]].values

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
        return True, f"GPU {gpu_fps / cpu_fps:.1f}x faster in benchmark"
    else:
        return False, f"CPU more efficient ({cpu_fps / gpu_fps:.1f}x) for this workload"


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def convert_cif_to_pdb(cif_file):
    """Convert CIF/mmCIF to PDB format using OpenBabel"""
    pdb_file = cif_file.rsplit(".", 1)[0] + ".pdb"

    print(f"Converting {cif_file} to PDB format...")
    try:
        subprocess.run(
            ["obabel", cif_file, "-O", pdb_file], check=True, capture_output=True, text=True
        )
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

        # Try to count aromatic rings in SMILES
        aromatic_count = smiles_string.lower().count("c") + smiles_string.lower().count("n")
        aromatic_rings_est = aromatic_count // 5  # Rough estimate

        # URL encode the SMILES string
        encoded_smiles = urllib.parse.quote(smiles_string, safe="")

        # First try to get SDF with 3D coordinates and aromatic bonds
        # SDF format preserves aromaticity better than PDB
        sdf_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/file?format=sdf&get3d=true"

        print("  Requesting 3D structure with aromatic bonds preserved...")

        # Get SDF first
        with urllib.request.urlopen(sdf_url) as response:
            sdf_content = response.read().decode("utf-8")

        # Check if we got an error
        if "Page not found" in sdf_content or "<html>" in sdf_content:
            print("Error: CACTUS could not process this SMILES string")
            return None

        # Save SDF file
        with open(sdf_file, "w") as f:
            f.write(sdf_content)

        # Convert SDF to PDB with OpenBabel to preserve bond orders
        try:
            subprocess.run(
                ["obabel", sdf_file, "-O", pdb_file, "-h", "-b"],
                check=True,
                capture_output=True,
                text=True,
            )
            with open(pdb_file, "r") as f:
                pdb_content = f.read()
        except Exception as e:
            print(f"OpenBabel conversion failed: {e}")
            print("  Falling back to CACTUS PDB (bond orders may be missing)...")
            pdb_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/file?format=pdb&get3d=true"
            with urllib.request.urlopen(pdb_url) as response:
                pdb_content = response.read().decode("utf-8")
            with open(pdb_file, "w") as f:
                f.write(pdb_content)

        # Count atoms and check aromaticity
        atom_count = pdb_content.count("HETATM")

        # Check SDF for aromatic bonds (bond type 4)
        aromatic_bonds = sdf_content.count("  4  ") + sdf_content.count(" 4 0 ")

        print(f"Generated {atom_count} atoms")
        if aromatic_bonds > 0:
            print(f"Preserved {aromatic_bonds} aromatic bonds")
        print(f"Created: {pdb_file} (for FluxMD)")
        print(f"Created: {sdf_file} (with aromatic bond info)")

        # Analyze structure
        if any(marker in smiles_string.lower() for marker in ["c1cc", "c1nc", "c1cn", "c1=c"]):
            print("\nNote: Aromatic system detected:")
            print("   - 3D coordinates generated with proper planarity")
            print("   - Aromatic bonds preserved in SDF format")
            print("   - PDB file contains 3D structure for FluxMD analysis")

        # For benzene specifically
        if smiles_string.lower() in ["c1ccccc1", "c1=cc=cc=c1"]:
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


def convert_smiles_with_aromatics(smiles_string, output_name="ligand"):
    """Enhanced SMILES to PDB conversion with aromatic ring detection and preservation"""
    import re

    # Count aromatic rings from SMILES patterns
    patterns = {
        "benzene": [r"c1ccccc1", r"c1=cc=cc=c1"],
        "pyridine": [r"c1ccncc1", r"n1ccccc1"],
        "pyrrole": [r"c1cc[nH]c1", r"n1cccc1"],
        "furan": [r"c1ccoc1", r"o1cccc1"],
        "thiophene": [r"c1ccsc1", r"s1cccc1"],
        "imidazole": [r"c1cnc[nH]1", r"n1ccnc1"],
        "thiadiazole": [r"c1nnsc1", r"s1nncc1", r"c1snns1"],
    }

    smiles_lower = smiles_string.lower()
    ring_count = 0
    ring_types = []

    # Count each type of ring
    for ring_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = len(re.findall(pattern, smiles_lower))
            if matches > 0:
                ring_count += matches
                ring_types.append(f"{matches} {ring_type}")
                break

    # Estimate from aromatic atoms
    aromatic_atoms = smiles_lower.count("c") + smiles_lower.count("n")
    estimated_rings = aromatic_atoms // 5

    print(f"\nAnalyzing SMILES: {smiles_string}")
    if ring_types:
        print(f"Detected aromatic patterns: {', '.join(ring_types)}")
    print(f"Estimated aromatic rings: ~{max(ring_count, estimated_rings)}")

    # Ask user to confirm
    user_rings = input(
        f"\nHow many aromatic rings are in this molecule? [{max(ring_count, estimated_rings)}]: "
    ).strip()
    if user_rings:
        try:
            actual_rings = int(user_rings)
        except:
            actual_rings = max(ring_count, estimated_rings)
    else:
        actual_rings = max(ring_count, estimated_rings)

    print(f"\nGenerating structure with {actual_rings} aromatic rings...")

    # Convert using OpenBabel PDBQT to preserve aromaticity
    smi_file = f"{output_name}.smi"
    pdbqt_file = f"{output_name}.pdbqt"
    pdb_file = f"{output_name}.pdb"

    try:
        # Write SMILES
        with open(smi_file, "w") as f:
            f.write(smiles_string)

        # Convert to PDBQT (preserves aromatic atom types)
        print("Converting to PDBQT format (preserves aromaticity)...")
        cmd = [
            "obabel",
            "-ismi",
            smi_file,
            "-opdbqt",
            "-O",
            pdbqt_file,
            "--gen3d",
            "-h",
            "-p",
            "7.4",
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Clean up
        if os.path.exists(smi_file):
            os.remove(smi_file)

        # Analyze PDBQT and convert to PDB
        if os.path.exists(pdbqt_file):
            aromatic_atom_nums = []

            with open(pdbqt_file, "r") as f:
                pdbqt_content = f.read()

            # Count aromatic atoms
            aromatic_c = pdbqt_content.count(" A  ")
            aromatic_n = pdbqt_content.count(" NA ")
            aromatic_o = pdbqt_content.count(" OA ")
            aromatic_s = pdbqt_content.count(" SA ")

            print(f"\nAromatic atoms detected:")
            print(f"  C (aromatic): {aromatic_c}")
            print(f"  N (aromatic): {aromatic_n}")
            print(f"  O (aromatic): {aromatic_o}")
            print(f"  S (aromatic): {aromatic_s}")
            print(f"  Total: {aromatic_c + aromatic_n + aromatic_o + aromatic_s}")

            # Convert PDBQT to PDB with aromatic annotations
            with open(pdbqt_file, "r") as f_in, open(pdb_file, "w") as f_out:
                f_out.write(f"REMARK   Generated from SMILES with {actual_rings} aromatic rings\n")
                f_out.write(f"REMARK   SMILES: {smiles_string}\n")

                for line in f_in:
                    if line.startswith(("ATOM", "HETATM")):
                        atom_type = line[77:79].strip() if len(line) > 78 else ""

                        # Track aromatic atoms
                        if atom_type in ["A", "NA", "OA", "SA"]:
                            atom_num = int(line[6:11].strip())
                            aromatic_atom_nums.append(atom_num)

                        # Convert atom type to element
                        element_map = {
                            "A": "C",
                            "NA": "N",
                            "OA": "O",
                            "SA": "S",
                            "C": "C",
                            "N": "N",
                            "O": "O",
                            "S": "S",
                            "CL": "Cl",
                            "BR": "Br",
                            "I": "I",
                            "F": "F",
                            "HD": "H",
                            "H": "H",
                        }
                        element = element_map.get(atom_type, atom_type[0] if atom_type else "C")

                        # Write PDB line with proper element
                        if len(line) >= 78:
                            pdb_line = line[:76] + f"{element:>2}" + "\n"
                        else:
                            pdb_line = (
                                line.rstrip()
                                + " " * (76 - len(line.rstrip()))
                                + f"{element:>2}"
                                + "\n"
                            )

                        f_out.write(pdb_line)
                    elif line.startswith("CONECT"):
                        f_out.write(line)

                # Add aromatic atom list
                if aromatic_atom_nums:
                    f_out.write(
                        f"REMARK   AROMATIC_ATOMS {','.join(map(str, aromatic_atom_nums))}\n"
                    )

            print(f"\nFiles created:")
            print(f"  {pdb_file} - Standard PDB for FluxMD")
            print(f"  {pdbqt_file} - PDBQT with aromatic types preserved")

            if actual_rings > 0:
                print(f"\nIMPORTANT: This molecule has {actual_rings} aromatic ring(s)")
                print("FluxMD will now properly detect π-π stacking interactions")

            # Clean up PDBQT if user doesn't need it
            keep_pdbqt = input("\nKeep PDBQT file for reference? (y/N): ").strip().lower()
            if keep_pdbqt != "y" and os.path.exists(pdbqt_file):
                os.remove(pdbqt_file)
                print(f"Removed {pdbqt_file}")

            return pdb_file

    except subprocess.CalledProcessError as e:
        print(f"OpenBabel error: {e}")
        print("Make sure OpenBabel is installed: conda install -c conda-forge openbabel")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def convert_smiles_to_pdb_openbabel(smiles_string, output_name="ligand"):
    """Simple SMILES to PDB conversion using OpenBabel (fallback method)"""
    smi_file = f"{output_name}.smi"
    pdb_file = f"{output_name}.pdb"

    try:
        # Write SMILES to file
        with open(smi_file, "w") as f:
            f.write(smiles_string)

        print(f"Converting SMILES to 3D structure using OpenBabel...")
        print(f"SMILES: {smiles_string}")

        # Simple one-step conversion with 3D generation
        cmd = ["obabel", "-ismi", smi_file, "-opdb", "-O", pdb_file, "--gen3d", "-h", "-p", "7.4"]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if result.stderr and "warning" not in result.stderr.lower():
            print(f"OpenBabel warnings: {result.stderr}")

        # Clean up
        if os.path.exists(smi_file):
            os.remove(smi_file)

        # Check output
        if os.path.exists(pdb_file):
            with open(pdb_file, "r") as f:
                content = f.read()
                atom_count = content.count("HETATM")

            print(f"Generated {atom_count} atoms")
            print(f"Created: {pdb_file}")

            # Warning for aromatics
            if any(marker in smiles_string.lower() for marker in ["c1cc", "c1nc", "c1cn"]):
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
        with open(params_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if ": " in line:
                key, value = line.split(": ", 1)

                # Extract trajectory parameters
                if key == "Steps per approach":
                    params["n_steps"] = int(value)
                elif key == "Number of iterations":
                    params["n_iterations"] = int(value)
                elif key == "Number of approaches":
                    params["n_approaches"] = int(value)
                elif key == "Approach distance" or key == "Initial approach distance":
                    # Handle various formats: "2.5", "2.5 Angstroms", "2.5 Å"
                    clean_value = value.replace(" Angstroms", "").replace(" Å", "").strip()
                    params["approach_distance"] = float(clean_value)
                elif key == "Starting distance":
                    # Handle various formats
                    clean_value = value.replace(" Angstroms", "").replace(" Å", "").strip()
                    params["starting_distance"] = float(clean_value)
                elif key == "Rotations per position":
                    params["n_rotations"] = int(value)
                elif key == "pH":
                    params["physiological_pH"] = float(value)
                elif key == "Physiological pH":
                    params["physiological_pH"] = float(value)
                elif key == "Protein":
                    params["protein_file"] = value
                elif key == "Ligand":
                    params["ligand_file"] = value
                # DNA workflow specific parameters
                elif key == "DNA (target)":
                    params["dna_file"] = value
                elif key == "Protein (mobile)":
                    params["protein_file"] = value
                elif key == "Protein name":
                    params["protein_name"] = value
                elif key == "Final distance":
                    # Handle final distance if present
                    clean_value = (
                        value.replace(" Angstroms", "").replace(" Å", "").replace("~", "").strip()
                    )
                    try:
                        params["final_distance"] = float(clean_value)
                    except:
                        pass  # Skip if can't parse
                elif key == "Mode":
                    # Store trajectory mode for reference
                    params["mode"] = value

        return params

    except Exception as e:
        print(f"Error parsing parameters file: {e}")
        return None


def validate_results(output_dir, protein_name):
    """Validate that results contain expected improvements"""
    print_banner("🔍 VALIDATING RESULTS")

    validation_passed = True

    # Check for processed flux data
    flux_file = os.path.join(output_dir, "processed_flux_data.csv")
    if os.path.exists(flux_file):
        df = pd.read_csv(flux_file)

        # Check for pi-stacking contribution
        if "is_aromatic" in df.columns:
            aromatic_residues = df[df["is_aromatic"] == 1]
            if len(aromatic_residues) > 0:
                aromatic_flux = aromatic_residues["average_flux"].mean()
                total_flux = df["average_flux"].mean()

                print(f"Found {len(aromatic_residues)} aromatic residues")
                print(f"  Average flux: {aromatic_flux:.3f} (vs overall {total_flux:.3f})")

                if aromatic_flux > total_flux:
                    print("  Aromatic residues show higher flux")
                else:
                    print("  Warning: Aromatic residues don't show enhanced flux")
            else:
                print("Warning: No aromatic residues found")

        # Check for statistical validation
        if "p_value" in df.columns:
            significant = df[df["p_value"] < 0.05]
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
            if "bond_type" in df.columns:
                total_interactions += len(df)
                pi_mask = df["bond_type"].str.contains("Pi-Stacking", na=False)
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
    print("5. Compute energy flux differentials (combined vector analysis)")
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
    if protein_file.lower().endswith((".cif", ".mmcif")):
        converted_file = convert_cif_to_pdb(protein_file)
        if converted_file is None:
            return
        protein_file = converted_file

    ligand_file = input("Enter ligand file (PDBQT/PDB) or SMILES: ").strip()

    # Check if input is SMILES
    if not os.path.exists(ligand_file) and not ligand_file.endswith((".pdbqt", ".pdb")):
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

    if use_existing == "y":
        params_file = input("Enter path to simulation_parameters.txt: ").strip()
        if os.path.exists(params_file):
            loaded_params = parse_simulation_parameters(params_file)
            if loaded_params:
                print("\nLoaded parameters from file:")
                for key, value in loaded_params.items():
                    print(f"  {key}: {value}")

                confirm = input("\nUse these parameters? (y/n): ").strip().lower()
                if confirm == "y":
                    # Use loaded parameters
                    n_steps = loaded_params.get("n_steps", 100)
                    n_iterations = loaded_params.get("n_iterations", 100)
                    n_approaches = loaded_params.get("n_approaches", 5)
                    approach_distance = loaded_params.get("approach_distance", 2.5)
                    starting_distance = loaded_params.get("starting_distance", 15)
                    n_rotations = loaded_params.get("n_rotations", 36)
                    physiological_pH = loaded_params.get("physiological_pH", 7.4)
                else:
                    use_existing = "n"  # Fall back to manual entry
            else:
                print("Failed to parse parameters file.")
                use_existing = "n"
        else:
            print(f"File not found: {params_file}")
            use_existing = "n"

    if use_existing != "y":
        # Manual parameter entry
        print("\nEnter parameters manually (press Enter for defaults)")
        n_steps = int(input("Steps per approach (default 100): ") or "100")
        n_iterations = int(input("Number of iterations (default 100): ") or "100")
        n_approaches = int(input("Number of approaches (default 5): ") or "5")
        approach_distance = float(input("Approach distance in Angstroms (default 2.5): ") or "2.5")
        starting_distance = float(input("Starting distance in Angstroms (default 15): ") or "15")

        # Add pH parameter
        physiological_pH = float(
            input("pH for protonation state calculation (default 7.4): ") or "7.4"
        )
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
        if "mps" in str(device) or "cuda" in str(device):
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
        if uma_choice == "y":
            print("\n   Redirecting to UMA workflow...")
            # Run UMA workflow with current files
            import subprocess

            cmd = [
                sys.executable,
                "fluxmd_uma.py",
                protein_file,
                ligand_file,
                "-o",
                output_dir,
                "-s",
                str(n_steps),
                "-i",
                str(n_iterations),
                "-a",
                str(n_approaches),
                "-d",
                str(starting_distance),
                "--approach-distance",
                str(approach_distance),
                "-r",
                str(n_rotations),
                "--ph",
                str(physiological_pH),
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
            decision_reason = (
                f"GPU faster ({gpu_time_per_frame:.2f}s vs {cpu_time_per_frame:.2f}s per frame)"
            )
        else:
            use_gpu = False
            decision_reason = (
                f"CPU faster ({cpu_time_per_frame:.2f}s vs {gpu_time_per_frame:.2f}s per frame)"
            )

        # Override for memory constraints
        gpu_memory_needed = (
            n_protein_atoms * n_ligand_atoms * n_rotations * 4 * 8
        )  # bytes (float32 + indices)
        if "cuda" in str(device):
            import torch

            gpu_memory_available = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory_needed > gpu_memory_available * 0.8:
                use_gpu = False
                decision_reason = "insufficient GPU memory"
        elif "mps" in str(device):
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
    print(f"  Total operations: {total_operations / 1e6:.1f} million")

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
        run_benchmark = (
            input("\nRun performance benchmark for optimal selection? (y/n): ").strip().lower()
        )
        if run_benchmark == "y":
            benchmark_use_gpu, benchmark_reason = benchmark_performance(
                protein_atoms,
                ligand_atoms,
                n_test_frames=min(5, n_steps),
                n_test_rotations=n_rotations,
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
            if override == "y":
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
        estimated_time = (
            total_operations / 1e6
        ) * 0.1  # Rough estimate: 0.1 sec per million on GPU
    else:
        cores = mp.cpu_count() if n_jobs == -1 else n_jobs
        estimated_time = (total_operations / 1e6) * 0.5 / cores  # 0.5 sec per million per core

    print(
        f"\nEstimated processing time: {estimated_time:.0f} seconds ({estimated_time / 60:.1f} minutes)"
    )

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
    if confirm != "y":
        print("Analysis cancelled.")
        return

    # Save parameters to file
    os.makedirs(output_dir, exist_ok=True)
    params_file = os.path.join(output_dir, "simulation_parameters.txt")

    with open(params_file, "w") as f:
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
        if "trajectory_step_size" in locals():
            f.write(f"Trajectory step size: {trajectory_step_size} Angstroms\n")
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
        analysis_kwargs = {
            "n_steps": n_steps,
            "n_iterations": n_iterations,
            "n_approaches": n_approaches,
            "approach_distance": approach_distance,
            "starting_distance": starting_distance,
            "n_jobs": n_jobs,
            "use_gpu": use_gpu,
            "n_rotations": n_rotations,
        }

        # Add trajectory_step_size if it was defined
        if "trajectory_step_size" in locals():
            analysis_kwargs["trajectory_step_size"] = trajectory_step_size

        iteration_data = trajectory_analyzer.run_complete_analysis(
            protein_file, ligand_file, output_dir, **analysis_kwargs
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
        if (
            use_gpu
            and hasattr(trajectory_analyzer, "gpu_trajectory_results")
            and trajectory_analyzer.gpu_trajectory_results is not None
        ):
            print("\nUsing integrated GPU flux pipeline (bypassing CSV parsing)...")
            # Use integrated GPU pipeline for maximum efficiency
            flux_data = flux_analyzer.create_integrated_flux_pipeline(
                protein_file, trajectory_analyzer.gpu_trajectory_results, output_dir
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
    if another == "y":
        print("\nTo compare ligands:")
        print("1. Run this workflow again with the new ligand")
        print("2. Use the same protein and output to a different directory")
        print("3. Compare the flux reports to identify different binding preferences")


def run_uma_workflow():
    """Wrapper for the UMA workflow"""
    print_banner("UMA-OPTIMIZED WORKFLOW")
    print("This uses zero-copy GPU processing for maximum performance.")
    print("Best for Apple Silicon Macs and systems with unified memory.\n")

    # Initialize variables
    protein_file = None
    ligand_file = None
    loaded_params = None

    # Ask if user wants to use existing parameters first
    use_existing = input("Load parameters from existing simulation? (y/n): ").strip().lower()

    if use_existing == "y":
        params_file = input("Enter path to simulation_parameters.txt: ").strip()
        if os.path.exists(params_file):
            loaded_params = parse_simulation_parameters(params_file)
            if loaded_params:
                print("\nLoaded parameters from file:")
                for key, value in loaded_params.items():
                    print(f"  {key}: {value}")

                # Ask about protein and ligand files
                if "protein_file" in loaded_params and "ligand_file" in loaded_params:
                    print(f"\nLoaded protein: {loaded_params['protein_file']}")
                    print(f"Loaded ligand: {loaded_params['ligand_file']}")
                    use_same = (
                        input("\nUse the same protein and ligand files? (y/n): ").strip().lower()
                    )

                    if use_same == "y":
                        protein_file = loaded_params["protein_file"]
                        ligand_file = loaded_params["ligand_file"]
                    else:
                        # Will ask for new files below
                        protein_file = None
                        ligand_file = None
                else:
                    # If files weren't in the loaded params, use them if available
                    if "protein_file" in loaded_params:
                        protein_file = loaded_params["protein_file"]
                    if "ligand_file" in loaded_params:
                        ligand_file = loaded_params["ligand_file"]
        else:
            print(f"File not found: {params_file}")
            use_existing = "n"

    # Get input files if not loaded from parameters
    if not protein_file:
        protein_file = input("\nEnter protein PDB file: ").strip()

    # Check if protein file exists, if not, handle missing files
    if not os.path.exists(protein_file):
        print(f"Error: {protein_file} not found!")

        # Check if it's a path issue from loaded parameters
        if loaded_params and "protein_file" in loaded_params:
            print("\nThe protein file path from the parameters file doesn't exist.")
            print("This might be because the files were moved or you're on a different machine.")

            # Try to find the file in current directory
            protein_basename = os.path.basename(protein_file)
            if os.path.exists(protein_basename):
                print(f"\nFound '{protein_basename}' in current directory.")
                use_current = input("Use this file? (y/n): ").strip().lower()
                if use_current == "y":
                    protein_file = protein_basename
                else:
                    protein_file = input("Enter correct path to protein PDB file: ").strip()
            else:
                # Ask for directory containing the files
                print("\nPlease provide the directory containing your protein and ligand files.")
                new_dir = input(
                    "Enter directory path (or press Enter to manually input file paths): "
                ).strip()

                if new_dir:
                    # Check if user entered a file path instead of directory
                    if os.path.isfile(new_dir):
                        print(f"\nYou entered a file path. Using: {new_dir}")
                        protein_file = new_dir
                    elif os.path.isdir(new_dir):
                        # Try to find the protein file in the new directory
                        potential_protein = os.path.join(new_dir, protein_basename)
                        if os.path.exists(potential_protein):
                            print(f"Found protein file: {potential_protein}")
                            protein_file = potential_protein
                        else:
                            # List PDB files in the directory
                            pdb_files = [f for f in os.listdir(new_dir) if f.endswith(".pdb")]
                            if pdb_files:
                                print(f"\nFound {len(pdb_files)} PDB files in {new_dir}:")
                                for i, f in enumerate(pdb_files):
                                    print(f"  {i + 1}. {f}")
                                choice = input(
                                    "\nSelect protein file by number (or press Enter to input manually): "
                                ).strip()
                                if choice.isdigit() and 1 <= int(choice) <= len(pdb_files):
                                    protein_file = os.path.join(new_dir, pdb_files[int(choice) - 1])
                                else:
                                    protein_file = input(
                                        "Enter full path to protein PDB file: "
                                    ).strip()
                            else:
                                protein_file = input(
                                    "Enter full path to protein PDB file: "
                                ).strip()

                        # Update ligand file path if it's also from loaded parameters
                        if loaded_params and "ligand_file" in loaded_params and ligand_file:
                            ligand_basename = os.path.basename(ligand_file)
                            potential_ligand = os.path.join(new_dir, ligand_basename)
                            if os.path.exists(potential_ligand):
                                print(f"Found ligand file: {potential_ligand}")
                                ligand_file = potential_ligand
                        else:
                            print(f"\nPath not found: {new_dir}")
                        protein_file = input("Enter full path to protein PDB file: ").strip()
                else:
                    protein_file = input("Enter full path to protein PDB file: ").strip()

        # Final check
        if not os.path.exists(protein_file):
            print(f"Error: Still cannot find {protein_file}")
            return

    if not ligand_file:
        ligand_file = input("Enter ligand PDB file: ").strip()

    # Similar handling for ligand file
    if not os.path.exists(ligand_file):
        print(f"Error: {ligand_file} not found!")

        if loaded_params and "ligand_file" in loaded_params:
            # Try to use the directory from protein file if it was updated
            protein_dir = os.path.dirname(protein_file)
            ligand_basename = os.path.basename(ligand_file)
            potential_ligand = os.path.join(protein_dir, ligand_basename)

            if os.path.exists(potential_ligand):
                print(f"\nFound '{ligand_basename}' in the same directory as protein.")
                use_this = input("Use this file? (y/n): ").strip().lower()
                if use_this == "y":
                    ligand_file = potential_ligand
                else:
                    ligand_file = input("Enter correct path to ligand file: ").strip()
            else:
                # List potential ligand files
                ligand_extensions = [".pdb", ".pdbqt", ".mol2", ".sdf"]
                ligand_files = [
                    f
                    for f in os.listdir(protein_dir)
                    if any(f.endswith(ext) for ext in ligand_extensions)
                    and f != os.path.basename(protein_file)
                ]

                if ligand_files:
                    print(f"\nFound {len(ligand_files)} potential ligand files:")
                    for i, f in enumerate(ligand_files):
                        print(f"  {i + 1}. {f}")
                    choice = input(
                        "\nSelect ligand file by number (or press Enter to input manually): "
                    ).strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(ligand_files):
                        ligand_file = os.path.join(protein_dir, ligand_files[int(choice) - 1])
                    else:
                        ligand_file = input("Enter full path to ligand file: ").strip()
                else:
                    ligand_file = input("Enter full path to ligand file: ").strip()
        else:
            ligand_file = input("Enter correct path to ligand file: ").strip()

        # Final check
        if not os.path.exists(ligand_file):
            print(f"Error: Still cannot find {ligand_file}")
            return

    output_dir = (
        input("Output directory (default 'flux_analysis_uma'): ").strip() or "flux_analysis_uma"
    )

    # Continue with parameter confirmation if loaded
    print("\nSIMULATION PARAMETERS")

    if use_existing == "y" and loaded_params:
        confirm = input("\nUse these parameters? (y/n): ").strip().lower()
        if confirm == "y":
            # Use loaded parameters
            n_steps = loaded_params.get("n_steps", 200)
            n_iterations = loaded_params.get("n_iterations", 10)
            n_approaches = loaded_params.get("n_approaches", 10)
            approach_distance = loaded_params.get("approach_distance", 2.5)
            starting_distance = loaded_params.get("starting_distance", 20.0)
            n_rotations = loaded_params.get("n_rotations", 36)
            physiological_pH = loaded_params.get("physiological_pH", 7.4)
        else:
            use_existing = "n"  # Fall back to manual entry
    else:
        if use_existing == "y":
            print("Failed to parse parameters file.")
        use_existing = "n"

    if use_existing != "y":
        # Manual parameter entry
        print("\nEnter parameters manually (press Enter for defaults)\n")

        n_steps = input("Steps per trajectory (default 200): ").strip()
        n_steps = int(n_steps) if n_steps else 200

        n_iterations = input("Number of iterations (default 10): ").strip()
        n_iterations = int(n_iterations) if n_iterations else 10

        n_approaches = input("Number of approach angles (default 10): ").strip()
        n_approaches = int(n_approaches) if n_approaches else 10

        starting_distance = input("Starting distance in Angstroms (default 20.0): ").strip()
        starting_distance = float(starting_distance) if starting_distance else 20.0

        approach_distance = input(
            "Distance step between approaches in Angstroms (default 2.5): "
        ).strip()
        approach_distance = float(approach_distance) if approach_distance else 2.5

        n_rotations = input("Rotations per position (default 36): ").strip()
        n_rotations = int(n_rotations) if n_rotations else 36

        physiological_pH = input("pH for protonation states (default 7.4): ").strip()
        physiological_pH = float(physiological_pH) if physiological_pH else 7.4

    # Ask about saving trajectories (default: yes)
    save_trajectories_input = input("\nSave trajectory files? (Y/n): ").strip().lower()
    save_trajectories = save_trajectories_input != "n"  # Default to yes unless 'n' is entered

    # Ask about showing detailed interaction breakdown (default: yes)
    show_interactions_input = input("Show detailed interaction breakdown? (Y/n): ").strip().lower()
    show_interactions = show_interactions_input != "n"  # Default to yes unless 'n' is entered

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

    confirm = input("\nProceed with UMA-optimized analysis? (Y/n): ").strip().lower()
    if confirm == "n":
        print("Analysis cancelled.")
        return

    # Run fluxmd_uma as subprocess with parameters
    import subprocess

    cmd = [
        sys.executable,
        "fluxmd_uma.py",
        protein_file,
        ligand_file,
        "-o",
        output_dir,
        "-s",
        str(n_steps),
        "-i",
        str(n_iterations),
        "-a",
        str(n_approaches),
        "-d",
        str(starting_distance),
        "--approach-distance",
        str(approach_distance),
        "-r",
        str(n_rotations),
        "--ph",
        str(physiological_pH),
    ]

    # Add save trajectories flag if requested
    if save_trajectories:
        cmd.append("--save-trajectories")

    # Add interaction details flag if requested
    if show_interactions:
        cmd.append("--interaction-details")

    subprocess.run(cmd)


def run_smiles_converter():
    """Wrapper for the SMILES to PDB converter"""
    print_banner("SMILES TO PDB CONVERTER")
    smiles = input("Enter SMILES string: ").strip()
    if smiles:
        name = input("Enter output name: ").strip() or "ligand"

        print("\nConversion options:")
        print("1. NCI CACTUS (web service - fast but limited aromatic preservation)")
        print("2. OpenBabel standard (local - basic PDB)")
        print("3. OpenBabel PDBQT (local - BEST for aromatic preservation)")

        method = input("\nSelect method (1-3) [3]: ").strip() or "3"

        if method == "1":
            convert_smiles_to_pdb_cactus(smiles, name)
        elif method == "2":
            convert_smiles_to_pdb_openbabel(smiles, name)
        else:
            # Use enhanced PDBQT method
            convert_smiles_with_aromatics(smiles, name)


def run_dna_generator():
    """Wrapper for the DNA generator"""
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
    valid_bases = set("ATGC")
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
        if confirm != "y":
            return

    output_name = input("Enter output filename (default: dna_structure.pdb): ").strip()
    if not output_name:
        output_name = "dna_structure.pdb"

    # Ensure .pdb extension
    if not output_name.endswith(".pdb"):
        output_name += ".pdb"

    # Import the DNA builder
    try:
        from fluxmd.utils.dna_to_pdb import DNABuilder
    except ImportError:
        print("Error: Could not import DNA builder")
        return

    print(f"\nGenerating B-DNA structure for: {sequence}")
    print(f"Sequence length: {len(sequence)} bp")
    print(f"Double helix will contain:")
    print(f"  - {len(sequence) * 2} nucleotides total")
    print(f"  - ~{len(sequence) * 35} atoms per strand")
    print(f"  - Helix length: ~{len(sequence) * 3.38:.1f} Angstroms")

    try:
        builder = DNABuilder()
        builder.build_dna(sequence)
        builder.write_pdb(output_name)

        print(f"\nStructure successfully written to: {output_name}")
        print(f"  Total atoms: {len(builder.atoms)}")
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


def get_recommended_parameters(protein_atoms, dna_atoms, device):
    """Recommend simulation parameters based on system size"""
    protein_size = len(protein_atoms)
    dna_size = len(dna_atoms)
    total_size = protein_size + dna_size

    # Base recommendations on protein size
    if protein_size < 2000:  # Small protein
        n_iterations = 10
        n_approaches = 10
        n_steps = 200
        n_rotations = 36
    elif protein_size < 5000:  # Medium protein
        n_iterations = 10
        n_approaches = 8
        n_steps = 150
        n_rotations = 24
    else:  # Large protein
        n_iterations = 5
        n_approaches = 6
        n_steps = 100
        n_rotations = 18

    # Adjust for DNA length
    if dna_size > 5000:  # Long DNA
        n_approaches = min(n_approaches, 6)

    return {
        "n_iterations": n_iterations,
        "n_approaches": n_approaches,
        "n_steps": n_steps,
        "n_rotations": n_rotations,
    }


def run_protein_dna_uma_workflow():
    """Wrapper for the Protein-DNA UMA workflow"""
    print_banner("Protein-DNA Interaction Analysis (UMA Workflow)")

    try:
        # Ask if user wants to load existing parameters
        use_existing = input("\nLoad parameters from existing simulation? (y/n): ").strip().lower()

        # Initialize variables
        loaded_params = None
        dna_file = ""
        protein_file = ""

        if use_existing == "y":
            params_file = input("Enter path to simulation_parameters.txt: ").strip()
            if os.path.exists(params_file):
                loaded_params = parse_simulation_parameters(params_file)
                if loaded_params:
                    print("\nLoaded parameters from file:")
                    for key, value in loaded_params.items():
                        print(f"  {key}: {value}")

                    # Extract file paths if available
                    dna_file = loaded_params.get("dna_file", "")
                    protein_file = loaded_params.get("protein_file", "")

                    if (
                        dna_file
                        and protein_file
                        and os.path.exists(dna_file)
                        and os.path.exists(protein_file)
                    ):
                        print(f"\nDNA file: {dna_file}")
                        print(f"Protein file: {protein_file}")
                        use_files = input("Use these files? (y/n): ").strip().lower()
                        if use_files != "y":
                            dna_file = ""
                            protein_file = ""
                    else:
                        dna_file = ""
                        protein_file = ""
            else:
                print(f"Parameters file not found: {params_file}")
                loaded_params = None

        # Get input files if not loaded from parameters
        if not dna_file:
            dna_file = input("Enter path to DNA PDB file: ").strip()
        if not protein_file:
            protein_file = input("Enter path to Protein PDB file: ").strip()

        if not os.path.exists(dna_file) or not os.path.exists(protein_file):
            print("Error: One or both input files not found.")
            return

        output_dir = input("Enter output directory name [flux_results_dna]: ").strip()
        if not output_dir:
            output_dir = "flux_results_dna"

        # Parse files first to get atom counts
        from fluxmd.utils.pdb_parser import PDBParser

        parser = PDBParser()
        dna_atoms = parser.parse(dna_file, is_dna=True)
        protein_atoms = parser.parse(protein_file)

        if dna_atoms is None or protein_atoms is None:
            print("Error parsing input files.")
            return

        print(f"\nSystem size: {len(protein_atoms)} protein atoms, {len(dna_atoms)} DNA atoms")

        # Get device and recommended parameters
        device = get_device()
        recommended = get_recommended_parameters(protein_atoms, dna_atoms, device)

        # Check if we have loaded parameters and ask to use them
        if loaded_params and use_existing == "y":
            confirm = input("\nUse loaded parameters? (y/n): ").strip().lower()
            if confirm == "y":
                # Use loaded parameters
                n_iterations = loaded_params.get("n_iterations", recommended["n_iterations"])
                n_approaches = loaded_params.get("n_approaches", recommended["n_approaches"])
                n_steps = loaded_params.get("n_steps", recommended["n_steps"])
                n_rotations = loaded_params.get("n_rotations", recommended["n_rotations"])
                starting_distance = loaded_params.get("starting_distance", 20.0)
                physiological_pH = loaded_params.get("physiological_pH", 7.4)

                print("\nUsing loaded parameters:")
                print(f"  Number of iterations: {n_iterations}")
                print(f"  Number of approaches: {n_approaches}")
                print(f"  Steps per trajectory: {n_steps}")
                print(f"  Rotations per position: {n_rotations}")
                print(f"  Starting distance: {starting_distance} Å")
                print(f"  Physiological pH: {physiological_pH}")
            else:
                loaded_params = None  # Fall through to manual selection

        # If not using loaded parameters, select analysis mode
        if not loaded_params or confirm != "y":
            print("\nSelect analysis mode:")
            print("1. Custom parameters (recommended)")
            print("2. Quick test (minimal sampling)")
            print("3. Standard analysis (balanced)")
            print("4. Deep analysis (comprehensive)")

            mode = input("Choice [1]: ").strip() or "1"

            if mode == "2":  # Quick test
                n_iterations = 2
                n_approaches = 4
                n_steps = 50
                n_rotations = 12
            elif mode == "3":  # Standard
                n_iterations = recommended["n_iterations"]
                n_approaches = recommended["n_approaches"]
                n_steps = recommended["n_steps"]
                n_rotations = recommended["n_rotations"]
            elif mode == "4":  # Deep analysis
                n_iterations = recommended["n_iterations"] * 2
                n_approaches = min(20, recommended["n_approaches"] + 5)
                n_steps = min(500, recommended["n_steps"] + 100)
                n_rotations = min(72, recommended["n_rotations"] * 2)
            else:  # Custom mode
                print("\nSimulation Parameters (recommendations based on system size):")
                n_iterations = int(
                    input(f"  Number of iterations [{recommended['n_iterations']}]: ")
                    or recommended["n_iterations"]
                )
                n_approaches = int(
                    input(f"  Number of approach angles [{recommended['n_approaches']}]: ")
                    or recommended["n_approaches"]
                )
                n_steps = int(
                    input(f"  Steps per trajectory [{recommended['n_steps']}]: ")
                    or recommended["n_steps"]
                )
                n_rotations = int(
                    input(f"  Rotations per position [{recommended['n_rotations']}]: ")
                    or recommended["n_rotations"]
                )

            # Additional parameters
            print("\nAdditional parameters:")
            starting_distance = float(input("Starting distance in Å [20.0]: ") or 20.0)
            physiological_pH = float(input("Physiological pH [7.4]: ") or 7.4)

        # Show analysis depth
        total_configs = n_iterations * n_approaches * n_steps * n_rotations
        print(f"\nTotal configurations to evaluate: {total_configs:,}")

        if total_configs < 100_000:
            print("Analysis depth: Light")
        elif total_configs < 1_000_000:
            print("Analysis depth: Moderate")
        elif total_configs < 10_000_000:
            print("Analysis depth: Comprehensive")
        else:
            print("Analysis depth: Very comprehensive")
            proceed = input("\n⚠️  This is a very comprehensive analysis. Continue? (y/n): ").lower()
            if proceed != "y":
                print("Analysis cancelled.")
                return

        # Ask about trajectory visualization
        save_trajectories = input("\nSave trajectory visualizations? (y/n) [y]: ").strip().lower()
        save_trajectories = save_trajectories != "n"  # Default to yes

        # Call the core workflow function
        from fluxmd.core.protein_dna_workflow import run_protein_dna_workflow

        run_protein_dna_workflow(
            dna_file=dna_file,
            protein_file=protein_file,
            output_dir=output_dir,
            n_iterations=n_iterations,
            n_steps=n_steps,
            n_approaches=n_approaches,
            starting_distance=starting_distance,
            n_rotations=n_rotations,
            physiological_pH=physiological_pH,
            force_cpu=False,
            save_trajectories=save_trajectories,
        )
    except Exception as e:
        print(f"\nAn error occurred during the Protein-DNA workflow: {e}")


def run_matryoshka_workflow():
    """Run the Matryoshka trajectory workflow with physics-based surface dynamics"""
    print_banner("MATRYOSHKA TRAJECTORY ANALYSIS")
    print("Advanced physics-based trajectory generation with Brownian dynamics")
    print("\nFeatures:")
    print("  ✓ Solvent-excluded surface (SES) based trajectories")
    print("  ✓ Physical Brownian-Langevin dynamics with proper diffusion")
    print("  ✓ Multi-layer exploration (nested Russian doll surfaces)")
    print("  ✓ PCA-based anchor points for objective trajectory endpoints")
    print("  ✓ GPU-accelerated energy calculations (when available)")

    try:
        # Check dependencies
        try:
            import skimage

            print("\n✓ scikit-image detected for marching cubes")
        except ImportError:
            print("\n⚠️  Warning: scikit-image not found. Surface mesh generation may be limited.")
            print("   Install with: pip install scikit-image")

        # Get protein file
        print("\n" + "=" * 80)
        print("Enter protein structure file (PDB format):")
        protein_file = input("Protein file path: ").strip()
        if not os.path.exists(protein_file):
            raise FileNotFoundError(f"Protein file not found: {protein_file}")

        # Get ligand file
        print("\nEnter ligand structure file (PDB format):")
        ligand_file = input("Ligand file path: ").strip()
        if not os.path.exists(ligand_file):
            raise FileNotFoundError(f"Ligand file not found: {ligand_file}")

        # Load existing parameters or collect new ones
        param_file = "simulation_parameters.txt"
        print(f"\nChecking for existing parameters in {param_file}...")

        if os.path.exists(param_file):
            use_existing = input("Found existing parameters. Use them? (y/n): ").lower() == "y"
            if use_existing:
                params = load_parameters(param_file)
                n_layers = params.get("n_layers", None)
                n_trajectories_per_layer = params.get("n_trajectories_per_layer", 10)
                layer_step = params.get("layer_step", 1.5)
                k_surf = params.get("k_surf", 2.0)
                k_guid = params.get("k_guid", 0.5)
                probe_radius = params.get("probe_radius", 0.75)
                physiological_pH = params.get("physiological_pH", 7.4)
                n_workers = params.get("n_workers", None)
                checkpoint_dir = params.get("checkpoint_dir", "./matryoshka_checkpoints")
            else:
                use_existing = False
        else:
            use_existing = False

        if not use_existing:
            # Collect Matryoshka-specific parameters
            print("\n" + "=" * 80)
            print("MATRYOSHKA PARAMETERS")
            print("=" * 80)

            print("\nLayer Configuration:")
            n_layers = input("Number of nested layers (default: auto-detect): ").strip()
            n_layers = int(n_layers) if n_layers else None

            n_trajectories_per_layer = int(input("Trajectories per layer (default 10): ") or "10")
            layer_step = float(input("Layer step size in Å (default 1.5): ") or "1.5")

            print("\nPhysics Parameters:")
            probe_radius = float(input("SES probe radius in Å (default 0.75): ") or "0.75")
            k_surf = float(
                input("Surface adherence force constant (default 2.0 kcal/mol/Å²): ") or "2.0"
            )
            k_guid = float(input("Guidance force constant (default 0.5 kcal/mol/Å²): ") or "0.5")

            print("\nComputation Settings:")
            physiological_pH = float(input("Physiological pH (default 7.4): ") or "7.4")
            n_workers_input = input("Number of parallel workers (default: auto): ").strip()
            n_workers = parse_workers(n_workers_input)
            print(f"  Workers set to: {format_workers_info(n_workers)}")

            checkpoint_dir = input(
                "Checkpoint directory (default ./matryoshka_checkpoints): "
            ).strip()
            checkpoint_dir = checkpoint_dir or "./matryoshka_checkpoints"

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"matryoshka_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nOutput directory: {output_dir}")

        # Save parameters
        save_matryoshka_parameters(
            output_dir,
            protein_file,
            ligand_file,
            n_layers,
            n_trajectories_per_layer,
            layer_step,
            probe_radius,
            k_surf,
            k_guid,
            physiological_pH,
            n_workers,
            checkpoint_dir,
        )

        # Run the Matryoshka analysis
        print("\n" + "=" * 80)
        print("STARTING MATRYOSHKA TRAJECTORY GENERATION")
        print("=" * 80)

        # Import here to avoid startup delays
        from fluxmd.core.trajectory_generator import (
            ProteinLigandFluxAnalyzer as TrajectoryGenerator,
        )
        from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator

        # Initialize trajectory generator (but we'll use it for helper methods)
        traj_gen = TrajectoryGenerator(
            protein_file, ligand_file, output_dir, energy_function="ref15"
        )

        # Load molecular structures
        print("\nLoading molecular structures...")

        # Use PDB parser to load structures
        parser = PDBParser()
        protein_df = parser.parse(protein_file)
        ligand_df = parser.parse(ligand_file)

        # Extract coordinates and prepare atom dictionaries
        protein_coords = protein_df[["x", "y", "z"]].values
        ligand_coords = ligand_df[["x", "y", "z"]].values

        # Get atomic properties
        protein_atoms = {
            "coords": protein_coords,
            "names": protein_df["name"].values,
            "radii": traj_gen.get_vdw_radii(protein_df),
            "masses": traj_gen.get_atomic_masses(protein_df),
            "resnames": protein_df.get(
                "resName", protein_df.get("resname", [""] * len(protein_df))
            ).values,
        }

        ligand_atoms = {
            "coords": ligand_coords,
            "names": ligand_df["name"].values,
            "masses": traj_gen.get_atomic_masses(ligand_df),
        }

        # Set up Matryoshka parameters
        matryoshka_params = {
            "T": 298.15,  # Room temperature
            "viscosity": 0.00089,  # Water viscosity in Pa·s
            "probe_radius": probe_radius,
            "layer_step": layer_step,
            "k_surf": k_surf,
            "k_guid": k_guid,
            "n_workers": n_workers,
            "checkpoint_dir": checkpoint_dir,
            "vdw_cutoff": 12.0,
        }

        # Check for GPU availability
        gpu_available = check_gpu_availability()
        if gpu_available:
            print("\n✓ GPU detected - will use for energy calculations")
            matryoshka_params["gpu_device"] = 0
        else:
            print("\n⚠️  No GPU detected - using CPU for energy calculations")

        # Initialize Matryoshka generator
        print("\nInitializing Matryoshka trajectory generator...")
        matryoshka_gen = MatryoshkaTrajectoryGenerator(
            protein_atoms, ligand_atoms, matryoshka_params
        )

        # Run trajectory generation
        print(f"\nGenerating trajectories...")
        print(f"  Layers: {n_layers if n_layers else 'auto-detect'}")
        print(f"  Trajectories per layer: {n_trajectories_per_layer}")

        start_time = time.time()
        trajectories = matryoshka_gen.run(n_layers=n_layers, n_iterations=n_trajectories_per_layer)
        end_time = time.time()

        print(
            f"\n✓ Generated {len(trajectories)} trajectories in {end_time - start_time:.1f} seconds"
        )

        # Convert and save trajectories
        print("\nSaving trajectory data...")
        save_matryoshka_trajectories(output_dir, trajectories, ligand_atoms)

        # Run energy analysis if REF15 calculator available
        if hasattr(traj_gen, "ref15_calculator"):
            print("\nRunning REF15 energy analysis...")
            analyze_matryoshka_energies(output_dir, trajectories, matryoshka_gen)

        # Generate analysis report
        print("\nGenerating analysis report...")
        generate_matryoshka_report(output_dir, trajectories, matryoshka_gen)

        print("\n" + "=" * 80)
        print("MATRYOSHKA ANALYSIS COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"\nAn error occurred during Matryoshka analysis: {e}")
        import traceback

        traceback.print_exc()


def save_matryoshka_parameters(
    output_dir,
    protein_file,
    ligand_file,
    n_layers,
    n_trajectories_per_layer,
    layer_step,
    probe_radius,
    k_surf,
    k_guid,
    physiological_pH,
    n_workers,
    checkpoint_dir,
):
    """Save Matryoshka simulation parameters"""
    param_file = os.path.join(output_dir, "matryoshka_parameters.txt")
    with open(param_file, "w") as f:
        f.write("MATRYOSHKA TRAJECTORY PARAMETERS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: Matryoshka (physics-based surface dynamics)\n")
        f.write(f"Protein file: {protein_file}\n")
        f.write(f"Ligand file: {ligand_file}\n")
        f.write(f"Number of layers: {n_layers if n_layers else 'auto-detect'}\n")
        f.write(f"Trajectories per layer: {n_trajectories_per_layer}\n")
        f.write(f"Layer step size: {layer_step} Å\n")
        f.write(f"SES probe radius: {probe_radius} Å\n")
        f.write(f"Surface force constant: {k_surf} kcal/mol/Å²\n")
        f.write(f"Guidance force constant: {k_guid} kcal/mol/Å²\n")
        f.write(f"Physiological pH: {physiological_pH}\n")
        f.write(f"Parallel workers: {n_workers if n_workers else 'auto'}\n")
        f.write(f"Checkpoint directory: {checkpoint_dir}\n")


def save_matryoshka_trajectories(output_dir, trajectories, ligand_atoms):
    """Save Matryoshka trajectories in FluxMD-compatible format"""
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    for i, traj in enumerate(trajectories):
        layer_idx = traj.get("layer_idx", 0)
        iter_idx = traj.get("iteration_idx", 0)

        # Save position trajectory
        pos_file = os.path.join(traj_dir, f"trajectory_L{layer_idx}_I{iter_idx}.csv")

        # Convert positions and quaternions to full ligand coordinates
        positions = []
        for pos, quat in zip(traj["pos"], traj["quat"]):
            # Here we would reconstruct full ligand coordinates
            # For now, just save COM positions
            positions.append(pos)

        # Save as CSV
        import pandas as pd

        df = pd.DataFrame(positions, columns=["x", "y", "z"])
        df["time_ps"] = traj["time"]
        df.to_csv(pos_file, index=False)

    print(f"  Saved {len(trajectories)} trajectory files")


def analyze_matryoshka_energies(output_dir, trajectories, generator):
    """Analyze energies from Matryoshka trajectories"""
    energy_data = []

    for traj in trajectories:
        if "sampled_energies" in traj:
            layer = traj["layer_idx"]
            energies = traj["sampled_energies"]
            energy_data.append(
                {
                    "layer": layer,
                    "mean_energy": np.mean(energies),
                    "min_energy": np.min(energies),
                    "std_energy": np.std(energies),
                }
            )

    if energy_data:
        import pandas as pd

        df = pd.DataFrame(energy_data)
        df.to_csv(os.path.join(output_dir, "layer_energies.csv"), index=False)
        print(f"  Saved energy analysis for {len(energy_data)} trajectories")


def generate_matryoshka_report(output_dir, trajectories, generator):
    """Generate comprehensive Matryoshka analysis report"""
    report_file = os.path.join(output_dir, "matryoshka_report.txt")

    with open(report_file, "w") as f:
        f.write("MATRYOSHKA TRAJECTORY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # System information
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Protein atoms: {len(generator.protein_atoms['coords'])}\n")
        f.write(f"Ligand atoms: {len(generator.ligand_atoms['coords'])}\n")
        f.write(f"Ligand radius of gyration: {generator.ligand_sphere['radius']:.2f} Å\n")
        f.write(f"Ligand mass: {generator.ligand_sphere['mass']:.1f} amu\n")
        f.write(
            f"Anchor separation: {np.linalg.norm(generator.anchors[1] - generator.anchors[0]):.1f} Å\n"
        )
        f.write(f"Geodesic distance: {generator.geodesic_distance:.1f} Å\n")
        f.write(f"Maximum useful layers: {generator.max_layers}\n\n")

        # Surface information
        f.write("SURFACE MESH INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Base surface vertices: {len(generator.base_surface.vertices)}\n")
        f.write(f"Base surface faces: {len(generator.base_surface.faces)}\n")
        f.write(f"Layer step size: {generator.layer_step} Å\n\n")

        # Trajectory statistics
        f.write("TRAJECTORY STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total trajectories: {len(trajectories)}\n")

        # Group by layer
        layer_counts = {}
        for traj in trajectories:
            layer = traj.get("layer_idx", 0)
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        f.write("\nTrajectories per layer:\n")
        for layer in sorted(layer_counts.keys()):
            f.write(f"  Layer {layer}: {layer_counts[layer]} trajectories\n")

        # Path length statistics
        path_lengths = []
        for traj in trajectories:
            if len(traj["pos"]) > 1:
                # Calculate total path length
                positions = np.array(traj["pos"])
                displacements = np.diff(positions, axis=0)
                path_length = np.sum(np.linalg.norm(displacements, axis=1))
                path_lengths.append(path_length)

        if path_lengths:
            f.write(f"\nPath length statistics:\n")
            f.write(f"  Mean: {np.mean(path_lengths):.1f} Å\n")
            f.write(f"  Std: {np.std(path_lengths):.1f} Å\n")
            f.write(f"  Min: {np.min(path_lengths):.1f} Å\n")
            f.write(f"  Max: {np.max(path_lengths):.1f} Å\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"  Generated analysis report: {report_file}")


def display_main_menu():
    """Display the main menu and return the user's choice"""
    print_banner("FLUXMD - PROTEIN-LIGAND FLUX ANALYSIS")

    print("Welcome to FluxMD - GPU-accelerated binding site prediction")
    print("\nOptions:")
    print("1. Run Matryoshka trajectory analysis ⟵ default")
    print("2. Convert SMILES to PDB (CACTUS with aromatics or OpenBabel)")
    print("3. Generate DNA structure from sequence")
    print("4. Protein-DNA Interaction Analysis (UMA)")
    print("5. Legacy workflows ▶")
    print("6. Exit")
    print("=" * 80)

    choice = input("Enter your choice [1-6]: ").strip()
    return choice


def handle_main_menu(choice):
    """Handle the user's choice and execute the corresponding workflow"""
    if choice == "1":
        run_matryoshka_workflow()
    elif choice == "2":
        run_smiles_converter()
    elif choice == "3":
        run_dna_generator()
    elif choice == "4":
        run_protein_dna_uma_workflow()
    elif choice == "5":
        print("\nLegacy workflows:")
        print("  a. Standard Cocoon (1.x)")
        print("  b. UMA Cocoon (1.x zero-copy GPU)")
        legacy = input("Select [a/b] or press Enter to cancel: ").lower()
        if legacy == "a":
            run_complete_workflow()
        elif legacy == "b":
            run_uma_workflow()
    elif choice == "6":
        print("\nExiting FluxMD. Goodbye!")
        sys.exit()
    else:
        print("Invalid choice!")
        main()


def run_batch_mode(config: dict):
    """Run FluxMD in batch mode with configuration file."""
    mode = config["mode"]

    print_banner(f"FLUXMD - {mode.upper()} MODE (BATCH)")
    print(f"Configuration loaded from: {config.get('_config_file', 'config')}")

    if mode == "matryoshka":
        # Set up parameters from config
        params = {
            "protein_file": config["protein_file"],
            "ligand_file": config["ligand_file"],
            "output_dir": config.get(
                "output_dir", f"matryoshka_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            "n_layers": config.get("n_layers", None),
            "n_trajectories_per_layer": config.get("n_trajectories_per_layer", 10),
            "layer_step": config.get("layer_step", 1.5),
            "probe_radius": config.get("probe_radius", 0.75),
            "k_surf": config.get("k_surf", 2.0),
            "k_guid": config.get("k_guid", 0.5),
            "physiological_pH": config.get("physiological_pH", 7.4),
            "n_workers": parse_workers(config.get("n_workers")),
            "checkpoint_dir": config.get("checkpoint_dir", "./matryoshka_checkpoints"),
            "groove_preference": config.get("groove_preference", "major"),
            "temperature": config.get("temperature", 298.15),
            "viscosity": config.get("viscosity", 0.00089),
            "max_steps": config.get("max_steps", 1_000_000),
            "use_gpu": config.get("use_gpu", True),
        }

        # Run Matryoshka workflow directly
        run_matryoshka_batch(params)

    elif mode == "cocoon":
        # Run standard workflow
        params = {
            "protein_file": config["protein_file"],
            "ligand_file": config["ligand_file"],
            "output_dir": config.get(
                "output_dir", f"fluxmd_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            "n_iterations": config.get("n_iterations", 100),
            "n_steps": config.get("n_steps", 100),
            "n_approaches": config.get("n_approaches", 5),
            "n_rotations": config.get("n_rotations", 36),
            "starting_distance": config.get("starting_distance", 15.0),
            "approach_distance": config.get("approach_distance", 2.5),
            "physiological_pH": config.get("physiological_pH", 7.4),
            "save_trajectories": config.get("save_trajectories", True),
            "use_gpu": config.get("use_gpu", True),
        }

        run_cocoon_batch(params)

    else:
        print(f"Batch mode not yet implemented for mode: {mode}")
        sys.exit(1)


def run_matryoshka_batch(params: dict):
    """Run Matryoshka workflow in batch mode."""
    try:
        # Import required modules
        from fluxmd.core.trajectory_generator import (
            ProteinLigandFluxAnalyzer as TrajectoryGenerator,
        )
        from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
        from fluxmd.utils.pdb_parser import PDBParser

        # Set up output directory
        os.makedirs(params["output_dir"], exist_ok=True)

        # Save parameters
        save_matryoshka_parameters(
            params["output_dir"],
            params["protein_file"],
            params["ligand_file"],
            params["n_layers"],
            params["n_trajectories_per_layer"],
            params["layer_step"],
            params["probe_radius"],
            params["k_surf"],
            params["k_guid"],
            params["physiological_pH"],
            params["n_workers"],
            params["checkpoint_dir"],
        )

        print("\nLoading molecular structures...")

        # Initialize trajectory generator for helper methods
        traj_gen = TrajectoryGenerator(
            params["protein_file"],
            params["ligand_file"],
            params["output_dir"],
            energy_function="ref15",
        )

        # Load structures
        parser = PDBParser()
        protein_df = parser.parse(params["protein_file"])
        ligand_df = parser.parse(params["ligand_file"])

        # Prepare atom dictionaries
        protein_atoms = {
            "coords": protein_df[["x", "y", "z"]].values,
            "names": protein_df["name"].values,
            "radii": traj_gen.get_vdw_radii(protein_df),
            "masses": traj_gen.get_atomic_masses(protein_df),
            "resnames": protein_df.get(
                "resName", protein_df.get("resname", [""] * len(protein_df))
            ).values,
        }

        ligand_atoms = {
            "coords": ligand_df[["x", "y", "z"]].values,
            "names": ligand_df["name"].values,
            "masses": traj_gen.get_atomic_masses(ligand_df),
        }

        # Set up Matryoshka parameters
        matryoshka_params = {
            "T": params["temperature"],
            "viscosity": params["viscosity"],
            "probe_radius": params["probe_radius"],
            "layer_step": params["layer_step"],
            "k_surf": params["k_surf"],
            "k_guid": params["k_guid"],
            "n_workers": params["n_workers"],
            "checkpoint_dir": params["checkpoint_dir"],
            "vdw_cutoff": 12.0,
            "groove_preference": params["groove_preference"],
            "max_steps": params["max_steps"],
            "use_ref15": params.get("use_gpu", True),
            "pH": params["physiological_pH"],
        }

        # Check GPU
        if params["use_gpu"] and check_gpu_availability():
            print("✓ GPU detected - will use for energy calculations")
            matryoshka_params["gpu_device"] = 0
        else:
            print("⚠️ Running in CPU mode")

        # Initialize and run
        print("\nInitializing Matryoshka trajectory generator...")
        matryoshka_gen = MatryoshkaTrajectoryGenerator(
            protein_atoms, ligand_atoms, matryoshka_params
        )

        print("\nGenerating trajectories...")
        start_time = time.time()
        trajectories = matryoshka_gen.run(
            n_layers=params["n_layers"], n_iterations=params["n_trajectories_per_layer"]
        )
        end_time = time.time()

        print(
            f"\n✓ Generated {len(trajectories)} trajectories in {end_time - start_time:.1f} seconds"
        )

        # Save results
        save_matryoshka_trajectories(params["output_dir"], trajectories, ligand_atoms)
        generate_matryoshka_report(params["output_dir"], trajectories, matryoshka_gen)

        print(f"\nResults saved to: {params['output_dir']}")

    except Exception as e:
        print(f"\nError in batch mode: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_cocoon_batch(params: dict):
    """Run standard cocoon workflow in batch mode."""
    # Implementation similar to run_complete_workflow but with params from config
    print("Cocoon batch mode implementation pending...")
    # TODO: Implement cocoon batch mode


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FluxMD - GPU-accelerated binding site prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python fluxmd.py
  
  # Batch mode with config file
  python fluxmd.py --config matryoshka_config.yaml
  
  # Dry run to check parameters
  python fluxmd.py --config matryoshka_config.yaml --dry-run
  
  # Create example configuration
  python fluxmd.py --create-example-config
        """,
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML or JSON)")

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print derived constants and exit without running",
    )

    parser.add_argument(
        "--create-example-config",
        action="store_true",
        help="Create an example configuration file and exit",
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (requires --config)",
    )

    return parser.parse_args()


def main():
    """Main entry point with argument parsing."""
    args = parse_arguments()

    # Handle example config creation
    if args.create_example_config:
        create_example_config()
        sys.exit(0)

    # Handle config file mode
    if args.config:
        try:
            config = load_config(args.config)
            config["_config_file"] = args.config

            # Dry run mode
            if args.dry_run:
                print_banner("FLUXMD - DRY RUN MODE")
                print(f"\nConfiguration file: {args.config}")
                print(f"Mode: {config['mode']}")
                print(f"Protein: {config['protein_file']}")
                print(f"Ligand: {config['ligand_file']}")
                print_derived_constants(config)
                sys.exit(0)

            # Run in batch mode
            run_batch_mode(config)
            sys.exit(0)

        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)

    # Check for non-interactive without config
    if args.non_interactive:
        print("Error: --non-interactive requires --config")
        sys.exit(1)

    # Default interactive mode
    while True:
        choice = display_main_menu()
        handle_main_menu(choice)
        input("\nPress Enter to return to the main menu...")
        # Clear screen for better UX
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
