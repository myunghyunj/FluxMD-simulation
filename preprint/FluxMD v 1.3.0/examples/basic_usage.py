#!/usr/bin/env python3
"""
Comprehensive FluxMD Usage Examples

This script demonstrates various FluxMD workflows including:
1. Basic protein-ligand analysis
2. p53-DNA complex analysis
3. SMILES conversion and analysis
4. DNA structure generation
5. Parameter optimization strategies
"""

import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For running from installed package, remove the sys.path line above


def run_basic_protein_ligand():
    """Basic protein-ligand binding site analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Protein-Ligand Analysis")
    print("=" * 60)

    # Example with ubiquitin and benzene
    protein_file = "examples/data/1ubq.pdb"  # Ubiquitin
    ligand_file = "examples/data/benzene.pdb"
    output_dir = "results/ubiquitin_benzene"

    print("""
This example analyzes ubiquitin (small protein) with benzene (aromatic ligand).
Expected results: Benzene should bind to the hydrophobic patch (Leu8, Ile44, Val70).

To run this example:
1. Download ubiquitin: https://www.rcsb.org/structure/1UBQ
2. Create benzene.pdb using SMILES (see Example 3)
3. Execute: fluxmd-uma 1ubq.pdb benzene.pdb -o results/
""")

    # Quick parameters for testing
    print("\nQuick test parameters:")
    print("fluxmd-uma protein.pdb ligand.pdb --steps 50 --iterations 5")

    # Production parameters
    print("\nProduction parameters:")
    print("fluxmd-uma protein.pdb ligand.pdb --steps 200 --iterations 100 --rotations 36")


def run_p53_dna_analysis():
    """Demonstrate p53-DNA complex analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: p53-DNA Recognition Complex")
    print("=" * 60)

    # Path to p53-DNA example files
    p53_dir = "examples/p53 and consensus binding site included DNA"
    p53_file = os.path.join(p53_dir, "p53_dbd.pdb")
    dna_file = os.path.join(p53_dir, "consensus_dna.pdb")
    output_dir = "results/p53_dna_complex"

    print(f"""
This example analyzes the p53 tumor suppressor binding to its consensus DNA sequence.
p53 is the "guardian of the genome" - mutations in p53 are found in >50% of cancers.

Files provided:
- {p53_file}
- {dna_file}

The consensus DNA sequence contains the p53 response element:
5'-RRRCWWGYYY-3' (R=purine, Y=pyrimidine, W=A/T)

To run this analysis:
""")

    # Method 1: Command line (recommended)
    print("\nMethod 1 - Command line (recommended for DNA):")
    print(f"cd '{p53_dir}'")
    print("fluxmd-uma p53_dbd.pdb consensus_dna.pdb -o p53_results/ --save-trajectories")

    # Method 2: Interactive mode
    print("\nMethod 2 - Interactive mode:")
    print("1. Run: fluxmd")
    print("2. Choose option 1 (Standard workflow)")
    print("3. Enter protein file: p53_dbd.pdb")
    print("4. Enter ligand file: consensus_dna.pdb")
    print("5. Use default parameters or customize")

    print("""
Expected key interactions:
- Arg248: Major groove contact (frequently mutated in cancer)
- Arg273: Major groove contact (hotspot mutation)
- Lys120: Minor groove contact
- Cys176, Cys238, Cys242, His179: Zinc coordination

DNA-specific features (automatic):
- Linear molecule detection
- Cylindrical trajectory generation
- Surface distance calculation
- Full backbone coverage
""")


def demonstrate_smiles_conversion():
    """Show how to convert SMILES to PDB for ligand preparation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: SMILES to PDB Conversion")
    print("=" * 60)

    print("""
FluxMD can convert SMILES strings to 3D structures for analysis.
This is useful when you don't have a PDB file for your ligand.

Interactive method (option 3 in main menu):
""")

    print("1. Run: fluxmd")
    print("2. Choose option 3 (SMILES converter)")
    print("3. Enter SMILES string")
    print("4. Enter output name")

    print("\nCommon SMILES examples:")
    print("- Benzene: c1ccccc1")
    print("- Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O")
    print("- Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    print("- ATP: C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N")

    print("\nProgrammatic method:")
    print("""
from fluxmd.utils.smiles_to_pdb import convert_smiles_to_pdb_cactus

# Convert SMILES to PDB
pdb_file = convert_smiles_to_pdb_cactus("c1ccccc1", "benzene")

# Use in analysis
fluxmd-uma protein.pdb benzene.pdb -o results/
""")


def generate_dna_structure():
    """Demonstrate DNA structure generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: DNA Structure Generation")
    print("=" * 60)

    print("""
FluxMD can generate B-form DNA structures from sequences.
This is useful for studying protein-DNA interactions.

Interactive method (option 4 in main menu):
""")

    print("1. Run: fluxmd")
    print("2. Choose option 4 (DNA generator)")
    print("3. Enter DNA sequence (e.g., ATCGATCG)")
    print("4. Enter output filename")

    print("\nCommand-line method:")
    print("fluxmd-dna GCGATCGC -o my_dna.pdb")

    print("\nExample sequences:")
    print("- p53 consensus: GGGCATGTCC")
    print("- EcoRI site: GAATTC")
    print("- AT-rich: AAAATTTT")
    print("- GC-rich: GGGGCCCC")

    print("""
Note: The generator creates:
- B-form double helix
- Proper base pairing (A-T, G-C)
- Sugar-phosphate backbone
- CONECT records for visualization
""")


def show_parameter_optimization():
    """Demonstrate parameter optimization strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Parameter Optimization")
    print("=" * 60)

    print("""
FluxMD parameters can be optimized for different scenarios:

1. QUICK SCREENING (5 minutes):
   fluxmd-uma protein.pdb ligand.pdb \\
     --steps 50 \\
     --iterations 5 \\
     --approaches 3 \\
     --rotations 12

2. STANDARD ANALYSIS (30 minutes):
   fluxmd-uma protein.pdb ligand.pdb \\
     --steps 100 \\
     --iterations 10 \\
     --approaches 5 \\
     --rotations 36

3. PUBLICATION QUALITY (2+ hours):
   fluxmd-uma protein.pdb ligand.pdb \\
     --steps 500 \\
     --iterations 20 \\
     --approaches 10 \\
     --rotations 72 \\
     --save-trajectories

4. DNA-SPECIFIC OPTIMIZATION:
   fluxmd-uma protein.pdb dna.pdb \\
     --steps 200 \\
     --iterations 10 \\
     --step-size 15 \\    # Smaller steps for groove sampling
     --approaches 8       # More angles for linear molecule

5. LARGE SYSTEMS (>10,000 atoms):
   # Use standard pipeline for memory efficiency
   fluxmd  # Choose option 1
   
6. pH-DEPENDENT ANALYSIS:
   # Physiological pH (default)
   fluxmd-uma protein.pdb ligand.pdb --ph 7.4
   
   # Acidic conditions (lysosome)
   fluxmd-uma protein.pdb ligand.pdb --ph 4.5
   
   # Basic conditions
   fluxmd-uma protein.pdb ligand.pdb --ph 9.0
""")


def analyze_results():
    """Guide for interpreting FluxMD results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Interpreting Results")
    print("=" * 60)

    print("""
FluxMD generates several output files:

1. processed_flux_data.csv
   - Primary results file
   - Columns: residue_index, residue_name, average_flux, std_flux, ci_lower_95, ci_upper_95
   - Sort by average_flux to find top binding sites

2. {protein}_flux_report.txt
   - Human-readable summary
   - Top 10 binding sites with confidence intervals
   - Statistical summary

3. {protein}_trajectory_flux_analysis.png
   - Visual heatmap of binding sites
   - Flux magnitude across protein sequence
   - Error bars show confidence intervals

4. iteration_*/ directories
   - Per-iteration trajectory data
   - Detailed interaction logs
   - Trajectory visualizations (if --save-trajectories used)

INTERPRETING FLUX VALUES:
- Flux > 1.0: Primary binding site (strong interaction)
- Flux 0.5-1.0: Secondary binding site
- Flux 0.2-0.5: Weak/transient interactions
- Flux < 0.2: No significant interaction

QUALITY INDICATORS:
- Narrow confidence intervals: Reliable binding site
- Wide confidence intervals: Dynamic or flexible region
- Consistent flux across iterations: Robust interaction
- Variable flux: Conformational changes or sampling issues

BIOLOGICAL INSIGHTS:
- Clusters of high flux: Binding pocket
- Isolated high flux: Specific contact points
- Path of moderate flux: Allosteric communication
- Flux at known sites: Validation of method
""")


def main():
    """Run all examples with user guidance."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     FluxMD Usage Examples                      ║
║                                                               ║
║  Learn how to use FluxMD for biomolecular binding analysis   ║
╚═══════════════════════════════════════════════════════════════╝
""")

    examples = [
        ("Basic Protein-Ligand Analysis", run_basic_protein_ligand),
        ("p53-DNA Recognition Complex", run_p53_dna_analysis),
        ("SMILES to PDB Conversion", demonstrate_smiles_conversion),
        ("DNA Structure Generation", generate_dna_structure),
        ("Parameter Optimization", show_parameter_optimization),
        ("Interpreting Results", analyze_results),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("7. Run all examples")
    print("0. Exit")

    while True:
        try:
            choice = input("\nSelect an example (0-7): ").strip()

            if choice == "0":
                print("Exiting...")
                break
            elif choice == "7":
                for name, func in examples:
                    func()
                print("\n" + "=" * 60)
                print("All examples completed!")
                break
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    examples[idx][1]()
                else:
                    print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
