#!/usr/bin/env python3
"""
Continue FluxMD analysis from where it failed
"""

import os
import sys

# Add FluxMD to path
fluxmd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, fluxmd_dir)

# Now import from the correct location
from fluxmd.analysis.flux_analyzer import TrajectoryFluxAnalyzer
import numpy as np
import pandas as pd


def continue_from_iterations(output_dir, protein_file, protein_name="protein"):
    """Continue analysis using already processed iteration data"""

    print("Continuing flux analysis from processed iterations...")

    # Check if all_iterations_flux.csv exists
    all_flux_file = os.path.join(output_dir, "all_iterations_flux.csv")

    if os.path.exists(all_flux_file):
        print(f"Found {all_flux_file}")

        # Load the data
        df = pd.read_csv(all_flux_file)

        # Group by residue and calculate statistics
        grouped = df.groupby("residue_id")

        processed_data = pd.DataFrame(
            {
                "residue_index": grouped.groups.keys(),
                "residue_name": grouped["residue_name"].first(),
                "average_flux": grouped["flux"].mean(),
                "std_flux": grouped["flux"].std(),
                "n_iterations": grouped.size(),
                "is_aromatic": grouped["is_aromatic"].first(),
            }
        )

        # Add simple statistics (no bootstrap)
        processed_data["p_value"] = 0.05  # Placeholder
        processed_data["ci_lower"] = (
            processed_data["average_flux"] - 1.96 * processed_data["std_flux"]
        )
        processed_data["ci_upper"] = (
            processed_data["average_flux"] + 1.96 * processed_data["std_flux"]
        )

        # Sort by flux
        processed_data = processed_data.sort_values("average_flux", ascending=False)

        # Save processed data
        output_file = os.path.join(output_dir, "processed_flux_data.csv")
        processed_data.to_csv(output_file, index=False)

        print(f"\nProcessed data saved to: {output_file}")

        # Create visualization
        try:
            analyzer = TrajectoryFluxAnalyzer()

            # Create flux_data dict for visualization
            flux_data = {
                "res_indices": processed_data["residue_index"].values,
                "res_names": processed_data["residue_name"].values,
                "avg_flux": processed_data["average_flux"].values,
                "std_flux": processed_data["std_flux"].values,
            }

            # Load CA coordinates from protein
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", protein_file)

            ca_coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if "CA" in residue:
                            ca_coords.append(residue["CA"].coord)

            flux_data["ca_coords"] = np.array(ca_coords)

            # Generate visualization
            analyzer.visualize_trajectory_flux(flux_data, protein_name, output_dir)

            # Generate report
            analyzer.generate_summary_report(flux_data, protein_name, output_dir)

            print("\nVisualization and report generated!")

        except Exception as e:
            print(f"\nCould not generate visualization: {e}")
            print("But processed data is saved successfully.")

        # Print top residues
        print("\nTop 10 residues by flux:")
        print(processed_data.head(10)[["residue_index", "residue_name", "average_flux"]])

        return True
    else:
        print(f"Error: {all_flux_file} not found")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python continue_analysis.py <output_dir> <protein_pdb> [protein_name]")
        sys.exit(1)

    output_dir = sys.argv[1]
    protein_file = sys.argv[2]
    protein_name = sys.argv[3] if len(sys.argv) > 3 else "protein"

    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} not found")
        sys.exit(1)

    if not os.path.exists(protein_file):
        print(f"Error: Protein file {protein_file} not found")
        sys.exit(1)

    success = continue_from_iterations(output_dir, protein_file, protein_name)

    if not success:
        print("\nFailed to continue analysis.")
        sys.exit(1)


if __name__ == "__main__":
    main()
