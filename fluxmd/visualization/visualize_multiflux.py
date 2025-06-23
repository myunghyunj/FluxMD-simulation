#!/usr/bin/env python
"""
visualize_multiflux.py - Create multi-panel flux visualization using matplotlib

This creates a grid visualization of multiple proteins colored by flux values
for publication-quality figures.

Usage:
    # Interactive mode (recommended)
    python visualize_multiflux.py

    # Command-line mode
    python visualize_multiflux.py \\
        --proteins protein1.pdb protein2.pdb protein3.pdb \\
        --fluxes flux1.csv flux2.csv flux3.csv \\
        --labels "WT" "Mutant-1" "Mutant-2" \\
        --output comparison.png

"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import PDB
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import splev, splprep

# Berlin color palette - professional blue-white-red diverging colormap
BERLIN_COLORS = [
    "#053061",  # Deep blue
    "#2166ac",  # Blue
    "#4393c3",  # Light blue
    "#92c5de",  # Pale blue
    "#d1e5f0",  # Very pale blue
    "#fddbc7",  # Very pale red
    "#f4a582",  # Pale red
    "#d6604d",  # Light red
    "#b2182b",  # Red
    "#67001f",  # Deep red
]

# Create Berlin colormap
berlin_rgb = []
for hex_color in BERLIN_COLORS:
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    berlin_rgb.append((r, g, b))

berlin_cmap = LinearSegmentedColormap.from_list("berlin", berlin_rgb, N=256)


def load_flux_data(csv_path):
    """Load flux data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        if "residue_index" not in df.columns or "average_flux" not in df.columns:
            print(f"Error: CSV must contain 'residue_index' and 'average_flux' columns")
            return None
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def get_backbone_coordinates(pdb_file):
    """Extract backbone atom coordinates for ribbon representation"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    backbone_atoms = {}

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chain_id not in backbone_atoms:
                backbone_atoms[chain_id] = []

            for residue in chain:
                # Skip hetero atoms and water
                if residue.get_id()[0] != " ":
                    continue

                res_data = {"res_id": residue.get_id()[1], "res_name": residue.get_resname()}

                # Get backbone atoms
                for atom_name in ["N", "CA", "C"]:
                    if atom_name in residue:
                        res_data[atom_name] = residue[atom_name].get_coord()

                # Only add if we have all backbone atoms
                if all(k in res_data for k in ["N", "CA", "C"]):
                    backbone_atoms[chain_id].append(res_data)

    return backbone_atoms


def calculate_ribbon_path(backbone_data, smooth_factor=10):
    """Calculate smoothed ribbon path from backbone atoms"""
    if len(backbone_data) < 4:
        return None

    # Extract CA positions
    ca_coords = np.array([res["CA"] for res in backbone_data])

    # Create parameter array (0 to 1)
    t = np.linspace(0, 1, len(ca_coords))

    # Fit splines to CA coordinates
    try:
        tck, u = splprep(
            [ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2]],
            s=smooth_factor,
            k=min(3, len(ca_coords) - 1),
        )

        # Evaluate spline at more points for smooth ribbon
        u_new = np.linspace(0, 1, len(ca_coords) * 5)
        ribbon_path = np.array(splev(u_new, tck)).T

        return ribbon_path, ca_coords, [res["res_id"] for res in backbone_data]
    except:
        return None


def plot_ribbon_protein(ax, pdb_file, flux_df, title):
    """Plot protein as ribbon structure colored by flux values"""
    # Get backbone coordinates
    backbone_data = get_backbone_coordinates(pdb_file)
    if not backbone_data:
        ax.text(
            0.5,
            0.5,
            0.5,
            f"No protein data found in\n{os.path.basename(pdb_file)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return

    # Process each chain
    for chain_id, chain_data in backbone_data.items():
        if len(chain_data) < 4:
            continue

        # Calculate ribbon path
        result = calculate_ribbon_path(chain_data)
        if result is None:
            continue

        ribbon_path, ca_coords, residue_ids = result

        # Create residue to flux mapping
        flux_map = {}
        for _, row in flux_df.iterrows():
            flux_map[int(row["residue_index"])] = row["average_flux"]

        # Map flux values to residues
        flux_values = np.array([flux_map.get(res_id, 0.0) for res_id in residue_ids])

        # Normalize flux for coloring
        if flux_values.max() > flux_values.min():
            norm_flux = (flux_values - flux_values.min()) / (flux_values.max() - flux_values.min())
        else:
            norm_flux = np.zeros_like(flux_values)

        # Create ribbon segments
        segments = []
        colors = []

        # Interpolate flux values for smooth ribbon
        flux_interp = np.interp(
            np.linspace(0, len(norm_flux) - 1, len(ribbon_path) - 1),
            np.arange(len(norm_flux)),
            norm_flux,
        )

        # Create ribbon as connected segments
        ribbon_width = 2.0

        for i in range(len(ribbon_path) - 1):
            # Calculate ribbon direction
            if i == 0:
                direction = ribbon_path[i + 1] - ribbon_path[i]
            else:
                direction = ribbon_path[i + 1] - ribbon_path[i - 1]

            # Create ribbon segments with width
            segments.append([ribbon_path[i], ribbon_path[i + 1]])
            # Use Berlin colormap for professional visualization
            colors.append(berlin_cmap(flux_interp[i]))

        # Plot ribbon as line collection
        lc = Line3DCollection(
            segments, colors=colors, linewidths=4, alpha=0.9, capstyle="round", joinstyle="round"
        )
        ax.add_collection(lc)

        # Add tube effect by plotting multiple lines with slight offset
        for offset in [-0.5, 0.5]:
            segments_offset = []
            for i in range(len(ribbon_path) - 1):
                seg = [ribbon_path[i] + offset, ribbon_path[i + 1] + offset]
                segments_offset.append(seg)

            lc_offset = Line3DCollection(segments_offset, colors=colors, linewidths=2, alpha=0.4)
            ax.add_collection(lc_offset)

    # Set axis properties
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add flux range info
    all_flux = flux_df["average_flux"].values
    flux_min, flux_max = all_flux.min(), all_flux.max()
    ax.text2D(
        0.5,
        0.02,
        f"Flux range: {flux_min:.2f} - {flux_max:.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )

    # Set proper limits
    if backbone_data:
        all_coords = []
        for chain_data in backbone_data.values():
            for res in chain_data:
                all_coords.extend([res["N"], res["CA"], res["C"]])
        all_coords = np.array(all_coords)

        margin = 5
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
        ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)

    # Adjust view angle
    ax.view_init(elev=15, azim=45)
    ax.set_box_aspect([1, 1, 0.8])


def visualize_multiflux(protein_flux_pairs, output_file="multiflux_comparison.png"):
    """Create multi-panel visualization of proteins colored by flux"""
    n_proteins = len(protein_flux_pairs)

    # Calculate grid dimensions
    if n_proteins <= 2:
        rows, cols = 1, n_proteins
    elif n_proteins <= 4:
        rows, cols = 2, 2
    elif n_proteins <= 6:
        rows, cols = 2, 3
    else:
        cols = math.ceil(math.sqrt(n_proteins))
        rows = math.ceil(n_proteins / cols)

    # Create figure with appropriate size
    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    # Plot each protein
    for idx, (pdb_file, csv_file, label) in enumerate(protein_flux_pairs):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        # Load flux data
        flux_df = load_flux_data(csv_file)
        if flux_df is None:
            continue

        # Plot protein ribbon
        plot_ribbon_protein(ax, pdb_file, flux_df, label)

        # Remove grid for cleaner look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # Add colorbar with Berlin colormap
    sm = cm.ScalarMappable(cmap=berlin_cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Energy Flux (normalized)", fontsize=12)

    plt.suptitle("Multi-Protein Flux Comparison", fontsize=16, fontweight="bold", y=0.98)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.05, wspace=0.05, hspace=0.2)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization to {output_file}")

    # Also save individual panels if requested
    save_individual = input("Save individual protein panels? (y/n): ").strip().lower()
    if save_individual == "y":
        for idx, (pdb_file, csv_file, label) in enumerate(protein_flux_pairs):
            fig_single = plt.figure(figsize=(8, 6))
            ax_single = fig_single.add_subplot(111, projection="3d")

            flux_df = load_flux_data(csv_file)
            if flux_df is not None:
                plot_ribbon_protein(ax_single, pdb_file, flux_df, label)

                # Remove grid
                ax_single.grid(False)
                ax_single.xaxis.pane.fill = False
                ax_single.yaxis.pane.fill = False
                ax_single.zaxis.pane.fill = False

                # Add colorbar with Berlin colormap
                sm = cm.ScalarMappable(cmap=berlin_cmap)
                sm.set_array([])
                cbar = fig_single.colorbar(
                    sm, ax=ax_single, orientation="vertical", pad=0.1, fraction=0.03
                )
                cbar.set_label("Energy Flux", fontsize=10)

                output_single = f"{label.replace(' ', '_')}_flux.png"
                plt.savefig(output_single, dpi=300, bbox_inches="tight", facecolor="white")
                print(f"  Saved {output_single}")

            plt.close(fig_single)

    plt.show()


def main():
    """Interactive main function"""
    print("\n=== Multi-Protein Ribbon Flux Visualization ===\n")
    print("This tool creates publication-quality figures comparing flux values across proteins.")
    print("\nRequired files:")
    print("  - PDB files: Your protein structures")
    print("  - CSV files: FluxMD output (processed_flux_data.csv from flux_analysis/)")
    print("\nTip: You can drag and drop file paths into the terminal!\n")

    # Get number of proteins
    n_proteins = int(input("How many proteins to compare? "))

    protein_flux_pairs = []

    for i in range(n_proteins):
        print(f"\n--- Protein {i+1} ---")
        print("Enter paths (drag & drop files or type path):")

        # Get PDB file
        pdb_file = input("  PDB file: ").strip().strip("'\"")  # Remove quotes if dragged
        if not os.path.exists(pdb_file):
            print(f"  ❌ File not found: {pdb_file}")
            alt_path = input("  Try again or press Enter to skip: ").strip().strip("'\"")
            if alt_path and os.path.exists(alt_path):
                pdb_file = alt_path
            else:
                print("  Skipping this protein...")
                continue

        # Get CSV file
        csv_file = input("  Flux CSV file: ").strip().strip("'\"")  # Remove quotes if dragged
        if not os.path.exists(csv_file):
            # Try common location
            base_dir = os.path.dirname(pdb_file)
            common_paths = [
                os.path.join(base_dir, "flux_analysis", "processed_flux_data.csv"),
                os.path.join(base_dir, "processed_flux_data.csv"),
                csv_file,
            ]

            found = False
            for path in common_paths:
                if os.path.exists(path):
                    csv_file = path
                    print(f"  ✓ Found flux data at: {csv_file}")
                    found = True
                    break

            if not found:
                print(f"  ❌ CSV file not found: {csv_file}")
                alt_path = input("  Try again or press Enter to skip: ").strip().strip("'\"")
                if alt_path and os.path.exists(alt_path):
                    csv_file = alt_path
                else:
                    print("  Skipping this protein...")
                    continue

        # Get label
        default_label = os.path.basename(pdb_file).replace(".pdb", "")
        label = input(f"  Label (default: {default_label}): ").strip()
        if not label:
            label = default_label

        protein_flux_pairs.append((pdb_file, csv_file, label))
        print(f"  ✓ Added: {label}")

    if not protein_flux_pairs:
        print("\n❌ No valid protein-flux pairs provided.")
        return

    print(f"\n✓ Ready to visualize {len(protein_flux_pairs)} proteins")

    output_file = input("\nOutput filename (default: multiflux_comparison.png): ").strip()
    if not output_file:
        output_file = "multiflux_comparison.png"

    print(f"\nCreating ribbon visualization...")
    visualize_multiflux(protein_flux_pairs, output_file)


if __name__ == "__main__":
    import sys

    # Check if called with arguments or interactively
    if len(sys.argv) == 1:
        main()
    else:
        parser = argparse.ArgumentParser(
            description="Visualize multiple proteins as ribbons colored by flux values"
        )
        parser.add_argument("--proteins", nargs="+", required=True, help="PDB files for proteins")
        parser.add_argument("--fluxes", nargs="+", required=True, help="CSV files with flux data")
        parser.add_argument("--labels", nargs="+", required=True, help="Labels for each protein")
        parser.add_argument("--output", default="multiflux_comparison.png", help="Output filename")

        args = parser.parse_args()

        if len(args.proteins) != len(args.fluxes) or len(args.proteins) != len(args.labels):
            print("Error: Number of proteins, flux files, and labels must match")
            sys.exit(1)

        pairs = list(zip(args.proteins, args.fluxes, args.labels))
        visualize_multiflux(pairs, args.output)
