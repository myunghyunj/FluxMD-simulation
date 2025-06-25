#!/usr/bin/env python3
"""
Signed Flux Analyzer for FluxMD
================================

This module implements signed flux analysis that preserves the sign of interaction energies,
providing a more physically meaningful representation of protein-ligand interactions.

Key differences from standard flux:
- Negative flux indicates attractive/stabilizing interactions
- Positive flux indicates repulsive/destabilizing interactions
- Net flux relates directly to binding affinity contributions

Author: FluxMD Development Team
"""

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class SignedFluxAnalyzer:
    """
    Analyzer for calculating signed flux that preserves energy directionality.

    The signed flux formula:
    Φᵢ = ⟨E⃗ᵢ⟩ · Cᵢ · (1 + τᵢ)

    Where E⃗ᵢ maintains the sign of the interaction energy:
    - Negative for attractive interactions (H-bonds, salt bridges, favorable vdW)
    - Positive for repulsive interactions (steric clashes)
    """

    def __init__(self, output_dir: str = "signed_flux_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_signed_flux(self, interactions_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Calculate signed flux for each residue preserving energy signs.

        Args:
            interactions_df: DataFrame with columns including:
                - protein_residue: Residue ID
                - bond_energy: Interaction energy (kcal/mol)
                - vector_x/y/z: Interaction vector components

        Returns:
            Dictionary mapping residue IDs to flux metrics
        """
        results = {}

        # Group by residue
        for res_id, group in interactions_df.groupby("protein_residue_id"):
            # Extract energies and vectors
            energies = group["bond_energy"].values
            vectors = group[["vector_x", "vector_y", "vector_z"]].values

            # Calculate SIGNED energy-weighted vectors (preserving energy sign)
            # Mean signed energy magnitude
            mean_signed_energy = np.mean(energies)

            # Coherence: ||∑v⃗|| / ∑||v⃗|| using RAW vectors (not energy-weighted)
            sum_vectors = np.sum(vectors, axis=0)
            sum_magnitudes = np.sum(np.linalg.norm(vectors, axis=1))

            if sum_magnitudes > 1e-10:
                coherence = np.linalg.norm(sum_vectors) / sum_magnitudes
            else:
                coherence = 0.0

            # Temporal factor using variance method
            if len(energies) > 1:
                variance = np.var(energies)
                mean_abs_energy = np.mean(np.abs(energies))
                if mean_abs_energy > 1e-10:
                    temporal_factor = 1.0 + np.sqrt(variance) / mean_abs_energy
                else:
                    temporal_factor = 1.0
            else:
                temporal_factor = 1.0

            # Signed flux calculation
            signed_flux = mean_signed_energy * coherence * temporal_factor

            # Store results
            results[res_id] = {
                "signed_flux": signed_flux,
                "mean_energy": mean_signed_energy,
                "coherence": coherence,
                "temporal_factor": temporal_factor,
                "n_interactions": len(energies),
                "attractive_fraction": (
                    np.sum(energies < 0) / len(energies) if len(energies) > 0 else 0
                ),
            }

        return results

    def analyze_trajectory(self, trajectory_dir: str, protein_name: str = "Protein") -> Dict:
        """
        Analyze all approach CSV files in a trajectory directory.

        Args:
            trajectory_dir: Directory containing approach CSV files
            protein_name: Name for labeling

        Returns:
            Comprehensive analysis results
        """
        trajectory_path = Path(trajectory_dir)
        approach_files = sorted(trajectory_path.glob("interactions_approach_*.csv"))

        if not approach_files:
            raise ValueError(f"No approach files found in {trajectory_dir}")

        print(f"\nAnalyzing {len(approach_files)} approach files...")

        all_results = []

        for approach_file in approach_files:
            # Load interactions
            df = pd.read_csv(approach_file)

            # Calculate signed flux
            flux_results = self.calculate_signed_flux(df)

            # Store results with approach info
            approach_num = int(approach_file.stem.split("_")[-1])
            for res_id, metrics in flux_results.items():
                metrics["approach"] = approach_num
                metrics["residue_id"] = res_id
                all_results.append(metrics)

        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_results)

        # Calculate per-residue statistics
        residue_stats = (
            results_df.groupby("residue_id")
            .agg(
                {
                    "signed_flux": ["mean", "std", "min", "max"],
                    "mean_energy": "mean",
                    "n_interactions": "sum",
                    "attractive_fraction": "mean",
                }
            )
            .round(4)
        )

        # Identify key binding residues (most negative flux)
        residue_stats["net_flux"] = residue_stats[("signed_flux", "mean")]
        binding_residues = residue_stats.nsmallest(20, "net_flux")

        return {
            "raw_data": results_df,
            "residue_stats": residue_stats,
            "binding_residues": binding_residues,
            "protein_name": protein_name,
        }

    def visualize_signed_flux(self, analysis_results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of signed flux analysis.
        """
        fig = plt.figure(figsize=(16, 12))

        residue_stats = analysis_results["residue_stats"]
        protein_name = analysis_results["protein_name"]

        # Sort by residue ID for plotting
        residue_ids = sorted(residue_stats.index)
        mean_flux = [residue_stats.loc[r, ("signed_flux", "mean")] for r in residue_ids]
        std_flux = [residue_stats.loc[r, ("signed_flux", "std")] for r in residue_ids]

        # 1. Signed flux profile
        ax1 = plt.subplot(3, 2, 1)
        bars = ax1.bar(range(len(residue_ids)), mean_flux, yerr=std_flux, capsize=2, width=0.8)

        # Color bars by sign
        for i, (bar, flux) in enumerate(zip(bars, mean_flux)):
            if flux < 0:
                bar.set_color("blue")
                bar.set_alpha(min(1.0, abs(flux) / 10))  # Intensity by magnitude
            else:
                bar.set_color("red")
                bar.set_alpha(min(1.0, flux / 5))

        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_xlabel("Residue Index")
        ax1.set_ylabel("Signed Flux (kcal/mol·Å)")
        ax1.set_title(f"{protein_name} - Signed Energy Flux\n(Blue: Attractive, Red: Repulsive)")
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative binding contribution
        ax2 = plt.subplot(3, 2, 2)
        sorted_flux = sorted(mean_flux)
        cumulative_binding = np.cumsum(sorted_flux)

        ax2.plot(cumulative_binding, "b-", linewidth=2)
        ax2.fill_between(
            range(len(cumulative_binding)),
            0,
            cumulative_binding,
            where=(cumulative_binding < 0),
            alpha=0.3,
            color="blue",
            label="Net Attractive",
        )

        ax2.set_xlabel("Residues (sorted by flux)")
        ax2.set_ylabel("Cumulative Flux (kcal/mol·Å)")
        ax2.set_title("Cumulative Binding Contribution")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Top binding residues
        ax3 = plt.subplot(3, 2, 3)
        top_binding = residue_stats.nsmallest(15, ("signed_flux", "mean"))

        y_pos = np.arange(len(top_binding))
        flux_values = top_binding[("signed_flux", "mean")].values

        ax3.barh(y_pos, flux_values, color="darkblue", alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"Res {idx}" for idx in top_binding.index])
        ax3.set_xlabel("Signed Flux (kcal/mol·Å)")
        ax3.set_title("Top 15 Binding Residues")
        ax3.grid(True, alpha=0.3, axis="x")

        # 4. Energy distribution
        ax4 = plt.subplot(3, 2, 4)
        mean_energies = [residue_stats.loc[r, ("mean_energy", "mean")] for r in residue_ids]

        ax4.hist(mean_energies, bins=30, color="purple", alpha=0.7, edgecolor="black")
        ax4.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax4.set_xlabel("Mean Interaction Energy (kcal/mol)")
        ax4.set_ylabel("Number of Residues")
        ax4.set_title("Distribution of Mean Interaction Energies")

        # 5. Attractive vs Repulsive breakdown
        ax5 = plt.subplot(3, 2, 5)
        attractive_residues = sum(1 for f in mean_flux if f < 0)
        repulsive_residues = sum(1 for f in mean_flux if f > 0)
        neutral_residues = sum(1 for f in mean_flux if f == 0)

        labels = ["Attractive\n(Negative Flux)", "Repulsive\n(Positive Flux)", "Neutral"]
        sizes = [attractive_residues, repulsive_residues, neutral_residues]
        colors = ["blue", "red", "gray"]

        ax5.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax5.set_title("Residue Interaction Types")

        # 6. Binding energy estimate
        ax6 = plt.subplot(3, 2, 6)

        # Estimate binding energy from negative flux contributions
        negative_flux = [f for f in mean_flux if f < 0]
        total_attractive = sum(negative_flux)

        # Apply Gaussian smoothing to account for spatial distribution
        sigma = 3  # ~3 residue smoothing
        smoothed_flux = gaussian_filter1d(mean_flux, sigma=sigma, mode="reflect")
        smoothed_negative = sum(f for f in smoothed_flux if f < 0)

        categories = ["Direct Sum", "Spatially Smoothed"]
        values = [total_attractive, smoothed_negative]

        bars = ax6.bar(categories, values, color=["lightblue", "darkblue"])
        ax6.set_ylabel("Estimated Binding Energy (kcal/mol·Å)")
        ax6.set_title("Binding Energy Estimates from Flux")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0, height, f"{val:.1f}", ha="center", va="bottom"
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved visualization to: {save_path}")

        plt.show()

        # Print summary statistics
        print("\n" + "=" * 60)
        print(f"SIGNED FLUX ANALYSIS SUMMARY - {protein_name}")
        print("=" * 60)
        print(f"Total residues analyzed: {len(residue_ids)}")
        print(
            "Attractive residues: "
            f"{attractive_residues} "
            f"({attractive_residues / len(residue_ids) * 100:.1f}%)"
        )
        print(
            "Repulsive residues: "
            f"{repulsive_residues} "
            f"({repulsive_residues / len(residue_ids) * 100:.1f}%)"
        )
        print(f"\nEstimated binding contribution: {total_attractive:.2f} kcal/mol·Å")
        print(f"Spatially smoothed estimate: {smoothed_negative:.2f} kcal/mol·Å")

        # Print top binding residues
        print("\nTop 10 Binding Residues:")
        print("-" * 40)
        for idx, row in residue_stats.nsmallest(10, ("signed_flux", "mean")).iterrows():
            flux = row[("signed_flux", "mean")]
            energy = row[("mean_energy", "mean")]
            n_int = row[("n_interactions", "sum")]
            print(
                f"Residue {idx}: Flux = {flux:.3f}, "
                f"Mean Energy = {energy:.3f} kcal/mol, N = {n_int}"
            )

        return fig


def analyze_case_study(
    trajectory_dir: str, protein_name: str = "GPX4", output_dir: str = "signed_flux_analysis"
):
    """
    Analyze a specific case study with signed flux.
    """
    analyzer = SignedFluxAnalyzer(output_dir)

    # Run analysis
    results = analyzer.analyze_trajectory(trajectory_dir, protein_name)

    # Create visualization
    save_path = os.path.join(output_dir, f"{protein_name}_signed_flux_analysis.png")
    analyzer.visualize_signed_flux(results, save_path)

    # Save detailed results
    results_file = os.path.join(output_dir, f"{protein_name}_signed_flux_results.csv")
    results["residue_stats"].to_csv(results_file)
    print(f"\nDetailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        trajectory_dir = sys.argv[1]
        protein_name = sys.argv[2] if len(sys.argv) > 2 else "Protein"
        analyze_case_study(trajectory_dir, protein_name)
    else:
        print("Usage: python signed_flux_analyzer.py <trajectory_dir> [protein_name]")
        print("Example: python signed_flux_analyzer.py /path/to/iteration_10 GPX4")
