#!/usr/bin/env python3
"""
Validate Matryoshka physics implementation.

This script runs comprehensive physics validation tests to ensure:
1. Energy conservation
2. Proper Brownian dynamics
3. Correct surface constraints
4. Valid Monte Carlo statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluxmd.core.dynamics.brownian_roller import BrownianSurfaceRoller
from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.surface.ses_builder import SESBuilder
from tests.test_matryoshka_integration import create_synthetic_ligand, create_synthetic_protein


class PhysicsValidator:
    """Validate physics implementation of Matryoshka engine."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def validate_diffusion_coefficient(self) -> Dict:
        """Validate that diffusion follows Einstein relation."""
        print("\n1. Validating diffusion coefficients...")

        # Create test sphere
        radius = 5.0  # Å
        viscosity = 1e-3  # Pa·s
        temp = 300.0  # K
        dt = 1e-4  # ps

        # Expected diffusion coefficient
        k_B = 1.380649e-23  # J/K
        D_expected = k_B * temp / (6 * np.pi * viscosity * radius * 1e-10)
        D_expected *= 1e20  # Convert to Å²/ps

        # Create spherical surface
        n_vertices = 1000
        phi = np.random.uniform(0, 2 * np.pi, n_vertices)
        theta = np.random.uniform(0, np.pi, n_vertices)

        vertices = 20 * np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
        )

        # Simple triangulation
        from scipy.spatial import SphericalVoronoi

        sv = SphericalVoronoi(vertices / 20)
        faces = []
        for region in sv.regions:
            if len(region) >= 3:
                for i in range(1, len(region) - 1):
                    faces.append([region[0], region[i], region[i + 1]])
        faces = np.array(faces)

        from fluxmd.core.surface.ses_builder import SurfaceMesh

        surface = SurfaceMesh(vertices=vertices, faces=faces)

        # Run Brownian dynamics without guidance
        roller = BrownianSurfaceRoller(
            surface=surface,
            viscosity=viscosity,
            temp=temp,
            dt=dt,
            ligand_radius=radius,
            anchor_force_constant=0.0,  # No guidance
        )

        # Track positions over time
        n_steps = 10000
        positions = []
        position = vertices[0]
        quaternion = np.array([1, 0, 0, 0])

        for _ in range(n_steps):
            position, quaternion = roller.step(position, quaternion)
            positions.append(position.copy())

        positions = np.array(positions)

        # Calculate mean squared displacement
        msd = []
        max_lag = min(1000, n_steps // 10)
        for lag in range(1, max_lag):
            displacements = positions[lag:] - positions[:-lag]
            # Project onto surface (approximate for sphere)
            radial = positions[:-lag] / np.linalg.norm(positions[:-lag], axis=1, keepdims=True)
            tangential_disp = (
                displacements - np.sum(displacements * radial, axis=1, keepdims=True) * radial
            )
            msd.append(np.mean(np.sum(tangential_disp**2, axis=1)))

        times = np.arange(1, max_lag) * dt
        msd = np.array(msd)

        # Fit MSD = 4*D*t for 2D diffusion on surface
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(times[:100], msd[:100])
        D_measured = slope / 4  # 2D diffusion

        # Plot MSD
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(times, msd, "o", alpha=0.5, label="Measured")
        ax.plot(times, 4 * D_measured * times, "r-", label=f"Fit: D={D_measured:.2f} Ų/ps")
        ax.plot(times, 4 * D_expected * times, "g--", label=f"Theory: D={D_expected:.2f} Ų/ps")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Mean Squared Displacement (Ų)")
        ax.set_title("Diffusion Coefficient Validation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "diffusion_validation.png", dpi=150)
        plt.close()

        # Check agreement
        relative_error = abs(D_measured - D_expected) / D_expected
        passed = relative_error < 0.2  # 20% tolerance

        result = {
            "D_expected": D_expected,
            "D_measured": D_measured,
            "relative_error": relative_error,
            "r_squared": r_value**2,
            "passed": passed,
        }

        print(f"  Expected D: {D_expected:.3f} Ų/ps")
        print(f"  Measured D: {D_measured:.3f} Ų/ps")
        print(f"  Relative error: {relative_error*100:.1f}%")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

        self.results["diffusion"] = result
        return result

    def validate_surface_constraint(self) -> Dict:
        """Validate that particles stay on surface."""
        print("\n2. Validating surface constraints...")

        # Create test protein and surface
        protein = create_synthetic_protein(n_atoms=200)
        builder = SESBuilder(probe_radius=0.75)
        surface = builder.build_ses0(protein)

        # Run dynamics
        roller = BrownianSurfaceRoller(
            surface=surface,
            viscosity=1e-3,
            temp=300.0,
            dt=1e-4,
            ligand_radius=2.0,
            anchor_force_constant=5.0,
        )

        # Set anchor
        anchor_idx = len(surface.vertices) // 2
        roller.set_anchor(surface.vertices[anchor_idx])

        # Track distances from surface
        n_steps = 1000
        distances = []
        position = surface.vertices[0]
        quaternion = np.array([1, 0, 0, 0])

        for _ in range(n_steps):
            position, quaternion = roller.step(position, quaternion)

            # Find closest surface point
            dist_to_vertices = np.linalg.norm(surface.vertices - position, axis=1)
            min_dist = np.min(dist_to_vertices)
            distances.append(min_dist)

        distances = np.array(distances)

        # Plot distance distribution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        ax1.plot(distances)
        ax1.axhline(y=0, color="r", linestyle="--", label="Surface")
        ax1.axhline(y=0.5, color="y", linestyle="--", label="0.5Å tolerance")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Distance from Surface (Å)")
        ax1.set_title("Surface Constraint Validation")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(distances, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(x=0.5, color="y", linestyle="--", label="0.5Å tolerance")
        ax2.set_xlabel("Distance from Surface (Å)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distance Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "surface_constraint.png", dpi=150)
        plt.close()

        # Check constraint satisfaction
        max_distance = np.max(distances)
        mean_distance = np.mean(distances)
        violations = np.sum(distances > 0.5)  # 0.5Å tolerance
        passed = violations == 0

        result = {
            "max_distance": max_distance,
            "mean_distance": mean_distance,
            "violations": int(violations),
            "total_steps": n_steps,
            "passed": passed,
        }

        print(f"  Max distance: {max_distance:.3f} Å")
        print(f"  Mean distance: {mean_distance:.3f} Å")
        print(f"  Violations (>0.5Å): {violations}/{n_steps}")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

        self.results["surface_constraint"] = result
        return result

    def validate_boltzmann_statistics(self) -> Dict:
        """Validate Monte Carlo layer hopping follows Boltzmann statistics."""
        print("\n3. Validating Boltzmann statistics...")

        # Create simple two-layer system
        protein = create_synthetic_protein(n_atoms=100)
        builder = SESBuilder()

        surface0 = builder.build_ses0(protein, probe_radius=0.75)
        surface1 = builder.build_ses0(protein, probe_radius=3.25)
        surfaces = [surface0, surface1]

        # Energy difference between layers (arbitrary units)
        delta_E = 2.0  # kT units
        temp = 300.0
        kT = 1.380649e-23 * temp * 6.022e23 / 1000  # kJ/mol

        # Initialize roller with layer hopping
        roller = BrownianSurfaceRoller(
            surface=surface0,
            viscosity=1e-3,
            temp=temp,
            dt=1e-4,
            ligand_radius=2.0,
            anchor_force_constant=0.0,  # No guidance
            enable_hopping=True,
            hop_probability=0.1,
        )

        roller.set_surfaces(surfaces, current_layer=0)

        # Mock energy function
        def mock_energy(layer_idx):
            return layer_idx * delta_E * kT

        # Monkey patch energy calculation
        original_calculate_energy = roller._calculate_local_energy
        roller._calculate_local_energy = lambda pos: mock_energy(roller.current_layer)

        # Run simulation and track layer occupancy
        n_steps = 100000
        layer_counts = [0, 0]

        position = surface0.vertices[0]
        quaternion = np.array([1, 0, 0, 0])

        for _ in range(n_steps):
            position, quaternion = roller.step(position, quaternion)
            layer_counts[roller.current_layer] += 1

        # Calculate occupancy ratio
        ratio_measured = layer_counts[0] / layer_counts[1] if layer_counts[1] > 0 else np.inf
        ratio_expected = np.exp(delta_E)  # Boltzmann factor

        # Plot layer occupancy over time
        fig, ax = plt.subplots(figsize=(8, 6))

        labels = ["Layer 0 (E=0)", f"Layer 1 (E={delta_E}kT)"]
        ax.bar(labels, layer_counts, alpha=0.7, edgecolor="black")
        ax.axhline(y=n_steps / 2, color="r", linestyle="--", alpha=0.5, label=f"Equal occupancy")

        # Add expected ratio line
        expected_counts = [
            n_steps * np.exp(-0) / (np.exp(-0) + np.exp(-delta_E)),
            n_steps * np.exp(-delta_E) / (np.exp(-0) + np.exp(-delta_E)),
        ]
        ax.plot(
            [0, 1],
            expected_counts,
            "go-",
            linewidth=2,
            markersize=10,
            label=f"Boltzmann (ratio={ratio_expected:.2f})",
        )

        ax.set_ylabel("Occupancy")
        ax.set_title(f"Boltzmann Statistics Validation (ΔE={delta_E}kT)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add text with measured ratio
        ax.text(
            0.5,
            0.95,
            f"Measured ratio: {ratio_measured:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.savefig(self.output_dir / "boltzmann_statistics.png", dpi=150)
        plt.close()

        # Check agreement
        relative_error = abs(ratio_measured - ratio_expected) / ratio_expected
        passed = relative_error < 0.1  # 10% tolerance

        result = {
            "delta_E_kT": delta_E,
            "ratio_expected": ratio_expected,
            "ratio_measured": ratio_measured,
            "layer_0_count": layer_counts[0],
            "layer_1_count": layer_counts[1],
            "relative_error": relative_error,
            "passed": passed,
        }

        print(f"  Energy difference: {delta_E} kT")
        print(f"  Expected ratio: {ratio_expected:.3f}")
        print(f"  Measured ratio: {ratio_measured:.3f}")
        print(f"  Relative error: {relative_error*100:.1f}%")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

        self.results["boltzmann"] = result
        return result

    def validate_energy_conservation(self) -> Dict:
        """Validate energy is properly tracked during dynamics."""
        print("\n4. Validating energy conservation...")

        # Create test system
        protein = create_synthetic_protein(n_atoms=300)
        ligand = create_synthetic_ligand()

        generator = MatryoshkaTrajectoryGenerator(
            temp=300.0,
            dt=1e-4,
            probe_radius=0.75,
            n_layers=1,  # Single layer for simplicity
            device="cpu",
        )

        # Generate short trajectory
        output_dir = self.output_dir / "energy_test"
        output_dir.mkdir(exist_ok=True)

        generator.generate_trajectories(
            protein_atoms=protein,
            ligand_atoms=ligand,
            output_base=output_dir,
            n_orientations=1,
            n_frames=100,
            samples_per_frame=12,
        )

        # Load energy trajectory
        energies = pd.read_csv(output_dir / "trajectory_1_energies.csv")

        # Plot energy over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(energies["total_energy"], alpha=0.7)
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Total Energy (kJ/mol)")
        ax1.set_title("Energy Conservation During Dynamics")
        ax1.grid(True, alpha=0.3)

        # Plot energy distribution
        ax2.hist(energies["total_energy"], bins=30, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Total Energy (kJ/mol)")
        ax2.set_ylabel("Count")
        ax2.set_title("Energy Distribution")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(self.output_dir / "energy_conservation.png", dpi=150)
        plt.close()

        # Check energy statistics
        mean_energy = energies["total_energy"].mean()
        std_energy = energies["total_energy"].std()
        cv = std_energy / abs(mean_energy) if mean_energy != 0 else np.inf

        # Energy should fluctuate but not drift systematically
        # Check for drift using linear regression
        from scipy import stats

        frames = np.arange(len(energies))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            frames, energies["total_energy"]
        )

        drift_per_frame = slope
        total_drift = slope * len(energies)
        relative_drift = abs(total_drift) / abs(mean_energy) if mean_energy != 0 else np.inf

        # Pass if no significant drift
        passed = relative_drift < 0.1 and cv < 0.5

        result = {
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "coefficient_of_variation": cv,
            "drift_per_frame": drift_per_frame,
            "total_drift": total_drift,
            "relative_drift": relative_drift,
            "passed": passed,
        }

        print(f"  Mean energy: {mean_energy:.1f} kJ/mol")
        print(f"  Std deviation: {std_energy:.1f} kJ/mol")
        print(f"  CV: {cv:.3f}")
        print(f"  Drift: {drift_per_frame:.3f} kJ/mol per frame")
        print(f"  Relative drift: {relative_drift*100:.1f}%")
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")

        self.results["energy_conservation"] = result
        return result

    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 60)
        print("MATRYOSHKA PHYSICS VALIDATION SUMMARY")
        print("=" * 60)

        all_passed = all(r.get("passed", False) for r in self.results.values())

        for test_name, result in self.results.items():
            status = "✅" if result.get("passed", False) else "❌"
            print(f"{status} {test_name.replace('_', ' ').title()}")

        print("\n" + "=" * 60)
        print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        print("=" * 60)

        # Save detailed results
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")
        print(f"Plots saved to: {self.output_dir}/")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate Matryoshka physics")
    parser.add_argument(
        "--output-dir", type=Path, default="validation_results", help="Output directory for results"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["diffusion", "surface", "boltzmann", "energy", "all"],
        default=["all"],
        help="Tests to run",
    )

    args = parser.parse_args()

    validator = PhysicsValidator(args.output_dir)

    # Run requested tests
    if "all" in args.tests:
        tests = ["diffusion", "surface", "boltzmann", "energy"]
    else:
        tests = args.tests

    for test in tests:
        if test == "diffusion":
            validator.validate_diffusion_coefficient()
        elif test == "surface":
            validator.validate_surface_constraint()
        elif test == "boltzmann":
            validator.validate_boltzmann_statistics()
        elif test == "energy":
            validator.validate_energy_conservation()

    # Generate report
    all_passed = validator.generate_report()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
