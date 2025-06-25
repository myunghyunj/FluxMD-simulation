"""
Performance benchmarks for Matryoshka trajectory engine.

These benchmarks track performance of critical operations and enforce
regression thresholds to prevent performance degradation.
"""

import time

import numpy as np
import pandas as pd
import pytest

from fluxmd.core.dynamics.brownian_roller import BrownianSurfaceRoller
from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.surface.ses_builder import SESBuilder
from tests.test_matryoshka_integration import create_synthetic_ligand, create_synthetic_protein


class BenchmarkSuite:
    """Benchmark suite for Matryoshka components."""

    def setup(self):
        """Setup benchmark data."""
        np.random.seed(42)
        self.protein_100 = create_synthetic_protein(n_atoms=100)
        self.protein_500 = create_synthetic_protein(n_atoms=500)
        self.protein_1000 = create_synthetic_protein(n_atoms=1000)
        self.protein_5000 = create_synthetic_protein(n_atoms=5000)
        self.ligand = create_synthetic_ligand()

    def test_ses_generation_100(self, benchmark):
        """Benchmark SES generation for 100 atoms."""
        builder = SESBuilder(probe_radius=0.75)
        benchmark(builder.build_ses0, self.protein_100)

    def test_ses_generation_500(self, benchmark):
        """Benchmark SES generation for 500 atoms."""
        builder = SESBuilder(probe_radius=0.75)
        benchmark(builder.build_ses0, self.protein_500)

    def test_ses_generation_1000(self, benchmark):
        """Benchmark SES generation for 1000 atoms."""
        builder = SESBuilder(probe_radius=0.75)
        benchmark(builder.build_ses0, self.protein_1000)

    def test_ses_generation_5000(self, benchmark):
        """Benchmark SES generation for 5000 atoms."""
        builder = SESBuilder(probe_radius=0.75)
        benchmark(builder.build_ses0, self.protein_5000)

    def test_pca_anchors_100(self, benchmark):
        """Benchmark PCA anchor detection for 100 atoms."""
        coords = self.protein_100[["x", "y", "z"]].values
        atom_names = self.protein_100["name"].tolist()

        pca = PCAAnchors()
        benchmark(pca.extreme_calpha_pairs, coords, atom_names)

    def test_pca_anchors_1000(self, benchmark):
        """Benchmark PCA anchor detection for 1000 atoms."""
        coords = self.protein_1000[["x", "y", "z"]].values
        atom_names = self.protein_1000["name"].tolist()

        pca = PCAAnchors()
        benchmark(pca.extreme_calpha_pairs, coords, atom_names)

    def test_brownian_step(self, benchmark):
        """Benchmark single Brownian dynamics step."""
        # Create spherical surface for consistent benchmarking
        n_vertices = 500
        phi = np.random.uniform(0, 2 * np.pi, n_vertices)
        theta = np.random.uniform(0, np.pi, n_vertices)

        vertices = 20 * np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
        )

        # Simple triangulation
        from scipy.spatial import Delaunay

        points_2d = np.column_stack([phi, theta])
        tri = Delaunay(points_2d)
        faces = tri.simplices

        from fluxmd.core.surface.ses_builder import SurfaceMesh

        surface = SurfaceMesh(vertices=vertices, faces=faces)

        roller = BrownianSurfaceRoller(
            surface=surface,
            viscosity=1e-3,
            temp=300.0,
            dt=1e-4,
            ligand_radius=2.0,
            anchor_force_constant=10.0,
        )

        position = np.array([20, 0, 0])
        quaternion = np.array([1, 0, 0, 0])
        roller.set_anchor(np.array([-20, 0, 0]))

        def run_step():
            return roller.step(position, quaternion)

        benchmark(run_step)

    def test_ref15_energy_calculation(self, benchmark):
        """Benchmark REF15 energy calculation."""
        generator = MatryoshkaTrajectoryGenerator(
            temp=300.0, dt=1e-4, probe_radius=0.75, n_layers=3, device="cpu"
        )

        # Prepare atom data
        protein_coords = self.protein_500[["x", "y", "z"]].values
        ligand_coords = self.ligand[["x", "y", "z"]].values

        def calculate_energy():
            return generator._calculate_ref15_energy(self.protein_500, ligand_coords, self.ligand)

        benchmark(calculate_energy)

    def test_full_trajectory_generation(self, benchmark, tmp_path):
        """Benchmark full trajectory generation for small system."""
        generator = MatryoshkaTrajectoryGenerator(
            temp=300.0, dt=1e-4, probe_radius=0.75, n_layers=2, device="cpu"
        )

        output_dir = tmp_path / "benchmark_full"

        def generate():
            if output_dir.exists():
                import shutil

                shutil.rmtree(output_dir)
            output_dir.mkdir()

            generator.generate_trajectories(
                protein_atoms=self.protein_100,
                ligand_atoms=self.ligand,
                output_base=output_dir,
                n_orientations=1,
                n_frames=10,
                samples_per_frame=12,
            )

        benchmark(generate)

    @pytest.mark.parametrize("n_layers", [1, 3, 5, 10])
    def test_layer_generation_scaling(self, benchmark, n_layers):
        """Benchmark layer generation scaling."""
        builder = SESBuilder(probe_radius=0.75)

        def generate_layers():
            surfaces = []
            current_probe = 0.75
            for i in range(n_layers):
                surface = builder.build_ses0(self.protein_500, probe_radius=current_probe)
                surfaces.append(surface)
                current_probe += 2.5
            return surfaces

        benchmark(generate_layers)

    def test_monte_carlo_hopping(self, benchmark):
        """Benchmark Monte Carlo layer hopping decision."""
        # Setup multi-layer surfaces
        builder = SESBuilder(probe_radius=0.75)
        surfaces = []
        for i in range(3):
            probe = 0.75 + i * 2.5
            surface = builder.build_ses0(self.protein_100, probe_radius=probe)
            surfaces.append(surface)

        # Create roller with hopping
        roller = BrownianSurfaceRoller(
            surface=surfaces[1],  # Middle layer
            viscosity=1e-3,
            temp=300.0,
            dt=1e-4,
            ligand_radius=2.0,
            anchor_force_constant=10.0,
            enable_hopping=True,
            hop_probability=0.1,
        )

        roller.set_surfaces(surfaces, current_layer=1)

        position = surfaces[1].vertices[0]
        quaternion = np.array([1, 0, 0, 0])

        def attempt_hop():
            return roller._attempt_layer_hop(position, quaternion)

        benchmark(attempt_hop)


class PerformanceRegressionTests:
    """Tests that enforce performance regression thresholds."""

    def test_ses_performance_regression(self):
        """Ensure SES generation hasn't regressed."""
        # Baseline: 1000 atoms should complete in < 1 second
        protein = create_synthetic_protein(n_atoms=1000)
        builder = SESBuilder(probe_radius=0.75)

        start = time.time()
        surface = builder.build_ses0(protein)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"SES generation too slow: {elapsed:.3f}s for 1000 atoms"
        assert len(surface.vertices) > 0, "SES generation failed"

    def test_trajectory_performance_regression(self, tmp_path):
        """Ensure trajectory generation meets performance target."""
        # Target: 1000 atoms × 3 layers × 3 paths ≤ 60s
        protein = create_synthetic_protein(n_atoms=1000)
        ligand = create_synthetic_ligand()

        generator = MatryoshkaTrajectoryGenerator(
            temp=300.0, dt=1e-4, probe_radius=0.75, n_layers=3, device="cpu"
        )

        output_dir = tmp_path / "perf_regression"
        output_dir.mkdir()

        start = time.time()
        generator.generate_trajectories(
            protein_atoms=protein,
            ligand_atoms=ligand,
            output_base=output_dir,
            n_orientations=3,
            n_frames=100,
            samples_per_frame=12,
        )
        elapsed = time.time() - start

        # Allow some margin but enforce < 60s target
        assert elapsed < 60.0, f"Trajectory generation too slow: {elapsed:.1f}s > 60s target"

    def test_memory_usage(self):
        """Ensure memory usage is reasonable."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate large protein
        protein = create_synthetic_protein(n_atoms=10000)
        builder = SESBuilder(probe_radius=0.75)

        # Generate 5 layers
        surfaces = []
        for i in range(5):
            surface = builder.build_ses0(protein, probe_radius=0.75 + i * 2.5)
            surfaces.append(surface)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should use less than 1GB for 10k atom system with 5 layers
        assert memory_increase < 1024, f"Memory usage too high: {memory_increase:.1f} MB"


def pytest_benchmark_compare_machine_info(config, benchmarksession):
    """Provide machine info for benchmark comparison."""
    import platform

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def pytest_benchmark_generate_json(
    config, benchmarks, include_data=False, machine_info=None, commit_info=None
):
    """Generate JSON output for benchmark comparison."""
    # Add custom fields to track Matryoshka version
    for benchmark in benchmarks:
        benchmark["extra_info"] = {
            "matryoshka_version": "1.0.0",
            "test_date": str(pd.Timestamp.now()),
        }
