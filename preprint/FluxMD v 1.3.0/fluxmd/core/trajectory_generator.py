"""
Protein-Ligand Trajectory Generator - Cocoon Mode
Advanced molecular dynamics simulation with GPU acceleration
Now with integrated intra-protein force field calculations

COCOON TRAJECTORY MODE:
- Maintains constant distance from protein surface (hovering)
- Uses physics-based Brownian dynamics with MW-dependent diffusion
- Samples multiple ligand orientations at each position
- Creates a "cocoon" of trajectories at different distances
- Inspired by legacy_cocoon_trajectory.py implementation
"""

# Set matplotlib backend before any other imports
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for blackbox operation

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# Import intra-protein interaction calculator
from .intra_protein_interactions import IntraProteinInteractions
from .protonation_aware_interactions import calculate_interactions_with_protonation
from .intelligent_cocoon_sampler import IntelligentCocoonSampler
from .ref15_energy import get_ref15_calculator
from .energy_config import DEFAULT_ENERGY_FUNCTION

# Import utils
from ..utils.pdb_parser import PDBParser


class CollisionDetector:
    """Efficient collision detection for protein-ligand systems"""

    def __init__(self, clash_tolerance=0.8):
        self.clash_tolerance = clash_tolerance

        # VDW radii in Angstroms
        self.vdw_radii = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "P": 1.80,
            "S": 1.80,
            "CL": 1.75,
            "BR": 1.85,
            "I": 1.98,
            "A": 1.70,  # PDBQT aromatic carbon
        }
        self.default_radius = 1.70

    def build_protein_tree(self, protein_coords, protein_atoms):
        """Build KD-tree for fast collision detection"""
        self.protein_tree = cKDTree(protein_coords)

        # Store VDW radii for protein atoms
        self.protein_radii = []
        for _, atom in protein_atoms.iterrows():
            element = atom.get("element", atom["name"][0]).upper()
            radius = self.vdw_radii.get(element, self.default_radius)
            self.protein_radii.append(radius)

        self.protein_radii = np.array(self.protein_radii)

    def check_collision(self, ligand_coords, ligand_atoms):
        """Check if ligand configuration has collision with protein"""
        for i, (coord, atom) in enumerate(zip(ligand_coords, ligand_atoms.itertuples())):
            # Get ligand atom radius
            element = getattr(atom, "element", getattr(atom, "name", "C")[0]).upper()
            lig_radius = self.vdw_radii.get(element, self.default_radius)

            # Find nearby protein atoms
            search_radius = lig_radius + max(self.protein_radii) + 1.0
            nearby_indices = self.protein_tree.query_ball_point(coord, search_radius)

            # Check detailed distances
            for prot_idx in nearby_indices:
                prot_radius = self.protein_radii[prot_idx]
                distance = np.linalg.norm(coord - self.protein_tree.data[prot_idx])

                # Minimum allowed distance with tolerance
                min_allowed = (lig_radius + prot_radius) * self.clash_tolerance

                if distance < min_allowed:
                    return True  # Collision detected

        return False  # No collision

    def find_collision_free_path(
        self, start_pos, end_pos, ligand_coords, ligand_atoms, n_steps=10, max_attempts=5
    ):
        """Find collision-free path from start to end position"""
        path = []
        current_pos = start_pos.copy()

        for step in range(n_steps):
            # Linear interpolation
            t = (step + 1) / n_steps
            target_pos = (1 - t) * start_pos + t * end_pos

            # Try direct movement first
            test_coords = ligand_coords + (target_pos - ligand_coords.mean(axis=0))

            if not self.check_collision(test_coords, ligand_atoms):
                path.append(target_pos)
                current_pos = target_pos
            else:
                # Try alternative paths
                found_alternative = False

                for attempt in range(max_attempts):
                    # Add random perturbation
                    perturbation = np.random.randn(3) * 2.0
                    alt_pos = target_pos + perturbation

                    test_coords = ligand_coords + (alt_pos - ligand_coords.mean(axis=0))

                    if not self.check_collision(test_coords, ligand_atoms):
                        path.append(alt_pos)
                        current_pos = alt_pos
                        found_alternative = True
                        break

                if not found_alternative:
                    # Stay at current position
                    path.append(current_pos)

        return np.array(path)


class ProteinLigandFluxAnalyzer:
    """
    Main class to orchestrate the flux analysis, from trajectory generation
    to interaction calculation and data processing.
    """

    def __init__(
        self,
        protein_file: str,
        ligand_file: str,
        output_dir: str,
        target_is_dna: bool = False,
        energy_function: str = None,
    ):
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.output_dir = output_dir
        self.target_is_dna = target_is_dna  # NEW: Flag for DNA target
        self.gpu_calculator = None

        # Get protein and ligand names
        self.protein_name = os.path.basename(protein_file).replace(".pdb", "").replace(".cif", "")
        self.ligand_name = os.path.basename(ligand_file).replace(".pdb", "").replace(".smi", "")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize intra-protein force field calculator
        self.intra_protein_calc = None
        self.intra_protein_vectors = None

        # Initialize collision detector
        self.collision_detector = CollisionDetector()

        # Initialize physiological pH
        self.physiological_pH = 7.4

        # Initialize energy function
        self.energy_function = energy_function or DEFAULT_ENERGY_FUNCTION
        if self.energy_function.startswith("ref15"):
            print(f"\n   Using REF15 energy function")
            self.ref15_calculator = get_ref15_calculator(self.physiological_pH)
            self.intelligent_sampler = None  # Will be initialized after protein loading
        else:
            print(f"\n   Using legacy energy function")
            self.ref15_calculator = None
            self.intelligent_sampler = None

        # Initialize residue property definitions
        self.init_residue_properties()

        # Initialize molecular weight dictionary
        self.ELEMENT_MASSES = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "S": 32.065,
            "P": 30.974,
            "F": 18.998,
            "CL": 35.453,
            "BR": 79.904,
            "I": 126.904,
            "FE": 55.845,
            "ZN": 65.38,
            "A": 12.011,  # PDBQT aromatic carbon if not caught by parser
        }

        # Initialize molecular weight
        self.molecular_weight = None

        # Initialize trajectory results
        self.gpu_trajectory_results = None

    def init_residue_properties(self):
        """Initialize residue property definitions"""
        # Hydrogen bond donors
        self.DONORS = {
            "SER": ["OG"],
            "THR": ["OG1"],
            "CYS": ["SG"],
            "TYR": ["OH"],
            "ASN": ["ND2"],
            "GLN": ["NE2"],
            "LYS": ["NZ"],
            "ARG": ["NE", "NH1", "NH2"],
            "HIS": ["ND1", "NE2"],
            "TRP": ["NE1"],
        }

        # Hydrogen bond acceptors
        self.ACCEPTORS = {
            "ASP": ["OD1", "OD2"],
            "GLU": ["OE1", "OE2"],
            "SER": ["OG"],
            "THR": ["OG1"],
            "CYS": ["SG"],
            "TYR": ["OH"],
            "ASN": ["OD1"],
            "GLN": ["OE1"],
            "HIS": ["ND1", "NE2"],
            "MET": ["SD"],
        }

        # Charged groups
        self.POSITIVE = {"ARG": ["CZ"], "LYS": ["NZ"], "HIS": ["CE1"]}
        self.NEGATIVE = {"ASP": ["CG"], "GLU": ["CD"]}

        # Hydrophobic residues
        self.HYDROPHOBIC = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"]

        # Aromatic residues
        self.AROMATIC = {
            "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
            "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        }

    def get_protein_surface_points(self, protein_coords, n_points=100, buffer_distance=5.0):
        """
        Generate starting points on protein surface using alpha shapes
        This ensures ligand starts OUTSIDE the protein
        """
        print("\n   Calculating protein surface points...")

        # Method 1: Convex hull with buffer
        try:
            hull = ConvexHull(protein_coords)

            # Sample points on hull surface
            surface_points = []

            for simplex in hull.simplices[:n_points]:
                # Get face vertices
                face_points = protein_coords[simplex]

                # Calculate face center
                face_center = np.mean(face_points, axis=0)

                # Calculate outward normal
                v1 = face_points[1] - face_points[0]
                v2 = face_points[2] - face_points[0]
                normal = np.cross(v1, v2)
                normal = normal / (np.linalg.norm(normal) + 1e-10)

                # Ensure normal points outward
                center_to_face = face_center - protein_coords.mean(axis=0)
                if np.dot(normal, center_to_face) < 0:
                    normal = -normal

                # Place point at buffer distance
                surface_point = face_center + normal * buffer_distance
                surface_points.append(surface_point)

            # If we need more points, interpolate
            while len(surface_points) < n_points:
                idx1, idx2 = np.random.choice(len(surface_points), 2, replace=False)
                interp_point = (surface_points[idx1] + surface_points[idx2]) / 2

                # Project back to surface + buffer
                direction = interp_point - protein_coords.mean(axis=0)
                direction = direction / np.linalg.norm(direction)

                # Find intersection with hull and add buffer
                surface_points.append(
                    protein_coords.mean(axis=0)
                    + direction
                    * (
                        np.max(cdist([protein_coords.mean(axis=0)], protein_coords)[0])
                        + buffer_distance
                    )
                )

            print(f"   ✓ Generated {len(surface_points)} surface points using convex hull")
            return np.array(surface_points[:n_points])

        except Exception as e:
            print(f"   ⚠️  Convex hull failed: {e}, using spherical sampling")

        # Method 2: Spherical sampling (fallback)
        center = protein_coords.mean(axis=0)
        max_radius = np.max(cdist([center], protein_coords)[0])

        # Generate points on sphere
        surface_points = []

        # Use golden spiral for even distribution
        golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

        for i in range(n_points):
            theta = golden_angle * i
            phi = np.arccos(1 - 2 * i / n_points)

            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            # Scale and translate
            point = center + np.array([x, y, z]) * (max_radius + buffer_distance)
            surface_points.append(point)

        print(f"   ✓ Generated {n_points} surface points using spherical sampling")
        return np.array(surface_points)

    def calculate_molecular_weight(self, ligand_atoms):
        """Calculate molecular weight from ligand atoms"""
        # Simple atomic mass table
        masses = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "S": 32.065,
            "P": 30.974,
            "F": 18.998,
            "CL": 35.453,
            "BR": 79.904,
            "I": 126.904,
            "FE": 55.845,
            "ZN": 65.38,
            "A": 12.011,
        }  # PDBQT aromatic carbon if not caught by parser

        total_mass = 0
        for _, atom in ligand_atoms.iterrows():
            element = atom["element"].upper()
            mass = masses.get(element, 12.0)  # Default to carbon if unknown
            total_mass += mass

        return total_mass

    def calculate_diffusion_coefficient(self, molecular_weight, temperature=36.5):
        """Calculate diffusion coefficient based on molecular weight (legacy method)"""
        # Physical constants
        k_B = 1.381e-23  # Boltzmann constant (J/K)

        temp_kelvin = temperature + 273.15  # Convert to Kelvin

        # Approximate radius from molecular weight (Å)
        # Using empirical relation: r ≈ 0.66 * MW^(1/3) for small molecules
        # This gives ~4.4 Å for MW=300, which is realistic
        radius_angstrom = 0.66 * (molecular_weight ** (1 / 3))
        radius_meter = radius_angstrom * 1e-10

        # Water viscosity at 36.5°C (Pa·s)
        viscosity = 0.00069

        # Diffusion coefficient (m²/s)
        D = (k_B * temp_kelvin) / (6 * np.pi * viscosity * radius_meter)

        # Convert to Å²/fs (femtoseconds)
        D_angstrom_fs = D * 1e20 / 1e15  # 1e20 for m² to Å², 1e15 for s to fs

        return D_angstrom_fs

    def analyze_molecular_geometry(self, coords):
        """
        Analyze molecular geometry using PCA to determine shape type.
        Returns shape classification and geometric parameters.
        """
        # Calculate center of mass
        center = np.mean(coords, axis=0)
        centered_coords = coords - center

        # Handle small molecules
        if len(coords) < 10:
            return {
                "shape_type": "small",
                "center": center,
                "principal_axes": np.eye(3),
                "eigenvalues": np.ones(3),
                "dimensions": np.ones(3) * 10.0,
                "aspect_ratios": (1.0, 1.0, 1.0),
            }

        # Compute covariance matrix and PCA
        try:
            cov_matrix = np.cov(centered_coords.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Sort by eigenvalue (largest first)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate dimensions along principal axes
            projected = centered_coords @ eigenvectors
            dimensions = np.array(
                [
                    np.ptp(projected[:, 0]),  # Range along major axis
                    np.ptp(projected[:, 1]),  # Range along middle axis
                    np.ptp(projected[:, 2]),  # Range along minor axis
                ]
            )

            # Calculate aspect ratios
            eps = 1e-6
            aspect_ratio_1 = dimensions[0] / (dimensions[1] + eps)
            aspect_ratio_2 = dimensions[0] / (dimensions[2] + eps)
            aspect_ratio_3 = dimensions[1] / (dimensions[2] + eps)

            # Classify shape
            if aspect_ratio_1 > 3.0 and aspect_ratio_2 > 3.0:
                shape_type = "linear"  # DNA, fibrous proteins
            elif aspect_ratio_1 < 2.0 and aspect_ratio_2 < 2.0:
                shape_type = "globular"  # Most proteins
            elif aspect_ratio_1 > 3.0 and aspect_ratio_3 < 2.0:
                shape_type = "planar"  # Sheet-like
            else:
                shape_type = "irregular"

            return {
                "shape_type": shape_type,
                "center": center,
                "principal_axes": eigenvectors,
                "eigenvalues": eigenvalues,
                "dimensions": dimensions,
                "aspect_ratios": (aspect_ratio_1, aspect_ratio_2, aspect_ratio_3),
            }

        except np.linalg.LinAlgError:
            # Fallback for degenerate cases
            return {
                "shape_type": "irregular",
                "center": center,
                "principal_axes": np.eye(3),
                "eigenvalues": np.ones(3),
                "dimensions": np.ones(3) * 20.0,
                "aspect_ratios": (1.0, 1.0, 1.0),
            }

    def calculate_surface_distance(self, point, coords):
        """
        Calculate distance from point to nearest surface atom.
        Much more appropriate for linear molecules than center distance.
        """
        distances = cdist([point], coords)[0]
        return np.min(distances)

    def generate_cocoon_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps=100,
        dt=40,
        target_distance=35.0,
        trajectory_step_size=None,
        approach_angle=0.0,
    ):
        """
        Generate adaptive cocoon trajectory based on molecular geometry.
        Automatically detects linear molecules (DNA) and uses appropriate method.

        Args:
            approach_angle: Starting angle offset (radians) for this approach
        """
        # Analyze target geometry
        geometry = self.analyze_molecular_geometry(protein_coords)

        print(f"\n   Molecular geometry analysis:")
        print(f"     Shape type: {geometry['shape_type']}")
        print(
            f"     Dimensions: {geometry['dimensions'][0]:.1f} x {geometry['dimensions'][1]:.1f} x {geometry['dimensions'][2]:.1f} Å"
        )
        print(
            f"     Aspect ratios: {geometry['aspect_ratios'][0]:.1f}:{geometry['aspect_ratios'][1]:.1f}:{geometry['aspect_ratios'][2]:.1f}"
        )

        # Calculate diffusion coefficient
        D = self.calculate_diffusion_coefficient(molecular_weight)

        if trajectory_step_size is not None:
            # Use user-defined step size
            step_size = trajectory_step_size
            print(f"     Using user-defined step size: {step_size:.1f} Å")
        else:
            # Calculate from diffusion
            step_size = np.sqrt(2 * D * dt)
            # Increase step size for more visible motion
            step_size *= 5.0
            print(f"     Calculated step size: {step_size:.1f} Å (from diffusion)")

        # Choose trajectory generation method based on shape
        if geometry["shape_type"] == "linear":
            print(f"     → Using cylindrical trajectory for linear molecule")
            return self.generate_linear_cocoon_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
                geometry,
                step_size,
                approach_angle,
            )
        else:
            print(f"     → Using spherical trajectory for {geometry['shape_type']} molecule")
            return self.generate_spherical_cocoon_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
                geometry,
                step_size,
                approach_angle,
            )

    def determine_dna_trajectory_mode(self, protein_coords, dna_coords, dna_geometry):
        """
        Determine the appropriate trajectory mode based on protein and DNA sizes.

        Returns:
            str: One of 'spiral', 'groove_following', 'sliding', 'docking', 'hybrid'
        """
        # Calculate protein size
        protein_center = np.mean(protein_coords, axis=0)
        protein_radius = np.max(np.linalg.norm(protein_coords - protein_center, axis=1))

        # DNA dimensions from geometry
        dna_radius = max(dna_geometry["dimensions"][1], dna_geometry["dimensions"][2]) / 2
        dna_length = dna_geometry["dimensions"][0]

        # Determine appropriate mode based on size ratio
        size_ratio = protein_radius / dna_radius

        print(f"\n   Trajectory mode selection:")
        print(f"     Protein radius: {protein_radius:.1f} Å")
        print(f"     DNA radius: {dna_radius:.1f} Å")
        print(f"     Size ratio: {size_ratio:.2f}")

        if size_ratio < 1.5:
            # Small peptides can spiral around DNA
            mode = "spiral"
            print(f"     Selected mode: SPIRAL (small peptide)")
        elif size_ratio < 3.0:
            # Medium proteins should follow grooves
            mode = "groove_following"
            print(f"     Selected mode: GROOVE FOLLOWING (medium protein)")
        elif size_ratio < 5.0:
            # Larger proteins slide along DNA
            mode = "sliding"
            print(f"     Selected mode: SLIDING (large protein)")
        else:
            # Very large proteins use docking approach
            mode = "docking"
            print(f"     Selected mode: DOCKING (very large protein)")

        return mode

    def generate_linear_cocoon_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps,
        dt,
        target_distance,
        geometry,
        step_size,
        approach_angle=0.0,
    ):
        """
        Generate adaptive trajectory for DNA based on protein size.

        This method automatically selects the appropriate trajectory type:
        - Spiral: For small peptides that can wrap around DNA
        - Groove following: For medium proteins that fit in grooves
        - Sliding: For large proteins that slide along DNA
        - Docking: For very large proteins

        IMPORTANT: In DNA-protein workflow context:
        - protein_coords = DNA coordinates (the target)
        - ligand_coords = protein coordinates (the mobile molecule)

        Args:
            approach_angle: Starting angle offset for trajectory
        """
        # Determine trajectory mode
        # CRITICAL: In DNA-protein workflow, the naming is reversed:
        # - protein_coords = DNA (the target)
        # - ligand_coords = protein (the mobile molecule)
        # So we pass ligand_coords as protein and protein_coords as DNA
        mode = self.determine_dna_trajectory_mode(ligand_coords, protein_coords, geometry)

        # Call appropriate trajectory generator
        if mode == "spiral":
            return self.generate_spiral_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
                geometry,
                step_size,
                approach_angle,
            )
        elif mode == "groove_following":
            return self.generate_groove_following_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
                geometry,
                step_size,
                approach_angle,
            )
        elif mode == "sliding":
            return self.generate_sliding_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
                geometry,
                step_size,
                approach_angle,
            )
        else:  # docking
            return self.generate_docking_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
                geometry,
                step_size,
                approach_angle,
            )

    def generate_spiral_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps,
        dt,
        target_distance,
        geometry,
        step_size,
        approach_angle=0.0,
    ):
        """
        Improved spiral trajectory with true helical motion along DNA axis.
        Creates a nail/screw-like trajectory that progresses steadily along DNA.
        """
        center = geometry["center"]
        axes = geometry["principal_axes"]
        dimensions = geometry["dimensions"]

        # Major axis is the first principal component
        major_axis = axes[:, 0]
        minor_axis_1 = axes[:, 1]
        minor_axis_2 = axes[:, 2]

        # Linear molecule parameters
        length = dimensions[0]
        radius = max(dimensions[1], dimensions[2]) / 2

        # Get actual DNA bounds in world coordinates
        # Project all DNA coords onto major axis to find true extent
        dna_projections = [(coord - center).dot(major_axis) for coord in protein_coords]
        z_min_local = min(dna_projections)
        z_max_local = max(dna_projections)
        actual_length = z_max_local - z_min_local

        # Debug: Print center and geometry info
        print(f"     DNA center (from geometry): {center}")
        print(f"     DNA actual bounds on major axis: [{z_min_local:.1f}, {z_max_local:.1f}] Å")
        print(f"     DNA actual length: {actual_length:.1f} Å (geometry length: {length:.1f} Å)")
        print(f"     DNA radius: {radius:.1f} Å")
        print(
            f"     Approach angle: {approach_angle:.2f} radians ({np.degrees(approach_angle):.1f}°)"
        )

        trajectory = []
        times = []

        # TRUE SPIRAL PARAMETERS - like a screw thread
        n_helical_turns = 8.0  # Total number of complete rotations along DNA
        pitch = actual_length / n_helical_turns  # Distance along axis per rotation

        # Add per-approach random state for reproducible but unique trajectories
        approach_random_state = np.random.RandomState(int(approach_angle * 1000) % 2**32)

        print(f"     TRUE HELICAL parameters: {n_helical_turns} turns")
        print(f"     Pitch: {pitch:.1f} Å per turn")
        print(f"     Angular velocity: {n_helical_turns * 360 / n_steps:.1f}°/step")

        # Determine starting end and direction based on approach angle
        approach_idx = int(approach_angle / (np.pi / 2))  # 0, 1, 2, or 3
        start_from_negative = (approach_idx % 2) == 1  # Alternate starting ends
        reverse_direction = approach_idx >= 2  # Reverse spiral direction for approaches 2,3

        # Add random variation to initial angle
        angle_variation = approach_random_state.uniform(-np.pi / 4, np.pi / 4)  # ±45° variation
        initial_theta = approach_angle + angle_variation

        print(
            f"     Approach {approach_idx + 1}: Starting from {'negative' if start_from_negative else 'positive'} Z end"
        )
        print(
            f"     Spiral direction: {'counter-clockwise' if not reverse_direction else 'clockwise'}"
        )
        print(f"     Initial angle: {np.degrees(initial_theta):.1f}°")

        # Calculate actual surface distance for proper hovering distance
        sample_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        surface_distances = []
        for ang in sample_angles:
            test_point = center + radius * (np.cos(ang) * minor_axis_1 + np.sin(ang) * minor_axis_2)
            min_dist = self.calculate_surface_distance(test_point, protein_coords)
            surface_distances.append(min_dist)

        actual_surface_radius = np.median(surface_distances)
        print(f"     Actual surface radius: {actual_surface_radius:.1f} Å")

        # Track collision statistics
        collision_count = 0
        successful_points = 0

        # ADAPTIVE Z-PROGRESSION: Track actual Z progress, not time-based
        z_progress = 0.0  # Current progress along DNA (0 to 1)
        z_step_base = 1.0 / n_steps  # Base step size for Z progression
        consecutive_failures = 0  # Track consecutive collision failures
        max_consecutive_failures = 10  # Force progression after this many failures

        # Track last successful Z position to ensure progression
        last_successful_z_local = z_min_local if start_from_negative else z_max_local

        i = 0
        while i < n_steps and z_progress < 0.99:  # Continue until we cover the DNA
            # Calculate current Z position based on actual progress
            if start_from_negative:
                z_local = z_min_local + z_progress * actual_length
            else:
                z_local = z_max_local - z_progress * actual_length

            # Calculate helical angle based on Z progress (not time)
            angle_progress = z_progress * n_helical_turns * 2 * np.pi
            if reverse_direction:
                theta = initial_theta - angle_progress
            else:
                theta = initial_theta + angle_progress

            # Radial distance with controlled variations
            r_base = actual_surface_radius + target_distance

            # Add radial variations for realistic motion
            # 1. Small oscillations to simulate breathing motion
            r_oscillation = 3.0 * np.sin(2 * np.pi * t * 5.3 + approach_angle)

            # 2. Random hovering motion
            r_random = approach_random_state.normal(0, 2.0)

            # 3. Periodic close approaches for interaction sampling
            close_approach = 0.0
            approach_frequency = 0.15  # 15% of the time
            if approach_random_state.random() < approach_frequency:
                close_approach = -min(10.0, target_distance * 0.3)  # Approach closer

            # Combine all radial components
            r = r_base + r_oscillation + r_random + close_approach
            r = max(actual_surface_radius + 2.0, r)  # Maintain minimum distance

            # Add small random perturbations to make trajectory more realistic
            theta_noise = approach_random_state.normal(0, 0.05)  # Small angular noise
            z_noise = approach_random_state.normal(0, pitch * 0.02)  # Small axial noise

            theta_final = theta + theta_noise
            z_local_final = z_local + z_noise

            # Convert cylindrical to Cartesian in local frame
            x_local = r * np.cos(theta_final)
            y_local = r * np.sin(theta_final)

            # Keep z_local separate for clarity
            z_position = z_local_final

            # Transform to world coordinates using principal axes
            local_pos = x_local * minor_axis_1 + y_local * minor_axis_2 + z_position * major_axis
            world_pos = center + local_pos

            # Check collision
            test_coords = ligand_coords + (world_pos - ligand_coords.mean(axis=0))

            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                trajectory.append(world_pos)
                times.append(i * dt)
                successful_points += 1
                consecutive_failures = 0  # Reset failure counter
                last_successful_z_local = z_local  # Update last successful Z

                # Advance Z-progress after successful point
                z_progress += z_step_base
            else:
                collision_count += 1
                consecutive_failures += 1
                point_added = False

                # Strategy 1: Try backing off radially
                for r_adjust in np.linspace(2, 10, 5):
                    r_test = r + r_adjust
                    x_test = r_test * np.cos(theta_final)
                    y_test = r_test * np.sin(theta_final)
                    local_test = (
                        x_test * minor_axis_1 + y_test * minor_axis_2 + z_position * major_axis
                    )
                    world_test = center + local_test
                    test_coords = ligand_coords + (world_test - ligand_coords.mean(axis=0))

                    if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                        trajectory.append(world_test)
                        times.append(i * dt)
                        successful_points += 1
                        consecutive_failures = 0
                        last_successful_z_local = z_local
                        z_progress += z_step_base
                        point_added = True
                        break

                # Strategy 2: If radial backoff failed, try small Z adjustments
                if not point_added:
                    for z_adjust in [-pitch * 0.1, pitch * 0.1, -pitch * 0.2, pitch * 0.2]:
                        z_test = z_position + z_adjust
                        local_test = (
                            x_local * minor_axis_1 + y_local * minor_axis_2 + z_test * major_axis
                        )
                        world_test = center + local_test
                        test_coords = ligand_coords + (world_test - ligand_coords.mean(axis=0))

                        if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                            trajectory.append(world_test)
                            times.append(i * dt)
                            successful_points += 1
                            consecutive_failures = 0
                            # Still advance Z even with adjustment
                            z_progress += z_step_base
                            point_added = True
                            break

                # Strategy 3: Force Z progression if stuck
                if not point_added and consecutive_failures >= max_consecutive_failures:
                    # Force progression by jumping ahead
                    z_progress += z_step_base * 3  # Triple step to escape stuck region
                    consecutive_failures = 0
                    print(f"     WARNING: Forced Z-progression at z_progress={z_progress:.3f}")

            i += 1

        trajectory = np.array(trajectory)
        times = np.array(times[: len(trajectory)])

        # Print statistics
        print(f"\n   ADAPTIVE SPIRAL trajectory statistics:")
        print(f"     Total iterations: {i}")
        print(f"     Successful points: {successful_points}")
        print(f"     Collisions handled: {collision_count}")
        print(f"     Z-progress achieved: {z_progress * 100:.1f}%")
        print(f"     Points per unit Z: {successful_points / max(z_progress, 0.01):.1f}")

        if len(trajectory) > 0:
            # Calculate Z-coverage
            trajectory_z = [(pos - center).dot(major_axis) for pos in trajectory]
            z_coverage = (max(trajectory_z) - min(trajectory_z)) / actual_length * 100

            # Calculate surface distances
            surface_distances = []
            for pos in trajectory:
                min_dist = self.calculate_surface_distance(pos, protein_coords)
                surface_distances.append(min_dist)

            surface_distances = np.array(surface_distances)

            print(f"\n   Coverage and distance statistics:")
            print(f"     Z-axis coverage: {z_coverage:.1f}% of DNA length")
            print(f"     Z range: [{min(trajectory_z):.1f}, {max(trajectory_z):.1f}] Å")
            print(
                f"     Distance from surface: {np.min(surface_distances):.1f} - {np.max(surface_distances):.1f} Å"
            )
            print(f"     Mean distance: {np.mean(surface_distances):.1f} Å")
            print(
                f"     Close approaches (<3.5Å): {np.sum(surface_distances < 3.5)} frames ({100 * np.sum(surface_distances < 3.5) / len(surface_distances):.1f}%)"
            )

            # Verify spiral motion
            trajectory_angles = []
            for pos in trajectory:
                rel_pos = pos - center
                x_proj = rel_pos.dot(minor_axis_1)
                y_proj = rel_pos.dot(minor_axis_2)
                angle = np.arctan2(y_proj, x_proj)
                trajectory_angles.append(angle)

            if len(trajectory_angles) > 1:
                angle_changes = np.diff(np.unwrap(trajectory_angles))
                total_rotation = np.sum(angle_changes)
                actual_turns = abs(total_rotation) / (2 * np.pi)
                print(
                    f"     Actual spiral rotation: {abs(np.degrees(total_rotation)):.1f}° ({actual_turns:.1f} turns)"
                )
                print(
                    f"     Spiral integrity: {'GOOD' if actual_turns > n_helical_turns * 0.8 else 'POOR'}"
                )

        # FALLBACK: If Z-coverage is too low, add sparse points along DNA
        if len(trajectory) > 0 and z_coverage < 50:
            print(
                f"\n   WARNING: Low Z-coverage ({z_coverage:.1f}%), adding sparse coverage points"
            )

            # Add points at larger radius to ensure no collisions
            fallback_radius = actual_surface_radius + target_distance * 2
            n_fallback = 20

            for j in range(n_fallback):
                t_fallback = j / (n_fallback - 1)

                if start_from_negative:
                    z_fallback = z_min_local + t_fallback * actual_length
                else:
                    z_fallback = z_max_local - t_fallback * actual_length

                theta_fallback = initial_theta + t_fallback * n_helical_turns * 2 * np.pi

                x_fallback = fallback_radius * np.cos(theta_fallback)
                y_fallback = fallback_radius * np.sin(theta_fallback)

                local_fallback = (
                    x_fallback * minor_axis_1 + y_fallback * minor_axis_2 + z_fallback * major_axis
                )
                world_fallback = center + local_fallback

                trajectory = np.vstack([trajectory, world_fallback])
                times = np.append(times, times[-1] + dt if len(times) > 0 else 0)

            print(f"     Added {n_fallback} fallback points for coverage")

        return trajectory, times

    def generate_groove_following_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps,
        dt,
        target_distance,
        geometry,
        step_size,
        approach_angle=0.0,
    ):
        """
        Generate trajectory that follows DNA major and minor grooves.
        Suitable for medium-sized proteins that can fit into grooves.
        """
        center = geometry["center"]
        axes = geometry["principal_axes"]
        dimensions = geometry["dimensions"]

        # DNA parameters
        major_axis = axes[:, 0]
        minor_axis_1 = axes[:, 1]
        minor_axis_2 = axes[:, 2]
        length = dimensions[0]
        radius = max(dimensions[1], dimensions[2]) / 2

        print(f"\n   GROOVE-FOLLOWING TRAJECTORY")
        print(f"     DNA length: {length:.1f} Å, radius: {radius:.1f} Å")

        trajectory = []
        times = []

        # DNA B-form parameters
        HELIX_PITCH = 34.0  # Å per complete turn
        BP_RISE = 3.4  # Å per base pair
        MAJOR_GROOVE_WIDTH = 22.0  # Å
        MINOR_GROOVE_WIDTH = 12.0  # Å
        MAJOR_GROOVE_DEPTH = 8.5  # Å
        MINOR_GROOVE_DEPTH = 7.5  # Å

        # Determine which groove to follow based on approach angle
        # Even approaches follow major groove, odd follow minor
        follow_major = (int(approach_angle / (np.pi / 4)) % 2) == 0
        groove_type = "major" if follow_major else "minor"
        groove_width = MAJOR_GROOVE_WIDTH if follow_major else MINOR_GROOVE_WIDTH
        groove_depth = MAJOR_GROOVE_DEPTH if follow_major else MINOR_GROOVE_DEPTH

        print(f"     Following {groove_type} groove")
        print(f"     Groove width: {groove_width:.1f} Å, depth: {groove_depth:.1f} Å")

        # Random state for this approach
        approach_random_state = np.random.RandomState(int(approach_angle * 1000) % 2**32)

        # Starting position along DNA
        start_z = approach_random_state.uniform(-length / 2 + 10, length / 2 - 10)

        # Groove angle offset (major groove at 0°, minor at 144°)
        groove_angle_offset = 0 if follow_major else np.radians(144)

        for i in range(n_steps):
            t = i / (n_steps - 1)

            # Progress along DNA axis
            z_position = start_z + (t - 0.5) * length * 0.8  # Cover 80% of DNA length
            z_position = np.clip(z_position, -length / 2 + 5, length / 2 - 5)

            # Helical position following DNA twist
            helix_angle = (
                (z_position / HELIX_PITCH) * 2 * np.pi + groove_angle_offset + approach_angle
            )

            # Position in groove with some lateral movement
            lateral_offset = groove_width * 0.3 * np.sin(t * 4 * np.pi)  # Oscillate within groove

            # Distance from DNA axis (in groove)
            r_groove = radius + groove_depth / 2 + lateral_offset

            # Add hovering motion
            hover_distance = target_distance + 5.0 * np.sin(t * 3 * np.pi)
            hover_distance += approach_random_state.normal(0, 2.0)
            r = r_groove + hover_distance

            # Convert to Cartesian coordinates
            x_local = r * np.cos(helix_angle)
            y_local = r * np.sin(helix_angle)
            z_local = z_position

            # Apply small perturbations
            x_local += approach_random_state.normal(0, step_size * 0.2)
            y_local += approach_random_state.normal(0, step_size * 0.2)
            z_local += approach_random_state.normal(0, step_size * 0.1)

            # Transform to world coordinates
            local_pos = x_local * minor_axis_1 + y_local * minor_axis_2 + z_local * major_axis
            world_pos = center + local_pos

            # Check collision
            test_coords = ligand_coords + (world_pos - ligand_coords.mean(axis=0))

            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                trajectory.append(world_pos)
                times.append(i * dt)

        trajectory = np.array(trajectory)
        times = np.array(times[: len(trajectory)])

        # Print statistics
        if len(trajectory) > 0:
            surface_distances = []
            for pos in trajectory:
                min_dist = self.calculate_surface_distance(pos, protein_coords)
                surface_distances.append(min_dist)

            surface_distances = np.array(surface_distances)
            print(f"\n   Groove-following statistics:")
            print(f"     Trajectory points: {len(trajectory)}")
            print(f"     Mean distance from surface: {np.mean(surface_distances):.1f} Å")
            print(f"     Groove type: {groove_type}")

        return trajectory, times

    def generate_sliding_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps,
        dt,
        target_distance,
        geometry,
        step_size,
        approach_angle=0.0,
    ):
        """
        Generate sliding trajectory along DNA axis.
        Suitable for large proteins that slide along DNA (1D diffusion).
        """
        center = geometry["center"]
        axes = geometry["principal_axes"]
        dimensions = geometry["dimensions"]

        # DNA parameters
        major_axis = axes[:, 0]
        minor_axis_1 = axes[:, 1]
        minor_axis_2 = axes[:, 2]
        length = dimensions[0]
        radius = max(dimensions[1], dimensions[2]) / 2

        print(f"\n   SLIDING TRAJECTORY")
        print(f"     DNA length: {length:.1f} Å, radius: {radius:.1f} Å")

        trajectory = []
        times = []

        # Random state for this approach
        approach_random_state = np.random.RandomState(int(approach_angle * 1000) % 2**32)

        # Initial contact position
        initial_angle = approach_angle + approach_random_state.uniform(-np.pi / 4, np.pi / 4)

        # Starting position along DNA
        start_z = approach_random_state.uniform(-length / 3, length / 3)

        # Calculate actual surface distance
        test_point = center + radius * (
            np.cos(initial_angle) * minor_axis_1 + np.sin(initial_angle) * minor_axis_2
        )
        actual_surface_dist = self.calculate_surface_distance(test_point, protein_coords)

        print(f"     Initial angle: {np.degrees(initial_angle):.1f}°")
        print(f"     Starting Z position: {start_z:.1f} Å")

        for i in range(n_steps):
            t = i / (n_steps - 1)

            # Slide along DNA with 1D diffusion
            z_drift = (t - 0.5) * length * 0.6  # Drift along DNA
            z_diffusion = approach_random_state.normal(0, step_size * 2.0) * np.sqrt(t + 0.1)
            z_position = start_z + z_drift + z_diffusion
            z_position = np.clip(z_position, -length / 2 + 10, length / 2 - 10)

            # Maintain contact with slow rotation
            rotation_speed = 0.5  # Slow rotation while sliding
            theta = initial_angle + t * rotation_speed * 2 * np.pi
            theta += approach_random_state.normal(0, 0.1)  # Small angular diffusion

            # Distance from DNA surface (maintaining close contact)
            contact_distance = target_distance + 3.0 * np.sin(t * 2 * np.pi)
            contact_distance += approach_random_state.normal(0, 1.0)
            contact_distance = max(2.0, contact_distance)  # Maintain minimum distance

            r = actual_surface_dist + contact_distance

            # Convert to Cartesian coordinates
            x_local = r * np.cos(theta)
            y_local = r * np.sin(theta)
            z_local = z_position

            # Transform to world coordinates
            local_pos = x_local * minor_axis_1 + y_local * minor_axis_2 + z_local * major_axis
            world_pos = center + local_pos

            # Check collision
            test_coords = ligand_coords + (world_pos - ligand_coords.mean(axis=0))

            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                trajectory.append(world_pos)
                times.append(i * dt)

        trajectory = np.array(trajectory)
        times = np.array(times[: len(trajectory)])

        # Print statistics
        if len(trajectory) > 0:
            print(f"\n   Sliding trajectory statistics:")
            print(f"     Trajectory points: {len(trajectory)}")
            print(
                f"     Z-range covered: {np.ptp([p.dot(major_axis) for p in trajectory - center]):.1f} Å"
            )

        return trajectory, times

    def generate_docking_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps,
        dt,
        target_distance,
        geometry,
        step_size,
        approach_angle=0.0,
    ):
        """
        Generate docking-style approach trajectory.
        Suitable for very large proteins that approach DNA from multiple angles.
        """
        center = geometry["center"]
        axes = geometry["principal_axes"]
        dimensions = geometry["dimensions"]

        # DNA parameters
        major_axis = axes[:, 0]
        minor_axis_1 = axes[:, 1]
        minor_axis_2 = axes[:, 2]
        length = dimensions[0]
        radius = max(dimensions[1], dimensions[2]) / 2

        print(f"\n   DOCKING TRAJECTORY")
        print(f"     DNA length: {length:.1f} Å, radius: {radius:.1f} Å")

        trajectory = []
        times = []

        # Random state for this approach
        approach_random_state = np.random.RandomState(int(approach_angle * 1000) % 2**32)

        # Docking parameters
        n_docking_attempts = 5  # Multiple approach attempts

        for attempt in range(n_docking_attempts):
            # Random approach vector for this attempt
            phi = approach_angle + attempt * (2 * np.pi / n_docking_attempts)
            theta_tilt = approach_random_state.uniform(
                np.pi / 4, 3 * np.pi / 4
            )  # Approach angle from DNA axis

            # Random position along DNA
            z_target = approach_random_state.uniform(-length / 3, length / 3)

            # Steps for this docking attempt
            steps_per_attempt = n_steps // n_docking_attempts

            for i in range(steps_per_attempt):
                t = i / (steps_per_attempt - 1)

                # Approach from far to close
                approach_dist = target_distance * 3 * (1 - t) + target_distance * t

                # Position with respect to DNA
                r = radius + approach_dist

                # Add oscillation during approach
                r += 5.0 * np.sin(t * 2 * np.pi)

                # Convert spherical to Cartesian
                x_local = r * np.sin(theta_tilt) * np.cos(phi)
                y_local = r * np.sin(theta_tilt) * np.sin(phi)
                z_local = z_target + (1 - t) * length * 0.2  # Slight drift during approach

                # Add small perturbations
                x_local += approach_random_state.normal(0, step_size * 0.3)
                y_local += approach_random_state.normal(0, step_size * 0.3)
                z_local += approach_random_state.normal(0, step_size * 0.2)

                # Transform to world coordinates
                local_pos = x_local * minor_axis_1 + y_local * minor_axis_2 + z_local * major_axis
                world_pos = center + local_pos

                # Check collision
                test_coords = ligand_coords + (world_pos - ligand_coords.mean(axis=0))

                if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                    trajectory.append(world_pos)
                    times.append((attempt * steps_per_attempt + i) * dt)

        # Convert to arrays
        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            times = np.array(times[: len(trajectory)])
        else:
            # FALLBACK: If no collision-free positions found, create minimal trajectory
            print(f"\n   WARNING: No collision-free positions found in docking trajectory")
            print(f"     Generating fallback trajectory at larger distance")

            # Try with much larger distances
            fallback_points = []
            for i in range(min(10, n_steps)):
                t = i / (min(10, n_steps) - 1)

                # Use much larger distance
                r = radius + target_distance * 5
                phi = approach_angle + t * np.pi
                theta_tilt = np.pi / 2

                x_local = r * np.cos(phi)
                y_local = r * np.sin(phi)
                z_local = 0

                local_pos = x_local * minor_axis_1 + y_local * minor_axis_2 + z_local * major_axis
                world_pos = center + local_pos

                fallback_points.append(world_pos)

            trajectory = np.array(fallback_points)
            times = np.linspace(0, n_steps * dt, len(trajectory))

        # Print statistics
        print(f"\n   Docking trajectory statistics:")
        print(f"     Trajectory points: {len(trajectory)}")
        print(f"     Docking attempts: {n_docking_attempts}")
        if len(trajectory) < n_steps / 2:
            print(f"     WARNING: Low success rate, consider increasing target distance")

        return trajectory, times

    def generate_spherical_cocoon_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps,
        dt,
        target_distance,
        geometry,
        step_size,
        approach_angle=0.0,
    ):
        """
        Original spherical cocoon trajectory for globular molecules.
        Enhanced with surface distance calculations.

        Args:
            approach_angle: Starting angle offset for spherical trajectory
        """
        protein_center = geometry["center"]
        principal_axes = geometry["principal_axes"]

        # Initialize spherical coordinates for winding motion with approach-specific angle
        # Use approach_angle to ensure different starting positions
        theta = approach_angle
        phi = np.pi / 2 + 0.3 * np.sin(approach_angle)  # Vary phi based on approach

        # Initial distance with more variation allowed
        current_radius = target_distance + np.max(cdist([protein_center], protein_coords)[0])

        # Angular velocities for winding motion
        theta_velocity = 2 * np.pi / (n_steps / 4)  # Complete ~4 winds
        phi_velocity = 0.0

        # FIXED: Allow much closer approaches for H-bond detection
        min_distance = 2.0  # Allow approach to 2.0Å (H-bonds need < 3.5Å)
        max_distance = target_distance * 2.5

        # Add periodic close approaches for sampling H-bonds
        close_approach_frequency = n_steps // 10  # Make 10 close approaches

        trajectory = []
        times = []

        # Momentum terms for smoother motion
        distance_momentum = 0.0
        theta_momentum = 0.0
        phi_momentum = 0.0

        for i in range(n_steps):
            # Update angular position with momentum (winding motion)
            theta_momentum += np.random.randn() * 0.1
            phi_momentum += np.random.randn() * 0.05

            # Apply damping to momentum
            theta_momentum *= 0.95
            phi_momentum *= 0.95

            # Update angles
            theta += theta_velocity + theta_momentum
            phi += phi_velocity + phi_momentum

            # Keep phi in valid range
            phi = np.clip(phi, 0.1, np.pi - 0.1)

            # Distance variation with momentum
            distance_force = np.random.randn() * 2.0

            # FIXED: Add periodic close approaches for H-bond sampling
            if i % close_approach_frequency == 0:
                # Force a close approach every N steps
                distance_force -= (
                    20.0  # ENHANCED: Increased attractive force for better H-bond sampling
                )

                # Override minimum distance temporarily for H-bond detection
                min_distance = 1.5  # Allow very close approach for H-bonds
            else:
                # Reset to normal minimum distance after close approach period
                min_distance = 2.0

            # Add oscillatory component for natural in/out motion
            oscillation = 5.0 * np.sin(2 * np.pi * i / (n_steps / 6))

            # Update distance momentum
            distance_momentum += distance_force + oscillation * 0.1
            distance_momentum *= 0.9  # Damping

            # Update radius
            current_radius += distance_momentum

            # Enforce boundaries with gentler force during H-bond sampling
            if current_radius < min_distance:
                # For H-bond sampling, allow temporary violations with gentler repulsion
                if i % close_approach_frequency < 5:  # 5 steps of close approach
                    distance_momentum += (
                        min_distance - current_radius
                    ) * 0.1  # FIXED: Reduced from 0.5 for H-bond sampling
                else:
                    distance_momentum += (min_distance - current_radius) * 0.5  # Normal repulsion
            elif current_radius > max_distance:
                distance_momentum -= (current_radius - max_distance) * 0.1

            # Convert spherical to Cartesian in protein's frame
            x = current_radius * np.sin(phi) * np.cos(theta)
            y = current_radius * np.sin(phi) * np.sin(theta)
            z = current_radius * np.cos(phi)

            # Transform to world coordinates using principal axes
            pos_local = np.array([x, y, z])
            pos_world = protein_center + principal_axes @ pos_local

            # Add Brownian displacement for additional randomness
            brownian = np.random.randn(3) * step_size * 0.5
            pos_world += brownian

            # Check collision
            test_coords = ligand_coords + (pos_world - ligand_coords.mean(axis=0))

            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                trajectory.append(pos_world.copy())
                times.append(i * dt)
            else:
                # Collision detected - try adjusting radius
                # For close approaches, try backing off gradually
                # FIXED: Allow closer approaches for H-bond sampling (down to 0.1Å adjustment)
                radius_adjustments = np.linspace(0.1, 2.0, 20)

                for radius_adjust in radius_adjustments:
                    test_radius = current_radius + radius_adjust

                    # Recalculate position with adjusted radius
                    x_adj = test_radius * np.sin(phi) * np.cos(theta)
                    y_adj = test_radius * np.sin(phi) * np.sin(theta)
                    z_adj = test_radius * np.cos(phi)

                    pos_local_adj = np.array([x_adj, y_adj, z_adj])
                    pos_world_adj = protein_center + principal_axes @ pos_local_adj

                    test_coords = ligand_coords + (pos_world_adj - ligand_coords.mean(axis=0))

                    if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                        trajectory.append(pos_world_adj.copy())
                        times.append(i * dt)
                        current_radius = test_radius
                        break

        # Convert to numpy arrays
        trajectory = np.array(trajectory)
        times = np.array(times[: len(trajectory)])

        # Calculate surface distances (improved method)
        if len(trajectory) > 0:
            surface_distances = []
            for pos in trajectory:
                min_dist = self.calculate_surface_distance(pos, protein_coords)
                surface_distances.append(min_dist)

            surface_distances = np.array(surface_distances)

            print(f"\n   Trajectory distance statistics:")
            print(f"     Min distance from surface: {np.min(surface_distances):.1f} Å")
            print(f"     Max distance from surface: {np.max(surface_distances):.1f} Å")
            print(f"     Mean distance from surface: {np.mean(surface_distances):.1f} Å")
            print(f"     Close approaches (<3.5Å): {np.sum(surface_distances < 3.5)} frames")
            print(
                f"     H-bond range (<3.5Å): {100 * np.sum(surface_distances < 3.5) / len(surface_distances):.1f}% of frames"
            )

        # Ensure we have enough points
        if len(trajectory) < n_steps // 2:
            print(f"Warning: Only {len(trajectory)} collision-free positions found")
            return self.generate_simple_cocoon_trajectory(
                protein_coords,
                ligand_coords,
                ligand_atoms,
                molecular_weight,
                n_steps,
                dt,
                target_distance,
            )

        return trajectory, times

    def generate_simple_cocoon_trajectory(
        self,
        protein_coords,
        ligand_coords,
        ligand_atoms,
        molecular_weight,
        n_steps=100,
        dt=40,
        target_distance=35.0,
    ):
        """Fallback simple cocoon trajectory for when winding fails"""
        # Calculate diffusion coefficient
        D = self.calculate_diffusion_coefficient(molecular_weight)
        step_size = np.sqrt(2 * D * dt) * 5.0

        protein_center = np.mean(protein_coords, axis=0)
        max_protein_radius = np.max(cdist([protein_center], protein_coords)[0])

        trajectory = []
        times = []

        # Generate random points on expanding/contracting sphere
        for i in range(n_steps):
            # Random spherical coordinates
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            # Varying radius
            radius_variation = 20.0 * np.sin(2 * np.pi * i / n_steps)
            radius = max_protein_radius + target_distance + radius_variation

            # Convert to Cartesian
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            pos = protein_center + np.array([x, y, z])

            # Add Brownian noise
            pos += np.random.randn(3) * step_size

            # Check collision
            test_coords = ligand_coords + (pos - ligand_coords.mean(axis=0))

            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                trajectory.append(pos)
                times.append(i * dt)

        return np.array(trajectory), np.array(times)

    def generate_brownian_trajectory_collision_free(
        self,
        start_pos,
        end_pos,
        n_steps,
        ligand_coords,
        ligand_atoms,
        molecular_weight=300.0,
        dt=40,
        biased=True,
    ):
        """
        Generate physics-based Brownian trajectory

        Args:
            start_pos: Starting position
            end_pos: Target end position
            n_steps: Number of trajectory steps
            ligand_coords: Ligand atom coordinates
            ligand_atoms: Ligand atom data
            molecular_weight: Molecular weight for diffusion calculation
            dt: Time step (fs)
            biased: If True, bias trajectory toward target (default). If False, pure random walk.
        """
        # Calculate diffusion coefficient
        D = self.calculate_diffusion_coefficient(molecular_weight)

        # Calculate step size using Brownian motion theory
        # For Brownian motion: <x²> = 2Dt
        step_size = np.sqrt(2 * D * dt)

        trajectory = [start_pos]
        times = [0]
        current_pos = start_pos.copy()

        # Calculate target distance (distance from start to protein center)
        # Note: protein center should be passed as parameter, using origin as approximation
        protein_center = np.array([0.0, 0.0, 0.0])  # Protein is centered at origin
        target_distance = np.linalg.norm(start_pos - protein_center)

        # Gradually decrease target distance toward end position
        end_distance = np.linalg.norm(end_pos - protein_center)

        for i in range(1, n_steps):
            # Generate random displacement (true Brownian motion)
            displacement = np.random.randn(3) * step_size

            # Propose new position
            proposed_pos = current_pos + displacement

            # Calculate target distance for this step (linear interpolation)
            t = i / n_steps
            current_target_distance = (1 - t) * target_distance + t * end_distance

            # Adjust position to maintain approximate target distance
            direction_to_proposed = proposed_pos - protein_center
            distance_to_proposed = np.linalg.norm(direction_to_proposed)

            if biased and distance_to_proposed > 0:
                # For biased trajectory: blend physics with distance constraint
                direction_unit = direction_to_proposed / distance_to_proposed
                distance_constrained_pos = protein_center + direction_unit * current_target_distance

                # Weighted combination (favor physics, but respect constraints)
                final_pos = 0.7 * proposed_pos + 0.3 * distance_constrained_pos
            else:
                # For unbiased trajectory: use pure random displacement
                final_pos = proposed_pos

            # Check collision
            test_coords = ligand_coords + (final_pos - ligand_coords.mean(axis=0))

            if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                current_pos = final_pos
            else:
                # Try smaller steps if collision detected
                for scale in [0.5, 0.25, 0.1]:
                    smaller_displacement = displacement * scale
                    smaller_pos = current_pos + smaller_displacement
                    test_coords = ligand_coords + (smaller_pos - ligand_coords.mean(axis=0))

                    if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                        current_pos = smaller_pos
                        break
                # If all collision attempts fail, stay at current position

            trajectory.append(current_pos.copy())
            times.append(i * dt)

        return np.array(trajectory)

    def generate_random_walk_trajectory(
        self,
        start_pos,
        n_steps,
        ligand_coords,
        ligand_atoms,
        molecular_weight=300.0,
        dt=40,
        max_distance=None,
        max_attempts=10,
    ):
        """
        Generate a truly random Brownian walk with no directional bias.

        Args:
            start_pos: Starting position
            n_steps: Number of trajectory steps
            ligand_coords: Ligand atom coordinates
            ligand_atoms: Ligand atom data
            molecular_weight: Molecular weight for diffusion calculation
            dt: Time step (fs)
            max_distance: Optional maximum distance from origin (for boundary)
            max_attempts: Maximum collision retries per step

        Returns:
            Array of positions shape (n_steps, 3)

        Note:
            This generates TRUE Brownian motion with no bias. The molecule
            will randomly diffuse according to Einstein's relation:
            <r²> = 6Dt

            With corrected diffusion coefficient, a 300 Da molecule will have:
            - D ≈ 7.4e-5 Å²/fs
            - RMS displacement after 1 ps: ~0.67 Å
            - RMS displacement after 1 ns: ~21 Å
        """
        # Calculate diffusion coefficient
        D = self.calculate_diffusion_coefficient(molecular_weight)

        # Calculate step size
        step_size = np.sqrt(2 * D * dt)

        trajectory = [start_pos]
        current_pos = start_pos.copy()
        protein_center = np.array([0.0, 0.0, 0.0])

        print(f"\nGenerating random walk:")
        print(f"  Molecular weight: {molecular_weight} Da")
        print(f"  Diffusion coefficient: {D:.6f} Å²/fs")
        print(f"  RMS step size: {step_size:.4f} Å per {dt} fs")
        print(f"  Total simulation: {n_steps * dt} fs = {n_steps * dt / 1000:.1f} ps")

        n_rejected = 0

        for i in range(1, n_steps):
            attempts = 0
            while attempts < max_attempts:
                displacement = np.random.randn(3) * step_size
                proposed_pos = current_pos + displacement

                if max_distance is not None:
                    distance_from_center = np.linalg.norm(proposed_pos - protein_center)
                    if distance_from_center > max_distance:
                        direction = (proposed_pos - protein_center) / distance_from_center
                        proposed_pos = protein_center + direction * (
                            2 * max_distance - distance_from_center
                        )

                test_coords = ligand_coords + (proposed_pos - ligand_coords.mean(axis=0))
                if not self.collision_detector.check_collision(test_coords, ligand_atoms):
                    current_pos = proposed_pos
                    break

                attempts += 1

            if attempts == max_attempts:
                n_rejected += 1

            trajectory.append(current_pos.copy())

        if n_rejected > 0:
            print(f"  Rejected {n_rejected}/{n_steps} steps due to collisions")

        return np.array(trajectory)

    def parse_structure(self, pdb_file, parse_heterogens=True):
        """Parse PDB structure and extract atom information

        Args:
            pdb_file: Path to PDB file
            parse_heterogens: If True, include HETATM records (needed for ligands)
        """
        structure = self.parser.get_structure("structure", pdb_file)

        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Handle both standard residues and heterogens
                    residue_id = residue.get_id()
                    hetatm_flag = residue_id[0]

                    # Skip water molecules
                    if residue.get_resname() in ["HOH", "WAT"]:
                        continue

                    # For proteins: only standard residues (hetatm_flag == ' ')
                    # For ligands: only heterogens (hetatm_flag != ' ')
                    if not parse_heterogens and hetatm_flag != " ":
                        continue

                    for atom in residue:
                        # Get element - BioPython should parse this from PDB
                        element = atom.element if hasattr(atom, "element") else None

                        # Fallback element detection
                        if not element or element.strip() == "":
                            atom_name = atom.get_name().strip()
                            # PDBQT atom type patterns
                            if atom_name.startswith("AC") or atom_name.startswith("A"):
                                element = "C"  # Aromatic carbon
                            elif atom_name.startswith("NA"):
                                element = "N"  # Aromatic nitrogen
                            elif atom_name.startswith("OA"):
                                element = "O"  # Aromatic oxygen
                            elif atom_name.startswith("SA"):
                                element = "S"  # Aromatic sulfur
                            # Common patterns (case-insensitive)
                            elif atom_name[:2].upper() in ["CL", "BR"] or atom_name in ["Cl", "Br"]:
                                element = (
                                    atom_name[:2].capitalize() if len(atom_name) > 1 else atom_name
                                )
                            elif atom_name and atom_name[0] in ["C", "N", "O", "S", "P", "H", "F"]:
                                element = atom_name[0]
                            else:
                                # Try harder to extract element
                                for e in ["C", "N", "O", "S", "P", "H", "F"]:
                                    if atom_name.startswith(e):
                                        element = e
                                        break
                                if not element:
                                    element = "C"  # Default to carbon
                                    print(
                                        f"Warning: Could not determine element for atom {atom_name}, defaulting to C"
                                    )

                        # Check if atom name indicates PDBQT aromatic type
                        is_aromatic_atom = False
                        if atom_name in ["A", "NA", "OA", "SA"] or atom_name.startswith(
                            ("AC", "NA", "OA", "SA")
                        ):
                            is_aromatic_atom = True

                        atom_info = {
                            "chain": chain.get_id(),
                            "resname": residue.get_resname(),
                            "resSeq": residue.get_id()[1],
                            "name": atom.get_name(),
                            "element": element.strip().upper() if element else "C",
                            "x": atom.get_coord()[0],
                            "y": atom.get_coord()[1],
                            "z": atom.get_coord()[2],
                            "atom_id": atom.get_serial_number(),
                            "is_hetatm": hetatm_flag != " ",
                            "residue_id": residue.get_id()[1],  # Add this for consistency
                            "is_aromatic": is_aromatic_atom,
                        }
                        atoms.append(atom_info)

        return pd.DataFrame(atoms)

    def parse_structure_robust(self, pdb_file, parse_heterogens=True):
        """
        Parse PDB structure with fallback to manual parsing for problematic files

        Args:
            pdb_file: Path to PDB file
            parse_heterogens: If True, include HETATM records (needed for ligands)

        Returns:
            pd.DataFrame: Parsed atom information
        """
        # First try BioPython parsing
        try:
            bio_atoms = self.parse_structure(pdb_file, parse_heterogens)

            # Check if we got reasonable number of atoms
            with open(pdb_file, "r") as f:
                lines = f.readlines()

            expected_atoms = sum(1 for line in lines if line.startswith(("ATOM", "HETATM")))

            if len(bio_atoms) >= expected_atoms * 0.9:  # Got at least 90% of atoms
                return bio_atoms
            else:
                print(f"  ⚠️  BioPython only parsed {len(bio_atoms)}/{expected_atoms} atoms")
                print(f"  Switching to manual parsing...")

        except Exception as e:
            print(f"  BioPython parsing failed: {e}")
            print(f"  Switching to manual parsing...")

        # Manual parsing fallback
        manual_atoms = []

        with open(pdb_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    record_type = line[0:6].strip()
                    atom_serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    res_seq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Element detection
                    element = ""
                    if len(line) >= 78:
                        element = line[76:78].strip()

                    # If element column is empty but line is long enough, check if element is at end of line
                    if not element and len(line) >= 77:
                        # OpenBabel sometimes puts element right-justified at end of line
                        potential_element = line.rstrip()[-2:].strip()
                        if potential_element in [
                            "H",
                            "C",
                            "N",
                            "O",
                            "F",
                            "P",
                            "S",
                            "CL",
                            "BR",
                            "I",
                        ]:
                            element = potential_element
                        elif len(line.rstrip()) >= 1 and line.rstrip()[-1] in [
                            "H",
                            "C",
                            "N",
                            "O",
                            "F",
                            "P",
                            "S",
                        ]:
                            element = line.rstrip()[-1]

                    # Check if element is PDBQT aromatic type
                    is_aromatic_atom = False
                    if element == "A":
                        element = "C"  # PDBQT aromatic carbon
                        is_aromatic_atom = True
                    elif element == "NA":
                        element = "N"  # PDBQT aromatic nitrogen
                        is_aromatic_atom = True
                    elif element == "OA":
                        element = "O"  # PDBQT aromatic oxygen
                        is_aromatic_atom = True
                    elif element == "SA":
                        element = "S"  # PDBQT aromatic sulfur
                        is_aromatic_atom = True

                    if not element:
                        # Guess from atom name (case-insensitive)
                        atom_name_upper = atom_name.upper()
                        if atom_name_upper.startswith("CL") or atom_name == "Cl":
                            element = "Cl"
                        elif atom_name_upper.startswith("BR") or atom_name == "Br":
                            element = "Br"
                        elif atom_name.startswith("AC") or atom_name.startswith("A"):
                            # PDBQT aromatic carbon
                            element = "C"
                        elif atom_name.startswith("NA"):
                            # PDBQT aromatic nitrogen
                            element = "N"
                        elif atom_name.startswith("OA"):
                            # PDBQT aromatic oxygen
                            element = "O"
                        elif atom_name.startswith("SA"):
                            # PDBQT aromatic sulfur
                            element = "S"
                        elif atom_name and atom_name[0] in ["C", "N", "O", "S", "P", "H", "F"]:
                            element = atom_name[0].upper()
                        else:
                            element = "C"  # Default

                    # Skip water
                    if res_name in ["HOH", "WAT"]:
                        continue

                    # Apply parse_heterogens filter
                    is_hetatm = record_type == "HETATM"
                    if not parse_heterogens and is_hetatm:
                        continue

                    manual_atoms.append(
                        {
                            "chain": chain_id if chain_id else "A",
                            "resname": res_name,
                            "resSeq": res_seq,
                            "name": atom_name,
                            "element": element,
                            "x": x,
                            "y": y,
                            "z": z,
                            "atom_id": atom_serial,
                            "is_hetatm": is_hetatm,
                            "residue_id": res_seq,
                            "is_aromatic": is_aromatic_atom,  # Track if from PDBQT aromatic type
                        }
                    )

                except Exception as e:
                    continue

        df = pd.DataFrame(manual_atoms)
        print(f"  ✓ Manual parsing succeeded: {len(df)} atoms")

        return df

    def calculate_pi_stacking(self, aromatic_atoms1, aromatic_atoms2):
        """Calculate pi-stacking interactions between aromatic rings"""
        if len(aromatic_atoms1) < 3 or len(aromatic_atoms2) < 3:
            return None

        # Calculate ring centers
        center1 = aromatic_atoms1[["x", "y", "z"]].mean()
        center2 = aromatic_atoms2[["x", "y", "z"]].mean()

        # Distance between centers
        distance = np.linalg.norm(center2 - center1)

        # Check if within pi-stacking range (3.4-4.5 Å typically)
        if distance > 4.5:  # Proper pi-stacking cutoff (was 7.0)
            return None

        # Calculate ring normals using SVD
        coords1 = aromatic_atoms1[["x", "y", "z"]].values - center1.values
        coords2 = aromatic_atoms2[["x", "y", "z"]].values - center2.values

        # SVD to find plane normal
        _, _, vh1 = np.linalg.svd(coords1)
        _, _, vh2 = np.linalg.svd(coords2)

        normal1 = vh1[2]
        normal2 = vh2[2]

        # Calculate angle between normals
        cos_angle = abs(np.dot(normal1, normal2))
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

        # Calculate offset
        center_vector = center2.values - center1.values
        offset = np.linalg.norm(center_vector - np.dot(center_vector, normal1) * normal1)

        # Determine stacking type
        if angle < 30:  # Parallel
            if offset < 2.0:
                stack_type = "Pi-Stacking-Parallel"
                energy = -5.0  # Stronger
            else:
                stack_type = "Pi-Stacking-Offset"
                energy = -3.5
        elif 60 < angle < 120:  # T-shaped
            stack_type = "Pi-Stacking-T-Shaped"
            energy = -4.0
        else:  # Angled
            stack_type = "Pi-Stacking-Angled"
            energy = -2.5

        # Apply distance penalty
        optimal_distance = 3.8
        distance_penalty = np.exp(-(((distance - optimal_distance) / 1.5) ** 2))
        energy *= distance_penalty

        return {
            "type": stack_type,
            "energy": energy,
            "distance": distance,
            "angle": angle,
            "offset": offset,
            "center1": center1.values,
            "center2": center2.values,
            "normal1": normal1,
            "normal2": normal2,
        }

    def calculate_interactions(self, protein_atoms, ligand_atoms, iteration_num):
        """Calculate all non-covalent interactions with protonation awareness"""
        # Use protonation-aware interaction detection
        interactions_df = calculate_interactions_with_protonation(
            protein_atoms, ligand_atoms, pH=self.physiological_pH, iteration_num=iteration_num
        )

        # Add intra-protein vectors to interactions
        if not interactions_df.empty and self.intra_protein_vectors:
            # Create residue IDs
            interactions_df["res_id"] = (
                interactions_df["protein_chain"].astype(str)
                + ":"
                + interactions_df["protein_residue"].astype(str)
            )

            # Initialize vector columns
            interactions_df["intra_vector_x"] = 0.0
            interactions_df["intra_vector_y"] = 0.0
            interactions_df["intra_vector_z"] = 0.0

            # Add intra-protein vectors
            for idx, row in interactions_df.iterrows():
                res_id = row["res_id"]
                if res_id in self.intra_protein_vectors:
                    intra_vector = self.intra_protein_vectors[res_id]
                    interactions_df.at[idx, "intra_vector_x"] = intra_vector[0]
                    interactions_df.at[idx, "intra_vector_y"] = intra_vector[1]
                    interactions_df.at[idx, "intra_vector_z"] = intra_vector[2]

            # Calculate inter-protein vectors (already in the df as vector_x/y/z)
            interactions_df["inter_vector_x"] = interactions_df["vector_x"]
            interactions_df["inter_vector_y"] = interactions_df["vector_y"]
            interactions_df["inter_vector_z"] = interactions_df["vector_z"]

            # Calculate combined vectors
            interactions_df["vector_x"] = (
                interactions_df["inter_vector_x"] + interactions_df["intra_vector_x"]
            )
            interactions_df["vector_y"] = (
                interactions_df["inter_vector_y"] + interactions_df["intra_vector_y"]
            )
            interactions_df["vector_z"] = (
                interactions_df["inter_vector_z"] + interactions_df["intra_vector_z"]
            )
            interactions_df["combined_magnitude"] = np.sqrt(
                interactions_df["vector_x"] ** 2
                + interactions_df["vector_y"] ** 2
                + interactions_df["vector_z"] ** 2
            )

            # Drop temporary column
            interactions_df = interactions_df.drop(columns=["res_id"])

        # Check for pi-stacking interactions (these are not yet protonation-aware)
        pi_stacking_interactions = self.detect_pi_stacking(
            protein_atoms, ligand_atoms, iteration_num
        )

        # Add intra-protein vectors to pi-stacking interactions
        for pi_interaction in pi_stacking_interactions:
            chain = pi_interaction.get("protein_chain", "A")
            residue_num = pi_interaction.get(
                "protein_residue", pi_interaction.get("protein_residue_id", 0)
            )
            res_id = f"{chain}:{residue_num}"

            # Get intra-protein vector
            intra_vector = np.zeros(3)
            if self.intra_protein_vectors and res_id in self.intra_protein_vectors:
                intra_vector = self.intra_protein_vectors[res_id]

            # Update pi-stacking interaction with vectors
            inter_vector = np.array(
                [
                    pi_interaction.get("vector_x", 0),
                    pi_interaction.get("vector_y", 0),
                    pi_interaction.get("vector_z", 0),
                ]
            )
            combined_vector = inter_vector + intra_vector

            pi_interaction.update(
                {
                    "inter_vector_x": inter_vector[0],
                    "inter_vector_y": inter_vector[1],
                    "inter_vector_z": inter_vector[2],
                    "intra_vector_x": intra_vector[0],
                    "intra_vector_y": intra_vector[1],
                    "intra_vector_z": intra_vector[2],
                    "vector_x": combined_vector[0],  # combined vector
                    "vector_y": combined_vector[1],
                    "vector_z": combined_vector[2],
                    "combined_magnitude": np.linalg.norm(combined_vector),
                    "pH": self.physiological_pH,  # Add pH info
                }
            )

        # Convert pi-stacking to dataframe and concatenate
        if pi_stacking_interactions:
            pi_df = pd.DataFrame(pi_stacking_interactions)
            interactions_df = pd.concat([interactions_df, pi_df], ignore_index=True)

        return interactions_df

    def detect_aromatic_rings_networkx(self, atoms_df):
        """Detect aromatic rings using networkx graph analysis"""
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx not installed. Using simple aromatic detection.")
            return []

        # Build molecular graph
        G = nx.Graph()

        # Add nodes (atoms)
        for idx, atom in atoms_df.iterrows():
            G.add_node(idx, element=atom["element"], coords=[atom["x"], atom["y"], atom["z"]])

        # Add edges (bonds) based on distance
        coords = atoms_df[["x", "y", "z"]].values
        for i in range(len(atoms_df)):
            for j in range(i + 1, len(atoms_df)):
                dist = np.linalg.norm(coords[i] - coords[j])
                # Typical C-C bond: 1.4-1.5 Å for aromatic, up to 1.6 Å for single
                if dist < 1.7:
                    G.add_edge(i, j)

        # Find cycles (rings)
        aromatic_rings = []
        cycles = nx.cycle_basis(G)

        for cycle in cycles:
            if 5 <= len(cycle) <= 6:  # 5 or 6-membered rings
                ring_atoms = atoms_df.iloc[cycle]
                # Check if ring is planar (aromatic)
                if len(ring_atoms) >= 4:
                    coords = ring_atoms[["x", "y", "z"]].values
                    # Calculate planarity using SVD
                    centered = coords - coords.mean(axis=0)
                    _, s, _ = np.linalg.svd(centered)
                    # If third singular value is small, atoms are coplanar
                    if s[2] < 0.3:  # Threshold for planarity
                        aromatic_rings.append(ring_atoms)

        return aromatic_rings

    def detect_pi_stacking(self, protein_atoms, ligand_atoms, iteration_num):
        """Detect pi-stacking interactions between aromatic systems"""
        pi_interactions = []

        # Find aromatic residues in protein
        for res_name in self.AROMATIC:
            # Get aromatic residues of this type
            aromatic_residues = protein_atoms[protein_atoms["resname"] == res_name][
                "resSeq"
            ].unique()

            for res_id in aromatic_residues:
                # Get atoms for this aromatic residue
                res_atoms = protein_atoms[
                    (protein_atoms["resSeq"] == res_id)
                    & (protein_atoms["resname"] == res_name)
                    & (protein_atoms["name"].isin(self.AROMATIC[res_name]))
                ]

                if len(res_atoms) >= 3:
                    # Detect aromatic rings in ligand using graph analysis
                    ligand_aromatic_rings = self.detect_aromatic_rings_networkx(ligand_atoms)

                    # If networkx detection fails, fall back to simple heuristic
                    if not ligand_aromatic_rings:
                        # Simple heuristic: check for connected C/N atoms
                        ligand_aromatic = ligand_atoms[(ligand_atoms["element"].isin(["C", "N"]))]

                        if len(ligand_aromatic) >= 5:  # Minimum for aromatic ring
                            # Check if atoms form a connected cluster
                            coords = ligand_aromatic[["x", "y", "z"]].values
                            distances = cdist(coords, coords)
                            # If most atoms are within bonding distance, likely aromatic
                            close_pairs = (distances < 1.7) & (distances > 0.1)
                            if np.sum(close_pairs) >= len(ligand_aromatic) * 1.5:
                                ligand_aromatic_rings = [ligand_aromatic]

                    # Check each ligand aromatic ring
                    for ligand_ring in ligand_aromatic_rings:
                        if len(ligand_ring) >= 3:
                            # Calculate pi-stacking
                            pi_result = self.calculate_pi_stacking(res_atoms, ligand_ring)

                            if pi_result:
                                interaction = {
                                    "frame": iteration_num,
                                    "protein_chain": res_atoms.iloc[0]["chain"],
                                    "protein_residue": res_id,
                                    "protein_resname": res_name,
                                    "protein_atom": "RING",
                                    "protein_atom_id": -1,  # Special marker for pi-stacking
                                    "protein_residue_id": res_id,  # CRITICAL: Add residue mapping!
                                    "ligand_atom": "RING",
                                    "ligand_ring_size": len(ligand_ring),
                                    "distance": pi_result["distance"],
                                    "bond_type": pi_result["type"],
                                    "bond_energy": pi_result["energy"],
                                    "angle": pi_result["angle"],
                                    "offset_distance": pi_result["offset"],
                                    "centroid1_x": pi_result["center1"][0],
                                    "centroid1_y": pi_result["center1"][1],
                                    "centroid1_z": pi_result["center1"][2],
                                    "centroid2_x": pi_result["center2"][0],
                                    "centroid2_y": pi_result["center2"][1],
                                    "centroid2_z": pi_result["center2"][2],
                                    "vector_x": pi_result["center2"][0] - pi_result["center1"][0],
                                    "vector_y": pi_result["center2"][1] - pi_result["center1"][1],
                                    "vector_z": pi_result["center2"][2] - pi_result["center1"][2],
                                }
                                pi_interactions.append(interaction)

        return pi_interactions

    def determine_interaction_type(self, protein_atom, ligand_atom, distance):
        """Determine the type of non-covalent interaction"""
        p_resname = protein_atom["resname"]
        p_atom_name = protein_atom["name"]
        l_atom_name = str(ligand_atom.get("name", "")).strip()
        l_element = str(ligand_atom.get("element", "")).strip().upper()

        # Ensure element is valid
        if not l_element:
            # Try to infer from atom name
            l_atom_name_upper = l_atom_name.upper()
            if l_atom_name_upper[:2] in ["CL", "BR"]:
                l_element = l_atom_name_upper[:2]
            elif l_atom_name_upper and l_atom_name_upper[0] in ["C", "N", "O", "S", "P", "H", "F"]:
                l_element = l_atom_name_upper[0]
            else:
                l_element = "C"  # Default
                print(f"Warning: Unknown element for ligand atom {l_atom_name}, defaulting to C")

        # Handle PDBQT aromatic carbon 'A' element
        if l_element == "A":
            l_element = "C"

        # Hydrogen bond detection
        if distance < 3.5:
            # Check if protein is donor and ligand is acceptor
            if p_resname in self.DONORS and p_atom_name in self.DONORS[p_resname]:
                if l_element in ["O", "N", "S", "F", "CL", "BR"]:  # Common acceptors
                    return "HBond", -2.5

            # Check if protein is acceptor and ligand is donor
            if p_resname in self.ACCEPTORS and p_atom_name in self.ACCEPTORS[p_resname]:
                # Ligand donor: H atom or polar H (simplified detection)
                if l_element == "H":
                    return "HBond", -2.5
                # Also check if it's a polar group that likely has H
                elif l_element in ["N", "O"] and ("H" in l_atom_name.upper() or l_element == "N"):
                    return "HBond", -2.5

        # Salt bridge detection
        if distance < 5.0:
            # Positive protein to negative ligand
            if p_resname in self.POSITIVE and p_atom_name in self.POSITIVE[p_resname]:
                # Check for negative ligand groups
                if l_element in ["O", "S"]:
                    # Look for carboxylate, phosphate, sulfate patterns or any O
                    return "Salt Bridge", -5.0

            # Negative protein to positive ligand
            if p_resname in self.NEGATIVE and p_atom_name in self.NEGATIVE[p_resname]:
                if l_element == "N":
                    # Nitrogen atoms are often positive
                    return "Salt Bridge", -5.0

        # Pi-cation interactions
        if distance < 6.0:
            # Aromatic protein to charged ligand
            if p_resname in self.AROMATIC and p_atom_name in self.AROMATIC[p_resname]:
                if l_element == "N":  # Nitrogen often positive
                    return "Pi-Cation", -3.0

            # Charged protein to aromatic ligand
            if p_resname in self.POSITIVE and p_atom_name in self.POSITIVE[p_resname]:
                if l_element == "C":  # Carbon could be aromatic
                    return "Pi-Cation", -3.0

        # Van der Waals - default for all close contacts
        if distance < 5.0:
            return "Van der Waals", -0.5

        return None, 0

    def extract_ca_backbone(self, protein_atoms):
        """Extract CA (alpha-carbon) coordinates for backbone visualization"""
        # Filter for CA atoms
        ca_atoms = protein_atoms[protein_atoms["name"] == "CA"].copy()

        if len(ca_atoms) == 0:
            # Fallback: use every 8th atom (approximate CA spacing)
            ca_coords = protein_atoms.iloc[::8][["x", "y", "z"]].values
        else:
            # Sort by residue sequence if available
            if "resSeq" in ca_atoms.columns:
                ca_atoms = ca_atoms.sort_values("resSeq")
            elif "residue_id" in ca_atoms.columns:
                ca_atoms = ca_atoms.sort_values("residue_id")

            ca_coords = ca_atoms[["x", "y", "z"]].values

        return ca_coords

    def smooth_backbone_trace(self, coords, smoothing_factor=3):
        """
        Smooth the backbone trace for professional visualization
        Similar to academic structure visualization
        """
        from scipy.interpolate import UnivariateSpline

        if len(coords) < 4:
            return coords

        # Parametric smoothing
        t = np.arange(len(coords))
        t_smooth = np.linspace(0, len(coords) - 1, len(coords) * smoothing_factor)

        # Smooth coordinates
        smooth_coords = []
        for i in range(3):  # x, y, z
            spl = UnivariateSpline(t, coords[:, i], s=len(coords) * 0.1)
            smooth_coords.append(spl(t_smooth))
        smooth_coords = np.array(smooth_coords).T

        return smooth_coords

    def visualize_trajectory_cocoon(
        self, protein_atoms, trajectory, iteration_num, approach_idx, output_dir
    ):
        """Visualize the thread-like winding trajectory around protein"""
        fig = plt.figure(figsize=(15, 12))

        # Extract CA backbone
        ca_coords = self.extract_ca_backbone(protein_atoms)

        # Smooth backbone for professional appearance
        smooth_backbone = self.smooth_backbone_trace(ca_coords, smoothing_factor=4)

        # Handle empty trajectory case
        trajectory = np.array(trajectory)
        if len(trajectory) == 0:
            print(f"  WARNING: Empty trajectory for visualization, creating placeholder figure")
            plt.text(
                0.5,
                0.5,
                "No trajectory points generated\n(All positions had collisions)",
                ha="center",
                va="center",
                transform=fig.transFigure,
                fontsize=20,
            )
            vis_filename = os.path.join(
                output_dir, f"trajectory_iteration_{iteration_num}_approach_{approach_idx + 1}.png"
            )
            plt.savefig(vis_filename, dpi=300, bbox_inches="tight")
            plt.close()
            return fig

        # 3D trajectory plot (main visualization)
        ax1 = fig.add_subplot(221, projection="3d")

        # Plot protein backbone as professional black smoothed line
        ax1.plot(
            smooth_backbone[:, 0],
            smooth_backbone[:, 1],
            smooth_backbone[:, 2],
            "k-",
            linewidth=3,
            alpha=0.4,
            label="Protein backbone",
            solid_capstyle="round",
        )

        # Plot trajectory with gradient coloring
        n_points = len(trajectory)
        colors = plt.cm.plasma(np.linspace(0, 1, n_points))

        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                ax1.plot(
                    trajectory[i : i + 2, 0],
                    trajectory[i : i + 2, 1],
                    trajectory[i : i + 2, 2],
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                    solid_capstyle="round",
                )

        # Start and end points with circles
        if len(trajectory) > 0:
            ax1.scatter(
                *trajectory[0],
                s=200,
                c="green",
                marker="o",
                edgecolors="darkgreen",
                linewidth=3,
                label="Start",
                zorder=5,
            )
            if len(trajectory) > 1:
                ax1.scatter(
                    *trajectory[-1],
                    s=200,
                    c="red",
                    marker="o",
                    edgecolors="darkred",
                    linewidth=3,
                    label="End",
                    zorder=5,
                )

        ax1.set_xlabel("X (Å)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Y (Å)", fontsize=12, fontweight="bold")
        ax1.set_zlabel("Z (Å)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Winding Trajectory - Iteration {iteration_num}, Approach {approach_idx + 1}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(fontsize=11)
        ax1.view_init(elev=20, azim=45)

        # Remove grid for cleaner look
        ax1.grid(False)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        # Z-axis coverage for linear molecules (DNA)
        ax2 = fig.add_subplot(222)

        # Check if this is a linear molecule by looking at backbone extent
        backbone_extent = np.ptp(ca_coords, axis=0)
        is_linear = np.max(backbone_extent) / np.min(backbone_extent) > 3.0

        if is_linear:
            # For linear molecules, show position along principal axis
            # Calculate principal axis of DNA
            dna_center = ca_coords.mean(axis=0)
            centered_dna = ca_coords - dna_center
            cov_matrix = np.cov(centered_dna.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            principal_axis = eigenvectors[:, eigenvalues.argmax()]

            # Project trajectory and DNA onto principal axis
            z_positions = [(pos - dna_center).dot(principal_axis) for pos in trajectory]
            dna_projections = [(pos - dna_center).dot(principal_axis) for pos in ca_coords]
            z_min, z_max = min(dna_projections), max(dna_projections)

            ax2.plot(range(len(trajectory)), z_positions, "b-", linewidth=2, alpha=0.8)
            ax2.axhline(y=z_min, color="r", linestyle="--", alpha=0.5, label="DNA ends")
            ax2.axhline(y=z_max, color="r", linestyle="--", alpha=0.5)
            ax2.fill_between(
                range(len(trajectory)),
                z_min,
                z_max,
                alpha=0.2,
                color="lightgray",
                label="DNA region",
            )
            ax2.set_ylabel("Z Position (Å)", fontsize=12, fontweight="bold")
            ax2.set_title("Axial Coverage (DNA Length)", fontsize=14, fontweight="bold")

            # Add coverage percentage
            z_coverage = (np.ptp(z_positions) / (z_max - z_min)) * 100
            ax2.text(
                0.95,
                0.95,
                f"Coverage: {z_coverage:.1f}%",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            # For globular molecules, show distance variation
            min_distances = []
            for pos in trajectory:
                distances = cdist([pos], ca_coords)[0]
                min_distances.append(np.min(distances))

            ax2.plot(
                range(len(trajectory)), min_distances, "b-", linewidth=2, label="Actual distance"
            )
            ax2.axhline(
                y=np.mean(min_distances),
                color="g",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(min_distances):.1f} Å",
            )
            ax2.fill_between(range(len(trajectory)), min_distances, alpha=0.3, color="lightblue")
            ax2.set_ylabel("Distance to Backbone (Å)", fontsize=12, fontweight="bold")
            ax2.set_title("Winding Distance Variation", fontsize=14, fontweight="bold")

        ax2.set_xlabel("Trajectory Step", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.2, color="gray", linewidth=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # XY projection (top view) with smoothed backbone
        ax3 = fig.add_subplot(223)
        # Protein backbone projection (smoothed)
        ax3.plot(
            smooth_backbone[:, 0],
            smooth_backbone[:, 1],
            "k-",
            linewidth=2,
            alpha=0.4,
            label="Backbone",
            solid_capstyle="round",
        )

        # Trajectory path
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                ax3.plot(
                    trajectory[i : i + 2, 0],
                    trajectory[i : i + 2, 1],
                    color=colors[i],
                    linewidth=2,
                    alpha=0.7,
                    solid_capstyle="round",
                )

        if len(trajectory) > 0:
            ax3.scatter(
                *trajectory[0, :2],
                s=150,
                c="green",
                marker="o",
                edgecolors="darkgreen",
                linewidth=3,
                zorder=5,
            )
            if len(trajectory) > 1:
                ax3.scatter(
                    *trajectory[-1, :2],
                    s=150,
                    c="red",
                    marker="o",
                    edgecolors="darkred",
                    linewidth=3,
                    zorder=5,
                )
        ax3.set_xlabel("X (Å)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Y (Å)", fontsize=12, fontweight="bold")
        ax3.set_title("XY Projection", fontsize=14, fontweight="bold")
        ax3.axis("equal")
        ax3.grid(True, alpha=0.2, color="gray", linewidth=0.5)
        ax3.legend(fontsize=11)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # XZ projection (side view) - critical for linear molecules
        ax4 = fig.add_subplot(224)
        # Protein backbone projection (smoothed)
        ax4.plot(
            smooth_backbone[:, 0],
            smooth_backbone[:, 2],
            "k-",
            linewidth=2,
            alpha=0.4,
            label="Backbone",
            solid_capstyle="round",
        )

        # Trajectory path with coverage indication
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                ax4.plot(
                    trajectory[i : i + 2, 0],
                    trajectory[i : i + 2, 2],
                    color=colors[i],
                    linewidth=2,
                    alpha=0.7,
                    solid_capstyle="round",
                )

        # Highlight starting position based on approach
        if len(trajectory) > 0:
            start_marker_color = "green" if trajectory[0, 2] > 0 else "cyan"
            ax4.scatter(
                trajectory[0, 0],
                trajectory[0, 2],
                s=150,
                c=start_marker_color,
                marker="o",
                edgecolors="black",
                linewidth=3,
                zorder=5,
                label=f"Start ({'top' if trajectory[0, 2] > 0 else 'bottom'})",
            )
            if len(trajectory) > 1:
                ax4.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 2],
                    s=150,
                    c="red",
                    marker="o",
                    edgecolors="darkred",
                    linewidth=3,
                    zorder=5,
                    label="End",
                )

        # Add shaded regions to show DNA extent
        if is_linear:
            z_min, z_max = ca_coords[:, 2].min(), ca_coords[:, 2].max()
            x_min, x_max = ax4.get_xlim()
            ax4.axhspan(z_min, z_max, alpha=0.1, color="gray", label="DNA region")

        ax4.set_xlabel("X (Å)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Z (Å)", fontsize=12, fontweight="bold")
        ax4.set_title("XZ Projection (Side View)", fontsize=14, fontweight="bold")
        ax4.axis("equal")
        ax4.grid(True, alpha=0.2, color="gray", linewidth=0.5)
        ax4.legend(fontsize=11, loc="best")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        plt.tight_layout()

        # Save figure
        vis_filename = os.path.join(
            output_dir, f"trajectory_iteration_{iteration_num}_approach_{approach_idx + 1}.png"
        )
        plt.savefig(vis_filename, dpi=300, bbox_inches="tight")
        print(f"  Saved trajectory visualization to {vis_filename}")

        # Close without displaying (blackbox record)
        plt.close()

        return fig

    def run_single_iteration(
        self,
        protein_file,
        ligand_file,
        output_dir,
        n_steps,
        n_approaches,
        approach_distance,
        starting_distance,
        iteration_num,
        use_gpu=False,
        n_rotations=36,
        n_jobs=-1,
        trajectory_step_size=None,
    ):
        """Run a single iteration of the flux analysis with cocoon trajectories

        Args:
            n_rotations: Number of rotations to try at each trajectory position
            trajectory_step_size: User-defined step size in Angstroms (optional)
        """
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration_num}")
        print(f"{'=' * 60}")

        # Parse structures
        print("Parsing structures...")
        # Parse protein (exclude heterogens)
        protein_atoms = self.parse_structure(protein_file, parse_heterogens=False)
        # Parse ligand (include heterogens) - use robust parser for ligands
        ligand_atoms = self.parse_structure_robust(ligand_file, parse_heterogens=True)

        # Initialize intra-protein force field (only on first iteration)
        if iteration_num == 1 and self.intra_protein_calc is None:
            print("\n📊 Calculating intra-protein force field (one-time computation)...")
            structure = self.parser.get_structure("protein", protein_file)
            self.intra_protein_calc = IntraProteinInteractions(
                structure, physiological_pH=self.physiological_pH
            )
            self.intra_protein_vectors = self.intra_protein_calc.calculate_all_interactions()
            print(
                f"  ✓ Calculated static force field for {len(self.intra_protein_vectors)} residues"
            )

        # For ligands, filter to only HETATM records if mixed file
        if "is_hetatm" in ligand_atoms.columns:
            hetatm_count = ligand_atoms["is_hetatm"].sum()
            non_hetatm_count = (~ligand_atoms["is_hetatm"]).sum()

            if hetatm_count > 0 and non_hetatm_count > 0:
                print(
                    f"  Ligand file contains both ATOM ({non_hetatm_count}) and HETATM ({hetatm_count}) records"
                )
                print(f"  Using HETATM records for ligand")
                ligand_atoms = ligand_atoms[ligand_atoms["is_hetatm"]]
            elif hetatm_count == 0:
                print(f"  Note: No HETATM records found in ligand file")
                # If no HETATM, use all atoms (might be a non-standard ligand file)

        print(f"  Parsed {len(protein_atoms)} protein atoms")
        print(f"  Parsed {len(ligand_atoms)} ligand atoms")

        if len(ligand_atoms) == 0:
            print("  ERROR: No ligand atoms found! Check if ligand file contains HETATM records.")
            return pd.DataFrame()

        # Verify element assignments
        print(f"  Protein elements: {protein_atoms['element'].value_counts().to_dict()}")
        print(f"  Ligand elements: {ligand_atoms['element'].value_counts().to_dict()}")

        # Continue with rest of the method...
        # Get coordinates
        protein_coords = protein_atoms[["x", "y", "z"]].values
        ligand_coords = ligand_atoms[["x", "y", "z"]].values

        # Extract CA coordinates for cocoon trajectory
        ca_coords = self.extract_ca_backbone(protein_atoms)

        # Calculate molecular weight
        ligand_mw = self.calculate_molecular_weight(ligand_atoms)
        print(f"  Ligand molecular weight: {ligand_mw:.1f} Da")

        # Build collision detection tree
        self.collision_detector.build_protein_tree(protein_coords, protein_atoms)

        # Create iteration directory
        iter_dir = os.path.join(output_dir, f"iteration_{iteration_num}")
        os.makedirs(iter_dir, exist_ok=True)

        all_interactions = []
        all_trajectories = []

        # Use GPU acceleration if available
        if use_gpu:
            try:
                from gpu_accelerated_flux import GPUAcceleratedInteractionCalculator

                gpu_calc = GPUAcceleratedInteractionCalculator(
                    physiological_pH=self.physiological_pH
                )
                gpu_calc.precompute_protein_properties_gpu(protein_atoms)
                gpu_calc.precompute_ligand_properties_gpu(ligand_atoms)

                # Pass intra-protein vectors to GPU if available
                if self.intra_protein_vectors:
                    gpu_calc.set_intra_protein_vectors(self.intra_protein_vectors)

                print("   ✓ GPU acceleration enabled!")

                # Generate winding trajectories for GPU processing
                print("\n🌀 Generating winding trajectories for GPU...")
                all_gpu_trajectories = []

                for approach_idx in range(n_approaches):
                    # Calculate initial distance for this approach (will vary during trajectory)
                    initial_distance = starting_distance - approach_idx * approach_distance

                    # Calculate unique angle for this approach
                    approach_angle = (2 * np.pi * approach_idx) / n_approaches
                    print(
                        f"   Approach {approach_idx + 1}/{n_approaches}: Initial {initial_distance:.1f} Å, Angle {np.degrees(approach_angle):.1f}°"
                    )

                    # Generate winding trajectory with unique angle
                    trajectory, times = self.generate_cocoon_trajectory(
                        protein_coords,
                        ligand_coords,
                        ligand_atoms,
                        ligand_mw,
                        n_steps=n_steps,
                        dt=40,
                        target_distance=initial_distance,
                        trajectory_step_size=trajectory_step_size,
                        approach_angle=approach_angle,
                    )
                    all_gpu_trajectories.append(trajectory)

                    # Save trajectory data
                    traj_df = pd.DataFrame(
                        {
                            "step": range(len(trajectory)),
                            "time_ps": times / 1000,
                            "x": trajectory[:, 0],
                            "y": trajectory[:, 1],
                            "z": trajectory[:, 2],
                            "approach": approach_idx,
                            "initial_distance": initial_distance,
                        }
                    )
                    traj_path = os.path.join(
                        iter_dir,
                        f"trajectory_iteration_{iteration_num}_approach_{approach_idx}.csv",
                    )
                    traj_df.to_csv(traj_path, index=False)

                # Combine all trajectories for GPU processing
                full_trajectory = np.vstack(all_gpu_trajectories)

                # Process on GPU with rotations
                print(f"\n   Starting GPU processing of {len(full_trajectory)} frames...")
                print(f"   This may take a few minutes for large systems...")

                gpu_results = gpu_calc.process_trajectory_batch_gpu(
                    full_trajectory, ligand_coords, n_rotations=n_rotations
                )

                # Convert GPU results to interaction dataframe
                # Group results by approach
                frames_per_approach = n_steps
                approach_interactions_gpu = [[] for _ in range(n_approaches)]

                for frame_idx, frame_result in enumerate(gpu_results):
                    if (
                        "best_interactions" in frame_result
                        and frame_result["best_interactions"] is not None
                    ):
                        # Determine which approach this frame belongs to
                        approach_idx = frame_idx // frames_per_approach
                        if approach_idx >= n_approaches:
                            approach_idx = n_approaches - 1

                        # Extract InteractionResult from GPU
                        gpu_interaction = frame_result["best_interactions"]

                        # Convert to pandas DataFrame with proper format
                        interaction_data = []

                        # Process each interaction from GPU result
                        for i in range(len(gpu_interaction.indices)):
                            protein_idx = gpu_interaction.indices[i, 0].item()
                            ligand_idx = gpu_interaction.indices[i, 1].item()

                            # Map interaction type to string
                            interaction_type_map = {
                                0: "Van der Waals",
                                1: "Hydrogen Bond",
                                2: "Salt Bridge",
                                3: "Pi-Pi Parallel",
                                4: "Pi-Cation",
                                5: "Pi-Pi T-Shaped",
                                6: "Pi-Pi Offset",
                            }

                            # Get protein atom info
                            p_atom = protein_atoms.iloc[protein_idx]
                            l_atom = ligand_atoms.iloc[ligand_idx]

                            # Calculate vector
                            vector = (
                                frame_result["position"]
                                + ligand_coords[ligand_idx]
                                - protein_coords[protein_idx]
                            )

                            interaction_dict = {
                                "frame": frame_idx,
                                "protein_chain": p_atom.get("chain", "A"),
                                "protein_residue": gpu_interaction.residue_ids[i].item(),
                                "protein_resname": p_atom["resname"],
                                "protein_atom": p_atom["name"],
                                "protein_atom_id": p_atom.get("atom_id", protein_idx),
                                "ligand_atom": l_atom["name"],
                                "ligand_atom_id": ligand_idx,
                                "distance": gpu_interaction.distances[i].item(),
                                "bond_type": interaction_type_map.get(
                                    gpu_interaction.types[i].item(), "Unknown"
                                ),
                                "bond_energy": gpu_interaction.energies[i].item(),
                                "vector_x": vector[0],
                                "vector_y": vector[1],
                                "vector_z": vector[2],
                                "protein_residue_id": gpu_interaction.residue_ids[i].item(),
                                "rotation": frame_idx % n_rotations,  # Add rotation info
                                "rotation_angle": (frame_idx % n_rotations) * (360 / n_rotations),
                            }
                            interaction_data.append(interaction_dict)

                        if interaction_data:
                            approach_interactions_gpu[approach_idx].extend(interaction_data)

                # Save interactions for each approach
                for approach_idx, approach_data in enumerate(approach_interactions_gpu):
                    if approach_data:
                        interactions_df = pd.DataFrame(approach_data)
                        all_interactions.append(interactions_df)

                        # Save detailed interaction data
                        interaction_file = os.path.join(
                            iter_dir, f"interactions_approach_{approach_idx}.csv"
                        )
                        interactions_df.to_csv(interaction_file, index=False)
                        print(
                            f"   Saved {len(interactions_df)} interactions for approach {approach_idx + 1}"
                        )

                # Store trajectories for visualization
                all_trajectories = all_gpu_trajectories

                # Store GPU results for integrated flux pipeline
                self.gpu_trajectory_results = gpu_results

                # Visualize trajectories for each approach
                for approach_idx, approach_trajectory in enumerate(all_trajectories):
                    self.visualize_trajectory_cocoon(
                        protein_atoms, approach_trajectory, iteration_num, approach_idx, iter_dir
                    )

                print(f"   ✓ GPU processing complete!")

            except Exception as e:
                print(f"   ⚠️  GPU acceleration failed: {e}")
                print("   Falling back to CPU processing...")
                use_gpu = False

        if not use_gpu:
            # CPU processing with winding trajectories
            print("\n🌀 Generating winding trajectories...")

            for approach_idx in range(n_approaches):
                print(f"\nApproach {approach_idx + 1}/{n_approaches}")

                # Calculate initial distance for this approach (will vary during trajectory)
                initial_distance = starting_distance - approach_idx * approach_distance

                # Calculate unique angle for this approach
                approach_angle = (2 * np.pi * approach_idx) / n_approaches
                print(
                    f"  Initial distance: {initial_distance:.1f} Å (will vary 5-{initial_distance * 2.5:.0f} Å)"
                )
                print(f"  Starting angle: {np.degrees(approach_angle):.1f}°")

                # Generate winding trajectory with unique angle
                trajectory, times = self.generate_cocoon_trajectory(
                    protein_coords,
                    ligand_coords,
                    ligand_atoms,
                    ligand_mw,
                    n_steps=n_steps,
                    dt=40,
                    target_distance=initial_distance,
                    trajectory_step_size=trajectory_step_size,
                    approach_angle=approach_angle,
                )

                all_trajectories.append(trajectory)

                # Save trajectory data
                traj_df = pd.DataFrame(
                    {
                        "step": range(len(trajectory)),
                        "time_ps": times / 1000,  # Convert fs to ps
                        "x": trajectory[:, 0],
                        "y": trajectory[:, 1],
                        "z": trajectory[:, 2],
                        "approach": approach_idx,
                        "initial_distance": initial_distance,
                    }
                )
                traj_path = os.path.join(
                    iter_dir, f"trajectory_iteration_{iteration_num}_approach_{approach_idx}.csv"
                )
                traj_df.to_csv(traj_path, index=False)
                print(f"  Saved trajectory to {traj_path}")

                # Visualize trajectory (cocoon)
                self.visualize_trajectory_cocoon(
                    protein_atoms, trajectory, iteration_num, approach_idx, iter_dir
                )

                # Calculate interactions at each position with rotations
                print(f"  Calculating interactions with {n_rotations} rotations per position...")
                approach_interactions = []

                # Use scipy for rotation calculations
                from scipy.spatial.transform import Rotation as R
                from joblib import Parallel, delayed

                # Pre-generate all rotation angles and matrices
                angles = np.linspace(0, 360, n_rotations, endpoint=False)

                # Define function for parallel processing
                def process_step_rotations(
                    step,
                    position,
                    protein_atoms,
                    ligand_atoms,
                    ligand_coords,
                    ca_coords,
                    collision_detector,
                    n_rotations,
                    approach_idx,
                    n_steps,
                    analyzer_self,
                ):
                    """Process all rotations for a single trajectory step in parallel"""
                    # Find closest CA for rotation axis
                    distances = cdist([position], ca_coords)[0]
                    closest_idx = np.argmin(distances)
                    closest_ca = ca_coords[closest_idx]

                    # Calculate normal vector for rotation axis
                    normal = (
                        closest_ca / np.linalg.norm(closest_ca)
                        if np.linalg.norm(closest_ca) > 0
                        else np.array([0, 0, 1])
                    )

                    # Get ligand center
                    ligand_center = ligand_coords.mean(axis=0)
                    centered_coords = ligand_coords - ligand_center

                    # Process rotations in parallel
                    def process_rotation(rot_idx):
                        angle = rot_idx * (360 / n_rotations)

                        # Create rotation using scipy
                        rotation = R.from_rotvec(normal * np.radians(angle))

                        # Apply rotation
                        rotated_coords = rotation.apply(centered_coords)

                        # Translate to trajectory position
                        final_coords = rotated_coords + position

                        # Update ligand atoms with new coordinates
                        ligand_atoms_rot = ligand_atoms.copy()
                        ligand_atoms_rot[["x", "y", "z"]] = final_coords

                        # Check collision
                        if not collision_detector.check_collision(final_coords, ligand_atoms_rot):
                            # Calculate interactions
                            interactions = analyzer_self.calculate_interactions(
                                protein_atoms, ligand_atoms_rot, approach_idx * n_steps + step
                            )

                            if len(interactions) > 0:
                                # Add rotation info
                                interactions["rotation"] = rot_idx
                                interactions["rotation_angle"] = angle
                                return interactions.to_dict("records")
                        return []

                    # Run rotations in parallel
                    rotation_results = Parallel(n_jobs=n_jobs, backend="threading")(
                        delayed(process_rotation)(rot_idx) for rot_idx in range(n_rotations)
                    )

                    # Flatten results
                    step_interactions = []
                    for result in rotation_results:
                        if result:
                            step_interactions.extend(result)

                    return step_interactions

                # Process all steps with progress reporting
                import multiprocessing as mp

                actual_cores = mp.cpu_count() if n_jobs == -1 else n_jobs
                print(f"\n  Using parallel CPU processing with {actual_cores} cores")

                step_results = []
                for step, position in enumerate(trajectory):
                    if step % 10 == 0:
                        print(f"\r    Progress: {step}/{n_steps} steps processed", end="")

                    # Process this step
                    step_interactions = process_step_rotations(
                        step,
                        position,
                        protein_atoms,
                        ligand_atoms,
                        ligand_coords,
                        ca_coords,
                        self.collision_detector,
                        n_rotations,
                        approach_idx,
                        n_steps,
                        self,
                    )

                    if step_interactions:
                        step_df = pd.DataFrame(step_interactions)
                        approach_interactions.append(step_df)

                print(f"\n  Found interactions at {len(approach_interactions)} positions")

                if approach_interactions:
                    combined_interactions = pd.concat(approach_interactions, ignore_index=True)
                    all_interactions.append(combined_interactions)

                    # Save detailed interaction data
                    interaction_file = os.path.join(
                        iter_dir, f"interactions_approach_{approach_idx}.csv"
                    )
                    combined_interactions.to_csv(interaction_file, index=False)
                    print(
                        f"  Saved {len(combined_interactions)} interactions to {interaction_file}"
                    )

        # Combine all interactions
        final_interactions = pd.DataFrame()  # Initialize to empty DataFrame

        if all_interactions:
            final_interactions = pd.concat(all_interactions, ignore_index=True)

            # Add residue mapping for all interactions
            final_interactions["protein_residue_id"] = final_interactions["protein_residue"]

            # Save results
            output_file = os.path.join(
                iter_dir, f"flux_iteration_{iteration_num}_output_vectors.csv"
            )
            final_interactions.to_csv(output_file, index=False)
            print(f"\nSaved {len(final_interactions)} interactions to {output_file}")

            # Print summary
            print("\nInteraction summary:")
            print(final_interactions["bond_type"].value_counts())

            # Check for pi-stacking
            pi_stacking_count = final_interactions["bond_type"].str.contains("Pi-Stacking").sum()
            if pi_stacking_count > 0:
                print(f"\n✓ Found {pi_stacking_count} pi-stacking interactions!")
                print("  Pi-stacking properly mapped to residues")
        else:
            print("\n⚠️  No interactions found in this iteration")

        # Visualize trajectory
        if all_trajectories:
            self.visualize_trajectory(
                protein_atoms,
                all_trajectories,
                ligand_coords,
                os.path.join(iter_dir, f"trajectory_iteration_{iteration_num}.png"),
            )

        return final_interactions

    def visualize_trajectory(self, protein_atoms, trajectories, ligand_coords, output_file):
        """Visualize the combined Brownian trajectories with professional backbone"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Extract CA backbone for professional visualization
        ca_coords = self.extract_ca_backbone(protein_atoms)

        # Smooth backbone for professional appearance
        smooth_backbone = self.smooth_backbone_trace(ca_coords, smoothing_factor=4)

        # Plot protein backbone as professional black smoothed line
        ax.plot(
            smooth_backbone[:, 0],
            smooth_backbone[:, 1],
            smooth_backbone[:, 2],
            "k-",
            linewidth=4,
            alpha=0.4,
            label="Protein backbone",
            solid_capstyle="round",
        )

        # Plot trajectories with distinct colors
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))

        for i, (trajectory, color) in enumerate(zip(trajectories, colors)):
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                alpha=0.9,
                linewidth=3,
                label=f"Approach {i + 1}",
                solid_capstyle="round",
            )

            # Mark start and end with circles
            ax.scatter(
                *trajectory[0],
                color=color,
                s=180,
                marker="o",
                edgecolor="black",
                linewidth=3,
                zorder=5,
            )
            ax.scatter(
                *trajectory[-1],
                color=color,
                s=180,
                marker="o",
                edgecolor="white",
                linewidth=3,
                zorder=5,
            )

        # Set labels with professional styling
        ax.set_xlabel("X (Å)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Y (Å)", fontsize=14, fontweight="bold")
        ax.set_zlabel("Z (Å)", fontsize=14, fontweight="bold")
        ax.set_title("Ligand Trajectory Exploration", fontsize=16, fontweight="bold")

        # Remove grid and background for cleaner academic look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        # Set equal aspect ratio based on backbone
        max_range = (
            np.array(
                [
                    ca_coords[:, 0].max() - ca_coords[:, 0].min(),
                    ca_coords[:, 1].max() - ca_coords[:, 1].min(),
                    ca_coords[:, 2].max() - ca_coords[:, 2].min(),
                ]
            ).max()
            / 1.5
        )

        mid_x = (ca_coords[:, 0].max() + ca_coords[:, 0].min()) * 0.5
        mid_y = (ca_coords[:, 1].max() + ca_coords[:, 1].min()) * 0.5
        mid_z = (ca_coords[:, 2].max() + ca_coords[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Better viewing angle
        ax.view_init(elev=15, azim=45)
        ax.legend(fontsize=12, frameon=False)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved trajectory visualization to {output_file}")

    def run_complete_analysis(
        self,
        protein_file,
        ligand_file,
        output_dir,
        n_steps=100,
        n_iterations=3,
        n_approaches=5,
        approach_distance=2.5,
        starting_distance=35,
        n_jobs=-1,
        use_gpu=False,
        n_rotations=36,
        trajectory_step_size=None,
    ):
        """Run complete flux analysis with cocoon trajectories

        Args:
            n_rotations: Number of rotations to sample at each trajectory position
            trajectory_step_size: User-defined step size in Angstroms (optional)
        """
        print("\n" + "=" * 80)
        print("PROTEIN-LIGAND FLUX ANALYSIS - COCOON TRAJECTORY MODE")
        print("=" * 80)
        print(f"Protein: {protein_file}")
        print(f"Ligand: {ligand_file}")
        print(f"Output directory: {output_dir}")
        print(f"Iterations: {n_iterations}")
        print(f"Steps per approach: {n_steps}")
        print(f"Number of approaches: {n_approaches}")
        print(f"Starting distance: {starting_distance} Å")
        print(f"Approach step: {approach_distance} Å")
        print(f"Rotations per position: {n_rotations}")
        print(f"GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED'}")
        print("=" * 80)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        response = input("\nContinue with analysis? (y/n): ").strip().lower()
        if response != "y":
            print("Analysis cancelled.")
            return None

        # Run iterations
        iteration_data = []

        for iteration in range(n_iterations):
            print(f"\n{'#' * 80}")
            print(f"STARTING ITERATION {iteration + 1} OF {n_iterations}")
            print(f"{'#' * 80}")

            iteration_start = time.time()

            # Run single iteration with cocoon trajectories
            interactions = self.run_single_iteration(
                protein_file,
                ligand_file,
                output_dir,
                n_steps,
                n_approaches,
                approach_distance,
                starting_distance,
                iteration + 1,
                use_gpu,
                n_rotations=n_rotations,
                n_jobs=n_jobs,
                trajectory_step_size=trajectory_step_size,
            )

            iteration_time = time.time() - iteration_start

            if not interactions.empty:
                iteration_data.append(
                    {
                        "iteration": iteration + 1,
                        "interactions": interactions,
                        "n_interactions": len(interactions),
                        "time": iteration_time,
                    }
                )

                print(f"\nIteration {iteration + 1} complete!")
                print(f"Time: {iteration_time:.1f} seconds")
                print(f"Total interactions: {len(interactions)}")
            else:
                print(f"\n⚠️  Iteration {iteration + 1} produced no interactions")

        # Summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)

        if iteration_data:
            total_interactions = sum(d["n_interactions"] for d in iteration_data)
            total_time = sum(d["time"] for d in iteration_data)

            print(f"Total iterations completed: {len(iteration_data)}")
            print(f"Total interactions found: {total_interactions}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average time per iteration: {total_time / len(iteration_data):.1f} seconds")

            # Check for pi-stacking
            pi_stacking_total = 0
            for iter_data in iteration_data:
                pi_count = iter_data["interactions"]["bond_type"].str.contains("Pi-Stacking").sum()
                pi_stacking_total += pi_count

            if pi_stacking_total > 0:
                print(f"\n✓ Total pi-stacking interactions: {pi_stacking_total}")
        else:
            print("No valid iterations completed.")

        return iteration_data

    def run_simulation_iterations(
        self, protein_atoms: pd.DataFrame, ligand_atoms: pd.DataFrame, n_iterations: int, **kwargs
    ):
        """
        Run the complete simulation across multiple iterations for statistical significance.
        """
        print_banner(f"🚀 STARTING FLUX SIMULATION: {self.protein_name} + {self.ligand_name}")

        # Create a pool of workers
        # NOTE: Using 'fork' start method can cause issues with GPU resources on some platforms
        # 'spawn' is safer but might be slower due to process creation overhead.
        start_method = "spawn" if platform.system() != "Linux" else "fork"

        try:
            with mp.get_context(start_method).Pool(processes=mp.cpu_count()) as pool:
                # Prepare arguments for each iteration
                args_list = [
                    (i, protein_atoms, ligand_atoms, self.output_dir, kwargs)
                    for i in range(n_iterations)
                ]

                # Run iterations in parallel
                pool.starmap(self._run_single_iteration_wrapper, args_list)

        except Exception as e:
            print(f"\nError during parallel processing: {e}")
            print("Switching to sequential execution for remaining iterations.")
            # Fallback to sequential execution
            for i in range(n_iterations):
                self._run_single_iteration_wrapper(
                    i, protein_atoms, ligand_atoms, self.output_dir, kwargs
                )

        print("\nAll simulation iterations complete.")

    def _run_single_iteration_wrapper(self, i, protein_atoms, ligand_atoms, output_dir, kwargs):
        """Wrapper to pass self to the iteration function for parallel execution"""
        # Re-initialize analyzer for each process to avoid sharing state
        analyzer = ProteinLigandFluxAnalyzer(
            self.protein_file, self.ligand_file, output_dir, self.target_is_dna
        )
        analyzer.set_gpu_calculator(self.gpu_calculator)
        analyzer.run_single_iteration(i, protein_atoms, ligand_atoms, **kwargs)


def main():
    """Main entry point"""
    analyzer = ProteinLigandFluxAnalyzer()

    # Example usage
    print("PROTEIN-LIGAND TRAJECTORY GENERATOR - COCOON MODE")
    print("=" * 60)
    print("This generator creates cocoon-style trajectories that maintain")
    print("constant distance from the protein surface while exploring")
    print("different orientations through Brownian dynamics.")
    print("-" * 60)

    protein_file = input("Enter protein PDB file: ").strip()
    ligand_file = input("Enter ligand PDB/PDBQT file: ").strip()
    output_dir = input("Output directory (default: flux_output): ").strip() or "flux_output"

    # Optional: ask for trajectory parameters
    print("\nTrajectory parameters (press Enter for defaults):")
    n_steps = input("  Steps per approach [100]: ").strip()
    n_steps = int(n_steps) if n_steps else 100

    n_approaches = input("  Number of approaches [5]: ").strip()
    n_approaches = int(n_approaches) if n_approaches else 5

    n_rotations = input("  Rotations per position [36]: ").strip()
    n_rotations = int(n_rotations) if n_rotations else 36

    # Run analysis
    analyzer.run_complete_analysis(
        protein_file,
        ligand_file,
        output_dir,
        n_steps=n_steps,
        n_iterations=3,
        n_approaches=n_approaches,
        approach_distance=2.5,
        starting_distance=35,
        use_gpu=True,  # Try to use GPU if available
        n_rotations=n_rotations,
    )


if __name__ == "__main__":
    main()
