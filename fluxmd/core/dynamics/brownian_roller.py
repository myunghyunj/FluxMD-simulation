# fluxmd/core/dynamics/brownian_roller.py
from __future__ import annotations

import heapq
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..surface.ses_builder import SurfaceMesh

__all__ = ["BrownianSurfaceRoller", "quaternion_multiply"]


class BrownianSurfaceRoller:
    """BAOAB Langevin dynamics for ligand rolling on molecular surfaces."""

    # Physical constants
    KB_KCAL = 0.0019872041  # Boltzmann constant in kcal/mol/K

    def __init__(
        self,
        surface: SurfaceMesh,
        ligand_sphere: Dict[str, np.ndarray],
        anchors: Tuple[np.ndarray, np.ndarray],
        T: float = 298.15,
        viscosity: float = 0.00089,
        dt_fs: float = 5.0,
        k_surf: float = 2.0,
        k_guid: float = 0.5,
        layer_generator=None,
        current_layer_idx: int = 0,
        energy_calculator=None,
        **kwargs,
    ) -> None:
        """Initialize roller with physics parameters.

        Args:
            surface: Target molecular surface mesh
            ligand_sphere: Dict with 'center', 'radius', 'mass', 'inertia'
            anchors: (start_pos, end_pos) trajectory endpoints
            T: Temperature in Kelvin
            viscosity: Solvent viscosity in Pa·s
            dt_fs: Timestep in femtoseconds
            k_surf: Surface adherence force constant (kcal/mol/Å²)
            k_guid: Guidance force constant (kcal/mol/Å²)
            layer_generator: MatryoshkaLayerGenerator for layer hopping
            current_layer_idx: Starting layer index
            energy_calculator: Function to calculate energy at a position
        """
        self.surface = surface
        self.ligand = ligand_sphere
        self.start_anchor, self.end_anchor = anchors
        self.T = T
        self.viscosity = viscosity
        self.k_surf = k_surf
        self.k_guid = k_guid

        # Layer hopping support
        self.layer_generator = layer_generator
        self.current_layer_idx = current_layer_idx
        self.energy_calculator = energy_calculator
        self.hop_attempt_interval = kwargs.get("hop_attempt_interval", 100)  # steps
        self.hop_probability = kwargs.get("hop_probability", 0.1)

        # Extract ligand properties
        self.ligand_radius = ligand_sphere["radius"]
        self.ligand_mass = ligand_sphere["mass"]
        self.ligand_inertia = ligand_sphere.get("inertia", self._sphere_inertia())

        # Thermal energy (needed for diffusion calculations)
        self.kT = self.KB_KCAL * self.T

        # Calculate diffusion coefficients
        self.D_t, self.D_r = self._calculate_diffusion_coefficients()

        # Set adaptive timestep
        self.dt_fs = self._calculate_adaptive_timestep()
        self.dt_ps = self.dt_fs / 1000.0  # Convert to picoseconds

        # Initialize RNG
        self.rng = np.random.default_rng(kwargs.get("seed", None))

        # Friction coefficients
        self.gamma_t = self.kT / self.D_t
        self.gamma_r = self.kT / self.D_r

        # BAOAB integrator constants
        self._setup_baoab_constants()

        # Guidance activation tracking
        self.guidance_active = False
        self.guidance_activation_time = None
        self.guidance_anneal_time = 1.0  # ps

        # Build KD-tree for efficient surface queries
        self.surface_kdtree = cKDTree(self.surface.vertices)

        # Pre-compute geodesic distance between anchors
        self.geodesic_distance = self._compute_geodesic_distance()
        print(f"  Geodesic distance: {self.geodesic_distance:.1f} Å")

        # Layer hopping tracking
        self.layer_history = [current_layer_idx]
        self.hop_attempts = 0
        self.successful_hops = 0

        # DNA groove detector (if available)
        self.groove_detector = kwargs.get("groove_detector", None)
        self.groove_preference = kwargs.get("groove_preference", "major")

    def _sphere_inertia(self) -> float:
        """Calculate moment of inertia for uniform sphere."""
        return (2.0 / 5.0) * self.ligand_mass * self.ligand_radius**2

    def _calculate_diffusion_coefficients(self) -> Tuple[float, float]:
        """Calculate translational and rotational diffusion coefficients.

        Returns:
            (D_t, D_r) in Å²/ps and rad²/ps
        """
        # Convert viscosity from Pa·s to kcal·ps/mol/Å²
        # 1 Pa·s = 1 kg/(m·s) = 6.022e23 amu·Å/(mol·ps)
        # Then multiply by energy conversion
        eta_converted = self.viscosity * 1.439e-4  # Approximate conversion

        # Stokes-Einstein relations
        D_t = self.kT / (6 * np.pi * eta_converted * self.ligand_radius)
        D_r = self.kT / (8 * np.pi * eta_converted * self.ligand_radius**3)

        return D_t, D_r

    def _calculate_adaptive_timestep(self) -> float:
        """Calculate timestep for target RMS displacement.

        Returns:
            Timestep in femtoseconds
        """
        target_rms = 0.3  # Target RMS displacement in Å

        # RMS displacement in 3D: sqrt(6*D_t*dt)
        # Solve for dt
        dt_ps = (target_rms**2) / (6 * self.D_t)
        dt_fs = dt_ps * 1000.0

        # Clamp to reasonable range, adjusted for tests
        return np.clip(dt_fs, 5.0, 50.0)

    def _setup_baoab_constants(self) -> None:
        """Precompute constants for BAOAB integrator."""
        # Ornstein-Uhlenbeck process constants
        self.ou_decay = np.exp(-self.gamma_t * self.dt_ps / self.ligand_mass)
        self.ou_noise = np.sqrt(self.kT * (1 - self.ou_decay**2) / self.ligand_mass)

        # Rotational OU constants
        self.ou_decay_rot = np.exp(-self.gamma_r * self.dt_ps / self.ligand_inertia)
        self.ou_noise_rot = np.sqrt(self.kT * (1 - self.ou_decay_rot**2) / self.ligand_inertia)

    def _compute_geodesic_distance(self) -> float:
        """Compute geodesic distance between anchors on surface mesh using Dijkstra.

        Returns:
            Shortest path distance on surface in Angstroms
        """
        vertices = self.surface.vertices
        faces = self.surface.faces

        # Find closest vertices to start and end anchors
        start_dists = np.linalg.norm(vertices - self.start_anchor, axis=1)
        end_dists = np.linalg.norm(vertices - self.end_anchor, axis=1)
        start_vertex = np.argmin(start_dists)
        end_vertex = np.argmin(end_dists)

        # Build adjacency graph with edge weights
        n_vertices = len(vertices)
        adjacency = [[] for _ in range(n_vertices)]

        # Add edges from mesh faces
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge_length = np.linalg.norm(vertices[v2] - vertices[v1])
                adjacency[v1].append((v2, edge_length))
                adjacency[v2].append((v1, edge_length))

        # Dijkstra's algorithm
        distances = np.full(n_vertices, np.inf)
        distances[start_vertex] = 0
        heap = [(0, start_vertex)]
        visited = set()

        while heap:
            current_dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current == end_vertex:
                break

            for neighbor, edge_weight in adjacency[current]:
                if neighbor not in visited:
                    new_dist = current_dist + edge_weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))

        geodesic_dist = distances[end_vertex]

        # Fallback to Euclidean distance if no path found
        if np.isinf(geodesic_dist):
            print("Warning: No geodesic path found, using Euclidean distance")
            geodesic_dist = np.linalg.norm(self.end_anchor - self.start_anchor)

        return geodesic_dist

    def _attempt_layer_hop(self, position: np.ndarray, quaternion: np.ndarray) -> bool:
        """Attempt Monte Carlo transition to adjacent layer.

        Args:
            position: Current position
            quaternion: Current orientation

        Returns:
            True if hop was accepted
        """
        if self.layer_generator is None or self.energy_calculator is None:
            return False

        self.hop_attempts += 1

        # Randomly choose to hop up or down
        hop_up = self.rng.random() < 0.5
        target_layer_idx = self.current_layer_idx + (1 if hop_up else -1)

        # Check bounds
        if target_layer_idx < 0:
            return False

        # Get target surface
        try:
            target_surface = self.layer_generator.get_layer(target_layer_idx)
        except:
            return False  # Layer doesn't exist

        # Calculate current energy
        current_energy = self._calculate_energy_at_position(position, quaternion)

        # Find closest point on target surface
        target_point, target_dist, target_normal = self._find_closest_point_on_surface_mesh(
            position, target_surface
        )

        # Project position to target surface
        target_position = target_point + target_normal * self.ligand_radius

        # Calculate energy at target position
        target_energy = self._calculate_energy_at_position(target_position, quaternion)

        # Calculate groove bias if detector available
        groove_bias = 1.0
        if self.groove_detector is not None:
            groove_bias = self.groove_detector.get_groove_bias(
                position, target_position, self.groove_preference
            )

        # Metropolis criterion with groove bias
        delta_E = target_energy - current_energy

        if delta_E < 0:
            # Always accept downhill moves (but still apply groove bias)
            accept_probability = groove_bias
        else:
            # Accept uphill moves with Boltzmann probability
            boltzmann_factor = np.exp(-delta_E / self.kT)
            accept_probability = boltzmann_factor * groove_bias

        accept = self.rng.random() < accept_probability

        if accept:
            # Update layer and surface
            self.current_layer_idx = target_layer_idx
            self.surface = target_surface
            self.surface_kdtree = cKDTree(target_surface.vertices)
            self.layer_history.append(target_layer_idx)
            self.successful_hops += 1
            return True

        return False

    def _calculate_energy_at_position(self, position: np.ndarray, quaternion: np.ndarray) -> float:
        """Calculate energy at given position and orientation.

        Args:
            position: Center of mass position
            quaternion: Orientation quaternion

        Returns:
            Energy in kcal/mol
        """
        if self.energy_calculator is None:
            # Simple distance-based energy
            dist_to_end = np.linalg.norm(position - self.end_anchor)
            return -5.0 * np.exp(-0.1 * dist_to_end)

        # Use provided energy calculator
        return self.energy_calculator(position, quaternion, self.current_layer_idx)

    def _find_closest_point_on_surface_mesh(
        self, point: np.ndarray, mesh: SurfaceMesh
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Find closest point on a specific surface mesh.

        Args:
            point: Query point
            mesh: Surface mesh to query

        Returns:
            (closest_point, distance, normal)
        """
        # Build temporary KD-tree if needed
        kdtree = cKDTree(mesh.vertices)
        distance, closest_idx = kdtree.query(point)
        closest_point = mesh.vertices[closest_idx]

        # Approximate normal
        normal = point - closest_point
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal /= norm
        else:
            normal = np.array([0, 0, 1])

        return closest_point, distance, normal

    def _find_closest_point_on_surface(
        self, point: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Find closest point on surface mesh using KD-tree.

        Args:
            point: Query point

        Returns:
            (closest_point, distance, normal)
        """
        # Use KD-tree for efficient nearest neighbor query
        distance, closest_idx = self.surface_kdtree.query(point)
        closest_point = self.surface.vertices[closest_idx]

        # Approximate normal as direction from surface to query point
        # (In production, would compute proper vertex normals)
        normal = point - closest_point
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal /= norm
        else:
            normal = np.array([0, 0, 1])

        return closest_point, distance, normal

    def _surface_force(self, position: np.ndarray, target_distance: float) -> np.ndarray:
        """Calculate harmonic force to maintain surface distance.

        Args:
            position: Current position
            target_distance: Desired distance from surface

        Returns:
            Force vector in kcal/mol/Å
        """
        closest_point, current_distance, normal = self._find_closest_point_on_surface(position)

        # Harmonic potential: F = -k * (d - d_target) * n
        deviation = current_distance - target_distance
        force_magnitude = -self.k_surf * deviation

        return force_magnitude * normal

    def _guidance_force(
        self,
        position: np.ndarray,
        path_length: float,
        current_time: float,
        total_length: float | None = None,
    ) -> np.ndarray:
        """Calculate late-stage guidance force with soft time-based ramp-up.

        Args:
            position: Current position
            path_length: Distance traveled so far
            current_time: Current simulation time in ps
            total_length: Deprecated argument, kept for backward compatibility with tests

        Returns:
            Force vector in kcal/mol/Å
        """
        # The 'total_length' argument is ignored and kept only for backward compatibility with tests.
        # Check if we should activate guidance (75% of geodesic distance)
        if not self.guidance_active and path_length >= 0.75 * self.geodesic_distance:
            self.guidance_active = True
            self.guidance_activation_time = current_time

        # Return zero force if not active
        if not self.guidance_active:
            return np.zeros(3)

        # Time-based soft ramp-up over ~1 ps
        time_since_activation = current_time - self.guidance_activation_time
        ramp_factor = min(1.0, time_since_activation / self.guidance_anneal_time)
        k_effective = self.k_guid * ramp_factor

        # Harmonic force: F = -k * (r - r_target)
        # Since displacement = r_target - r, we get F = k * displacement
        displacement = self.end_anchor - position
        return k_effective * displacement

    def _quaternion_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis and angle."""
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        return np.array(
            [
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ]
        )

    def run(self, max_steps: int = 1_000_000) -> Dict[str, np.ndarray]:
        """Execute Brownian dynamics trajectory.

        Args:
            max_steps: Maximum integration steps

        Returns:
            Dict with 'pos', 'quat', 'time', 'energy' arrays
        """
        # Initialize trajectory storage
        trajectory = {"pos": [], "quat": [], "time": [], "energy": [], "layer": []}

        # Initialize at start anchor with surface offset
        closest_point, dist, normal = self._find_closest_point_on_surface(self.start_anchor)
        position = self.start_anchor + normal * (self.ligand_radius + 2.0)
        velocity = np.zeros(3)

        # Initialize orientation (identity quaternion)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        angular_velocity = np.zeros(3)

        # Path tracking
        path_length = 0.0
        last_position = position.copy()
        stuck_counter = 0

        # Main integration loop
        for step in range(max_steps):
            # Store current state
            trajectory["pos"].append(position.copy())
            trajectory["quat"].append(quaternion.copy())
            trajectory["time"].append(step * self.dt_ps)
            trajectory["layer"].append(self.current_layer_idx)

            # Calculate forces
            current_time = step * self.dt_ps
            f_surface = self._surface_force(position, self.ligand_radius)
            f_guidance = self._guidance_force(position, path_length, current_time)
            total_force = f_surface + f_guidance

            # BAOAB integration
            # B step (velocity update - half)
            velocity += (0.5 * self.dt_ps / self.ligand_mass) * total_force

            # A step (position update - half)
            position += (0.5 * self.dt_ps) * velocity

            # O step (Ornstein-Uhlenbeck)
            velocity = self.ou_decay * velocity + self.ou_noise * self.rng.normal(size=3)
            angular_velocity = (
                self.ou_decay_rot * angular_velocity + self.ou_noise_rot * self.rng.normal(size=3)
            )

            # A step (position update - half)
            position += (0.5 * self.dt_ps) * velocity

            # Update orientation
            if np.linalg.norm(angular_velocity) > 1e-10:
                angle = np.linalg.norm(angular_velocity) * self.dt_ps
                axis = angular_velocity / np.linalg.norm(angular_velocity)
                rotation = self._quaternion_from_axis_angle(axis, angle)
                quaternion = quaternion_multiply(rotation, quaternion)
                quaternion /= np.linalg.norm(quaternion)  # Normalize

            # Recalculate forces at new position
            f_surface = self._surface_force(position, self.ligand_radius)
            f_guidance = self._guidance_force(position, path_length, current_time)
            total_force = f_surface + f_guidance

            # B step (velocity update - half)
            velocity += (0.5 * self.dt_ps / self.ligand_mass) * total_force

            # Attempt layer hop periodically
            if (
                step > 0
                and step % self.hop_attempt_interval == 0
                and self.rng.random() < self.hop_probability
            ):
                hop_accepted = self._attempt_layer_hop(position, quaternion)
                if hop_accepted:
                    # Recalculate surface force after layer change
                    f_surface = self._surface_force(position, self.ligand_radius)

            # Update path length
            displacement = np.linalg.norm(position - last_position)
            path_length += displacement

            # Stuck detection
            if displacement < 0.001:  # Less than 0.001 Å
                stuck_counter += 1
                if stuck_counter > 1000:  # ~1 ps
                    # Re-thermalize
                    velocity = self.ou_noise * self.rng.normal(size=3) * 3.0
                    stuck_counter = 0
            else:
                stuck_counter = 0

            last_position = position.copy()

            # Check termination
            if np.linalg.norm(position - self.end_anchor) < 1.5:
                print(f"Reached end anchor after {step} steps")
                break

            # Energy placeholder (would calculate actual interaction energy)
            trajectory["energy"].append(0.0)

        # Convert to arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])

        # Add layer hopping statistics
        trajectory["hop_attempts"] = self.hop_attempts
        trajectory["successful_hops"] = self.successful_hops
        trajectory["layer_history"] = np.array(self.layer_history)

        print(
            f"Layer hops: {self.successful_hops}/{self.hop_attempts} "
            + f"({100 * self.successful_hops / (self.hop_attempts + 1e-10):.1f}% success)"
        )

        return trajectory

    @staticmethod
    def _quaternion_multiply(q1, q2):
        """DEPRECATED – kept for tests written against FluxMD 1.x."""
        return quaternion_multiply(q1, q2)


# -----------------------------------------------------------------------------
# Quaternion utilities (module-level so they can be reused & pickle-friendly)
# -----------------------------------------------------------------------------


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of two quaternions.

    Both *q1* and *q2* are expected as 4-element arrays ``[w, x, y, z]``.
    The result is a new quaternion following the right-handed convention.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
