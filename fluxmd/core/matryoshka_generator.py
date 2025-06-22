# fluxmd/core/matryoshka_generator.py
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .dynamics.brownian_roller import BrownianSurfaceRoller, quaternion_multiply as _quat_mul
from .geometry.pca_anchors import extreme_calpha_pairs
from .surface.layer_stream import MatryoshkaLayerGenerator
from .surface.ses_builder import SESBuilder
from .ref15_energy import get_ref15_calculator
from ..utils.cpu import parse_workers, format_workers_info

__all__ = ["MatryoshkaTrajectoryGenerator"]


class MatryoshkaTrajectoryGenerator:
    """Orchestrates Matryoshka trajectory generation with Brownian dynamics."""
    
    def __init__(
        self,
        protein_atoms: Dict[str, np.ndarray],
        ligand_atoms: Dict[str, np.ndarray],
        params: Dict[str, Any]
    ) -> None:
        """Initialize generator with molecular structures.
        
        Args:
            protein_atoms: Dict with 'coords', 'names', 'radii', 'masses' arrays
            ligand_atoms: Dict with 'coords', 'names', 'masses' arrays
            params: Simulation parameters (T, viscosity, force constants, etc.)
        """
        self.protein_atoms = protein_atoms
        self.ligand_atoms = ligand_atoms
        self.params = params
        
        # Extract parameters with defaults
        self.temperature = params.get('T', 298.15)
        self.viscosity = params.get('viscosity', 0.00089)
        self.probe_radius = params.get('probe_radius', 0.75)
        self.layer_step = params.get('layer_step', 1.5)
        self.k_surf = params.get('k_surf', 2.0)
        self.k_guid = params.get('k_guid', 0.5)

        n_workers_param = params.get('n_workers', 'auto')
        if n_workers_param in (None, '', 'auto'):
            self.n_workers = max(1, os.cpu_count() - 1)
        else:
            try:
                self.n_workers = max(1, int(n_workers_param))
            except ValueError as e:
                raise ValueError(
                    f"Invalid n_workers value: {n_workers_param}. Must be 'auto' or a positive integer."
                ) from e
        
        self.checkpoint_dir = params.get('checkpoint_dir', None)
        self.gpu_device = params.get('gpu_device', None)
        self.collision_detector = params.get('collision_detector', None)
        
        # Initialize REF15 calculator if not provided
        self.ref15_calculator = params.get('ref15_calculator', None)
        if self.ref15_calculator is None and params.get('use_ref15', True):
            print("Initializing REF15 energy calculator...")
            self.ref15_calculator = get_ref15_calculator(params.get('pH', 7.4))
        
        # Calculate PCA anchors from protein backbone
        print("Calculating trajectory anchors via PCA...")
        self.anchors = extreme_calpha_pairs(
            self.protein_atoms['coords'],
            self.protein_atoms['names']
        )
        anchor_distance = np.linalg.norm(self.anchors[1] - self.anchors[0])
        print(f"  Anchor separation: {anchor_distance:.1f} Å")
        
        # Build ligand pseudo-sphere
        print("Building ligand pseudo-sphere...")
        self.ligand_sphere = self._calculate_ligand_sphere()
        print(f"  Radius of gyration: {self.ligand_sphere['radius']:.2f} Å")
        print(f"  Total mass: {self.ligand_sphere['mass']:.1f} amu")
        
        # Initialize SES builder
        print("Initializing surface builder...")
        self.ses_builder = SESBuilder(
            self.protein_atoms['coords'],
            self.protein_atoms['radii'],
            probe_radius=self.probe_radius,
            atom_names=self.protein_atoms.get('names', None),
            resnames=self.protein_atoms.get('resnames', None)
        )
        
        # Build base surface (expensive, do once)
        print("Building base SES surface (this may take a moment)...")
        start_time = time.time()
        self.base_surface = self.ses_builder.build_ses0()
        build_time = time.time() - start_time
        print(f"  Surface built in {build_time:.1f}s")
        print(f"  Vertices: {len(self.base_surface.vertices)}")
        print(f"  Faces: {len(self.base_surface.faces)}")
        
        # Initialize layer generator
        self.layer_generator = MatryoshkaLayerGenerator(
            self.base_surface,
            step=self.layer_step
        )
        
        # Calculate maximum useful layers
        self.max_layers = self.layer_generator.get_max_useful_layers(
            ligand_radius=self.ligand_sphere['radius'],
            cutoff=self.params.get('vdw_cutoff', 12.0)
        )
        print(f"  Maximum useful layers: {self.max_layers}")
        
        # Set up checkpoint directory
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"  Workers: {format_workers_info(self.n_workers)}")
    
    def _calculate_ligand_sphere(self) -> Dict[str, Any]:
        """Calculate ligand pseudo-sphere properties.
        
        Returns:
            Dict with radius, mass, center, inertia
        """
        coords = self.ligand_atoms['coords']
        masses = self.ligand_atoms['masses']
        
        # Center of mass
        total_mass = np.sum(masses)
        center_of_mass = np.average(coords, weights=masses, axis=0)
        
        # Radius of gyration
        centered_coords = coords - center_of_mass
        rgyr_squared = np.average(
            np.sum(centered_coords**2, axis=1),
            weights=masses
        )
        radius_of_gyration = np.sqrt(rgyr_squared)
        
        # Inertia tensor
        inertia_tensor = np.zeros((3, 3))
        for i, (coord, mass) in enumerate(zip(centered_coords, masses)):
            # I = sum(m * (r²I - rr^T))
            r_squared = np.dot(coord, coord)
            inertia_tensor += mass * (r_squared * np.eye(3) - np.outer(coord, coord))
        
        # For simplicity, use trace/3 as effective moment (isotropic approximation)
        effective_inertia = np.trace(inertia_tensor) / 3.0
        
        return {
            'radius': radius_of_gyration,
            'mass': total_mass,
            'center': center_of_mass,
            'inertia': effective_inertia,
            'inertia_tensor': inertia_tensor,
            'coords_centered': centered_coords
        }
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def _reconstruct_ligand_coords(
        self,
        position: np.ndarray,
        quaternion: np.ndarray
    ) -> np.ndarray:
        """Reconstruct full ligand coordinates from position and orientation.
        
        Args:
            position: Center of mass position
            quaternion: Orientation quaternion
            
        Returns:
            (N, 3) array of ligand atom coordinates
        """
        # Get rotation matrix
        R = self._quaternion_to_rotation_matrix(quaternion)
        
        # Rotate centered coordinates and translate
        rotated_coords = self.ligand_sphere['coords_centered'] @ R.T
        full_coords = rotated_coords + position
        
        return full_coords
    
    def _calculate_ref15_energy(
        self,
        protein_atoms: Dict[str, np.ndarray],
        ligand_coords: np.ndarray,
        ligand_atoms: Dict[str, np.ndarray]
    ) -> float:
        """Calculate REF15 energy between protein and ligand.
        
        Args:
            protein_atoms: Protein atom dictionary
            ligand_coords: Current ligand coordinates
            ligand_atoms: Ligand atom dictionary
            
        Returns:
            Interaction energy in kcal/mol
        """
        if self.ref15_calculator is None:
            # Fallback to distance-based estimate
            ligand_com = ligand_coords.mean(axis=0)
            return -10.0 * np.exp(-0.1 * np.linalg.norm(ligand_com - self.anchors[1]))
        
        total_energy = 0.0
        protein_coords = protein_atoms['coords']
        
        # Create atom contexts for ligand atoms
        ligand_contexts = []
        for i, (coord, name) in enumerate(zip(ligand_coords, ligand_atoms['names'])):
            atom_dict = {
                'x': coord[0], 'y': coord[1], 'z': coord[2],
                'name': name,
                'element': name[0],  # Simple element extraction
                'resname': 'LIG',
                'resSeq': 1
            }
            context = self.ref15_calculator.create_atom_context(atom_dict)
            ligand_contexts.append(context)
        
        # Calculate pairwise interactions (simplified - real implementation would be more efficient)
        cutoff_sq = 36.0  # 6 Angstrom cutoff squared
        
        for lig_idx, lig_context in enumerate(ligand_contexts):
            lig_coord = ligand_coords[lig_idx]
            
            # Find nearby protein atoms
            distances_sq = np.sum((protein_coords - lig_coord)**2, axis=1)
            nearby_mask = distances_sq < cutoff_sq
            nearby_indices = np.where(nearby_mask)[0]
            
            for prot_idx in nearby_indices:
                # Create protein atom context
                prot_atom_dict = {
                    'x': protein_coords[prot_idx, 0],
                    'y': protein_coords[prot_idx, 1], 
                    'z': protein_coords[prot_idx, 2],
                    'name': protein_atoms['names'][prot_idx],
                    'element': protein_atoms['names'][prot_idx][0],
                    'resname': 'PRO',  # Simplified
                    'resSeq': prot_idx // 10  # Rough residue assignment
                }
                prot_context = self.ref15_calculator.create_atom_context(prot_atom_dict)
                
                # Calculate interaction energy
                distance = np.sqrt(distances_sq[prot_idx])
                energy = self.ref15_calculator.calculate_interaction_energy(
                    prot_context, lig_context, distance
                )
                total_energy += energy
        
        return total_energy
    
    def _generate_orientation_variants(
        self,
        base_quaternion: np.ndarray,
        n_variants: int = 4
    ) -> np.ndarray:
        """Generate orientation variants around base quaternion.
        
        Args:
            base_quaternion: Starting orientation
            n_variants: Number of variants to generate
            
        Returns:
            (n_variants, 4) array of quaternions
        """
        variants = np.zeros((n_variants, 4))
        variants[0] = base_quaternion  # Include original
        
        # Generate small rotations
        for i in range(1, n_variants):
            # Random small rotation axis
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            
            # Small angle (5-15 degrees)
            angle = np.random.uniform(5, 15) * np.pi / 180
            
            # Create rotation quaternion
            half_angle = angle / 2
            rot_quat = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle)
            ])
            
            # Apply rotation
            variants[i] = _quat_mul(rot_quat, base_quaternion)
            variants[i] /= np.linalg.norm(variants[i])
        
        return variants
    
    def _calculate_ref15_energy_batch(
        self,
        position: np.ndarray,
        quaternions: np.ndarray,
        sample_fraction: float = 0.1
    ) -> np.ndarray:
        """Calculate REF15 energies for multiple orientations efficiently.
        
        Args:
            position: Ligand center of mass
            quaternions: (N, 4) array of orientations
            sample_fraction: Fraction of protein atoms to sample
            
        Returns:
            (N,) array of energies
        """
        n_orientations = len(quaternions)
        energies = np.zeros(n_orientations)
        
        # Subsample protein atoms for speed
        n_protein = len(self.protein_atoms['coords'])
        n_sample = max(100, int(n_protein * sample_fraction))
        sample_indices = np.random.choice(n_protein, n_sample, replace=False)
        
        protein_coords_sample = self.protein_atoms['coords'][sample_indices]
        protein_names_sample = self.protein_atoms['names'][sample_indices]
        
        # Process each orientation
        for i, quat in enumerate(quaternions):
            # Reconstruct ligand coordinates
            ligand_coords = self._reconstruct_ligand_coords(position, quat)
            
            # Create simplified protein atoms dict for sampled atoms
            protein_sample = {
                'coords': protein_coords_sample,
                'names': protein_names_sample
            }
            
            # Calculate energy
            energies[i] = self._calculate_ref15_energy(
                protein_sample, ligand_coords, self.ligand_atoms
            )
        
        # Scale energies to account for sampling
        energies *= (n_protein / n_sample)
        
        return energies
    
    def _run_single_trajectory(
        self,
        layer_idx: int,
        iteration_idx: int,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a single trajectory on a specific layer.
        
        Args:
            layer_idx: Which Matryoshka layer
            iteration_idx: Which iteration on this layer
            seed: Random seed for reproducibility
            
        Returns:
            Trajectory results dictionary
        """
        # Get the appropriate surface layer
        surface = self.layer_generator.get_layer(layer_idx)
        
        # Create roller with unique seed
        if seed is None:
            seed = hash((layer_idx, iteration_idx, time.time())) % 2**32
        
        # Create energy calculator function for layer hopping
        def energy_calculator(pos, quat, layer):
            ligand_coords = self._reconstruct_ligand_coords(pos, quat)
            return self._calculate_ref15_energy(
                self.protein_atoms, ligand_coords, self.ligand_atoms
            )
        
        roller = BrownianSurfaceRoller(
            surface=surface,
            ligand_sphere=self.ligand_sphere,
            anchors=self.anchors,
            T=self.temperature,
            viscosity=self.viscosity,
            k_surf=self.k_surf,
            k_guid=self.k_guid,
            layer_generator=self.layer_generator,
            current_layer_idx=layer_idx,
            energy_calculator=energy_calculator if self.ref15_calculator else None,
            hop_attempt_interval=100,
            hop_probability=0.1,
            groove_detector=self.ses_builder.groove_detector,
            groove_preference=self.params.get('groove_preference', 'major'),
            seed=seed
        )
        
        # Run trajectory
        print(f"  Running trajectory L{layer_idx}-I{iteration_idx}...")
        trajectory = roller.run(max_steps=self.params.get('max_steps', 1_000_000))
        
        # Add metadata
        trajectory['layer_idx'] = layer_idx
        trajectory['iteration_idx'] = iteration_idx
        trajectory['seed'] = seed
        
        # If we have REF15 calculator, compute actual energies
        if self.ref15_calculator and len(trajectory['pos']) > 0:
            # Sample subset of positions for energy calculation
            n_samples = min(len(trajectory['pos']), 100)
            sample_indices = np.linspace(
                0, len(trajectory['pos']) - 1, n_samples, dtype=int
            )
            
            energies = []
            for idx in sample_indices:
                pos = trajectory['pos'][idx]
                quat = trajectory['quat'][idx]
                
                # Reconstruct ligand coordinates
                ligand_coords = self._reconstruct_ligand_coords(pos, quat)
                
                # Calculate REF15 energy with multiple orientations
                if hasattr(self, '_calculate_ref15_energy_batch'):
                    # Generate 4 random orientations around current
                    orientations = self._generate_orientation_variants(quat, n_variants=4)
                    energies_batch = self._calculate_ref15_energy_batch(
                        pos, orientations, sample_fraction=0.1
                    )
                    energy = np.min(energies_batch)  # Take best orientation
                else:
                    # Fallback to single orientation
                    energy = self._calculate_ref15_energy(
                        self.protein_atoms, ligand_coords, self.ligand_atoms
                    )
                energies.append(energy)
            
            trajectory['sampled_energies'] = np.array(energies)
            trajectory['sample_indices'] = sample_indices
        
        return trajectory
    
    def _worker_process(
        self,
        work_queue: mp.Queue,
        result_queue: mp.Queue,
        worker_id: int
    ) -> None:
        """Worker process for parallel trajectory generation.
        
        Args:
            work_queue: Queue of (layer_idx, iteration_idx, seed) tuples
            result_queue: Queue for results
            worker_id: Worker identifier
        """
        while True:
            try:
                work_item = work_queue.get(timeout=1)
                if work_item is None:  # Poison pill
                    break
                
                layer_idx, iteration_idx, seed = work_item
                
                # Run trajectory
                result = self._run_single_trajectory(layer_idx, iteration_idx, seed)
                
                # Send result
                result_queue.put((layer_idx, iteration_idx, result))
                
            except mp.queues.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                result_queue.put((layer_idx, iteration_idx, {'error': str(e)}))
    
    def _save_checkpoint(
        self,
        trajectories: List[Dict[str, Any]],
        layer_idx: int,
        iteration_idx: int
    ) -> None:
        """Save checkpoint to disk.
        
        Args:
            trajectories: List of completed trajectories
            layer_idx: Current layer
            iteration_idx: Current iteration
        """
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'trajectories': trajectories,
            'layer_idx': layer_idx,
            'iteration_idx': iteration_idx,
            'timestamp': time.time()
        }
        
        checkpoint_path = Path(self.checkpoint_dir) / f'checkpoint_L{layer_idx}_I{iteration_idx}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Checkpoint saved: {checkpoint_path.name}")
    
    def _load_checkpoint(self) -> Tuple[List[Dict[str, Any]], int, int]:
        """Load most recent checkpoint.
        
        Returns:
            (trajectories, layer_idx, iteration_idx)
        """
        if not self.checkpoint_dir or not Path(self.checkpoint_dir).exists():
            return [], 0, 0
        
        checkpoints = list(Path(self.checkpoint_dir).glob('checkpoint_L*_I*.pkl'))
        if not checkpoints:
            return [], 0, 0
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        print(f"Loading checkpoint: {latest.name}")
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return (
            checkpoint['trajectories'],
            checkpoint['layer_idx'],
            checkpoint['iteration_idx']
        )
    
    def run(
        self,
        n_layers: Optional[int] = None,
        n_iterations: int = 10
    ) -> List[Dict[str, np.ndarray]]:
        """Generate trajectories across nested surface layers.
        
        Args:
            n_layers: Number of Matryoshka layers (None = adaptive)
            n_iterations: Trajectories per layer
            
        Returns:
            List of trajectory dictionaries
        """
        assert isinstance(self.n_workers, int) and self.n_workers >= 1, (
            "Runtime error: n_workers must be a positive integer."
        )

        # Use adaptive layer count if not specified
        if n_layers is None:
            n_layers = min(self.max_layers, 10)
        else:
            n_layers = min(n_layers, self.max_layers)
        
        print(f"\nStarting Matryoshka trajectory generation:")
        print(f"  Layers: {n_layers}")
        print(f"  Iterations per layer: {n_iterations}")
        print(f"  Total trajectories: {n_layers * n_iterations}")
        
        # Ensure n_workers is valid
        if self.n_workers is None:
            self.n_workers = parse_workers(None)  # Will return auto-detected value
            
        print(f"  Workers: {format_workers_info(self.n_workers)}")
        
        # Load checkpoint if available
        trajectories, start_layer, start_iteration = self._load_checkpoint()
        
        # Calculate total work items
        total_items = n_layers * n_iterations
        completed_items = len(trajectories)
        
        if self.n_workers > 1 and total_items - completed_items > 1:
            # Parallel execution
            trajectories.extend(self._run_parallel(
                n_layers, n_iterations, start_layer, start_iteration
            ))
        else:
            # Serial execution
            with tqdm(total=total_items, initial=completed_items) as pbar:
                for layer_idx in range(n_layers):
                    if layer_idx < start_layer:
                        continue
                    
                    for iteration_idx in range(n_iterations):
                        if layer_idx == start_layer and iteration_idx < start_iteration:
                            continue
                        
                        # Generate seed for reproducibility
                        seed = hash((layer_idx, iteration_idx, 42)) % 2**32
                        
                        # Run trajectory
                        trajectory = self._run_single_trajectory(
                            layer_idx, iteration_idx, seed
                        )
                        trajectories.append(trajectory)
                        
                        # Update progress
                        pbar.update(1)
                        
                        # Checkpoint periodically
                        if len(trajectories) % 5 == 0:
                            self._save_checkpoint(
                                trajectories, layer_idx, iteration_idx + 1
                            )
        
        print(f"\nCompleted {len(trajectories)} trajectories")
        
        # Final checkpoint
        self._save_checkpoint(trajectories, n_layers, 0)
        
        return trajectories
    
    def _run_parallel(
        self,
        n_layers: int,
        n_iterations: int,
        start_layer: int = 0,
        start_iteration: int = 0
    ) -> List[Dict[str, Any]]:
        """Run trajectories in parallel.
        
        Args:
            n_layers: Number of layers
            n_iterations: Iterations per layer
            start_layer: Resume from this layer
            start_iteration: Resume from this iteration
            
        Returns:
            List of trajectory results
        """
        # Create work queue
        work_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Populate work items
        n_items = 0
        for layer_idx in range(n_layers):
            if layer_idx < start_layer:
                continue
                
            for iteration_idx in range(n_iterations):
                if layer_idx == start_layer and iteration_idx < start_iteration:
                    continue
                
                seed = hash((layer_idx, iteration_idx, 42)) % 2**32
                work_queue.put((layer_idx, iteration_idx, seed))
                n_items += 1
        
        # Add poison pills
        for _ in range(self.n_workers):
            work_queue.put(None)
        
        # Start workers
        workers = []
        for i in range(self.n_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(work_queue, result_queue, i)
            )
            p.start()
            workers.append(p)
        
        # Collect results
        trajectories = []
        with tqdm(total=n_items) as pbar:
            for _ in range(n_items):
                layer_idx, iteration_idx, result = result_queue.get()
                trajectories.append(result)
                pbar.update(1)
                
                # Checkpoint periodically
                if len(trajectories) % 10 == 0:
                    self._save_checkpoint(
                        trajectories, layer_idx, iteration_idx + 1
                    )
        
        # Wait for workers to finish
        for p in workers:
            p.join()
        
        return trajectories