"""
REF15-Enhanced Trajectory Generator Extension
Adds intelligent sampling capabilities to FluxMD trajectory generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .trajectory_generator import CocoonTrajectoryGenerator
from .intelligent_cocoon_sampler import IntelligentCocoonSampler
from .ref15_energy import get_ref15_calculator

logger = logging.getLogger(__name__)


class REF15TrajectoryGenerator(CocoonTrajectoryGenerator):
    """
    Enhanced trajectory generator using REF15 energy-guided sampling
    Inherits from CocoonTrajectoryGenerator and adds intelligent features
    """

    def __init__(
        self, protein_file: str, ligand_file: str, output_dir: str, target_is_dna: bool = False
    ):
        # Initialize parent with REF15 energy function
        super().__init__(
            protein_file, ligand_file, output_dir, target_is_dna, energy_function="ref15"
        )

        # Will be initialized after protein/ligand loading
        self.intelligent_sampler = None
        self.ligand_properties = None

    def analyze_ligand_properties(self, ligand_atoms: pd.DataFrame) -> Dict:
        """Analyze ligand chemical properties for intelligent sampling"""
        properties = {
            "charge": 0.0,
            "has_aromatic": False,
            "has_hbond_donor": False,
            "has_hbond_acceptor": False,
            "molecular_weight": 0.0,
        }

        # Calculate net charge
        if "formal_charge" in ligand_atoms.columns:
            properties["charge"] = ligand_atoms["formal_charge"].sum()

        # Check for aromatic atoms
        aromatic_elements = {"C", "N"}  # Simplified
        for _, atom in ligand_atoms.iterrows():
            if atom.get("element", "") in aromatic_elements:
                # Simple aromaticity check
                if "AR" in atom.get("name", "").upper() or atom.get("is_aromatic", False):
                    properties["has_aromatic"] = True
                    break

        # Check for H-bond capability
        for _, atom in ligand_atoms.iterrows():
            element = atom.get("element", "")
            if element in ["N", "O", "S"]:
                properties["has_hbond_acceptor"] = True
            if element in ["N", "O"] and "H" in atom.get("name", ""):
                properties["has_hbond_donor"] = True

        # Molecular weight
        properties["molecular_weight"] = self.calculate_molecular_weight(ligand_atoms)

        logger.info(f"Ligand properties: {properties}")
        return properties

    def generate_approach_trajectories_intelligent(self, n_trajectories: int) -> List[np.ndarray]:
        """
        Generate approach trajectories using REF15-guided intelligent sampling

        Returns:
            List of starting points for trajectories
        """
        if self.intelligent_sampler is None:
            # Initialize sampler with protein data
            protein_atoms = self.parse_target_structure()
            self.intelligent_sampler = IntelligentCocoonSampler(
                self.protein_coords, protein_atoms.to_dict("records"), self.ref15_calculator
            )

        # Analyze ligand if not done
        if self.ligand_properties is None:
            ligand_atoms = self.parse_mobile_structure()
            self.ligand_properties = self.analyze_ligand_properties(ligand_atoms)

        # Generate intelligent approach points
        approach_points = self.intelligent_sampler.generate_approach_points(
            n_trajectories,
            ligand_charge=self.ligand_properties["charge"],
            ligand_has_aromatic=self.ligand_properties["has_aromatic"],
            ligand_has_hbond=(
                self.ligand_properties["has_hbond_donor"]
                or self.ligand_properties["has_hbond_acceptor"]
            ),
        )

        return approach_points

    def run_enhanced_workflow(self, n_iterations: int = 100):
        """
        Run enhanced FluxMD workflow with REF15 and intelligent sampling
        """
        print("\n" + "=" * 80)
        print("Starting REF15-Enhanced FluxMD Workflow")
        print("=" * 80)

        # Parse structures
        print("\n1. Parsing structures...")
        protein_atoms = self.parse_target_structure()
        ligand_atoms = self.parse_mobile_structure()

        # Calculate intra-protein forces with REF15
        print("\n2. Calculating REF15 intra-protein forces...")
        self.calculate_intra_protein_forces()

        # Initialize intelligent sampler
        print("\n3. Initializing intelligent cocoon sampler...")
        self.intelligent_sampler = IntelligentCocoonSampler(
            self.protein_coords, protein_atoms.to_dict("records"), self.ref15_calculator
        )

        # Analyze ligand
        print("\n4. Analyzing ligand properties...")
        self.ligand_properties = self.analyze_ligand_properties(ligand_atoms)

        # Generate approach points
        print(f"\n5. Generating {n_iterations} intelligent approach trajectories...")
        approach_points = self.generate_approach_trajectories_intelligent(n_iterations)

        # Run simulations
        print("\n6. Running trajectory simulations...")
        all_interactions = []

        for i, start_point in enumerate(approach_points):
            if i % 10 == 0:
                print(f"   Progress: {i}/{n_iterations} trajectories")

            # Generate trajectory from this approach point
            trajectory = self.generate_single_trajectory(
                start_point,
                self.protein_coords,
                self.ligand_coords,
                ligand_atoms,
                self.ligand_properties["molecular_weight"],
            )

            # Calculate interactions along trajectory
            for frame_idx, position in enumerate(trajectory):
                # Move ligand to position
                ligand_coords_moved = self.move_ligand_to_position(self.ligand_coords, position)

                # Calculate REF15 interactions
                interactions = self.calculate_ref15_interactions(
                    protein_atoms,
                    ligand_atoms,
                    ligand_coords_moved,
                    iteration_num=i,
                    frame_num=frame_idx,
                )

                all_interactions.append(interactions)

        # Aggregate results
        print("\n7. Aggregating results...")
        results_df = pd.concat(all_interactions, ignore_index=True)

        # Save results
        output_file = os.path.join(self.output_dir, "ref15_interactions.csv")
        results_df.to_csv(output_file, index=False)
        print(f"   Saved interactions to {output_file}")

        # Analyze flux
        print("\n8. Analyzing REF15-based flux...")
        flux_results = self.analyze_ref15_flux(results_df)

        print("\n" + "=" * 80)
        print("REF15-Enhanced FluxMD Workflow Complete!")
        print("=" * 80)

        return results_df, flux_results

    def generate_single_trajectory(
        self,
        start_point: np.ndarray,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        ligand_atoms: pd.DataFrame,
        molecular_weight: float,
        n_steps: int = 100,
    ) -> np.ndarray:
        """Generate a single trajectory from starting point"""
        # Use parent's cocoon trajectory generation
        # but starting from the intelligent approach point

        # Temporarily move ligand to start point
        ligand_center = ligand_coords.mean(axis=0)
        offset = start_point - ligand_center
        ligand_coords_start = ligand_coords + offset

        # Generate trajectory
        trajectory = self.generate_cocoon_trajectory(
            protein_coords, ligand_coords_start, ligand_atoms, molecular_weight, n_steps=n_steps
        )

        return trajectory

    def calculate_ref15_interactions(
        self,
        protein_atoms: pd.DataFrame,
        ligand_atoms: pd.DataFrame,
        ligand_coords: np.ndarray,
        iteration_num: int,
        frame_num: int,
    ) -> pd.DataFrame:
        """Calculate interactions using REF15 energy function"""
        # This would use the GPU-accelerated REF15 if available
        # For now, use the protonation-aware detector which we updated

        # Update ligand coordinates in dataframe
        ligand_atoms_moved = ligand_atoms.copy()
        ligand_atoms_moved[["x", "y", "z"]] = ligand_coords

        # Use the updated interaction calculator
        interactions = calculate_interactions_with_protonation(
            protein_atoms, ligand_atoms_moved, pH=self.physiological_pH, iteration_num=iteration_num
        )

        interactions["frame"] = frame_num

        return interactions

    def move_ligand_to_position(
        self, ligand_coords: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        """Move ligand center to specified position"""
        center = ligand_coords.mean(axis=0)
        offset = position - center
        return ligand_coords + offset

    def analyze_ref15_flux(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze flux using REF15 energies"""
        # Group by residue and calculate flux
        flux_data = []

        for (chain, resnum, resname), group in interactions_df.groupby(
            ["protein_chain", "protein_residue", "protein_resname"]
        ):
            # Calculate flux as sum of energy magnitudes
            # REF15 energies are properly weighted
            total_flux = np.abs(group["bond_energy"]).sum()

            # Get interaction counts by type
            interaction_counts = group["bond_type"].value_counts().to_dict()

            flux_data.append(
                {
                    "chain": chain,
                    "residue": resnum,
                    "resname": resname,
                    "flux": total_flux,
                    "n_interactions": len(group),
                    **interaction_counts,
                }
            )

        flux_df = pd.DataFrame(flux_data)
        flux_df = flux_df.sort_values("flux", ascending=False)

        # Save flux results
        flux_file = os.path.join(self.output_dir, "ref15_flux_analysis.csv")
        flux_df.to_csv(flux_file, index=False)
        print(f"   Saved flux analysis to {flux_file}")

        # Print top residues
        print("\n   Top 10 residues by REF15 flux:")
        for _, row in flux_df.head(10).iterrows():
            print(f"     {row['chain']}:{row['resname']}{row['residue']} - Flux: {row['flux']:.2f}")

        return flux_df


# Convenience function
def create_ref15_trajectory_generator(
    protein_file: str, ligand_file: str, output_dir: str, target_is_dna: bool = False
) -> REF15TrajectoryGenerator:
    """Create REF15-enhanced trajectory generator"""
    return REF15TrajectoryGenerator(protein_file, ligand_file, output_dir, target_is_dna)
