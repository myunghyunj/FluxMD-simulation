"""
REF15 Energy Calculator
Core implementation of Rosetta Energy Function 2015
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ref15_params import fade_function, get_ref15_params, sigmoidal_dielectric, switch_function
from .rosetta_atom_types import AtomTypeInfo, get_atom_typer

logger = logging.getLogger(__name__)


@dataclass
class AtomContext:
    """Context information for an atom needed for energy calculations"""

    atom_type: str
    coords: np.ndarray
    formal_charge: float = 0.0
    is_backbone: bool = False
    is_donor: bool = False
    is_acceptor: bool = False
    is_aromatic: bool = False
    neighbor_count: int = 0  # For burial calculation
    attached_hydrogens: List[int] = None  # Indices of attached H atoms
    residue_id: int = 0
    residue_name: str = ""


class REF15EnergyCalculator:
    """
    Full implementation of REF15 energy function
    Handles all major energy terms with proper physics
    """

    def __init__(self, physiological_pH: float = 7.4):
        self.pH = physiological_pH
        self.params = get_ref15_params()
        self.atom_typer = get_atom_typer()

        # Validate parameters to catch zero volumes early
        self._validate_parameters()

        # Energy component tracking for analysis
        self.last_energy_components = {}

    def _validate_parameters(self):
        """Validate REF15 parameters and warn about potential issues"""
        zero_volume_types = []

        # Check LK parameters for zero volumes
        for atom_type in self.params.lk_params:
            dgfree, lambda_val, volume = self.params.get_lk_params(atom_type)
            if volume == 0.0:
                zero_volume_types.append(atom_type)

        if zero_volume_types:
            logger.warning(
                f"Found {len(zero_volume_types)} atom types with zero volume: {zero_volume_types[:5]}..."
                f"{'(showing first 5)' if len(zero_volume_types) > 5 else ''}"
            )
            logger.warning(
                "Zero volumes may cause division by zero errors. "
                "These will be handled with fallback values during calculations."
            )

    def calculate_interaction_energy(
        self, atom1: AtomContext, atom2: AtomContext, distance: float, detailed: bool = False
    ) -> float:
        """
        Calculate total REF15 energy between two atoms

        Args:
            atom1: First atom context
            atom2: Second atom context
            distance: Distance between atoms
            detailed: If True, store component breakdown

        Returns:
            Total interaction energy in kcal/mol
        """
        energy_components = {}

        # Skip if too far
        if distance > 6.0:
            return 0.0

        # 1. Lennard-Jones (attractive + repulsive)
        e_lj_atr, e_lj_rep = self._calculate_lj_energy(atom1, atom2, distance)
        energy_components["fa_atr"] = e_lj_atr
        energy_components["fa_rep"] = e_lj_rep

        # 2. Solvation (Lazaridis-Karplus)
        e_sol = self._calculate_solvation_energy(atom1, atom2, distance)
        energy_components["fa_sol"] = e_sol

        # 3. Electrostatics
        if atom1.formal_charge != 0 and atom2.formal_charge != 0:
            e_elec = self._calculate_electrostatic_energy(atom1, atom2, distance)
            energy_components["fa_elec"] = e_elec
        else:
            energy_components["fa_elec"] = 0.0

        # 4. Hydrogen bonding (if applicable)
        e_hbond = 0.0
        if self._can_form_hbond(atom1, atom2):
            # Simplified H-bond without full geometry for now
            e_hbond = self._calculate_hbond_energy_simple(atom1, atom2, distance)
            energy_components["hbond"] = e_hbond

        # Sum weighted components
        total_energy = 0.0
        for term, value in energy_components.items():
            weight = self.params.get_weight(term)
            total_energy += weight * value

        if detailed:
            self.last_energy_components = energy_components

        return total_energy

    def _calculate_lj_energy(
        self, atom1: AtomContext, atom2: AtomContext, distance: float
    ) -> Tuple[float, float]:
        """
        Calculate Lennard-Jones attractive and repulsive components
        REF15 uses split attractive/repulsive with different functional forms
        """
        # Get LJ parameters
        lj_radius, lj_wdepth = self.params.get_lj_params(atom1.atom_type, atom2.atom_type)

        # Reduced distance
        x = distance / lj_radius

        # REF15 uses different functions for different regions
        if x < 0.6:
            # Linear repulsive wall
            e_rep = 8.666 * (0.6 - x)
            e_atr = 0.0
        elif x < 0.67:
            # Transition region - cubic polynomial
            # This ensures smooth transition at x=0.6
            dx = x - 0.6
            poly_a = -46.592
            poly_b = 9.482
            poly_c = -0.648
            e_rep = poly_a * dx * dx * dx + poly_b * dx * dx + poly_c * dx
            e_atr = 0.0
        else:
            # Standard LJ region with split attractive/repulsive
            x2 = x * x
            x3 = x2 * x
            x6 = x3 * x3
            x12 = x6 * x6

            # Repulsive: standard 1/r^12
            if x < 1.0:
                e_rep = 1.0 / x12 - 2.0 / x6 + 1.0  # Shifted to 0 at x=1
            else:
                e_rep = 0.0

            # Attractive: -2*(1/r^6 - 1/r^3) with switching
            if x < 2.0:  # REF15 attractive cutoff
                e_atr_base = -2.0 * (1.0 / x6 - 1.0 / x3)

                # Apply switching function
                switch_params = self.params.get_switch_params("fa_atr")
                switch = switch_function(
                    distance, switch_params["switch_dis"], switch_params["cutoff_dis"]
                )
                e_atr = e_atr_base * switch * lj_wdepth
            else:
                e_atr = 0.0

        return e_atr, e_rep

    def _calculate_solvation_energy(
        self, atom1: AtomContext, atom2: AtomContext, distance: float
    ) -> float:
        """
        Calculate Lazaridis-Karplus solvation energy
        Based on Gaussian exclusion model
        """
        # Get LK parameters
        dgfree1, lambda1, volume1 = self.params.get_lk_params(atom1.atom_type)
        dgfree2, lambda2, volume2 = self.params.get_lk_params(atom2.atom_type)

        # Skip if both have zero solvation
        if dgfree1 == 0.0 and dgfree2 == 0.0:
            return 0.0

        # Gaussian exclusion
        # Atom 2 excludes volume from atom 1's solvation shell
        gaussian1 = np.exp(-((distance / lambda1) ** 2))
        gaussian2 = np.exp(-((distance / lambda2) ** 2))

        # Burial functions (simplified - should use actual neighbor counts)
        burial1 = self._burial_function(atom1.neighbor_count)
        burial2 = self._burial_function(atom2.neighbor_count)

        # Guard against zero atomic volumes to avoid singularity
        if volume1 == 0 or volume2 == 0:
            warnings.warn(
                f"Zero atomic volume detected (types: {atom1.atom_type}/{atom2.atom_type}); "
                "using fallback 1.0 Å³ to avoid singularity.",
                RuntimeWarning,
            )
            volume1 = volume1 or 1.0
            volume2 = volume2 or 1.0

        # Solvation energy change
        e_sol1 = dgfree1 * burial1 * gaussian2 * (volume2 / volume1)
        e_sol2 = dgfree2 * burial2 * gaussian1 * (volume1 / volume2)

        # Apply switching
        switch_params = self.params.get_switch_params("fa_sol")
        switch = switch_function(distance, switch_params["switch_dis"], switch_params["cutoff_dis"])

        return (e_sol1 + e_sol2) * switch

    def _burial_function(self, neighbor_count: int) -> float:
        """
        Calculate burial function for solvation
        Maps neighbor count to exposure level
        """
        params = self.params.burial_params

        if neighbor_count <= params["burial_min"]:
            return 1.0  # Fully exposed
        elif neighbor_count >= params["burial_max"]:
            return 0.0  # Fully buried
        else:
            # Linear interpolation
            frac = (neighbor_count - params["burial_min"]) / (
                params["burial_max"] - params["burial_min"]
            )
            return 1.0 - frac

    def _calculate_electrostatic_energy(
        self, atom1: AtomContext, atom2: AtomContext, distance: float
    ) -> float:
        """
        Calculate electrostatic energy with distance-dependent dielectric
        """
        if distance == 0:
            return 0.0

        # Get charges
        q1 = atom1.formal_charge
        q2 = atom2.formal_charge

        if q1 == 0 or q2 == 0:
            return 0.0

        # Distance-dependent dielectric
        dielectric = sigmoidal_dielectric(distance, self.params.elec_params)

        # Coulomb's law (332.0637 is conversion factor for kcal/mol)
        e_elec = 332.0637 * q1 * q2 / (dielectric * distance)

        # Apply switching
        switch_params = self.params.get_switch_params("fa_elec")
        switch = switch_function(distance, switch_params["switch_dis"], switch_params["cutoff_dis"])

        # Counterpair correction (removes double counting at close range)
        if distance < self.params.elec_params["countpair_distance"]:
            # Simple linear correction
            cp_factor = distance / self.params.elec_params["countpair_distance"]
            e_elec *= cp_factor

        return e_elec * switch

    def _can_form_hbond(self, atom1: AtomContext, atom2: AtomContext) -> bool:
        """Check if two atoms can form hydrogen bond"""
        return (atom1.is_donor and atom2.is_acceptor) or (atom2.is_donor and atom1.is_acceptor)

    def _calculate_hbond_energy_simple(
        self, atom1: AtomContext, atom2: AtomContext, distance: float
    ) -> float:
        """
        Simplified H-bond energy without full geometric dependencies
        For full implementation, would need H positions and angles
        """
        # Determine donor and acceptor
        if atom1.is_donor and atom2.is_acceptor:
            donor_type = "Hpol"  # Simplified
            acceptor_type = atom2.atom_type
        elif atom2.is_donor and atom1.is_acceptor:
            donor_type = "Hpol"  # Simplified
            acceptor_type = atom1.atom_type
        else:
            return 0.0

        # Get polynomial coefficients
        poly = self.params.get_hbond_poly(donor_type, acceptor_type)

        # Evaluate polynomial
        r = distance
        e_dist = (
            poly[0] + poly[1] * r + poly[2] * r * r + poly[3] * r * r * r + poly[4] * r * r * r * r
        )

        # Apply switching
        switch_params = self.params.get_switch_params("hbond")
        switch = switch_function(distance, switch_params["switch_dis"], switch_params["cutoff_dis"])

        # Simplified angular factor (would need actual angles)
        angular_factor = 0.8  # Assume reasonable geometry

        return e_dist * switch * angular_factor

    def calculate_hbond_energy_full(
        self,
        donor_heavy: AtomContext,
        hydrogen: AtomContext,
        acceptor: AtomContext,
        acceptor_base: Optional[AtomContext] = None,
    ) -> float:
        """
        Full H-bond energy with all geometric dependencies

        Args:
            donor_heavy: Heavy atom donor (N, O, S)
            hydrogen: Hydrogen atom
            acceptor: Acceptor atom
            acceptor_base: Base atom for acceptor (for chi angle)

        Returns:
            H-bond energy
        """
        # Calculate distances and angles
        h_coords = hydrogen.coords
        d_coords = donor_heavy.coords
        a_coords = acceptor.coords

        # H-A distance
        r_ha = np.linalg.norm(a_coords - h_coords)

        # D-H-A angle (should be ~180°)
        vec_dh = h_coords - d_coords
        vec_ha = a_coords - h_coords
        cos_dha = np.dot(vec_dh, vec_ha) / (np.linalg.norm(vec_dh) * np.linalg.norm(vec_ha))

        # B-A-H angle (acceptor angle)
        cos_bah = 0.0  # Default if no base
        if acceptor_base is not None:
            b_coords = acceptor_base.coords
            vec_ba = a_coords - b_coords
            vec_ah = h_coords - a_coords
            cos_bah = np.dot(vec_ba, vec_ah) / (np.linalg.norm(vec_ba) * np.linalg.norm(vec_ah))

        # Get polynomial for distance dependence
        donor_h_type = hydrogen.atom_type
        acceptor_type = acceptor.atom_type
        poly = self.params.get_hbond_poly(donor_h_type, acceptor_type)

        # Distance component
        e_dist = (
            poly[0]
            + poly[1] * r_ha
            + poly[2] * r_ha * r_ha
            + poly[3] * r_ha * r_ha * r_ha
            + poly[4] * r_ha * r_ha * r_ha * r_ha
        )

        # Angular components
        params = self.params.hbond_angular

        # D-H-A angle factor (fade from 0.75 to 1.0)
        e_dha = fade_function(cos_dha, params["fade_AHD_min"], params["fade_AHD_max"])

        # B-A-H angle factor (fade from -0.5 to 0.75)
        e_bah = fade_function(cos_bah, params["fade_BAH_min"], params["fade_BAH_max"])

        # Chi angle factor (would need dihedral calculation)
        e_chi = 1.0  # Simplified

        # Combine all factors
        e_hbond = e_dist * e_dha * e_bah * e_chi

        # Apply weight based on H-bond type
        if donor_heavy.is_backbone and acceptor.is_backbone:
            weight_key = (
                "hbond_sr_bb"
                if abs(donor_heavy.residue_id - acceptor.residue_id) <= 4
                else "hbond_lr_bb"
            )
        elif donor_heavy.is_backbone or acceptor.is_backbone:
            weight_key = "hbond_bb_sc"
        else:
            weight_key = "hbond_sc"

        return e_hbond * self.params.get_weight(weight_key)

    def create_atom_context(self, atom_dict: Dict) -> AtomContext:
        """
        Create AtomContext from atom dictionary

        Args:
            atom_dict: Dictionary with atom information

        Returns:
            AtomContext object
        """
        # Get Rosetta atom type
        atom_type = self.atom_typer.get_atom_type(atom_dict)
        type_info = self.atom_typer.get_type_info(atom_type)

        # Create context
        context = AtomContext(
            atom_type=atom_type,
            coords=np.array([atom_dict["x"], atom_dict["y"], atom_dict["z"]]),
            formal_charge=atom_dict.get("formal_charge", type_info.charge),
            is_backbone=atom_dict.get("name", "") in ["N", "CA", "C", "O", "H"],
            is_donor=type_info.is_donor,
            is_acceptor=type_info.is_acceptor,
            is_aromatic=type_info.is_aromatic,
            residue_id=atom_dict.get("resSeq", 0),
            residue_name=atom_dict.get("resname", ""),
        )

        return context


# Global instance
_ref15_calculator = None


def get_ref15_calculator(pH: float = 7.4) -> REF15EnergyCalculator:
    """Get the REF15 calculator instance"""
    global _ref15_calculator
    if _ref15_calculator is None or _ref15_calculator.pH != pH:
        _ref15_calculator = REF15EnergyCalculator(pH)
    return _ref15_calculator
