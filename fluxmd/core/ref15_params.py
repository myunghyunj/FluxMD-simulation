"""
REF15 Energy Function Parameters
Complete parameter set for Rosetta Energy Function 2015
"""

from typing import Dict, List, Tuple

import numpy as np

# REF15 Weight Set
REF15_WEIGHTS = {
    "fa_atr": 1.0,
    "fa_rep": 0.55,
    "fa_sol": 1.0,
    "fa_intra_rep": 0.005,
    "fa_intra_sol_xover4": 1.0,
    "lk_ball_wtd": 1.0,
    "fa_elec": 1.0,
    "hbond_lr_bb": 1.17,
    "hbond_sr_bb": 1.17,
    "hbond_bb_sc": 1.17,
    "hbond_sc": 1.1,
    "dslf_fa13": 1.0,
    "omega": 0.6,
    "fa_dun": 0.7,
    "p_aa_pp": 0.4,
    "yhh_planarity": 0.625,
    "ref": 1.0,
    "rama_prepro": 0.45,
}

# Lennard-Jones Parameters
# Format: (atom_type1, atom_type2): (radius, well_depth)
LJ_PARAMS = {
    # Carbon-Carbon interactions
    ("CAbb", "CAbb"): (3.8159, 0.1209),
    ("CAbb", "CObb"): (3.8159, 0.1209),
    ("CAbb", "CH1"): (4.0654, 0.0832),
    ("CAbb", "CH2"): (4.0654, 0.0832),
    ("CAbb", "CH3"): (4.0654, 0.0832),
    ("CObb", "CObb"): (3.8159, 0.1209),
    ("CObb", "CH1"): (4.0654, 0.0832),
    ("CH1", "CH1"): (4.3149, 0.0573),
    ("CH1", "CH2"): (4.3149, 0.0573),
    ("CH1", "CH3"): (4.3149, 0.0573),
    ("CH2", "CH2"): (4.3149, 0.0573),
    ("CH2", "CH3"): (4.3149, 0.0573),
    ("CH3", "CH3"): (4.3149, 0.0573),
    # Carbon-Nitrogen interactions
    ("CAbb", "Nbb"): (3.6909, 0.1613),
    ("CAbb", "Nhis"): (3.6909, 0.1613),
    ("CAbb", "Nlys"): (3.6909, 0.1613),
    ("CObb", "Nbb"): (3.6909, 0.1613),
    ("CH1", "Nbb"): (3.9404, 0.1110),
    ("CH2", "Nbb"): (3.9404, 0.1110),
    ("CH3", "Nbb"): (3.9404, 0.1110),
    # Carbon-Oxygen interactions
    ("CAbb", "OCbb"): (3.5659, 0.1755),
    ("CAbb", "OH"): (3.6159, 0.1755),
    ("CObb", "OCbb"): (3.5659, 0.1755),
    ("CObb", "OH"): (3.6159, 0.1755),
    ("CH1", "OCbb"): (3.8154, 0.1209),
    ("CH1", "OH"): (3.8654, 0.1209),
    ("CH2", "OCbb"): (3.8154, 0.1209),
    ("CH3", "OCbb"): (3.8154, 0.1209),
    # Nitrogen-Nitrogen interactions
    ("Nbb", "Nbb"): (3.5659, 0.2151),
    ("Nbb", "Nhis"): (3.5659, 0.2151),
    ("Nbb", "Nlys"): (3.5659, 0.2151),
    ("Nhis", "Nhis"): (3.5659, 0.2151),
    ("Nhis", "Nlys"): (3.5659, 0.2151),
    ("Nlys", "Nlys"): (3.5659, 0.2151),
    # Nitrogen-Oxygen interactions
    ("Nbb", "OCbb"): (3.4409, 0.2339),
    ("Nbb", "OH"): (3.4909, 0.2339),
    ("Nhis", "OCbb"): (3.4409, 0.2339),
    ("Nlys", "OCbb"): (3.4409, 0.2339),
    ("Nlys", "OH"): (3.4909, 0.2339),
    # Oxygen-Oxygen interactions
    ("OCbb", "OCbb"): (3.3159, 0.2547),
    ("OCbb", "OH"): (3.3659, 0.2547),
    ("OH", "OH"): (3.4159, 0.2547),
    # Sulfur interactions
    ("S", "S"): (4.0359, 0.2000),
    ("S", "SH1"): (4.0359, 0.2000),
    ("SH1", "SH1"): (4.0359, 0.2000),
    ("S", "CAbb"): (4.0009, 0.1414),
    ("S", "CH1"): (4.2504, 0.0973),
    ("S", "Nbb"): (3.8759, 0.1833),
    ("S", "OCbb"): (3.7509, 0.1995),
    # Hydrogen interactions (minimal LJ)
    ("Hpol", "Hpol"): (2.2489, 0.0157),
    ("Hpol", "Hapo"): (2.5489, 0.0108),
    ("Hapo", "Hapo"): (2.8489, 0.0074),
    ("Haro", "Haro"): (2.6000, 0.0100),
    # Add default values for missing pairs
    ("default", "default"): (3.8000, 0.1000),
}

# Hydrogen bond parameters
# Format: (donor_type, acceptor_type): [polynomial_coefficients]
HBOND_POLYNOMIALS = {
    # Backbone-backbone
    ("HNbb", "OCbb"): [-1.0856, 2.5003, -2.1487, 0.7973, -0.1115],
    # Backbone-sidechain
    ("HNbb", "OH"): [-0.8677, 1.9996, -1.7190, 0.6378, -0.0892],
    ("HNbb", "OOC"): [-1.3020, 3.0004, -2.5788, 0.9566, -0.1338],
    # Sidechain-backbone
    ("Hpol", "OCbb"): [-0.8677, 1.9996, -1.7190, 0.6378, -0.0892],
    ("HS", "OCbb"): [-0.6507, 1.4997, -1.2893, 0.4784, -0.0669],
    # Sidechain-sidechain
    ("Hpol", "OH"): [-0.7802, 1.7997, -1.5471, 0.5740, -0.0803],
    ("Hpol", "OOC"): [-1.0416, 2.4003, -2.0628, 0.7653, -0.1071],
    ("HS", "OH"): [-0.5206, 1.1998, -1.0314, 0.3827, -0.0535],
    # Default
    ("default", "default"): [-0.8000, 1.8000, -1.5000, 0.5500, -0.0800],
}

# H-bond angular parameters
HBOND_ANGULAR = {
    "fade_BAH_min": -0.5,
    "fade_BAH_max": 0.75,
    "fade_AHD_min": 0.75,
    "fade_AHD_max": 1.0,
    "chi_scale": {
        "sp2": 0.8,  # Strength of chi angle dependence for sp2
        "sp3": 0.0,  # No chi dependence for sp3
    },
}

# Solvation parameters (Lazaridis-Karplus)
LK_PARAMS = {
    # Format: atom_type: (dgfree, lambda, volume)
    "CAbb": (-0.24, 3.5, 14.7),
    "CObb": (-0.24, 3.5, 14.7),
    "CH1": (-0.04, 3.5, 23.7),
    "CH2": (-0.04, 3.5, 23.7),
    "CH3": (0.00, 3.5, 23.7),
    "aroC": (-0.24, 3.5, 18.4),
    "Nbb": (-3.20, 2.7, 11.2),
    "Npro": (-1.55, 2.7, 11.2),
    "NH2O": (-3.20, 2.7, 11.2),
    "Nlys": (-3.20, 2.7, 11.2),
    "Ntrp": (-3.20, 2.7, 11.2),
    "Nhis": (-3.20, 2.7, 11.2),
    "NtrR": (-3.20, 2.7, 11.2),
    "Narg": (-3.20, 2.7, 11.2),
    "OCbb": (-2.85, 2.6, 10.8),
    "OH": (-2.85, 2.6, 10.8),
    "OOC": (-2.85, 2.6, 10.8),
    "ONH2": (-2.85, 2.6, 10.8),
    "S": (-0.45, 3.5, 17.0),
    "SH1": (-0.60, 3.5, 17.0),
    "Hpol": (0.00, 1.0, 0.0),
    "Hapo": (0.00, 1.0, 0.0),
    "Haro": (0.00, 1.0, 0.0),
    "HNbb": (0.00, 1.0, 0.0),
    "HS": (0.00, 1.0, 0.0),
    "default": (0.00, 3.0, 15.0),
}

# Electrostatic parameters
ELEC_PARAMS = {
    "dielectric_model": "sigmoidal",  # REF15 uses sigmoidal dielectric
    "die_offset": 0.0,
    "die_slope": 0.12,
    "die_min": 10.0,
    "die_max": 80.0,
    "switch_dis": 5.5,
    "cutoff_dis": 6.0,
    "no_dis_dep_die": False,
    "countpair_distance": 5.5,
}

# Switching function parameters
SWITCH_PARAMS = {
    "fa_atr": {"switch_dis": 5.5, "cutoff_dis": 6.0, "smooth_type": "x3"},  # Cubic switching
    "fa_rep": {
        "switch_dis": 0.0,
        "cutoff_dis": 3.0,
        "smooth_type": "hard",  # No switching for repulsive
    },
    "fa_sol": {"switch_dis": 5.2, "cutoff_dis": 5.5, "smooth_type": "x3"},
    "hbond": {"switch_dis": 1.9, "cutoff_dis": 3.3, "smooth_type": "x3"},
    "fa_elec": {"switch_dis": 5.5, "cutoff_dis": 6.0, "smooth_type": "x3"},
}

# Burial function parameters for solvation
BURIAL_PARAMS = {
    "neighbor_radius": 5.2,  # Angstroms
    "burial_min": 0,  # Minimum neighbors for full burial
    "burial_max": 17,  # Maximum neighbors for no solvation
    "burial_shift": 0.0,
    "burial_slope": 0.1,
}


class REF15Parameters:
    """Singleton class to manage REF15 parameters"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(REF15Parameters, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.weights = REF15_WEIGHTS
        self.lj_params = self._expand_lj_params()
        self.hbond_poly = HBOND_POLYNOMIALS
        self.hbond_angular = HBOND_ANGULAR
        self.lk_params = LK_PARAMS
        self.elec_params = ELEC_PARAMS
        self.switch_params = SWITCH_PARAMS
        self.burial_params = BURIAL_PARAMS

        self._initialized = True

    def _expand_lj_params(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Expand LJ parameters to include all symmetric pairs"""
        expanded = {}

        for (type1, type2), params in LJ_PARAMS.items():
            expanded[(type1, type2)] = params
            expanded[(type2, type1)] = params  # Symmetric

        return expanded

    def get_lj_params(self, type1: str, type2: str) -> Tuple[float, float]:
        """Get LJ parameters for an atom pair"""
        key = (type1, type2)
        if key in self.lj_params:
            return self.lj_params[key]
        else:
            # Use combining rules or default
            return self._combine_lj_params(type1, type2)

    def _combine_lj_params(self, type1: str, type2: str) -> Tuple[float, float]:
        """Apply Lorentz-Berthelot combining rules"""
        # Try to get individual parameters
        params1 = self.lj_params.get((type1, type1), LJ_PARAMS[("default", "default")])
        params2 = self.lj_params.get((type2, type2), LJ_PARAMS[("default", "default")])

        # Arithmetic mean for radius
        radius = (params1[0] + params2[0]) / 2.0

        # Geometric mean for well depth
        well_depth = np.sqrt(params1[1] * params2[1])

        return (radius, well_depth)

    def get_hbond_poly(self, donor: str, acceptor: str) -> List[float]:
        """Get H-bond polynomial coefficients"""
        key = (donor, acceptor)
        if key in self.hbond_poly:
            return self.hbond_poly[key]
        else:
            return self.hbond_poly[("default", "default")]

    def get_lk_params(self, atom_type: str) -> Tuple[float, float, float]:
        """Get Lazaridis-Karplus solvation parameters"""
        if atom_type in self.lk_params:
            return self.lk_params[atom_type]
        else:
            return self.lk_params["default"]

    def get_weight(self, term: str) -> float:
        """Get weight for an energy term"""
        return self.weights.get(term, 0.0)

    def get_switch_params(self, term: str) -> Dict:
        """Get switching function parameters for an energy term"""
        return self.switch_params.get(term, self.switch_params["fa_atr"])


# Global instance
_ref15_params = None


def get_ref15_params() -> REF15Parameters:
    """Get the singleton REF15 parameters instance"""
    global _ref15_params
    if _ref15_params is None:
        _ref15_params = REF15Parameters()
    return _ref15_params


# Switching functions
def switch_function(
    distance: float, switch_dis: float, cutoff_dis: float, smooth_type: str = "x3"
) -> float:
    """
    REF15 switching function for smooth cutoffs

    Args:
        distance: Inter-atomic distance
        switch_dis: Distance where switching begins
        cutoff_dis: Distance where function goes to zero
        smooth_type: Type of switching ('x3' for cubic, 'hard' for step)

    Returns:
        Switching factor between 0 and 1
    """
    if smooth_type == "hard":
        return 1.0 if distance < cutoff_dis else 0.0

    if distance <= switch_dis:
        return 1.0
    elif distance >= cutoff_dis:
        return 0.0
    else:
        # Cubic switching function
        x = (distance - switch_dis) / (cutoff_dis - switch_dis)
        return 1.0 - 3.0 * x * x + 2.0 * x * x * x


def fade_function(value: float, min_val: float, max_val: float) -> float:
    """
    Fade function for angular dependencies

    Args:
        value: Input value (e.g., cosine of angle)
        min_val: Value where fade starts
        max_val: Value where fade reaches 1

    Returns:
        Fade factor between 0 and 1
    """
    if value <= min_val:
        return 0.0
    elif value >= max_val:
        return 1.0
    else:
        x = (value - min_val) / (max_val - min_val)
        return 3.0 * x * x - 2.0 * x * x * x


def sigmoidal_dielectric(distance: float, params: Dict) -> float:
    """
    REF15 sigmoidal distance-dependent dielectric function

    Args:
        distance: Inter-atomic distance
        params: Electrostatic parameters dictionary

    Returns:
        Dielectric constant
    """
    if params.get("no_dis_dep_die", False):
        return params["die_min"]

    # Sigmoidal transition
    x = (distance - params["die_offset"]) * params["die_slope"]
    sigmoid = 1.0 / (1.0 + np.exp(-x))

    # Interpolate between min and max dielectric
    die = params["die_min"] + (params["die_max"] - params["die_min"]) * sigmoid

    return die
