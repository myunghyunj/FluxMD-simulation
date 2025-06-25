"""
Rosetta REF15 Atom Type Classification System
Maps PDB atoms to Rosetta atom types for accurate energy calculations
"""

import numpy as np
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AtomTypeInfo:
    """Information about a Rosetta atom type"""

    rosetta_type: str
    element: str
    hybridization: str
    lj_radius: float  # Angstroms
    lj_wdepth: float  # kcal/mol
    lk_dgfree: float  # kcal/mol - solvation free energy
    lk_lambda: float  # Angstroms - correlation length
    lk_volume: float  # Cubic angstroms
    is_donor: bool
    is_acceptor: bool
    is_polar: bool
    is_aromatic: bool
    charge: float = 0.0


class RosettaAtomTyper:
    """
    Convert PDB atoms to REF15 atom types based on:
    - Element
    - Hybridization state
    - Chemical environment
    - Connectivity
    """

    def __init__(self):
        # Initialize REF15 atom type database
        self._init_atom_types()
        self._init_residue_specific_rules()
        self._init_connectivity_patterns()

    def _init_atom_types(self):
        """Initialize the REF15 atom type parameters"""
        self.atom_types = {
            # Carbons
            "CAbb": AtomTypeInfo(
                "CAbb", "C", "sp2", 1.75, 0.160, -0.24, 3.5, 14.7, False, False, False, True
            ),
            "CObb": AtomTypeInfo(
                "CObb", "C", "sp2", 1.75, 0.160, -0.24, 3.5, 14.7, False, False, False, False
            ),
            "CNH2": AtomTypeInfo(
                "CNH2", "C", "sp2", 1.75, 0.160, -0.24, 3.5, 14.7, False, False, False, False
            ),
            "COO": AtomTypeInfo(
                "COO", "C", "sp2", 1.75, 0.160, -0.24, 3.5, 14.7, False, False, False, False
            ),
            "CH1": AtomTypeInfo(
                "CH1", "C", "sp3", 2.00, 0.080, -0.04, 3.5, 23.7, False, False, False, False
            ),
            "CH2": AtomTypeInfo(
                "CH2", "C", "sp3", 2.00, 0.080, -0.04, 3.5, 23.7, False, False, False, False
            ),
            "CH3": AtomTypeInfo(
                "CH3", "C", "sp3", 2.00, 0.080, 0.00, 3.5, 23.7, False, False, False, False
            ),
            "aroC": AtomTypeInfo(
                "aroC", "C", "sp2", 1.75, 0.160, -0.24, 3.5, 18.4, False, False, False, True
            ),
            # Nitrogens
            "Nbb": AtomTypeInfo(
                "Nbb", "N", "sp2", 1.65, 0.170, -3.20, 2.7, 11.2, True, False, True, False
            ),
            "Npro": AtomTypeInfo(
                "Npro", "N", "sp2", 1.65, 0.170, -1.55, 2.7, 11.2, False, False, True, False
            ),
            "NH2O": AtomTypeInfo(
                "NH2O", "N", "sp2", 1.65, 0.170, -3.20, 2.7, 11.2, True, False, True, False
            ),
            "Nlys": AtomTypeInfo(
                "Nlys", "N", "sp3", 1.65, 0.170, -3.20, 2.7, 11.2, True, False, True, False, 1.0
            ),
            "Ntrp": AtomTypeInfo(
                "Ntrp", "N", "sp2", 1.65, 0.170, -3.20, 2.7, 11.2, True, False, True, False
            ),
            "Nhis": AtomTypeInfo(
                "Nhis", "N", "sp2", 1.65, 0.170, -3.20, 2.7, 11.2, True, True, True, True
            ),
            "NtrR": AtomTypeInfo(
                "NtrR", "N", "sp2", 1.65, 0.170, -3.20, 2.7, 11.2, True, False, True, False, 0.75
            ),
            "Narg": AtomTypeInfo(
                "Narg", "N", "sp2", 1.65, 0.170, -3.20, 2.7, 11.2, True, False, True, False, 0.75
            ),
            # Oxygens
            "OCbb": AtomTypeInfo(
                "OCbb", "O", "sp2", 1.50, 0.210, -2.85, 2.6, 10.8, False, True, True, False
            ),
            "OH": AtomTypeInfo(
                "OH", "O", "sp3", 1.55, 0.210, -2.85, 2.6, 10.8, True, True, True, False
            ),
            "OOC": AtomTypeInfo(
                "OOC", "O", "sp2", 1.50, 0.210, -2.85, 2.6, 10.8, False, True, True, False, -0.5
            ),
            "ONH2": AtomTypeInfo(
                "ONH2", "O", "sp2", 1.50, 0.210, -2.85, 2.6, 10.8, False, True, True, False
            ),
            # Sulfurs
            "S": AtomTypeInfo(
                "S", "S", "sp3", 1.85, 0.200, -0.45, 3.5, 17.0, False, False, False, False
            ),
            "SH1": AtomTypeInfo(
                "SH1", "S", "sp3", 1.85, 0.200, -0.60, 3.5, 17.0, True, False, True, False
            ),
            # Hydrogens
            "Hpol": AtomTypeInfo(
                "Hpol", "H", "s", 1.00, 0.046, 0.00, 1.0, 0.0, True, False, True, False
            ),
            "Hapo": AtomTypeInfo(
                "Hapo", "H", "s", 1.00, 0.046, 0.00, 1.0, 0.0, False, False, False, False
            ),
            "Haro": AtomTypeInfo(
                "Haro", "H", "s", 1.10, 0.015, 0.00, 1.0, 0.0, False, False, False, True
            ),
            "HNbb": AtomTypeInfo(
                "HNbb", "H", "s", 0.95, 0.046, 0.00, 1.0, 0.0, True, False, True, False
            ),
            "HS": AtomTypeInfo(
                "HS", "H", "s", 0.95, 0.046, 0.00, 1.0, 0.0, True, False, True, False
            ),
            # Special
            "VIRT": AtomTypeInfo(
                "VIRT", "X", "none", 1.00, 0.000, 0.00, 1.0, 0.0, False, False, False, False
            ),
        }

    def _init_residue_specific_rules(self):
        """Residue-specific atom typing rules"""
        self.residue_rules = {
            # Amino acids
            "ALA": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH3",
                "H": "HNbb",
                "HA": "Hapo",
            },
            "ARG": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH2",
                "CD": "CH2",
                "NE": "NtrR",
                "CZ": "CNH2",
                "NH1": "Narg",
                "NH2": "Narg",
                "H": "HNbb",
                "HE": "Hpol",
                "HH11": "Hpol",
                "HH12": "Hpol",
                "HH21": "Hpol",
                "HH22": "Hpol",
            },
            "ASN": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CNH2",
                "OD1": "ONH2",
                "ND2": "NH2O",
                "H": "HNbb",
                "HD21": "Hpol",
                "HD22": "Hpol",
            },
            "ASP": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "COO",
                "OD1": "OOC",
                "OD2": "OOC",
                "H": "HNbb",
            },
            "CYS": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "SG": "SH1",
                "H": "HNbb",
                "HG": "HS",
            },
            "GLN": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH2",
                "CD": "CNH2",
                "OE1": "ONH2",
                "NE2": "NH2O",
                "H": "HNbb",
                "HE21": "Hpol",
                "HE22": "Hpol",
            },
            "GLU": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH2",
                "CD": "COO",
                "OE1": "OOC",
                "OE2": "OOC",
                "H": "HNbb",
            },
            "GLY": {"N": "Nbb", "CA": "CH2", "C": "CObb", "O": "OCbb", "H": "HNbb"},
            "HIS": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "aroC",
                "ND1": "Nhis",
                "CD2": "aroC",
                "CE1": "aroC",
                "NE2": "Nhis",
                "H": "HNbb",
                "HD1": "Hpol",
                "HE2": "Hpol",
            },
            "ILE": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH1",
                "CG1": "CH2",
                "CG2": "CH3",
                "CD1": "CH3",
                "H": "HNbb",
            },
            "LEU": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH1",
                "CD1": "CH3",
                "CD2": "CH3",
                "H": "HNbb",
            },
            "LYS": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH2",
                "CD": "CH2",
                "CE": "CH2",
                "NZ": "Nlys",
                "H": "HNbb",
                "HZ1": "Hpol",
                "HZ2": "Hpol",
                "HZ3": "Hpol",
            },
            "MET": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH2",
                "SD": "S",
                "CE": "CH3",
                "H": "HNbb",
            },
            "PHE": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CAbb",
                "CD1": "CAbb",
                "CD2": "CAbb",
                "CE1": "CAbb",
                "CE2": "CAbb",
                "CZ": "CAbb",
                "H": "HNbb",
                "HD1": "Haro",
                "HD2": "Haro",
                "HE1": "Haro",
                "HE2": "Haro",
                "HZ": "Haro",
            },
            "PRO": {
                "N": "Npro",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CH2",
                "CD": "CH2",
            },
            "SER": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "OG": "OH",
                "H": "HNbb",
                "HG": "Hpol",
            },
            "THR": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH1",
                "OG1": "OH",
                "CG2": "CH3",
                "H": "HNbb",
                "HG1": "Hpol",
            },
            "TRP": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CAbb",
                "CD1": "CAbb",
                "CD2": "aroC",
                "NE1": "Ntrp",
                "CE2": "aroC",
                "CE3": "CAbb",
                "CZ2": "CAbb",
                "CZ3": "CAbb",
                "CH2": "CAbb",
                "H": "HNbb",
                "HE1": "Hpol",
                "HD1": "Haro",
                "HE3": "Haro",
                "HZ2": "Haro",
                "HZ3": "Haro",
                "HH2": "Haro",
            },
            "TYR": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH2",
                "CG": "CAbb",
                "CD1": "CAbb",
                "CD2": "CAbb",
                "CE1": "CAbb",
                "CE2": "CAbb",
                "CZ": "CAbb",
                "OH": "OH",
                "H": "HNbb",
                "HH": "Hpol",
                "HD1": "Haro",
                "HD2": "Haro",
                "HE1": "Haro",
                "HE2": "Haro",
            },
            "VAL": {
                "N": "Nbb",
                "CA": "CH1",
                "C": "CObb",
                "O": "OCbb",
                "CB": "CH1",
                "CG1": "CH3",
                "CG2": "CH3",
                "H": "HNbb",
            },
        }

    def _init_connectivity_patterns(self):
        """Initialize connectivity patterns for atom typing"""
        self.connectivity_patterns = {
            # Patterns for identifying hybridization states
            "sp2_carbon": {"bond_count": [3], "neighbor_elements": {"C", "N", "O", "S"}},
            "sp3_carbon": {"bond_count": [4], "neighbor_elements": {"C", "N", "O", "S", "H"}},
            "aromatic": {"ring_size": [5, 6], "planar": True},
        }

    def get_atom_type(self, atom_dict: Dict) -> str:
        """
        Determine REF15 atom type from atom properties

        Args:
            atom_dict: Dictionary containing:
                - 'element': Element symbol
                - 'name': Atom name (e.g., 'CA', 'CB')
                - 'resname': Residue name
                - 'neighbors': List of bonded atoms (optional)
                - 'is_aromatic': Boolean (optional)

        Returns:
            REF15 atom type string
        """
        element = atom_dict.get("element", "").upper()
        atom_name = atom_dict.get("name", "").upper()
        res_name = atom_dict.get("resname", "").upper()

        # First check residue-specific rules
        if res_name in self.residue_rules:
            if atom_name in self.residue_rules[res_name]:
                return self.residue_rules[res_name][atom_name]

        # Fallback to general rules
        return self._apply_general_rules(atom_dict)

    def _apply_general_rules(self, atom_dict: Dict) -> str:
        """Apply general atom typing rules when residue-specific rules don't match"""
        element = atom_dict.get("element", "").upper()
        atom_name = atom_dict.get("name", "").upper()

        # Simple element-based fallbacks
        if element == "C":
            if atom_dict.get("is_aromatic", False):
                return "CAbb"
            elif "O" in atom_name or "N" in atom_name:  # Likely carbonyl
                return "CObb"
            else:
                # Count hydrogens to determine CH1/CH2/CH3
                h_count = atom_dict.get("hydrogen_count", 1)
                return f"CH{h_count}" if h_count <= 3 else "CH1"

        elif element == "N":
            if atom_name == "N":  # Backbone nitrogen
                return "Nbb"
            elif atom_dict.get("is_aromatic", False):
                return "Nhis"
            else:
                return "Nlys"  # Generic nitrogen

        elif element == "O":
            if "C" in atom_name:  # Carbonyl oxygen
                return "OCbb"
            else:
                return "OH"  # Hydroxyl

        elif element == "S":
            if atom_dict.get("has_hydrogen", False):
                return "SH1"
            else:
                return "S"

        elif element == "H":
            # Check what it's attached to
            if atom_dict.get("attached_to_polar", False):
                return "Hpol"
            elif atom_dict.get("attached_to_aromatic", False):
                return "Haro"
            else:
                return "Hapo"

        # Default fallback
        logger.warning(f"Unknown atom type for element {element}, using CH3 as default")
        return "CH3"

    def get_type_info(self, atom_type: str) -> AtomTypeInfo:
        """Get complete information for a REF15 atom type"""
        if atom_type not in self.atom_types:
            logger.warning(f"Unknown atom type {atom_type}, using CH3 as default")
            return self.atom_types["CH3"]
        return self.atom_types[atom_type]

    def classify_ligand_atom(self, atom_dict: Dict) -> str:
        """
        Special handling for ligand atoms which may not follow protein conventions
        """
        element = atom_dict.get("element", "").upper()

        # For ligands, use more generic typing based on element and connectivity
        if element == "C":
            # Check aromaticity first
            if atom_dict.get("is_aromatic", False):
                return "aroC"

            # Check for carbonyl carbon
            neighbors = atom_dict.get("neighbors", [])
            if any(n.get("element") == "O" and n.get("bond_order", 1) > 1 for n in neighbors):
                return "CObb"

            # Count hydrogens
            h_count = sum(1 for n in neighbors if n.get("element") == "H")
            if h_count == 0:
                return "CH1"  # Assuming one implicit H
            elif h_count == 1:
                return "CH1"
            elif h_count == 2:
                return "CH2"
            else:
                return "CH3"

        elif element == "N":
            if atom_dict.get("is_aromatic", False):
                return "Nhis"
            elif atom_dict.get("formal_charge", 0) > 0:
                return "Nlys"
            else:
                return "NH2O"

        elif element == "O":
            # Check if connected to carbon
            neighbors = atom_dict.get("neighbors", [])
            if any(n.get("element") == "C" for n in neighbors):
                if atom_dict.get("formal_charge", 0) < 0:
                    return "OOC"
                elif any(n.get("bond_order", 1) > 1 for n in neighbors):
                    return "ONH2"
                else:
                    return "OH"
            return "OH"

        elif element == "S":
            if atom_dict.get("has_hydrogen", False):
                return "SH1"
            return "S"

        elif element == "H":
            # Check what it's bonded to
            neighbors = atom_dict.get("neighbors", [])
            if neighbors:
                attached = neighbors[0].get("element", "")
                if attached in ["N", "O", "S"]:
                    return "Hpol"
                elif atom_dict.get("is_aromatic", False):
                    return "Haro"
            return "Hapo"

        elif element == "F":
            return "F"  # Add fluorine if needed
        elif element == "CL":
            return "Cl"  # Add chlorine if needed
        elif element == "BR":
            return "Br"  # Add bromine if needed

        # Default
        logger.warning(f"Unknown ligand atom element {element}")
        return "CH3"


# Singleton instance
_atom_typer = None


def get_atom_typer() -> RosettaAtomTyper:
    """Get the singleton atom typer instance"""
    global _atom_typer
    if _atom_typer is None:
        _atom_typer = RosettaAtomTyper()
    return _atom_typer
