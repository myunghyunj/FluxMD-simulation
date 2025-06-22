"""
PDB file parser for FluxMD.
Handles both protein and DNA structures.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class PDBParser:
    """Parser for PDB files supporting both proteins and DNA."""

    def __init__(self):
        # Standard amino acids
        self.amino_acids = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }

        # DNA bases
        self.dna_bases = {"DA", "DT", "DG", "DC", "A", "T", "G", "C"}

        # RNA bases (for future extension)
        self.rna_bases = {"A", "U", "G", "C", "RA", "RU", "RG", "RC"}

    def parse(self, pdb_file: str, is_dna: bool = False) -> Optional[pd.DataFrame]:
        """
        Parse a PDB file and return a DataFrame of atoms.

        Args:
            pdb_file: Path to the PDB file
            is_dna: Whether to parse as DNA (affects residue validation)

        Returns:
            DataFrame with columns: atom_id, name, resname, chain, resSeq, x, y, z, element
        """
        atoms = []

        try:
            with open(pdb_file, "r") as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            # Parse PDB line according to standard format
                            atom_id = int(line[6:11])
                            name = line[12:16].strip()
                            resname = line[17:20].strip()
                            chain = line[21]
                            resSeq = int(line[22:26])
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])

                            # Try to get element from columns 76-78
                            element = line[76:78].strip() if len(line) > 76 else ""

                            # If element is missing, infer from atom name
                            if not element:
                                element = self._infer_element(name)

                            # Validate residue type
                            if is_dna:
                                # For DNA, accept DNA bases
                                if resname not in self.dna_bases and resname not in ["HOH", "WAT"]:
                                    # Allow modified bases but warn
                                    if resname not in self.amino_acids:  # Not a protein residue
                                        print(f"Warning: Unknown DNA residue '{resname}'")
                            else:
                                # For proteins, check amino acids
                                if resname not in self.amino_acids and resname not in [
                                    "HOH",
                                    "WAT",
                                ]:
                                    # Could be a ligand or modified residue
                                    if line.startswith("HETATM"):
                                        pass  # Accept HETATM records
                                    else:
                                        print(f"Warning: Unknown residue '{resname}'")

                            atoms.append(
                                {
                                    "atom_id": atom_id,
                                    "name": name,
                                    "resname": resname,
                                    "chain": chain,
                                    "resSeq": resSeq,
                                    "residue_id": resSeq,  # Alias for compatibility
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "element": element.upper(),
                                    "is_dna": is_dna and resname in self.dna_bases,
                                }
                            )

                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse line: {line.strip()}")
                            continue

            if not atoms:
                print(f"Error: No atoms found in {pdb_file}")
                return None

            df = pd.DataFrame(atoms)

            # Add additional properties for DNA
            if is_dna:
                df["base_type"] = (
                    df["resname"]
                    .map(
                        {
                            "DA": "A",
                            "A": "A",
                            "DT": "T",
                            "T": "T",
                            "DG": "G",
                            "G": "G",
                            "DC": "C",
                            "C": "C",
                        }
                    )
                    .fillna("")
                )

            print(f"Parsed {len(df)} atoms from {pdb_file}")
            if is_dna:
                n_bases = df[df["is_dna"]]["resSeq"].nunique()
                print(f"Found {n_bases} DNA nucleotides")

            return df

        except FileNotFoundError:
            print(f"Error: File {pdb_file} not found")
            return None
        except Exception as e:
            print(f"Error parsing PDB file: {str(e)}")
            return None

    def _infer_element(self, atom_name: str) -> str:
        """Infer element from atom name."""
        # Common patterns
        if not atom_name:
            return "C"

        # Single letter elements
        if atom_name[0] in ["H", "C", "N", "O", "S", "P", "F"]:
            return atom_name[0]

        # Two-letter elements
        if atom_name[:2] in ["CL", "BR", "FE", "ZN", "MG", "CA", "NA", "MN"]:
            return atom_name[:2]

        # DNA-specific
        if atom_name in ["O5'", "C5'", "O3'", "C3'", "O4'", "C4'", "C1'", "C2'"]:
            return atom_name[0]
        if atom_name in ["OP1", "OP2", "O1P", "O2P"]:
            return "O"

        # Default to carbon
        return "C"

    def get_sequence(self, df: pd.DataFrame, chain: str = "A") -> str:
        """
        Extract sequence from parsed DNA DataFrame.

        Args:
            df: DataFrame from parse()
            chain: Chain ID to extract

        Returns:
            DNA sequence as string (5' to 3')
        """
        if "base_type" not in df.columns:
            return ""

        chain_df = df[(df["chain"] == chain) & (df["is_dna"])].drop_duplicates("resSeq")
        sequence = "".join(chain_df.sort_values("resSeq")["base_type"].values)
        return sequence
