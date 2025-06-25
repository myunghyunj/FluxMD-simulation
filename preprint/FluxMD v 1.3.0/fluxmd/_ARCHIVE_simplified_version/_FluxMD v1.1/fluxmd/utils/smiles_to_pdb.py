#!/usr/bin/env python3
"""
SMILES to PDB Converter - Enhanced version with proper aromatic bond handling
Converts SMILES strings to 3D PDB structures using NCI CACTUS and OpenBabel
"""

import os
import sys
import urllib.parse
import urllib.request
import subprocess
import tempfile
from typing import Optional, Tuple


class SMILESConverter:
    """Converts SMILES strings to PDB format with aromatic bond preservation."""

    # CACTUS web service URL
    CACTUS_BASE_URL = "https://cactus.nci.nih.gov/chemical/structure"

    def __init__(self):
        """Initialize the SMILES converter."""
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            result = subprocess.run(["obabel", "-V"], capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError
        except FileNotFoundError:
            print("Warning: OpenBabel not found. Some features may be limited.")
            print("Install with: conda install -c conda-forge openbabel")

    def convert(self, smiles_string: str, output_name: str = "ligand") -> Optional[str]:
        """
        Convert SMILES to PDB with proper aromatic bond representation.

        Args:
            smiles_string: SMILES string to convert
            output_name: Base name for output files (without extension)

        Returns:
            Path to PDB file if successful, None otherwise
        """
        pdb_file = f"{output_name}.pdb"
        sdf_file = f"{output_name}.sdf"

        try:
            print(f"Converting SMILES to 3D structure...")
            print(f"SMILES: {smiles_string}")

            # Get SDF from CACTUS
            sdf_content = self._fetch_from_cactus(smiles_string, "sdf")
            if not sdf_content:
                return None

            # Save SDF file
            with open(sdf_file, "w") as f:
                f.write(sdf_content)

            # Convert to PDB with bond orders
            if self._convert_sdf_to_pdb(sdf_file, pdb_file):
                self._analyze_structure(pdb_file, sdf_content)
                return pdb_file
            else:
                # Fallback to direct CACTUS PDB
                print("  Falling back to CACTUS PDB...")
                pdb_content = self._fetch_from_cactus(smiles_string, "pdb")
                if pdb_content:
                    with open(pdb_file, "w") as f:
                        f.write(pdb_content)
                    return pdb_file

        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback

            traceback.print_exc()

        return None

    def _fetch_from_cactus(self, smiles_string: str, format_type: str) -> Optional[str]:
        """
        Fetch structure from NCI CACTUS web service.

        Args:
            smiles_string: SMILES string to convert
            format_type: Output format ('sdf' or 'pdb')

        Returns:
            Content string if successful, None otherwise
        """
        try:
            # URL encode the SMILES string
            encoded_smiles = urllib.parse.quote(smiles_string, safe="")

            # Build URL
            url = f"{self.CACTUS_BASE_URL}/{encoded_smiles}/file?format={format_type}&get3d=true"

            print(f"  Fetching 3D structure from NCI CACTUS...")

            with urllib.request.urlopen(url) as response:
                content = response.read().decode("utf-8")

            # Check if we got an error
            if "Page not found" in content or "<html>" in content:
                print("Error: CACTUS could not process this SMILES string")
                return None

            return content

        except urllib.error.URLError as e:
            print(f"Error connecting to CACTUS service: {e}")
            print("Please check your internet connection")
            return None

    def _convert_sdf_to_pdb(self, sdf_file: str, pdb_file: str) -> bool:
        """
        Convert SDF to PDB using OpenBabel with bond order preservation.

        Args:
            sdf_file: Input SDF file path
            pdb_file: Output PDB file path

        Returns:
            True if successful, False otherwise
        """
        try:
            print("  Converting to PDB with proper bond orders...")

            # OpenBabel options:
            # -h: Add hydrogens
            # -b: Output bond orders in CONECT records
            cmd = ["obabel", sdf_file, "-O", pdb_file, "-h", "-b"]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"OpenBabel error: {result.stderr}")
                return False

            return True

        except FileNotFoundError:
            print("OpenBabel not available for enhanced conversion")
            return False

    def _analyze_structure(self, pdb_file: str, sdf_content: str):
        """
        Analyze the generated structure and print statistics.

        Args:
            pdb_file: Path to PDB file
            sdf_content: Original SDF content
        """
        with open(pdb_file, "r") as f:
            pdb_content = f.read()

        # Count atoms and bonds
        atom_count = pdb_content.count("HETATM")
        conect_lines = [line for line in pdb_content.split("\n") if line.startswith("CONECT")]

        # Check for aromatic bonds in SDF (bond type 4)
        sdf_aromatic_bonds = sdf_content.count("  4  ") + sdf_content.count(" 4 0 ")

        # Check for multiple bonds in CONECT records
        multiple_bonds = 0
        for line in conect_lines:
            parts = line.split()
            if len(parts) > 2:
                # Count duplicate connections (indicates multiple bonds)
                connections = parts[1:]
                for atom in set(connections):
                    if connections.count(atom) > 1:
                        multiple_bonds += 1

        print(f"\nGenerated structure:")
        print(f"  Atoms: {atom_count}")
        print(f"  CONECT records: {len(conect_lines)}")
        if sdf_aromatic_bonds > 0:
            print(f"  Aromatic bonds in SDF: {sdf_aromatic_bonds}")
        if multiple_bonds > 0:
            print(
                f"  Multiple bonds in PDB: {multiple_bonds // 2}"
            )  # Divide by 2 as each bond is counted twice

        # Display sample CONECT lines
        if conect_lines:
            print("\nSample CONECT records:")
            for line in conect_lines[:5]:
                print(f"  {line}")
            if len(conect_lines) > 5:
                print(f"  ... ({len(conect_lines) - 5} more)")


def convert_smiles_to_pdb_improved(
    smiles_string: str, output_name: str = "ligand"
) -> Optional[str]:
    """
    Convenience function for backward compatibility.

    Args:
        smiles_string: SMILES string to convert
        output_name: Base name for output files

    Returns:
        Path to PDB file if successful, None otherwise
    """
    converter = SMILESConverter()
    return converter.convert(smiles_string, output_name)


def main():
    """Command-line interface for SMILES to PDB conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SMILES to PDB with proper aromatic bond handling"
    )
    parser.add_argument("smiles", help="SMILES string to convert")
    parser.add_argument(
        "-o", "--output", default="ligand", help="Output file base name (default: ligand)"
    )

    args = parser.parse_args()

    converter = SMILESConverter()
    result = converter.convert(args.smiles, args.output)

    if result:
        print(f"\nSuccess! PDB file created: {result}")
        print(f"SDF file also created: {args.output}.sdf")
    else:
        print("\nConversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
