"""
Command-line interface for FluxMD
"""

import importlib.util
import os
import sys

# Add parent directory to path for backwards compatibility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main entry point for fluxmd command"""
    # Import main function directly from the fluxmd.py script
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fluxmd_script_path = os.path.join(parent_dir, "fluxmd.py")

    # Import the script module dynamically

    spec = importlib.util.spec_from_file_location("fluxmd_script", fluxmd_script_path)
    fluxmd_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fluxmd_script)

    # Call the main function
    fluxmd_script.main()


def main_uma():
    """Main entry point for fluxmd-uma command"""
    # Import main function directly from the fluxmd_uma.py script
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uma_script_path = os.path.join(parent_dir, "fluxmd_uma.py")

    # Import the script module dynamically

    spec = importlib.util.spec_from_file_location("fluxmd_uma_script", uma_script_path)
    uma_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uma_script)

    # Call the main function
    uma_script.main()


def main_protein_dna_uma():
    """Main entry point for fluxmd-protein-dna-uma command"""
    from fluxmd.core.protein_dna_uma import main as protein_dna_main

    protein_dna_main()


if __name__ == "__main__":
    main()
