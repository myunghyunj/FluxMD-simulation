"""
Command-line interface for FluxMD
"""

import sys
import os

# Add parent directory to path for backwards compatibility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main entry point for fluxmd command"""
    # Import from parent directory
    import fluxmd as fluxmd_module

    fluxmd_module.main()


def main_uma():
    """Main entry point for fluxmd-uma command"""
    # Import from parent directory
    import fluxmd_uma as uma_module

    uma_module.main()


def main_protein_dna_uma():
    """Main entry point for fluxmd-protein-dna-uma command"""
    from fluxmd.core.protein_dna_uma import main as protein_dna_main

    protein_dna_main()


if __name__ == "__main__":
    main()
