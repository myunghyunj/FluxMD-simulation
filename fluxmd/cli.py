"""
Command-line interface for FluxMD
"""

import importlib.util
import os
import sys
import argparse

# Add parent directory to path for backwards compatibility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main entry point for fluxmd command"""
    parser = argparse.ArgumentParser(description="FluxMD CLI")
    parser.add_argument("--protein", required=False)
    parser.add_argument("--ligand", required=False)
    parser.add_argument("--params", required=False)
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()

    required = [args.protein, args.ligand, args.params]
    for p in required:
        if p and not os.path.exists(p):
            print(f"Error: Required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Build argv for underlying script
    sys.argv = [sys.argv[0]] + unknown
    if args.params:
        sys.argv.extend(["--config", args.params])
    if args.dry_run:
        sys.argv.append("--dry-run")

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
