"""Command-line interface for FluxMD."""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main entry point for standard FluxMD."""
    import fluxmd
    fluxmd.main()

def main_uma():
    """Main entry point for UMA-optimized FluxMD."""
    import fluxmd_uma
    fluxmd_uma.main()

if __name__ == "__main__":
    main()