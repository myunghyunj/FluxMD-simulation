# FluxMD Examples

This directory contains example scripts showing how to use FluxMD.

## basic_usage.py

A simple example demonstrating the basic FluxMD workflow. To run this example, you'll need to:

1. Provide your own protein and ligand PDB files
2. Update the file paths in the script
3. Run: `python examples/basic_usage.py`

The example uses reduced parameters for faster testing. For production use, consider:
- `n_steps`: 200+ for thorough sampling
- `n_iterations`: 100+ for statistical significance  
- `n_rotations`: 36 for complete rotation sampling

For interactive analysis with more options, use the main CLI:
```bash
fluxmd
```