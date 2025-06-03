PyMOL Multiple Protein Flux Visualization - Complete Example
===========================================================

1. Start PyMOL and navigate to FluxMD directory:
   PyMOL> cd /Users/myunghyun/Documents/GitHub/FluxMD

2. Run the script:
   PyMOL> run pymol_multiflux.py

3. Load your protein structures:
   PyMOL> load /path/to/GPX4-wt.pdb
   PyMOL> load /path/to/GPX4-single.pdb
   PyMOL> load /path/to/GPX4-double.pdb

4. Run multiflux command:
   PyMOL> multiflux

5. You will see prompts like this:
   [multiflux] Found 3 protein object(s): GPX4-wt, GPX4-single, GPX4-double
   
   [multiflux] Enter CSV file for each protein (or press Enter to skip):
     GPX4-wt: /path/to/your/wt_flux_analysis/processed_flux_data.csv
     GPX4-single: /path/to/your/single_flux_analysis/processed_flux_data.csv
     GPX4-double: /path/to/your/double_flux_analysis/processed_flux_data.csv

6. After entering all CSV paths, the proteins will be colored and arranged in grid view.

Alternative - Specify CSV files directly:
=========================================

PyMOL> multiflux GPX4-wt=/path/to/wt.csv,GPX4-single=/path/to/single.csv,GPX4-double=/path/to/double.csv

Tips:
=====
- CSV files are typically in the flux_analysis/ directory after running FluxMD
- The CSV file should be named "processed_flux_data.csv"
- You can use relative or absolute paths
- Press Enter to skip a protein if you don't have its CSV file