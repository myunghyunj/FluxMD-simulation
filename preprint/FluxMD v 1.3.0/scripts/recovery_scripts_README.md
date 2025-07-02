# FluxMD Recovery Scripts

This directory contains utility scripts for recovering and continuing FluxMD analysis when the main workflow is interrupted.

## Scripts Overview

### 1. `process_completed_iterations.py`
**Purpose**: Process completed iteration files when FluxMD analysis is interrupted before the final flux analysis step.

**When to use**: 
- When you have completed iteration directories (e.g., `iteration_1/`, `iteration_2/`, etc.) but no final analysis
- When you see files like `flux_iteration_*_output_vectors.csv` in each iteration directory
- When the main FluxMD process was interrupted during or after trajectory generation

**Usage**:
```bash
python process_completed_iterations.py <output_dir> <protein_pdb> [protein_name]
```

**Example**:
```bash
python process_completed_iterations.py /path/to/flux_analysis /path/to/protein.pdb T4_L99A
```

**What it does**:
1. Finds all `flux_iteration_*_output_vectors.csv` files
2. Combines flux data from all iterations into `all_iterations_flux.csv`
3. Calculates per-residue statistics (mean, std, confidence intervals)
4. Generates `processed_flux_data.csv` with ranked binding sites
5. Creates visualization plots and summary reports

### 2. `continue_analysis.py`
**Purpose**: Continue analysis from a previously created `all_iterations_flux.csv` file.

**When to use**:
- When `all_iterations_flux.csv` exists but final processing wasn't completed
- When you need to regenerate visualizations or reports
- After running `process_completed_iterations.py`

**Usage**:
```bash
python continue_analysis.py <output_dir> <protein_pdb> [protein_name]
```

**Example**:
```bash
python continue_analysis.py /path/to/flux_analysis /path/to/protein.pdb T4_L99A
```

**What it does**:
1. Reads existing `all_iterations_flux.csv`
2. Groups data by residue and calculates statistics
3. Generates/updates `processed_flux_data.csv`
4. Creates visualization heatmaps
5. Generates summary reports

## Common Scenarios

### Scenario 1: FluxMD crashed during bootstrap analysis
If you see an error like:
```
joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated
```

**Solution**:
```bash
# First, process all completed iterations
python scripts/process_completed_iterations.py flux_analysis/ protein.pdb MyProtein

# If needed, regenerate visualizations
python scripts/continue_analysis.py flux_analysis/ protein.pdb MyProtein
```

### Scenario 2: Analysis completed but need different visualizations
```bash
# Use continue_analysis.py to regenerate from existing data
python scripts/continue_analysis.py flux_analysis/ protein.pdb MyProtein
```

### Scenario 3: Unsure which script to use
Check what files exist in your output directory:
- If you have `iteration_*/` directories → use `process_completed_iterations.py`
- If you have `all_iterations_flux.csv` → use `continue_analysis.py`

## Output Files

Both scripts generate:
- `processed_flux_data.csv` - Ranked residues with statistics
- `*_flux_summary.png` - Visualization plots
- `*_flux_report.txt` - Detailed text report

## Notes

- These scripts skip the bootstrap validation step to avoid memory/parallelization issues
- P-values are approximated using simple t-tests instead of bootstrap
- Results are still valid for identifying top binding sites
- Original FluxMD with bootstrap provides more rigorous statistical validation

## Error Handling

If you encounter column name errors like `KeyError: 'residue_index'`, the scripts automatically handle different FluxMD output formats:
- Old format: `residue_id`, `flux`
- New format: `protein_residue_id`, `combined_magnitude`

The scripts will automatically detect and convert between formats.