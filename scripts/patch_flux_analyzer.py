#!/usr/bin/env python3
"""
Patch for flux_analyzer to fix joblib serialization error
"""

import os
import sys
import numpy as np

# Add the FluxMD directory to Python path
fluxmd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, fluxmd_dir)

# Import the analyzer
from fluxmd.analysis.flux_analyzer import TrajectoryFluxAnalyzer

# Monkey-patch the bootstrap method to use sequential processing
def bootstrap_flux_analysis_sequential(self, all_flux_data, residue_indices, n_bootstrap=None):
    """Perform bootstrap analysis on flux data - sequential version"""
    if n_bootstrap is not None:
        self.n_bootstrap = n_bootstrap
    
    n_iterations = len(all_flux_data)
    n_residues = len(residue_indices)
    
    print(f"   Running {self.n_bootstrap} bootstrap iterations (sequential)...")
    
    # Sequential bootstrap (no joblib)
    bootstrap_results = []
    for i in range(self.n_bootstrap):
        if i % 10 == 0:
            print(f"     Progress: {i}/{self.n_bootstrap}")
        result = self._single_bootstrap(all_flux_data, n_iterations, n_residues)
        bootstrap_results.append(result)
    
    # Rest of the function remains the same
    residue_distributions = {res_id: [] for res_id in residue_indices}
    
    for bootstrap_flux in bootstrap_results:
        for i, res_id in enumerate(residue_indices):
            residue_distributions[res_id].append(bootstrap_flux[i])
    
    statistics = {}
    
    for i, res_id in enumerate(residue_indices):
        distribution = np.array(residue_distributions[res_id])
        
        mean_flux = np.mean(distribution)
        std_flux = np.std(distribution)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(distribution, (alpha/2) * 100)
        ci_upper = np.percentile(distribution, (1 - alpha/2) * 100)
        
        if mean_flux > 0:
            p_value = 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0))
        else:
            p_value = 1.0
        
        effect_size = mean_flux / (std_flux + 1e-10)
        
        statistics[res_id] = {
            'mean': mean_flux,
            'std': std_flux,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'effect_size': effect_size,
            'distribution': distribution
        }
    
    return statistics

# Apply the monkey patch
TrajectoryFluxAnalyzer.bootstrap_validator.bootstrap_flux_analysis = bootstrap_flux_analysis_sequential

def main():
    """Run the patched analyzer"""
    if len(sys.argv) != 3:
        print("Usage: python patch_flux_analyzer.py <results_dir> <protein_pdb>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    protein_pdb = sys.argv[2]
    
    # Run the analyzer with patched bootstrap
    analyzer = TrajectoryFluxAnalyzer()
    
    print("Running flux analysis with patched bootstrap...")
    flux_data = analyzer.process_trajectory_iterations(results_dir, protein_pdb)
    
    if flux_data:
        # Save results
        analyzer.save_processed_data(flux_data, results_dir)
        print(f"\nAnalysis complete! Results saved to {results_dir}")

if __name__ == "__main__":
    main()