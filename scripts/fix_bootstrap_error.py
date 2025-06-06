#!/usr/bin/env python3
"""
Temporary fix for bootstrap error - processes flux data without bootstrap validation
"""

import os
import sys
import numpy as np
import pandas as pd

def process_flux_without_bootstrap(output_dir):
    """Process flux data without bootstrap validation"""
    
    # Read all iteration flux data
    all_flux_file = os.path.join(output_dir, 'all_iterations_flux.csv')
    if not os.path.exists(all_flux_file):
        print(f"Error: {all_flux_file} not found")
        return
    
    # Load data
    df = pd.read_csv(all_flux_file)
    
    # Calculate average flux per residue
    avg_flux = df.groupby('residue_index')['flux'].mean()
    std_flux = df.groupby('residue_index')['flux'].std()
    
    # Create processed data without bootstrap stats
    processed_data = pd.DataFrame({
        'residue_index': avg_flux.index,
        'average_flux': avg_flux.values,
        'std_flux': std_flux.values,
        'p_value': 0.05,  # Placeholder
        'ci_lower': avg_flux.values - 1.96 * std_flux.values,
        'ci_upper': avg_flux.values + 1.96 * std_flux.values
    })
    
    # Sort by flux
    processed_data = processed_data.sort_values('average_flux', ascending=False)
    
    # Save processed data
    output_file = os.path.join(output_dir, 'processed_flux_data_no_bootstrap.csv')
    processed_data.to_csv(output_file, index=False)
    
    print(f"Processed flux data saved to: {output_file}")
    print(f"\nTop 10 residues by flux:")
    print(processed_data.head(10))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_bootstrap_error.py <output_directory>")
        sys.exit(1)
    
    process_flux_without_bootstrap(sys.argv[1])