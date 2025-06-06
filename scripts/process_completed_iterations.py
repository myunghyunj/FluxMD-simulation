#!/usr/bin/env python3
"""
Process completed FluxMD iterations without bootstrap
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from Bio.PDB import PDBParser

def process_flux_iterations(output_dir, protein_file, protein_name="protein"):
    """Process flux data from completed iterations"""
    
    print("Processing completed FluxMD iterations...")
    
    # Find all flux CSV files
    flux_files = glob(os.path.join(output_dir, "iteration_*/flux_iteration_*_output_vectors.csv"))
    flux_files.sort()
    
    print(f"Found {len(flux_files)} iteration files")
    
    if not flux_files:
        print("Error: No flux files found!")
        return False
    
    # Collect all flux data
    all_flux_data = []
    
    for i, flux_file in enumerate(flux_files):
        if i % 10 == 0:
            print(f"Processing iteration {i+1}/{len(flux_files)}...")
        
        try:
            df = pd.read_csv(flux_file)
            if 'residue_id' in df.columns and 'flux' in df.columns:
                all_flux_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {flux_file}: {e}")
    
    if not all_flux_data:
        print("Error: No valid flux data found!")
        return False
    
    # Combine all data
    print("\nCombining flux data from all iterations...")
    combined_df = pd.concat(all_flux_data, ignore_index=True)
    
    # Save combined data
    all_flux_file = os.path.join(output_dir, "all_iterations_flux.csv")
    combined_df.to_csv(all_flux_file, index=False)
    print(f"Saved combined data to: {all_flux_file}")
    
    # Calculate statistics per residue
    print("\nCalculating per-residue statistics...")
    
    # Group by residue and calculate statistics
    grouped = combined_df.groupby('residue_id')
    
    stats_data = []
    for res_id, group in grouped:
        stats = {
            'residue_index': res_id,
            'residue_name': group['residue_name'].iloc[0] if 'residue_name' in group.columns else f"RES{res_id}",
            'average_flux': group['flux'].mean(),
            'std_flux': group['flux'].std(),
            'median_flux': group['flux'].median(),
            'min_flux': group['flux'].min(),
            'max_flux': group['flux'].max(),
            'n_observations': len(group),
            'is_aromatic': group['is_aromatic'].iloc[0] if 'is_aromatic' in group.columns else 0
        }
        
        # Calculate confidence intervals (95%)
        stats['ci_lower'] = stats['average_flux'] - 1.96 * stats['std_flux'] / np.sqrt(stats['n_observations'])
        stats['ci_upper'] = stats['average_flux'] + 1.96 * stats['std_flux'] / np.sqrt(stats['n_observations'])
        
        # Simple p-value estimate (t-test against zero)
        if stats['std_flux'] > 0 and stats['n_observations'] > 1:
            t_stat = stats['average_flux'] / (stats['std_flux'] / np.sqrt(stats['n_observations']))
            # Approximate p-value (two-tailed)
            from scipy import stats as scipy_stats
            stats['p_value'] = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), stats['n_observations'] - 1))
        else:
            stats['p_value'] = 1.0
        
        stats_data.append(stats)
    
    # Create DataFrame and sort by flux
    processed_df = pd.DataFrame(stats_data)
    processed_df = processed_df.sort_values('average_flux', ascending=False)
    
    # Save processed data
    output_file = os.path.join(output_dir, 'processed_flux_data.csv')
    processed_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Print top residues
    print("\nTop 20 residues by average flux:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Residue':<10} {'Name':<5} {'Avg Flux':<10} {'Std':<10} {'P-value':<10}")
    print("-" * 60)
    
    for i, row in processed_df.head(20).iterrows():
        print(f"{i+1:<5} {int(row['residue_index']):<10} {row['residue_name']:<5} "
              f"{row['average_flux']:<10.4f} {row['std_flux']:<10.4f} {row['p_value']:<10.4f}")
    
    # Create simple visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top 30 residues bar plot
        top_30 = processed_df.head(30)
        x = range(len(top_30))
        ax1.bar(x, top_30['average_flux'], yerr=top_30['std_flux'], capsize=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{int(r['residue_index'])}\n{r['residue_name']}" 
                             for _, r in top_30.iterrows()], rotation=45, ha='right')
        ax1.set_ylabel('Average Flux')
        ax1.set_title(f'{protein_name} - Top 30 Residues by Flux')
        ax1.grid(True, alpha=0.3)
        
        # Distribution of flux values
        ax2.hist(processed_df['average_flux'], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Average Flux')
        ax2.set_ylabel('Number of Residues')
        ax2.set_title('Distribution of Flux Values Across All Residues')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_file = os.path.join(output_dir, f'{protein_name}_flux_summary.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {viz_file}")
        
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {e}")
    
    # Create summary report
    report_file = os.path.join(output_dir, f'{protein_name}_flux_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"FluxMD Analysis Report - {protein_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total iterations processed: {len(flux_files)}\n")
        f.write(f"Total residues analyzed: {len(processed_df)}\n")
        f.write(f"Total flux observations: {len(combined_df)}\n\n")
        
        f.write("Top 20 Binding Sites (by average flux):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<5} {'Residue':<10} {'Name':<5} {'Avg Flux':<12} {'P-value':<10} {'Aromatic':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, row in processed_df.head(20).iterrows():
            aromatic = "Yes" if row.get('is_aromatic', 0) == 1 else "No"
            f.write(f"{i+1:<5} {int(row['residue_index']):<10} {row['residue_name']:<5} "
                   f"{row['average_flux']:<12.6f} {row['p_value']:<10.4f} {aromatic:<10}\n")
        
        f.write("\n\nStatistical Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean flux across all residues: {processed_df['average_flux'].mean():.6f}\n")
        f.write(f"Std dev of flux: {processed_df['average_flux'].std():.6f}\n")
        f.write(f"Max flux: {processed_df['average_flux'].max():.6f}\n")
        f.write(f"Min flux: {processed_df['average_flux'].min():.6f}\n")
        
        significant = processed_df[processed_df['p_value'] < 0.05]
        f.write(f"\nResidues with significant flux (p < 0.05): {len(significant)}\n")
        
        if 'is_aromatic' in processed_df.columns:
            aromatic_res = processed_df[processed_df['is_aromatic'] == 1]
            if len(aromatic_res) > 0:
                f.write(f"\nAromatic residues: {len(aromatic_res)}\n")
                f.write(f"Average flux of aromatic residues: {aromatic_res['average_flux'].mean():.6f}\n")
    
    print(f"\nReport saved to: {report_file}")
    
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python process_completed_iterations.py <output_dir> <protein_pdb> [protein_name]")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    protein_file = sys.argv[2]
    protein_name = sys.argv[3] if len(sys.argv) > 3 else "protein"
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} not found")
        sys.exit(1)
    
    if not os.path.exists(protein_file):
        print(f"Error: Protein file {protein_file} not found")
        sys.exit(1)
    
    success = process_flux_iterations(output_dir, protein_file, protein_name)
    
    if success:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()