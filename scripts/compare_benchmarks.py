#!/usr/bin/env python3
"""
Compare benchmark results and enforce regression thresholds.

This script is used by CI to ensure performance doesn't regress beyond
acceptable thresholds.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_benchmark_json(filepath: Path) -> Dict:
    """Load benchmark JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_benchmarks(data: Dict) -> Dict[str, float]:
    """Extract benchmark name -> mean time mapping."""
    benchmarks = {}
    
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        # Use mean time as the comparison metric
        mean_time = bench["stats"]["mean"]
        benchmarks[name] = mean_time
    
    return benchmarks


def compare_benchmarks(baseline: Dict[str, float], 
                      current: Dict[str, float], 
                      threshold: float) -> List[Tuple[str, float, float, float]]:
    """
    Compare baseline and current benchmarks.
    
    Returns list of (name, baseline_time, current_time, regression_percent) for failures.
    """
    failures = []
    
    for name, baseline_time in baseline.items():
        if name not in current:
            print(f"Warning: Benchmark '{name}' not found in current results")
            continue
        
        current_time = current[name]
        
        # Calculate regression percentage
        if baseline_time > 0:
            regression = (current_time - baseline_time) / baseline_time
        else:
            regression = 0
        
        # Check if regression exceeds threshold
        if regression > threshold:
            failures.append((name, baseline_time, current_time, regression))
    
    return failures


def print_comparison_report(baseline: Dict[str, float], 
                          current: Dict[str, float],
                          failures: List[Tuple[str, float, float, float]]):
    """Print detailed comparison report."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON REPORT")
    print("="*80)
    
    # Show all benchmarks
    all_names = sorted(set(baseline.keys()) | set(current.keys()))
    
    print(f"\n{'Benchmark':<40} {'Baseline (s)':>15} {'Current (s)':>15} {'Change':>10}")
    print("-" * 80)
    
    for name in all_names:
        baseline_time = baseline.get(name, 0)
        current_time = current.get(name, 0)
        
        if baseline_time > 0 and current_time > 0:
            change = (current_time - baseline_time) / baseline_time * 100
            change_str = f"{change:+.1f}%"
            
            # Color code based on change
            if change > 5:
                marker = "⚠️ "
            elif change < -5:
                marker = "✅ "
            else:
                marker = "  "
        else:
            change_str = "N/A"
            marker = "❓ "
        
        print(f"{marker}{name:<38} {baseline_time:>15.6f} {current_time:>15.6f} {change_str:>10}")
    
    # Show failures
    if failures:
        print("\n" + "="*80)
        print("PERFORMANCE REGRESSIONS DETECTED")
        print("="*80)
        
        for name, baseline_time, current_time, regression in failures:
            print(f"\n❌ {name}")
            print(f"   Baseline: {baseline_time:.6f}s")
            print(f"   Current:  {current_time:.6f}s")
            print(f"   Regression: {regression*100:.1f}% (exceeds {args.threshold*100:.0f}% threshold)")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON file")
    parser.add_argument("current", type=Path, help="Current benchmark JSON file")
    parser.add_argument("--threshold", type=float, default=0.05,
                       help="Regression threshold (default: 0.05 = 5%%)")
    parser.add_argument("--output", type=Path, help="Output comparison JSON file")
    
    args = parser.parse_args()
    
    # Load benchmark data
    baseline_data = load_benchmark_json(args.baseline)
    current_data = load_benchmark_json(args.current)
    
    # Extract benchmark times
    baseline_benchmarks = extract_benchmarks(baseline_data)
    current_benchmarks = extract_benchmarks(current_data)
    
    # Compare benchmarks
    failures = compare_benchmarks(baseline_benchmarks, current_benchmarks, args.threshold)
    
    # Print report
    print_comparison_report(baseline_benchmarks, current_benchmarks, failures)
    
    # Save comparison if requested
    if args.output:
        comparison = {
            "baseline_file": str(args.baseline),
            "current_file": str(args.current),
            "threshold": args.threshold,
            "baseline_benchmarks": baseline_benchmarks,
            "current_benchmarks": current_benchmarks,
            "failures": [
                {
                    "name": name,
                    "baseline_time": baseline_time,
                    "current_time": current_time,
                    "regression_percent": regression * 100
                }
                for name, baseline_time, current_time, regression in failures
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison saved to: {args.output}")
    
    # Exit with error if regressions detected
    if failures:
        print(f"\n❌ {len(failures)} benchmark(s) regressed beyond {args.threshold*100:.0f}% threshold")
        sys.exit(1)
    else:
        print(f"\n✅ All benchmarks within {args.threshold*100:.0f}% threshold")
        sys.exit(0)


if __name__ == "__main__":
    main()