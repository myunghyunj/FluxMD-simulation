import json
import subprocess
import sys
from pathlib import Path
import warnings


def test_perf_regression():
    baseline_file = Path('benchmarks/bench_baseline.json')
    current_file = Path('benchmarks/bench_current.json')
    if not baseline_file.exists() or not current_file.exists():
        reason = 'missing baseline or current file'
        import pytest
        pytest.skip(reason)

    baseline = json.loads(baseline_file.read_text())
    current = json.loads(current_file.read_text())

    delta = (current['mean'] - baseline['mean']) / baseline['mean']
    assert delta <= 0.05, f'Runtime regression {delta:.2%} exceeds 5%'

    try:
        mem_out = subprocess.check_output(
            [sys.executable, 'scripts/bench_runtime.py', '--profile-mem'],
            text=True
        )
        if mem_out.strip():
            mem_json = json.loads(mem_out)
            if (
                'max_mem' in baseline
                and mem_json.get('max_mem', 0) > 2 * baseline['max_mem']
            ):
                warnings.warn('Memory usage doubled', RuntimeWarning)
    except subprocess.CalledProcessError:
        pass
