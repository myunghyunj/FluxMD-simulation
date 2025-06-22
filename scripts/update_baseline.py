import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_bench(repeat: int) -> float:
    out = subprocess.check_output([
        sys.executable,
        'scripts/bench_runtime.py',
        '--repeat',
        str(repeat),
    ], text=True)
    data = json.loads(out)
    return float(data['mean'])


def main():
    parser = argparse.ArgumentParser(description='Update performance baselines')
    parser.add_argument('--force', action='store_true', help='overwrite baseline')
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()

    current = run_bench(args.repeat)
    current_file = Path('benchmarks/bench_current.json')
    baseline_file = Path('benchmarks/bench_baseline.json')

    current_file.write_text(json.dumps({'mean': current}))

    if args.force:
        baseline_file.write_text(json.dumps({'mean': current}))
        print('Baseline overwritten')
    else:
        if not baseline_file.exists():
            baseline_file.write_text(json.dumps({'mean': current}))
            print('Baseline created')
        else:
            print('Baseline left unchanged')


if __name__ == '__main__':
    main()
