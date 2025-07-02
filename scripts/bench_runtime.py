import time
import json
import argparse
import sys
import numpy as np
from fluxmd.utils.dna_to_pdb import DNABuilder
from fluxmd.core.cylindrical_sampler import FastCylindricalSampler


np.random.seed(0)


def bench(n_frames: int = 100000) -> float:
    builder = DNABuilder()
    seq = "ATCG" * 3  # 12-bp duplex
    builder.build_dna(seq)
    length = builder.RISE * len(seq)
    sampler = FastCylindricalSampler(length, builder.RADIUS + 5.0)
    start = time.perf_counter()
    sampler.sample(n_frames)
    return time.perf_counter() - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile-mem", action="store_true")
    args = parser.parse_args()
    if args.profile_mem:
        try:
            from memory_profiler import memory_usage
        except ImportError:
            print("memory_profiler not installed; skipping", file=sys.stderr)
            sys.exit(0)
        peak = max(memory_usage((bench, (), {})))
        print(json.dumps({"max_mem": peak}))
    else:
        times = [bench() for _ in range(args.repeat)]
        print(json.dumps({"mean": sum(times) / len(times), "runs": times}))
