import os
import time
import json
import urllib.request
import numpy as np
from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer
from fluxmd.utils.dna_to_pdb import DNABuilder

CAP_SEQ = "AAATGTGATCTAGATCAC"
PDB_URL = "https://files.rcsb.org/download/1LMB.pdb"
PDB_PATH = os.path.join("benchmarks", "1LMB.pdb")


def ensure_pdb():
    os.makedirs("benchmarks", exist_ok=True)
    if not os.path.exists(PDB_PATH):
        urllib.request.urlretrieve(PDB_URL, PDB_PATH)


def run_benchmark():
    ensure_pdb()
    builder = DNABuilder()
    builder.build_dna(CAP_SEQ)
    dna_path = os.path.join("benchmarks", "cap_dna.pdb")
    builder.write_pdb(dna_path)

    analyzer = ProteinLigandFluxAnalyzer(PDB_PATH, dna_path, ".")
    analyzer.dna_builder = builder
    geometry = {
        "shape_type": "linear",
        "dimensions": (len(CAP_SEQ) * builder.RISE, builder.RADIUS * 2, builder.RADIUS * 2),
    }
    start = time.perf_counter()
    pts, _ = analyzer.generate_uniform_linear_trajectory(geometry, 50000, 5.0)
    runtime = time.perf_counter() - start
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    hist, _ = np.histogramdd((theta, z), bins=(20, 20))
    cv = float(np.std(hist) / np.mean(hist))
    print(json.dumps({"runtime": runtime, "cv": cv}))


if __name__ == "__main__":
    run_benchmark()
