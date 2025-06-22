import numpy as np
import time

from fluxmd.utils.dna_to_pdb import DNABuilder
from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer

np.random.seed(0)


def test_fluxmd_end2end():
    builder = DNABuilder()
    seq = "ATCG" * 3
    builder.build_dna(seq)
    coords = np.array([a['coord'] for a in builder.atoms])

    tg = ProteinLigandFluxAnalyzer("dummy.pdb", "dummy.pdb", "./tmp")
    tg.dna_builder = builder
    geom = tg.analyze_molecular_geometry(coords)

    start = time.perf_counter()
    pts, _ = tg.generate_uniform_linear_trajectory(geom, 200000, 5.0)
    runtime = time.perf_counter() - start

    theta = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    hist, _ = np.histogramdd((theta, z), bins=(20, 20))
    cv = np.std(hist) / np.mean(hist)

    assert cv < 0.15
    assert runtime < 3.0
