#!/usr/bin/env python3
"""Optional groove width validation using 3DNA if available."""

import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

from fluxmd.utils.dna_to_pdb import DNABuilder


def main() -> None:
    if shutil.which("x3dna-dssr") is None:
        print("x3dna-dssr not found, skipping")
        return

    builder = DNABuilder()
    seq = "ATCG" * 3
    builder.build_dna(seq)
    tmp = Path("tmp_dna.pdb")
    builder.write_pdb(tmp)
    res = subprocess.run(["x3dna-dssr", str(tmp)], capture_output=True, text=True)
    tmp.unlink()
    widths = []
    for line in res.stdout.splitlines():
        if "minor-groove" in line and "width" in line:
            try:
                widths.append(float(line.split()[-2]))
            except ValueError:
                pass
    if widths:
        print(f"Mean minor groove width: {np.mean(widths):.2f} Å")
    else:
        print("No groove data found")


if __name__ == "__main__":
    main()
