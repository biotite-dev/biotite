import itertools
import json
from pathlib import Path
import mdtraj
import numpy as np

TEST_FILE = Path(__file__).parents[1] / "waterbox.gro"
OUTPUT_FILE = Path(__file__).parent / "rdf.json"
INTERVAL = [0, 10]
N_BINS = 100


if __name__ == "__main__":
    traj = mdtraj.load(TEST_FILE)
    ow = [a.index for a in traj.topology.atoms if a.name == "O"]
    pairs = itertools.product([ow[0]], ow)
    mdtraj_bins, mdtraj_g_r = mdtraj.compute_rdf(
        # Note the conversion from Angstrom to nm
        traj,
        list(pairs),
        r_range=np.array(INTERVAL) / 10,
        n_bins=N_BINS,
        periodic=False,
    )

    with open(OUTPUT_FILE, "w") as file:
        json.dump(
            {"bins": (mdtraj_bins * 10).tolist(), "g_r": mdtraj_g_r.tolist()},
            file,
            indent=4,
        )
