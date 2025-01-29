import json
import warnings
from pathlib import Path
import mdtraj
from biotite.structure.info.radii import _SINGLE_ELEMENT_VDW_RADII as SINGLE_RADII

PDB_IDS = ["1l2y", "1gya"]
STRUCTURE_DIR = Path(__file__).parents[1]
OUTPUT_FILE = Path(__file__).parent / "sasa.json"


def compute_mdtraj_sasa(pdb_path):
    # Use the same atom radii as Biotite
    radii = {
        element.capitalize(): radius / 10 for element, radius in SINGLE_RADII.items()
    }
    # Ignore warning about dummy unit cell vector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = mdtraj.load(pdb_path)
    # Conversion from nm^2 to A^2
    return mdtraj.shrake_rupley(traj, change_radii=radii, n_sphere_points=5000)[0] * 100


if __name__ == "__main__":
    data = {}
    for pdb_id in PDB_IDS:
        pdb_path = STRUCTURE_DIR / f"{pdb_id}.pdb"
        data[pdb_id] = compute_mdtraj_sasa(pdb_path).tolist()

    with open(OUTPUT_FILE, "w") as file:
        json.dump(data, file, indent=4)
