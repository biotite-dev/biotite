import json
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
import mdtraj
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

PDB_IDS = ["1l2y", "1gya", "1igy"]
STRUCTURE_DIR = Path(__file__).parents[1]
OUTPUT_FILE = Path(__file__).parent / "hbond.json"


def compute_mdtraj_hbonds(bcif_path, use_all_models):
    pdbx_file = pdbx.BinaryCIFFile.read(bcif_path)
    model = None if use_all_models else 1
    atoms = pdbx.get_structure(pdbx_file, model=model)
    # Only consider amino acids for consistency
    # with bonded hydrogen detection in MDTraj
    atoms = atoms[..., struc.filter_amino_acids(atoms)]

    temp = NamedTemporaryFile("w+", suffix=".pdb")
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atoms)
    pdb_file.write(temp.name)

    # Compute hbonds with MDTraj
    # Ignore warning about dummy unit cell vector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = mdtraj.load(temp.name)
    temp.close()
    return mdtraj.baker_hubbard(traj, freq=0, periodic=False).tolist()


if __name__ == "__main__":
    data = {}
    for pdb_id in PDB_IDS:
        bcif_path = STRUCTURE_DIR / f"{pdb_id}.bcif"
        data_for_pdb_id = {}
        for use_all_models in [False, True]:
            key = "single_model" if not use_all_models else "all_models"
            data_for_pdb_id[key] = compute_mdtraj_hbonds(bcif_path, use_all_models)
        data[pdb_id] = data_for_pdb_id

    with open(OUTPUT_FILE, "w") as file:
        json.dump(data, file, indent=4)
