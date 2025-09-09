import json
from pathlib import Path
import mdtraj
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

TEST_FILE = Path(__file__).parents[1] / "1l2y.cif"
OUTPUT_FILE = Path(__file__).parent / "dihedrals.json"


if __name__ == "__main__":
    pdbx_file = pdbx.CIFFile.read(TEST_FILE)
    atoms = pdbx.get_structure(pdbx_file, model=1)

    traj = mdtraj.load(TEST_FILE.as_posix())
    dihedral_dict = {}
    for angle_name, angle_func in [
        ("phi", mdtraj.compute_phi),
        ("psi", mdtraj.compute_psi),
        ("omega", mdtraj.compute_omega),
        ("chi1", mdtraj.compute_chi1),
        ("chi2", mdtraj.compute_chi2),
        ("chi3", mdtraj.compute_chi3),
        ("chi4", mdtraj.compute_chi4),
    ]:
        indices, dihedrals = angle_func(traj)
        # MDTraj only outputs the dihedral angles only for residues,
        # where they are applicable
        # -> Map the angles to the correct residues using the returned indices
        # amd keep the inapplicable residues as NaN
        mapped_dihedrals = np.full((struc.get_residue_count(atoms)), np.nan)
        # Use the second atom of each angle to infer the residue,
        # to handle the edge case of 'phi'
        # where the first atom stems from the previous residue
        residue_indices = struc.get_residue_positions(atoms, indices[:, 1])
        # For testing purposes checking against the first model in sufficient
        mapped_dihedrals[residue_indices] = dihedrals[0]
        dihedral_dict[angle_name] = mapped_dihedrals.tolist()

    with open(OUTPUT_FILE, "w") as file:
        json.dump(dihedral_dict, file, indent=4)
