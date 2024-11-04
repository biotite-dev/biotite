import json
import re
import subprocess
import tempfile
from pathlib import Path
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx

PDB_IDS = ["1l2y", "1gya"]


def atoms_to_temporary_cif(atoms):
    file = pdbx.CIFFile()
    pdbx.set_structure(file, atoms)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".cif", mode="w")
    file.write(tmp_file)
    tmp_file.flush()
    return tmp_file


def tm_score_from_us_align(reference, subject):
    reference_file = atoms_to_temporary_cif(reference)
    subject_file = atoms_to_temporary_cif(subject)

    # Do not run superposition to be able to use the original structure in the test
    # -> "-se"
    completed_process = subprocess.run(
        ["USalign", "-se", subject_file.name, reference_file.name],
        check=True,
        capture_output=True,
        text=True,
    )
    tm_score_match = re.search(r"TM-score= ([\d|\.]*)", completed_process.stdout)
    return float(tm_score_match.group(1))


if __name__ == "__main__":
    tm_scores = {}
    for pdb_id in PDB_IDS:
        pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))
        atoms = pdbx.get_structure(pdbx_file)
        tm_scores[pdb_id] = [
            tm_score_from_us_align(atoms[0], subject) for subject in atoms
        ]
    with open(Path(__file__).parent / "tm_scores.json", "w") as file:
        json.dump(tm_scores, file, indent=4)
