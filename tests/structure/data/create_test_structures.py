import argparse
import shutil
import subprocess
from pathlib import Path
import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
from biotite.database import RequestError


def create(pdb_id, directory, include_gro):
    # Create *.pdb", *.cif and *.bcif
    for file_format in ["pdb", "cif", "bcif"]:
        try:
            rcsb.fetch(pdb_id, file_format, directory, overwrite=True)
        except RequestError:
            # PDB entry is not provided in this format
            pass
    # Create *.gro files using GROMACS
    if include_gro:
        try:
            pdbx_file = pdbx.BinaryCIFFile.read(directory / pdb_id + ".bcif")
            atoms = pdbx.get_structure(pdbx_file)
        except biotite.InvalidFileError:
            # Structure probably contains multiple models with different
            # number of atoms
            # -> Cannot load AtomArrayStack
            # -> Skip writing GRO file
            return
        # Clean PDB file -> remove inscodes and altlocs
        cleaned_file_name = biotite.temp_file("pdb")
        strucio.save_structure(cleaned_file_name, atoms)
        # Run GROMACS for file conversion
        subprocess.run(
            [
                "gmxeditconf",
                "-f",
                cleaned_file_name,
                "-o",
                str(directory / pdb_id + ".gro"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create structure files for unit tests "
        "in all supported formats from PDB ID "
        "(excluding GROMACS trajectory files)"
    )
    parser.add_argument(
        "--dir",
        "-d",
        dest="directory",
        default=".",
        help="the Biotite project directory to put the test files into",
    )
    parser.add_argument("--id", "-i", dest="id", help="the PDB ID")
    parser.add_argument(
        "--file",
        "-f",
        dest="file",
        help="read mutliple PDB IDs from text file (line break separated IDs)",
    )
    parser.add_argument(
        "--gromacs",
        "-g",
        action="store_true",
        dest="include_gro",
        help="Create '*.gro' files using the Gromacs software",
    )
    args = parser.parse_args()

if __name__ == "__main__":
    data_dir = Path(__file__).parent
    include_gro = shutil.which("gmx") is not None
    with open(data_dir / "ids.txt", "r") as file:
        pdb_ids = [
            pdb_id.strip().lower()
            for pdb_id in file.read().split("\n")
            if len(pdb_id.strip()) != 0
        ]

    for i, pdb_id in enumerate(pdb_ids):
        print(f"{i:2d}/{len(pdb_ids):2d}: {pdb_id}", end="\r")
        create(pdb_id, data_dir, args.include_gro)
