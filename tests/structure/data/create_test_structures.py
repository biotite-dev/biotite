import argparse
import subprocess
from os.path import join
import logging
import sys
import biotite
from biotite.database import RequestError
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio


def create(pdb_id, directory, include_gro):
    # Create *.pdb", *.cif and *.bcif
    for file_format in ["pdb", "cif", "bcif"]:
        try:
            rcsb.fetch(pdb_id, file_format, directory, overwrite=True)
        except RequestError:
            # PDB entry is not provided in this format
            pass
    try:
        array = strucio.load_structure(join(directory, pdb_id+".pdb"))
    except biotite.InvalidFileError:
        # Structure probably contains multiple models with different
        # number of atoms
        # -> Cannot load AtomArrayStack
        # -> Skip writing GRO file
        return
    # Create *.gro files using GROMACS
    # Clean PDB file -> remove inscodes and altlocs
    if include_gro:
        cleaned_file_name = biotite.temp_file("pdb")
        strucio.save_structure(cleaned_file_name, array)
        # Run GROMACS for file conversion
        subprocess.run([
            "editconf",
            "-f", cleaned_file_name,
            "-o", join(directory, pdb_id+".gro")
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create structure files for unit tests "
                    "in all supported formats from PDB ID "
                    "(excluding GROMACS trajectory files)"
    )
    parser.add_argument(
        "--dir",  "-d", dest="directory", default=".",
        help="the Biotite project directory to put the test files into"
    )
    parser.add_argument(
        "--id", "-i", dest="id",
        help="the PDB ID"
    )
    parser.add_argument(
        "--file", "-f", dest="file",
        help="read mutliple PDB IDs from text file (line break separated IDs)"
    )
    parser.add_argument(
        "--gromacs", "-g", action="store_true", dest="include_gro",
        help="Create '*.gro' files using the Gromacs software"
    )
    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r") as file:
            pdb_ids = [pdb_id.strip().lower() for pdb_id
                       in file.read().split("\n") if len(pdb_id.strip()) != 0]
    elif args.id is not None:
        pdb_ids = [args.id.lower()]
    else:
        logging.error("Must specifiy PDB ID(s)")
        sys.exit()

    for i, pdb_id in enumerate(pdb_ids):
        print(f"{i:2d}/{len(pdb_ids):2d}: {pdb_id}", end="\r")
        try:
            create(pdb_id, args.directory, args.include_gro)
        except:
            print()
            raise