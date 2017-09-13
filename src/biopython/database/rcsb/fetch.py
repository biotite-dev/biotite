# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import requests
import os.path
import os
import glob

__all__ = ["fetch"]


_standard_url = "https://files.rcsb.org/download/"
_mmtf_url = "https://mmtf.rcsb.org/v1.0/full/"

def fetch(pdb_ids, format, target_path, overwrite=False, verbose=False):
    # If only a single PDB id is present,
    # put it into a single element list
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    file_names = []
    for i, id in enumerate(pdb_ids):
        # Verbose output
        if verbose:
            print("Fetching file {:d} / {:d} ({:})..."
                  .format(i+1, len(pdb_ids), id), end="\r")
        # Fetch file from database
        file_name = os.path.join(target_path, id + "." + format)
        file_names.append(file_name)
        if not os.path.isfile(file_name) or overwrite == True:
            if format == "pdb":
                r = requests.get(_standard_url + id + ".pdb")
                content = r.text
                with open(file_name, "w+") as f:
                    f.write(content)
            elif format == "cif":
                r = requests.get(_standard_url + id + ".cif")
                content = r.text
                with open(file_name, "w+") as f:
                    f.write(content)
            elif format == "mmtf":
                r = requests.get(_mmtf_url + id)
                content = r.content
                with open(file_name, "wb+") as f:
                    f.write(content)
            else:
                raise ValueError("Format '{:}' is not supported"
                                 .format(format))
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return file_names[0]
    else:
        return file_names