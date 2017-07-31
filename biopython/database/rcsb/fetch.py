# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import requests
import os.path
import os
import glob

_dowload_url = "https://files.rcsb.org/download/"

def fetch(pdb_ids, format, target_path, overwrite=False, quiet=True):
    # If only a single PDB id is present,
    # put it into a single element list
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    file_names = []
    for i, id in enumerate(pdb_ids):
        if not quiet:
            print("Fetching file " + str(i+1) + "/" + str(len(ids))
                  + " (" + id + ")...")
        file_name = os.path.join(target_path, id + "." + format)
        file_names.append(file_name)
        if not os.path.isfile(file_name) or overwrite == True:
            r = requests.get(_dowload_url + id + "." + format)
            content = r.text
            with open(file_name, "w+") as f:
                    f.write(content)
    return file_names