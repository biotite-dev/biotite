# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["fetch"]

import requests
import os.path
import os
import glob


_standard_url = "https://files.rcsb.org/download/"
_mmtf_url = "https://mmtf.rcsb.org/v1.0/full/"

def fetch(pdb_ids, format, target_path, overwrite=False, verbose=False):
    """
    Download structure files from the RCSB PDB in various formats.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    pdb_ids : str or iterable object of str
        A single PDB ID or a list of PDB IDs of the structure(s)
        to be downloaded .
    format : str
        The format of the files to be downloaded.
        'pdb', 'pdbx', 'cif' or 'mmtf' are allowed.
    target_path : str
        The target directory of the downloaded files.
    overwrite : bool, optional
        If true, existing files will be overwritten. Otherwise the
        respective file will only be downloaded if the file does not
        exist yet in the specified target directory. (Default: False)
    verbose: bool, optional
        If true, the function will output the download progress.
        (Default: False)
    
    Returns
    -------
    files : str or list of str
        The file path(s) to the downloaded files.
        If a single string (a single ID) was given in `pdb_ids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
    
    Examples
    --------
    
    >>> files = fetch(["1l2y", "3o5r"], "cif", "path/to/files")
    >>> print(files)
    ['/path/to/files/1l2y.cif', '/path/to/files/3o5r.cif']
    """
    # If only a single PDB ID is present,
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
            print(f"Fetching file {i+1:d} / {len(pdb_ids):d} ({id})...",
                  end="\r")
        # Fetch file from database
        file_name = os.path.join(target_path, id + "." + format)
        file_names.append(file_name)
        if not os.path.isfile(file_name) or overwrite == True:
            if format == "pdb":
                r = requests.get(_standard_url + id + ".pdb")
                content = r.text
                _assert_valid_file(content, id)
                with open(file_name, "w+") as f:
                    f.write(content)
            elif format == "cif" or format == "pdbx":
                r = requests.get(_standard_url + id + ".cif")
                content = r.text
                _assert_valid_file(content, id)
                with open(file_name, "w+") as f:
                    f.write(content)
            elif format == "mmtf":
                r = requests.get(_mmtf_url + id)
                content = r.content
                _assert_valid_file(r.text, id)
                with open(file_name, "wb+") as f:
                    f.write(content)
            else:
                raise ValueError(f"Format '{format}' is not supported")
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return file_names[0]
    else:
        return file_names


def _assert_valid_file(response_text, pdb_id):
    """
    Checks whether the response is an actual structure file
    or the response a *404* error due to invalid PDB ID.
    """
    if "404 Not Found" in response_text:
        raise ValueError("PDB ID {:} is invalid".format(pdb_id))
