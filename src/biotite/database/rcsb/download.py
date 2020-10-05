# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.rcsb"
__author__ = "Patrick Kunzmann"
__all__ = ["fetch"]

import requests
from os.path import isdir, isfile, join, getsize
import os
import glob
import io
from ..error import RequestError


_standard_url = "https://files.rcsb.org/download/"
_mmtf_url = "https://mmtf.rcsb.org/v1.0/full/"
_fasta_url = "https://www.rcsb.org/fasta/entry/"

_binary_formats = ["mmtf"]


def fetch(pdb_ids, format, target_path=None, overwrite=False, verbose=False):
    """
    Download structure files (or sequence files) from the RCSB PDB in
    various formats.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    pdb_ids : str or iterable object of str
        A single PDB ID or a list of PDB IDs of the structure(s)
        to be downloaded .
    format : {'pdb', 'pdbx', 'cif', 'mmcif', 'mmtf', 'fasta'}
        The format of the files to be downloaded.
        ``'pdbx'``, ``'cif'`` and ``'mmcif'`` are synonyms for
        the same format.
    target_path : str, optional
        The target directory of the downloaded files.
        By default, the file content is stored in a file-like object
        (`StringIO` or `BytesIO`, respectively).
    overwrite : bool, optional
        If true, existing files will be overwritten. Otherwise the
        respective file will only be downloaded if the file does not
        exist yet in the specified target directory or if the file is
        empty. (Default: False)
    verbose: bool, optional
        If true, the function will output the download progress.
        (Default: False)
    
    Returns
    -------
    files : str or StringIO or BytesIO or list of (str or StringIO or BytesIO)
        The file path(s) to the downloaded files.
        If a single string (a single ID) was given in `pdb_ids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
        If no `target_path` was given, the file contents are stored in
        either `StringIO` or `BytesIO` objects.
    
    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.
    
    Examples
    --------
    
    >>> import os.path
    >>> file = fetch("1l2y", "cif", path_to_directory)
    >>> print(os.path.basename(file))
    1l2y.cif
    >>> files = fetch(["1l2y", "3o5r"], "cif", path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['1l2y.cif', '3o5r.cif']
    """
    # If only a single PDB ID is present,
    # put it into a single element list
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not os.path.isdir(target_path):
        os.makedirs(target_path)
    
    files = []
    for i, id in enumerate(pdb_ids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i+1:d} / {len(pdb_ids):d} ({id})...",
                  end="\r")
        
        # Fetch file from database
        if target_path is not None:
            file = join(target_path, id + "." + format)
        else:
            # 'file = None' -> store content in a file-like object
            file = None
        
        if file is None \
           or not isfile(file) \
           or getsize(file) == 0 \
           or overwrite:
                if format == "pdb":
                    r = requests.get(_standard_url + id + ".pdb")
                    content = r.text
                    _assert_valid_file(content, id)
                elif format in ["cif", "mmcif", "pdbx"]:
                    r = requests.get(_standard_url + id + ".cif")
                    content = r.text
                    _assert_valid_file(content, id)
                elif format == "mmtf":
                    r = requests.get(_mmtf_url + id)
                    content = r.content
                    _assert_valid_file(r.text, id)
                elif format == "fasta":
                    r = requests.get(_fasta_url + id)
                    content = r.text
                    _assert_valid_file(content, id)
                else:
                    raise ValueError(f"Format '{format}' is not supported")
                
                if file is None:
                    if format in _binary_formats:
                        file = io.BytesIO(content)
                    else:
                        file = io.StringIO(content)
                else:
                    mode = "wb+" if format in _binary_formats else "w+"
                    with open(file, mode) as f:
                        f.write(content)
        
        files.append(file)
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return files[0]
    else:
        return files


def _assert_valid_file(response_text, pdb_id):
    """
    Checks whether the response is an actual structure file
    or the response a *404* error due to invalid PDB ID.
    """
    # Structure file and FASTA file retrieval
    # have different error messages
    if any(err_msg in response_text for err_msg in [
        "404 Not Found",
        "<title>RCSB Protein Data Bank Error Page</title>",
        "No fasta files were found."
    ]):
        raise RequestError("PDB ID {:} is invalid".format(pdb_id))
