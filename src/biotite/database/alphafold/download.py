# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.alphafold"
__author__ = "Jonathan Funk"
__all__ = ["fetch"]

import requests
from os.path import isdir, isfile, join, getsize
import io

_standard_url = 'https://alphafold.ebi.ac.uk/files/AF-'


def fetch(uniprot_ids, format, target_path=None, overwrite=False, verbose=False):
    """
    Downloads file from EBI AlphaFold database in various formats.

    This function requires internet connection.

    Parameters
    ----------
    uniprot_id: str
        A single UniProt ID of a protein
    target_path: str
        Path for saving the file to. If path does not exist it will be created.
    format: {'pdb', 'cif', 'mmcif', 'pdbx'}
        Format of the file to be downloaded.
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
    files : str or StringIO or list of (str or StringIO)
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
    >>> file = fetch("P0DPA9", "cif", path_to_directory)
    >>> print(os.path.basename(file))
    P0DPA9-F1-model_v3..cif
    >>> files = fetch(["P0DPA8", "P0DPA9"], "pdb", path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['P0DPA8-F1-model_v3..pdb', 'P0DPA9-F1-model_v3..pdb']
    """
    # If only a single PDB ID is present,
    # put it into a single element list
    if isinstance(uniprot_ids, str):
        uniprot_ids = [uniprot_ids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not os.path.isdir(target_path):
        os.makedirs(target_path)

    files = []
    for i, id in enumerate(uniprot_ids):

        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(uniprot_ids):d} ({id})...",
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
                r = requests.get(_standard_url + id + '-F1-model_v3' + ".pdb")
                content = r.text
                _assert_valid_file(content, id)
            elif format in ["cif", "mmcif", "pdbx"]:
                r = requests.get(_standard_url + id + '-F1-model_v3' + ".cif")
                content = r.text
                _assert_valid_file(content, id)
            else:
                raise ValueError(f"Format '{format}' is not supported")

            if file is None:
                file = io.StringIO(content)

            else:
                with open(file, "w+") as f:
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
        "No fasta files were found.",
        "No valid PDB IDs were submitted.",
    ]):
        raise RequestError("PDB ID {:} is invalid".format(pdb_id))