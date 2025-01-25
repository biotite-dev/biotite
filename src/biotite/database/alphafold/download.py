# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.alphafold"
__author__ = "Alex Carlin"
__all__ = ["fetch"]

import io
import os
from os.path import getsize, isdir, isfile, join
import requests
from biotite.database.alphafold.check import assert_valid_response

_fetch_url = "https://alphafold.com/api/prediction"


def fetch(ids, target_path=None, format="pdb", overwrite=False, verbose=False):
    """
    Download predicted protein structures from the AlphaFold DB.

    This function requires an internet connection.

    Parameters
    ----------
    ids : str or iterable object of str
        A single ID or a list of IDs of the file(s)
        to be downloaded.
    target_path : str, optional
        The target directory of the downloaded files.
        By default, the file content is stored in a file-like object
        (`StringIO` or `BytesIO`, respectively).
    format : {"pdb", "cif", "bcif"}
        The format of the files to be downloaded.
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
        If a single string (a single ID) was given in `ids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
        If no `target_path` was given, the file contents are stored in
        either `StringIO` or `BytesIO` objects.

    Examples
    --------

    >>> import os.path
    >>> file = fetch("P12345", path_to_directory)
    >>> print(os.path.basename(file))
    P12345.pdb
    >>> files = fetch(["P12345", "Q8K9I1"], path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['P12345.pdb', 'Q8K9I1.pdb']
    """

    # If only a single ID is present,
    # put it into a single element list
    if isinstance(ids, str):
        ids = [ids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not isdir(target_path):
        os.makedirs(target_path)
    files = []
    for i, id in enumerate(ids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(ids):d} ({id})...", end="\r")
        # Fetch file from database
        if target_path is not None:
            file = join(target_path, id + "." + format)
        else:
            # 'file = None' -> store content in a file-like object
            file = None
        if file is None or not isfile(file) or getsize(file) == 0 or overwrite:
            if format in ["pdb", "cif"]:
                metadata_response = requests.get(f"{_fetch_url}/{id}")
                assert_valid_response(metadata_response.status_code)
                metadata_json = metadata_response.json()[0]
                # a list of length 1 is always returned
                file_url = metadata_json[f"{format}Url"]
                file_response = requests.get(file_url)
                assert_valid_response(file_response.status_code)
                content = file_response.text
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
