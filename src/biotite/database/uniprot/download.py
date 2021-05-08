# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.uniprot"
__author__ = "Maximilian Greil"
__all__ = ["get_database_name", "fetch", "_assert_valid_file", "_sanitize_db_name"]

from os.path import isdir, isfile, join, getsize
import os
import io
import requests
from ..error import RequestError

fetch_url = "https://www.uniprot.org/"

_databases = {"UniProtKB": "uniprot",
              "UniRef": "uniref",
              "UniParc": "uniparc"}


def get_database_name(database):
    """
    Map a common UniProt database name to an E-utility database
    name.

    Parameters
    ----------
    database : str
    Uniprot database name.

    Returns
    -------
    name : str
    E-utility database name.

    Examples
    --------

    >>> print(get_database_name("UniProtKB"))
    uniprot
    """
    return _databases[database]


def fetch(uids, db_name, format, target_path=None,
          overwrite=False, verbose=False):
    """
    Download structure files (or sequence files) from the RCSB PDB in
    various formats.

    This function requires an internet connection.

    Parameters
    ----------
    uids : str or iterable object of str
        A single ID or a list of IDs of the file(s)
        to be downloaded.
    db_name : str:
        E-utility or common database name.
    format : {'fasta'}
        The format of the files to be downloaded.
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
    >>> file = fetch("P12345", "fasta", path_to_directory)
    >>> print(os.path.basename(file))
    P12345.fasta
    >>> files = fetch(["P12345", "Q8K9I1"], "fasta", path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['P12345.fasta', 'Q8K9I1.fasta']
    """
    # If only a single UID is present,
    # put it into a single element list
    if isinstance(uids, str):
        uids = [uids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not isdir(target_path):
        os.makedirs(target_path)
    files = []
    for i, id in enumerate(uids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(uids):d} ({id})...",
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
            if format == "fasta":
                r = requests.get(fetch_url + _sanitize_db_name(db_name) + "/" + id + ".fasta")
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


def _assert_valid_file(response_text, uid):
    """
    Checks whether the response is an actual file
    or not.
    """
    # Structure file and FASTA file retrieval
    # have different error messages
    if any(err_msg in response_text for err_msg in [
        "Bad request. There is a problem with your input.",
        "Not found. The resource you requested doesn't exist.",
        "Gone. The resource you requested was removed.",
        "Internal server error. Most likely a temporary problem, but if the problem persists please contact us.",
        "Service not available. The server is being updated, try again later."
    ]):
        raise RequestError("ID {:} is invalid".format(uid))


def _sanitize_db_name(db_name):
    if db_name in _databases.keys():
        # Convert into E-utility database name
        return _databases[db_name]
    elif db_name in _databases.values():
        # Is already E-utility database name
        return db_name
    else:
        raise ValueError("Database '{db_name}' is not existing")
