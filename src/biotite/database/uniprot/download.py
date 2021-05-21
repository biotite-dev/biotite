# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.uniprot"
__author__ = "Maximilian Greil"
__all__ = ["fetch"]

from os.path import isdir, isfile, join, getsize
import os
import io
import requests
from ..error import RequestError

_fetch_url = "https://www.uniprot.org/"


def _get_database_name(id):
    """
    Get the correct UniProt database from the ID of the file to be downloaded.

    Parameters
    ----------
    id: str
        ID of the file to be downloaded.

    Returns
    -------
    name : str
        E-utility UniProt database name.
    """
    if id[:3] == "UPI":
        return "uniparc"
    elif id[:6] == "UniRef":
        return "uniref"
    return "uniprot"


def fetch(ids, format, target_path=None,
          overwrite=False, verbose=False):
    """
    Download structure files (or sequence files) from the UniProt in
    various formats.

    Available databases are UniProtKB, UniRef and UniParc.

    This function requires an internet connection.

    Parameters
    ----------
    ids : str or iterable object of str
        A single ID or a list of IDs of the file(s)
        to be downloaded.
    format : {'fasta', 'gff', 'txt', 'xml', 'rdf', 'tab'}
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
        db_name = _get_database_name(id)
        # Verbose output
        if verbose:
            print(f"Fetching file {i + 1:d} / {len(ids):d} ({id})...",
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
            if format in ["fasta", "gff", "txt", "xml", "rdf", "tab"]:
                r = requests.get(_fetch_url + db_name + "/" + id + "." + format)
                content = r.text
                _assert_valid_file(r.status_code, id, format)
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


def _assert_valid_file(response_status_code, id, format):
    """
    Checks whether the response is an actual file
    or not.

    Parameters
    ----------
    response_status_code: int
        Status code of request.get.
    id: str
        ID of the file to be downloaded.
    format: str
        Format of the file to be downloaded.
    """
    if response_status_code == 400:
        raise RequestError("Bad request. There is a problem with your input {:}.".format(id + "." + format))
    elif response_status_code == 404:
        raise RequestError("Not found. The resource {:} you requested doesn't exist.".format(id + "." + format))
    elif response_status_code == 410:
        raise RequestError("Gone. The resource {:} you requested was removed.".format(id + "." + format))
    elif response_status_code == 500:
        raise RequestError(
            "Internal server error. Most likely a temporary problem, but if the problem persists please contact UniProt team.")
    elif response_status_code == 503:
        raise RequestError("Service not available. The server is being updated, try again later.")
