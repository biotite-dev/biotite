# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = ["fetch"]

import numbers
import requests
from os.path import isdir, isfile, join, getsize
import os
import io
from .throttle import ThrottleStatus
from .error import parse_error_details
from ..error import RequestError


_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
_binary_formats = ["png", "asnb"]


def fetch(cids, format="sdf", target_path=None, overwrite=False, verbose=False,
          throttle_threshold=0.5, return_throttle_status=False):
    """
    Download structure files (or sequence files) from the RCSB PDB in
    various formats.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    cids : int or iterable object or int
        A single PDB ID or a list of PDB IDs of the structure(s)
        to be downloaded .
    format : {'sdf', 'asnt' 'asnb', 'xml', 'json', 'jsonp', 'png'}
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
    # If only a single CID is present,
    # put it into a single element list
    if isinstance(cids, numbers.Integral):
        cids = [cids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if target_path is not None and not isdir(target_path):
        os.makedirs(target_path)
    
    files = []
    for i, cid in enumerate(cids):
        # Prevent IDs as strings, this could be a common error, as other
        # database interfaces of Biotite use string IDs
        if isinstance(cid, str):
            raise TypeError("CIDs must be given as integers, not as string")
        # Verbose output
        if verbose:
            print(f"Fetching file {i+1:d} / {len(cids):d} ({cid})...",
                  end="\r")
        
        # Fetch file from database
        if target_path is not None:
            file = join(target_path, str(cid) + "." + format)
        else:
            # 'file = None' -> store content in a file-like object
            file = None
        
        if file is None \
           or not isfile(file) \
           or getsize(file) == 0 \
           or overwrite:
                r = requests.get(
                    _base_url
                    + f"compound/cid/{cid}/{format.upper()}"
                )
                if not r.ok:
                    raise RequestError(parse_error_details(r.text))
                
                if format.lower() in _binary_formats:
                    content = r.content
                else:
                    content = r.text
                
                if file is None:
                    if format in _binary_formats:
                        file = io.BytesIO(content)
                    else:
                        file = io.StringIO(content)
                else:
                    mode = "wb+" if format in _binary_formats else "w+"
                    with open(file, mode) as f:
                        f.write(content)
                
                throttle_status = ThrottleStatus.from_response(r)
                if throttle_threshold is not None:
                    throttle_status.wait_if_busy(throttle_threshold)
                
        
        files.append(file)
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return_value = files[0]
    else:
        return_value = files
    if return_throttle_status:
        return return_value, throttle_status
    else:
        return return_value
