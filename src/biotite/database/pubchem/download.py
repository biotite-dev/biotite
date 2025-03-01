# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = ["fetch", "fetch_property"]

import io
import numbers
import os
from os.path import getsize, isdir, isfile, join
import requests
from biotite.database.error import RequestError
from biotite.database.pubchem.error import parse_error_details
from biotite.database.pubchem.throttle import ThrottleStatus

_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
_binary_formats = ["png", "asnb"]


def fetch(
    cids,
    format="sdf",
    target_path=None,
    as_structural_formula=False,
    overwrite=False,
    verbose=False,
    throttle_threshold=0.5,
    return_throttle_status=False,
):
    """
    Download structure files from *PubChem* in various formats.

    This function requires an internet connection.

    Parameters
    ----------
    cids : int or iterable object or int
        A single compound ID (CID) or a list of CIDs of the structure(s)
        to be downloaded.
    format : {'sdf', 'asnt' 'asnb', 'xml', 'json', 'jsonp', 'png'}
        The format of the files to be downloaded.
    target_path : str, optional
        The target directory of the downloaded files.
        By default, the file content is stored in a file-like object
        (:class:`StringIO` or :class:`BytesIO`, respectively).
    as_structural_formula : bool, optional
        If set to true, the structural formula is download instead of
        an 3D conformer.
        This means that coordinates lie in th xy-plane and represent
        the positions atoms would have an a structural formula
        representation.
    overwrite : bool, optional
        If true, existing files will be overwritten.
        Otherwise the respective file will only be downloaded, if the
        file does not exist yet in the specified target directory or if
        the file is empty.
    verbose : bool, optional
        If set to true, the function will output the download progress.
    throttle_threshold : float or None, optional
        A value between 0 and 1.
        If the load of either the request time or count exceeds this
        value the execution is halted.
        See :class:`ThrottleStatus` for more information.
        If ``None`` is given, the execution is never halted.
    return_throttle_status : float, optional
        If set to true, the :class:`ThrottleStatus` of the final request
        is also returned.

    Returns
    -------
    files : str or StringIO or BytesIO or list of (str or StringIO or BytesIO)
        The file path(s) to the downloaded files.
        If a single CID was given in `cids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
        If no `target_path` was given, the file contents are stored in
        either :class:`StringIO` or :class:`BytesIO` objects.
    throttle_status : ThrottleStatus
        The :class:`ThrottleStatus` obtained from the server response.
        If multiple CIDs are requested, the :class:`ThrottleStatus` of
        of the final response is returned.
        This can be used for custom request throttling, for example.
        Only returned, if `return_throttle_status` is set to true.

    Examples
    --------

    >>> import os.path
    >>> file = fetch(2244, "sdf", path_to_directory)
    >>> print(os.path.basename(file))
    2244.sdf
    >>> files = fetch([2244, 5950], "sdf", path_to_directory)
    >>> print([os.path.basename(file) for file in files])
    ['2244.sdf', '5950.sdf']
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
            print(f"Fetching file {i + 1:d} / {len(cids):d} ({cid})...", end="\r")

        # Fetch file from database
        if target_path is not None:
            file = join(target_path, str(cid) + "." + format)
        else:
            # 'file = None' -> store content in a file-like object
            file = None

        if file is None or not isfile(file) or getsize(file) == 0 or overwrite:
            record_type = "2d" if as_structural_formula else "3d"
            r = requests.get(
                _base_url + f"compound/cid/{cid}/{format.upper()}",
                params={"record_type": record_type},
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


def fetch_property(cids, name, throttle_threshold=0.5, return_throttle_status=False):
    """
    Download the given property for the given CID(s).

    This function requires an internet connection.

    Parameters
    ----------
    cids : int or iterable object or int
        A single compound ID (CID) or a list of CIDs to get the property
        for.
    name : str
        The name of the desired property.
        Valid properties are given in the *PubChem* REST API
        `documentation <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=Compound-Property-Tables>`_.
    throttle_threshold : float or None, optional
        A value between 0 and 1.
        If the load of either the request time or count exceeds this
        value the execution is halted.
        See :class:`ThrottleStatus` for more information.
        If ``None`` is given, the execution is never halted.
    return_throttle_status : float, optional
        If set to true, the :class:`ThrottleStatus` of the final request
        is also returned.

    Returns
    -------
    property : str or list of str
        The requested property for each given CID.
        If a single CID was given in `cids`,
        a single string is returned.
        If a list (or other iterable
        object) was given, a list of strings is returned.
    throttle_status : ThrottleStatus
        The :class:`ThrottleStatus` obtained from the server response.
        This can be used for custom request throttling, for example.
        Only returned, if `return_throttle_status` is set to true.

    Examples
    --------

    >>> butane_cids = np.array(search(FormulaQuery("C4H10")))
    >>> # Filter natural isotopes...
    >>> n_iso = np.array(fetch_property(butane_cids, "IsotopeAtomCount"), dtype=int)
    >>> # ...and neutral compounds
    >>> charge = np.array(fetch_property(butane_cids, "Charge"), dtype=int)
    >>> butane_cids = butane_cids[(n_iso == 0) & (charge == 0)]
    >>> print(sorted(butane_cids.tolist()))
    [6360, 7843, 18402699, 19029854, 19048342, 157632982, 158271732, 158934736, 161295599, 161897780]
    >>> # Get the IUPAC names for each compound
    >>> iupac_names = fetch_property(butane_cids, "IUPACName")
    >>> # Compounds with multiple molecules use ';' as separator
    >>> print(iupac_names)
    ['butane', '2-methylpropane', 'methane;prop-1-ene', 'ethane;ethene', 'cyclopropane;methane', 'cyclobutane;molecular hydrogen', 'acetylene;methane', 'carbanide;propane', 'carbanylium;propane', 'methylcyclopropane;molecular hydrogen']
    """
    # If only a single CID is present,
    # put it into a single element list
    if isinstance(cids, numbers.Integral):
        cids = [cids]
        single_element = True
    else:
        single_element = False

    # Property names may only contain letters and numbers
    if not name.isalnum():
        raise ValueError(f"Property '{name}' contains invalid characters")

    # Use TXT format instead of CSV to avoid issues with ',' characters
    # within table elements
    r = requests.post(
        _base_url + f"compound/cid/property/{name}/TXT",
        data={"cid": ",".join([str(cid) for cid in cids])},
    )
    if not r.ok:
        raise RequestError(parse_error_details(r.text))
    throttle_status = ThrottleStatus.from_response(r)
    if throttle_threshold is not None:
        throttle_status.wait_if_busy(throttle_threshold)

    # Each line contains the property for one CID
    properties = r.text.splitlines()

    # If input was a single ID, return only a single value
    if single_element:
        return_value = properties[0]
    else:
        return_value = properties
    if return_throttle_status:
        return return_value, throttle_status
    else:
        return return_value
