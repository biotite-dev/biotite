# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["temp_file", "temp_dir"]

import shutil
import atexit
import os
import tempfile


_temp_dir = ""

def _create_temp_dir():
    global _temp_dir
    if _temp_dir == "":
        cwd = os.getcwd()
        _temp_dir = os.path.join(cwd, ".biotitetemp")
        if not os.path.isdir(_temp_dir):
            os.makedirs(_temp_dir)
        atexit.register(_delete_temp)


def _delete_temp():
    global _temp_dir
    # Condition only for savety reasons
    if ".biotitetemp" in _temp_dir: 
        shutil.rmtree(_temp_dir)


def temp_file(suffix=""):
    """
    Get a file path to a temporary file.
    
    All temporary files will be deleted after script execution.
    
    Parameters
    ----------
    suffix : str
        Suffix of the file.
        By default no suffix will be appended.
    
    Returns
    -------
    temp_file_name : str
        a file name in the temporary directory.
    """
    global _temp_dir
    _create_temp_dir()
    if suffix != "" and not suffix.startswith("."):
        suffix = "." + suffix
    return tempfile.mktemp(suffix=suffix, dir=_temp_dir)
     

def temp_dir():
    """
    Get the temporary directory path.
    
    The temporary directory will be deleted after script execution.
    
    Returns
    -------
    temp_dir : str
        Path of the temporary directory.
    """
    global _temp_dir
    _create_temp_dir()
    return _temp_dir