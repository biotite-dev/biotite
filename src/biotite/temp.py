# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import shutil
import atexit
import os

__all__ = ["temp_file", "temp_dir"]


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


def temp_file(file_name):
    """
    Get a file path to a temporary file with the given file name.
    
    All temporary files will be deleted after script execution.
    
    Parameters
    ----------
    file_name : str
        The base name of the file.
    
    Returns
    -------
    temp_file_name : str
        `file_name` appended to the path of the temporary directory.
    """
    global _temp_dir
    _create_temp_dir()
    return os.path.join(_temp_dir, file_name)
     

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