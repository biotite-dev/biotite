# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

__all__ = ["has_c_extensions", "enable_c_extensions", "uses_c_extensions"]


def has_c_extensions():
    """
    Check if the Biotite distribution contains successfully built
    Cython extensions.
    
    Returns
    -------
    bool
        True, if Cython extensions exist, false otherwise.
    """
    try:
        from .cextensions import c_check_extensions
        return c_check_extensions()
    except ImportError:
        return False


def enable_c_extensions(enable):
    """
    Make Biotite use Cython accelerated functions.
    
    By default Biotite uses C-acceleration, if the distribution was
    successfully built with Cython extensions.
    The application of this function lasts until the end of the script
    or interactive python session.
    
    Parameters
    ----------
    enable : bool
        If true, Biotite will use C-acceleration if applicable,
        if false, Biotite will always use the pure Python
        implementation.
    
    Raises
    ------
    ValueError
        If `enable` is true, but the Biotite distribution does
        not contain Cython extensions.
    """
    global _extensions_enabled
    if enable:
        if has_c_extensions():
            _extensions_enabled = True
        else:
            raise ValueError("The Biotite distribution does not "
                             "contain Cython extensions")
    else:
        _extensions_enabled = False


def uses_c_extensions():
    """
    Check if the Biotite is currently using Cython extensions.
    
    This is only true, if the distribution contains Cython extensions
    and its usage is enbaled via `enable_c_extensions()` (default).
    
    Returns
    -------
    bool
        True, if Biotite uses Cython extensions, false otherwise.
    """
    global _extensions_enabled
    return _extensions_enabled


_extensions_enabled = has_c_extensions()
