# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

__all__ = ["has_c_extensions", "enable_c_extensions", "uses_c_extensions"]


def has_c_extensions():
    """
    Check if the Biopython distribution contains successfully built
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
    Make Biopython use Cython accelerated functions.
    
    By default Biopython uses C-acceleration, if the distribution was
    successfully built with Cython extensions.
    The application of this function lasts until the end of the script
    or interactive python session.
    
    Parameters
    ----------
    enable : bool
        If true, Biopython will use C-acceleration if applicable,
        if false, Biopython will always use the pure Python
        implementation.
    
    Raises
    ------
    ValueError
        If `enable` is true, but the Biopython distribution does
        not contain Cython extensions.
    """
    global _extensions_enabled
    if enable:
        if has_c_extensions():
            _extensions_enabled = True
        else:
            raise ValueError("The Biopython distribution does not "
                             "contain Cython extensions")
    else:
        _extensions_enabled = False


def uses_c_extensions():
    """
    Check if the Biopython is currently using Cython extensions.
    
    This is only true, if the distribution contains Cython extensions
    and its usage is enbaled via `enable_c_extensions()` (default).
    
    Returns
    -------
    bool
        True, if Biopython uses Cython extensions, false otherwise.
    """
    global _extensions_enabled
    return _extensions_enabled


_extensions_enabled = has_c_extensions()
