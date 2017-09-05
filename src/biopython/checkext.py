# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

__all__ = ["has_c_extensions"]


def has_c_extensions():
    try:
        from .ccheckext import c_check_extensions
        return c_check_extensions()
    except ImportError:
        return False