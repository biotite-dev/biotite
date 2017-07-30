# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module contains all possible errors of the `Structure` subpackage.
"""

class BadStructureError(Exception):
    """
    Indicates that a structure is not suitable for a certain operation.
    """
    pass