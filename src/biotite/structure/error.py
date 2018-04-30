# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information..

"""
This module contains all possible errors of the `structure` subpackage.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["BadStructureError"]


class BadStructureError(Exception):
    """
    Indicates that a structure is not suitable for a certain operation.
    """
    pass