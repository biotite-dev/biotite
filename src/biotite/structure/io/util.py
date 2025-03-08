# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Common functions used by a number of subpackages.
"""

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = ["number_of_integer_digits"]

import numpy as np


def number_of_integer_digits(values):
    """
    Get the maximum number of characters needed to represent the
    pre-decimal positions of the given numeric values.

    Parameters
    ----------
    values : ndarray, dtype=float
        The values to be checked.

    Returns
    -------
    n_digits : int
        The maximum number of characters needed to represent the
        pre-decimal positions of the given numeric values.
    """
    if len(values) == 0:
        return 0
    values = values.astype(int, copy=False)
    n_digits = 0
    n_digits = max(n_digits, len(str(np.min(values))))
    n_digits = max(n_digits, len(str(np.max(values))))
    return n_digits
