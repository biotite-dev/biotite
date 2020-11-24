# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module handles conversion of RNA structures to
dot-bracket-notation.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["dot_bracket"]

import numpy as np
from .basepairs import base_pairs
from .pseudoknots import pseudoknots
from .residues import get_residue_starts

_OPENING_BRACKETS = "([<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CLOSING_BRACKETS = ")]>abcdefghijklmnopqrstuvwxyz"


def dot_bracket(nucleic_acid_strand, scoring=None):
    """
    Represent a nucleic-acid-strand in dot-bracket-letter-notation
    (DBL-notation) [1]_.

    Parameters
    ----------
    atom_array : AtomArray
        The nucleic-acid-strand to be represented in DBL-notation.
    scoring : ndarray, dtype=int, shape=(n,) (default: None)
        The score for each basepair, which is passed on to
        :func:``pseudoknots()``

    Returns
    -------
    notations : list [string, ...]
        The DBL-notation for each solution from :func:``pseudoknots()``.

    See Also
    --------
    base_pairs
    pseudoknots

    References
    ----------

    .. [1] M Antczak, M Popenda and T Zok et al.,
       "New algorithms to represent complex pseudoknotted RNA structures
        in dot-bracket notation.",
       Bioinformatics, 34(8), 1304-1312 (2018).
    """
    # Analyze structure
    basepairs = base_pairs(nucleic_acid_strand)
    pseudoknot_order = pseudoknots(basepairs, scoring=scoring)
    residue_starts = get_residue_starts(nucleic_acid_strand)

    # Each optimal pseudoknot order solution is represented in
    # dot-bracket-notation
    notations = [""]*len(pseudoknot_order)

    for s, solution in enumerate(pseudoknot_order):
        # Bases whose partners have an opened bracket
        opened_brackets = set()
        for residue_start in residue_starts:
            if residue_start not in basepairs:
                notations[s] += "."
            else:
                # Get position in ``basepairs`` and ``pseudoknot_order``
                pos = np.where(basepairs == residue_start)[0][0]

                if residue_start in opened_brackets:
                    notations[s] += _CLOSING_BRACKETS[solution[pos]]
                else:
                    for base in basepairs[pos]:
                        if base != residue_start:
                            opened_brackets.add(base)
                    notations[s] += _OPENING_BRACKETS[solution[pos]]

    return notations
