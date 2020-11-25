# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module handles conversion of RNA structures to
dot-bracket-notation.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["dot_bracket_from_structure", "dot_bracket"]

import numpy as np
from .basepairs import base_pairs
from .pseudoknots import pseudoknots
from .residues import get_residue_count, get_residue_positions

_OPENING_BRACKETS = "([<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CLOSING_BRACKETS = ")]>abcdefghijklmnopqrstuvwxyz"


def dot_bracket_from_structure(nucleic_acid_strand, scoring=None):
    """
    Represent a nucleic-acid-strand in dot-bracket-letter-notation
    (DBL-notation) [1]_.

    Parameters
    ----------
    atom_array : AtomArray
        The nucleic-acid-strand to be represented in DBL-notation.
    scoring : ndarray, dtype=int, shape=(n,) (default: None)
        The score for each basepair, which is passed on to
        :func:`pseudoknots()`

    Returns
    -------
    notations : list [str, ...]
        The DBL-notation for each solution from :func:`pseudoknots()`.

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
    basepairs = base_pairs(nucleic_acid_strand)
    basepairs = get_residue_positions(nucleic_acid_strand, basepairs)
    length = get_residue_count(nucleic_acid_strand)
    return dot_bracket(basepairs, length, scoring=scoring)

def dot_bracket(basepairs, length, scoring=None):
    """
    Represent a nucleic-acid-strand in dot-bracket-letter-notation
    (DBL-notation) [1]_.

    The nucleic acid strand is represented as continuously counted
    nucleotide sequence, where the nucleotides are counted from zero.

    Parameters
    ----------
    basepairs : ndarray, shape=(n,2)
        Each row corresponds to the positions of the bases in the
        strand.
    length : int
        The number of bases in the strand.
    scoring : ndarray, dtype=int, shape=(n,) (default: None)
        The score for each basepair, which is passed on to
        :func:`pseudoknots()`

    Returns
    -------
    notations : list [str, ...]
        The DBL-notation for each solution from :func:`pseudoknots()`.

    Examples
    --------
    The sequence ``ACGTC`` has a length of 5. If there was to be a
    pairing interaction between the ``A`` and ``T``, ``basepairs`` would
    have the form:

    >>> import numpy as np
    >>> basepairs = np.array([[0, 3]])

    The DBL Notation can then be found with ``dot_bracket()``:

    >>> dot_bracket(basepairs, 5)[0]
    '(..).'


    See Also
    --------
    dot_bracket_from_structure
    base_pairs
    pseudoknots

    References
    ----------

    .. [1] M Antczak, M Popenda and T Zok et al.,
       "New algorithms to represent complex pseudoknotted RNA structures
        in dot-bracket notation.",
       Bioinformatics, 34(8), 1304-1312 (2018).
    """
    pseudoknot_order = pseudoknots(basepairs, scoring=scoring)

    # Each optimal pseudoknot order solution is represented in
    # dot-bracket-notation
    notations = [""]*len(pseudoknot_order)

    for s, solution in enumerate(pseudoknot_order):
        # Bases whose partners have an opened bracket
        opened_brackets = set()
        for pos in range(length):
            if pos not in basepairs:
                notations[s] += "."
            else:
                # Get position in ``basepairs`` and ``pseudoknot_order``
                bp_pos = np.where(basepairs == pos)[0][0]
                if pos in opened_brackets:
                    notations[s] += _CLOSING_BRACKETS[solution[bp_pos]]
                else:
                    for base in basepairs[bp_pos]:
                        if base != pos:
                            opened_brackets.add(base)
                    notations[s] += _OPENING_BRACKETS[solution[bp_pos]]

    return notations
