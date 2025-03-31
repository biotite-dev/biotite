# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module handles conversion of RNA structures to
dot-bracket-notation.
"""

__name__ = "biotite.structure"
__author__ = "Tom David Müller"
__all__ = ["dot_bracket_from_structure", "dot_bracket", "base_pairs_from_dot_bracket"]

import numpy as np
from biotite.structure.basepairs import base_pairs
from biotite.structure.pseudoknots import pseudoknots
from biotite.structure.residues import get_residue_count, get_residue_positions

_OPENING_BRACKETS = "([{<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_OPENING_BRACKETS_BYTES = _OPENING_BRACKETS.encode()
_CLOSING_BRACKETS = ")]}>abcdefghijklmnopqrstuvwxyz"
_CLOSING_BRACKETS_BYTES = _CLOSING_BRACKETS.encode()


def dot_bracket_from_structure(
    nucleic_acid_strand, scores=None, max_pseudoknot_order=None
):
    """
    Represent a nucleic-acid-strand in dot-bracket-letter-notation
    (DBL-notation). :footcite:`Antczak2018`

    Parameters
    ----------
    nucleic_acid_strand : AtomArray
        The nucleic acid strand to be represented in DBL-notation.
    scores : ndarray, dtype=int, shape=(n,)
        The score for each base pair, which is passed on to
        :func:`pseudoknots()`.
    max_pseudoknot_order : int
        The maximum pseudoknot order to be found. If a base pair would
        be of a higher order, it is represented as unpaired. If ``None``
        is given, all base pairs are evaluated.

    Returns
    -------
    notations : list [str, ...]
        The DBL-notation for each solution from :func:`pseudoknots()`.

    See Also
    --------
    base_pairs : Compute the base pairs from a structure as passed to this function.
    dot_bracket : Compute the dot bracket notation directly from base pairs.
    pseudoknots : Get the pseudoknot order for each base pair.

    References
    ----------

    .. footbibliography::
    """
    basepairs = base_pairs(nucleic_acid_strand)
    if len(basepairs) == 0:
        return [""]
    basepairs = get_residue_positions(nucleic_acid_strand, basepairs)
    length = get_residue_count(nucleic_acid_strand)
    return dot_bracket(
        basepairs, length, scores=scores, max_pseudoknot_order=max_pseudoknot_order
    )


def dot_bracket(basepairs, length, scores=None, max_pseudoknot_order=None):
    """
    Represent a nucleic acid strand in dot-bracket-letter-notation
    (DBL-notation). :footcite:`Antczak2018`

    The nucleic acid strand is represented as nucleotide sequence,
    where the nucleotides are counted continiously from zero.

    Parameters
    ----------
    basepairs : ndarray, shape=(n,2)
        Each row corresponds to the positions of the bases in the
        strand.
    length : int
        The number of bases in the strand.
    scores : ndarray, dtype=int, shape=(n,)
        The score for each base pair, which is passed on to :func:`pseudoknots()`.
    max_pseudoknot_order : int
        The maximum pseudoknot order to be found. If a base pair would
        be of a higher order, it is represented as unpaired. If ``None``
        is given, all pseudoknot orders are evaluated.

    Returns
    -------
    notations : list [str, ...]
        The DBL-notation for each solution from :func:`pseudoknots()`.

    See Also
    --------
    base_pairs_from_dot_bracket : The reverse operation.
    dot_bracket_from_structure : Compute the dot bracket notation from a structure.
    base_pairs : Compute the base pairs from a structure as passed to this function.
    pseudoknots : Get the pseudoknot order for each base pair.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    The sequence ``ACGTC`` has a length of 5. If there was to be a
    pairing interaction between the ``A`` and ``T``, `basepairs` would
    have the form:

    >>> import numpy as np
    >>> basepairs = np.array([[0, 3]])

    The DBL Notation can then be found with ``dot_bracket()``:

    >>> dot_bracket(basepairs, 5)[0]
    '(..).'
    """
    # Make sure the lower residue is on the left for each row
    basepairs = np.sort(basepairs, axis=1)

    # Get pseudoknot order
    pseudoknot_order = pseudoknots(
        basepairs, scores=scores, max_pseudoknot_order=max_pseudoknot_order
    )

    # Each optimal pseudoknot order solution is represented in
    # dot-bracket-notation
    notations = [bytearray((b"." * length)) for _ in range(len(pseudoknot_order))]
    for s, solution in enumerate(pseudoknot_order):
        for basepair, order in zip(basepairs, solution):
            if order == -1:
                continue
            notations[s][basepair[0]] = _OPENING_BRACKETS_BYTES[order]
            notations[s][basepair[1]] = _CLOSING_BRACKETS_BYTES[order]
    return [notation.decode() for notation in notations]


def base_pairs_from_dot_bracket(dot_bracket_notation):
    """
    Extract the base pairs from a nucleic-acid-strand in
    dot-bracket-letter-notation (DBL-notation). :footcite:`Antczak2018`

    The nucleic acid strand is represented as nucleotide sequence,
    where the nucleotides are counted continiously from zero.

    Parameters
    ----------
    dot_bracket_notation : str
        The DBL-notation.

    Returns
    -------
    basepairs : ndarray, shape=(n,2)
        Each row corresponds to the positions of the bases in the
        sequence.

    See Also
    --------
    dot_bracket : The reverse operation.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    The notation string ``'(..).'`` contains a base pair between the
    indices 0 and 3. This pairing interaction can be extracted
    conveniently by the use of :func:`base_pairs_from_dot_bracket()`:

    >>> base_pairs_from_dot_bracket('(..).')
    array([[0, 3]])
    """
    basepairs = []
    opened_brackets = [[] for _ in range(len(_OPENING_BRACKETS))]

    # Iterate through input string and extract base pairs
    for pos, symbol in enumerate(dot_bracket_notation):
        if symbol in _OPENING_BRACKETS:
            # Add opening residues to list (separate list for each
            # bracket type)
            index = _OPENING_BRACKETS.index(symbol)
            opened_brackets[index].append(pos)

        elif symbol in _CLOSING_BRACKETS:
            # For each closing bracket, the the base pair consists out
            # of the current index and the last index added to the list
            # in `opened_brackets` corresponding to the same bracket
            # type.
            index = _CLOSING_BRACKETS.index(symbol)
            basepairs.append((opened_brackets[index].pop(), pos))

        else:
            if symbol != ".":
                raise ValueError(f"'{symbol}' is an invalid character for DBL-notation")

    for not_closed in opened_brackets:
        if not_closed != []:
            raise ValueError(
                "Invalid DBL-notation, not all opening brackets have a closing bracket"
            )

    # Sort the base pair indices in ascending order
    basepairs = np.array(basepairs)
    if len(basepairs) > 0:
        basepairs = basepairs[np.argsort(basepairs[:, 0])]
    return basepairs
