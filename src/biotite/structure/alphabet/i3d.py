# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
NumPy port of the ``foldseek`` code for encoding structures to 3di.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde"
__all__ = ["I3DSequence", "to_3di"]

import warnings
from collections.abc import Iterable
from typing import ClassVar
import numpy as np
from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.sequence import Sequence
from biotite.structure.alphabet.encoder import Encoder
from biotite.structure.atoms import AtomArray
from biotite.structure.chains import get_chain_starts
from biotite.structure.util import coord_for_atom_name_per_residue
from biotite.typing import K, N, NDArray1


class I3DSequence(Sequence):
    """
    Representation of a structure in the 3Di structural alphabet.
    :footcite:`VanKempen2024`

    Parameters
    ----------
    sequence : iterable object, optional
        The 3Di sequence.
        This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.

    See Also
    --------
    to_3di : Create 3Di sequences from a structure.

    References
    ----------

    .. footbibliography::
    """

    alphabet: ClassVar[LetterAlphabet] = LetterAlphabet("acdefghiklmnpqrstvwy")
    undefined_symbol = "d"

    def __init__(self, sequence: str | Iterable[str] = "") -> None:
        if isinstance(sequence, str):
            sequence = sequence.lower()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        super().__init__(sequence)

    def get_alphabet(self) -> LetterAlphabet:
        return I3DSequence.alphabet

    def __repr__(self) -> str:
        return f'I3DSequence("{"".join(self.symbols)}")'


def to_3di(
    atoms: AtomArray[N],
) -> tuple[list[I3DSequence], NDArray1[K, np.integer]]:
    """
    Encode each chain in the given structure to the 3Di structure alphabet.
    :footcite:`VanKempen2024`

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode.
        May contain multiple chains.

    Returns
    -------
    sequences : list of I3DSequence, length=n
        The encoded 3Di sequence for each peptide chain in the structure.
    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> sequences, chain_starts = to_3di(atom_array)
    >>> print(sequences[0])
    dqqvvcvvcpnvvnvdhgdd
    """
    sequences = []
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        sequence = I3DSequence()
        if chain.array_length() == 0:
            warnings.warn("Ignoring empty chain")
        else:
            sequence.code = (
                Encoder()
                .encode(
                    *coord_for_atom_name_per_residue(chain, ["CA", "CB", "N", "C"]),
                )
                .filled()
            )
        sequences.append(sequence)
    return sequences, chain_start_indices[:-1]
