"""
Function for converting a structure into a sequence.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["to_sequence"]

import numpy as np
from biotite.sequence import Sequence
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.chains import get_chain_starts
from biotite.structure.error import BadStructureError
from biotite.structure.info.groups import amino_acid_names, nucleotide_names
from biotite.structure.info.misc import one_letter_code
from biotite.structure.residues import get_residues
from biotite.typing import K, M, N, NDArray1

HETERO_PLACEHOLDER = "."


def to_sequence(
    atoms: AtomArray[N] | AtomArrayStack[M, N],
    allow_hetero: bool = False,
) -> tuple[list[Sequence], NDArray1[K, np.integer]]:
    """
    Convert each chain in a structure into a sequence.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure.
        May contain multiple chains.
        Each chain must be either a peptide or a nucleic acid.
    allow_hetero : bool, optional
        If true, residues inside a amino acid or nucleotide chain,
        that have no one-letter code, are replaced by the respective
        '*any*' symbol (`"X"` or `"N"`, respectively).
        The same is true for amino acids in nucleotide chains and vice
        versa.
        By default, an exception is raised.

    Returns
    -------
    sequences : list of Sequence, length=n
        The sequence for each chain in the structure.
    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.

    Notes
    -----
    Residues are considered amino acids or nucleotides based on their
    appearance :func:`info.amino_acid_names()` or
    :func:`info.nucleotide_names()`, respectively.

    Examples
    --------

    >>> sequences, chain_starts = to_sequence(atom_array)
    >>> print(sequences)
    [ProteinSequence("NLYIQWLKDGGPSSGRPPPS", ambiguous=False)]
    """
    sequences = []
    # Sets allow much faster membership checks than `np.isin()` with the long lists
    amino_acids = frozenset(amino_acid_names())
    nucleotides = frozenset(nucleotide_names())
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        _, residues = get_residues(chain)
        one_letter_symbols = np.array(
            [one_letter_code(res) or HETERO_PLACEHOLDER for res in residues]
        )
        hetero_mask = one_letter_symbols == HETERO_PLACEHOLDER

        is_amino_acid = np.array([res in amino_acids for res in residues], dtype=bool)
        is_nucleotide = np.array([res in nucleotides for res in residues], dtype=bool)
        aa_count = np.count_nonzero(is_amino_acid)
        nuc_count = np.count_nonzero(is_nucleotide)
        if aa_count == 0 and nuc_count == 0:
            raise BadStructureError(
                f"Chain {chain.chain_id[0]} contains neither amino acids "
                "nor nucleotides"
            )
        elif aa_count > nuc_count:
            # Chain is a peptide
            hetero_mask |= ~is_amino_acid
            if not allow_hetero and np.any(hetero_mask):
                hetero_indices = np.where(hetero_mask)[0]
                raise BadStructureError(
                    f"Hetero residue(s) "
                    f"{', '.join(residues[hetero_indices])} in peptide"
                )
            one_letter_symbols[hetero_mask] = "X"
            # Replace selenocysteine and pyrrolysine
            one_letter_symbols[one_letter_symbols == "U"] = "C"
            one_letter_symbols[one_letter_symbols == "O"] = "K"
            sequences.append(ProteinSequence("".join(one_letter_symbols)))
        else:
            # Chain is a nucleic acid
            hetero_mask |= ~is_nucleotide
            if not allow_hetero and np.any(hetero_mask):
                hetero_indices = np.where(hetero_mask)[0]
                raise BadStructureError(
                    f"Hetero residue(s) "
                    f"{', '.join(residues[hetero_indices])} in nucleic acid"
                )
            one_letter_symbols[hetero_mask] = "N"
            # Replace uracil
            one_letter_symbols[one_letter_symbols == "U"] = "T"
            sequences.append(NucleotideSequence("".join(one_letter_symbols)))

    # Remove exclusive stop
    return sequences, chain_start_indices[:-1]
