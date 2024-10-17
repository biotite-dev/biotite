# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Conversion of structures into the *CLePAPS* structural alphabet.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Patrick Kunzmann"
__all__ = ["ClepapsSequence", "to_clepaps"]

import functools
from collections.abc import Iterable
from typing import Any, ClassVar, cast
import numpy as np
from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.sequence import Sequence
from biotite.structure.atoms import AtomArray
from biotite.structure.chains import get_chain_starts
from biotite.structure.filter import filter_amino_acids
from biotite.structure.geometry import angle, dihedral
from biotite.structure.util import coord_for_atom_name_per_residue
from biotite.typing import K, M, N, NDArray1, NDArray2, NDArray3

# The parameters of the Gaussian mixture model for the 17 conformational letters,
# taken from Table 1 of the CLePAPS paper.
# Each letter is a cluster in the phase space spanned by the three angles
# (theta, tau, theta') of four consecutive CA atoms:
# the two *bending* angles theta, theta' and the torsion angle tau.
# The rows are ordered according to the letters of the alphabet defined below (A-Q).

# The cluster centers (mu)
CLEPAPS_CENTERS = np.array(
    [
       [ 1.02, -2.  ,  1.55],
       [ 1.06, -2.94,  1.34],
       [ 1.01, -1.88,  1.14],
       [ 0.79, -2.3 ,  1.03],
       [ 1.02, -2.98,  0.95],
       [ 1.09, -2.72,  0.91],
       [ 1.49,  2.09,  1.05],
       [ 1.55,  0.88,  1.55],
       [ 1.52,  0.83,  1.52],
       [ 1.58,  1.05,  1.55],
       [ 1.48,  0.7 ,  1.43],
       [ 1.4 ,  0.75,  0.84],
       [ 1.47,  1.64,  1.44],
       [ 1.12,  0.14,  1.49],
       [ 1.54, -1.89,  1.48],
       [ 1.24, -2.98,  1.49],
       [ 0.86, -0.37,  1.01],
    ]
)  # fmt: skip

# The cluster weights (pi)
CLEPAPS_WEIGHTS = np.array(
    [ 4.3, 3.9, 7.5, 5.4, 11.6, 4.9, 5.6, 16.2, 8.2, 7.3, 5.9, 5.3, 3.7, 3.1, 2.1, 3.2, 1.7 ]
)  # fmt: skip

# The inverse covariance matrices (Sigma^-1) of the clusters.
# Table 1 lists the six unique entries of each symmetric 3x3 matrix in the
# order (theta-theta, tau-theta, tau-tau, theta'-theta, theta'-tau, theta'-theta').
_CLEPAPS_INV_COVARIANCES = np.array(
    [
       [  30.5,   9.1,    8.7,    6.0,     5.7,  228.6],
       [  26.9,   4.6,    4.9,    9.5,    -5.0,   54.3],
       [  28.0,   4.1,    6.2,    2.3,    -5.1,   69.4],
       [  56.2,   3.8,    4.2,  -10.8,    -2.1,   30.1],
       [  34.3,   4.2,   15.2,   -9.3,   -22.5,   56.8],
       [  24.1,   1.9,   10.9,  -11.2,    -8.8,   53.0],
       [ 163.9,   0.6,    3.8,    2.0,    -3.7,   32.3],
       [ 706.6, -93.9,  245.5,  128.9,  -171.8,  786.1],
       [ 275.4, -28.3,   84.3,  106.9,   -46.1,  214.4],
       [ 314.3, -10.3,   46.0,   37.8,   -70.0,  332.8],
       [  73.8, -13.7,   21.5,   15.5,   -25.3,   75.7],
       [  43.7,   2.5,    1.4,   -7.0,    -2.9,   34.5],
       [  72.9,   2.1,    4.8,    1.9,    -7.9,   72.9],
       [  25.3,   3.2,    3.1,    9.9,     0.9,   83.0],
       [ 170.8,  -0.7,    3.7,   -4.1,     3.1,   98.7],
       [  48.0,   8.2,    7.3,   -4.9,    -6.6,  155.6],
       [  28.4,   1.5,    1.2,    3.4,     0.1,   19.5],
    ]
)  # fmt: skip


class ClepapsSequence(Sequence):
    """
    Representation of a structure in the *CLePAPS* structural alphabet.
    :footcite:`Wang2008`

    Parameters
    ----------
    sequence : iterable object, optional
        The *CLePAPS* sequence.
        This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.

    See Also
    --------
    to_clepaps : Create *CLePAPS* sequences from a structure.

    References
    ----------

    .. footbibliography::
    """

    alphabet: ClassVar[LetterAlphabet] = LetterAlphabet("ABCDEFGHIJKLMNOPQR")
    unknown_symbol = "R"

    def __init__(self, sequence: str | Iterable[str] = "") -> None:
        if isinstance(sequence, str):
            sequence = sequence.upper()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        super().__init__(sequence)

    def get_alphabet(self) -> LetterAlphabet:
        return ClepapsSequence.alphabet


def to_clepaps(
    atoms: AtomArray[N],
) -> tuple[list[ClepapsSequence], NDArray1[K, np.integer]]:
    """
    Encode each chain in the given structure to the *CLePAPS* structural
    alphabet.
    :footcite:`Wang2008`

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode.
        May contain multiple chains.

    Returns
    -------
    sequences : list of Sequence, length=n
        The encoded *CLePAPS* sequence for each peptide chain in the structure.
    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> sequences, chain_starts = to_clepaps(atom_array)
    >>> print(sequences[0])
    RRHHHHHIHOCNJMAGDCCR
    """
    sequences = []
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        sequences.append(_to_clepaps(chain))
    return sequences, chain_start_indices[:-1]


def _to_clepaps(chain: AtomArray[N]) -> ClepapsSequence:
    amino_acid_mask = filter_amino_acids(chain)

    # Coordinates for dihedral angle calculation
    (coord_ca,) = coord_for_atom_name_per_residue(
        chain,
        ("CA",),
        amino_acid_mask,
    )

    # The two angles are *bending* angles, i.e. the angle between successive CA-CA
    # pseudobond vectors, which is the supplement of the CA-CA-CA valence angle
    # returned by `angle()`
    bending = cast(
        "NDArray1[Any, np.floating]",
        np.pi - angle(coord_ca[:-2], coord_ca[1:-1], coord_ca[2:]),
    )
    theta_1 = bending[:-1]
    theta_2 = bending[1:]
    tau = dihedral(coord_ca[:-3], coord_ca[1:-2], coord_ca[2:-1], coord_ca[3:])
    clepaps_angles = np.stack([theta_1, tau, theta_2], axis=-1)

    # Assign each angle triple to the most likely cluster of the Gaussian mixture model
    # Due to the definition of Biotite symbol codes,
    # the cluster index is directly the symbol code
    clepaps_seq_code = _assign_cluster(
        clepaps_angles,
        CLEPAPS_CENTERS,
        _log_weights(),
        _inverse_covariances(),
        # Only 'tau' is periodic
        periodic_dim=np.array([False, True, False]),
    )
    # Undefined clusters (from missing atoms/residues or chain ends) become the unknown
    # symbol
    clepaps_seq_code[clepaps_seq_code == -1] = ClepapsSequence.alphabet.encode(
        ClepapsSequence.unknown_symbol
    )
    # Put the array of symbol codes into actual sequence objects
    clepaps_sequence = ClepapsSequence()
    # Since every symbols comprises 4 residues, the sequence length is shortened by 3
    # By definition of CLePAPS, the first two and the last residue are undefined
    clepaps_sequence.code = np.full(
        coord_ca.shape[0],
        ClepapsSequence.alphabet.encode(ClepapsSequence.unknown_symbol),
    )
    clepaps_sequence.code[2:-1] = clepaps_seq_code
    return clepaps_sequence


def _assign_cluster(
    values: NDArray2[N, M, np.floating],
    centers: NDArray2[K, M, np.floating],
    weights: NDArray1[K, np.floating],
    inv_covariances: NDArray3[K, M, M, np.floating],
    periodic_dim: NDArray1[M, np.bool_] | None = None,
) -> NDArray1[N, np.integer]:
    """
    Assign each value to the most likely cluster of a Gaussian mixture model.

    Parameters
    ----------
    values : ndarray, shape=(n, d)
        The values to be assigned to clusters.
    centers : ndarray, shape=(k, d)
        The cluster centers (mu).
    weights : ndarray, shape=(k,)
        The cluster-dependent additive term of the log-density,
        i.e. ``log(pi) + log(sqrt(|Sigma^-1|))``.
    inv_covariances : ndarray, shape=(k, d, d)
        The inverse covariance matrices (Sigma^-1) of the clusters.
    periodic_dim : ndarray, shape=(d,), dtype=bool, optional
        A boolean mask over the value dimensions.
        The difference along each dimension marked as ``True`` is wrapped to
        ``[-pi, pi]`` (e.g. for torsion angles).
        By default, no dimension is treated as periodic.

    Returns
    -------
    cluster_indices : ndarray, shape=(n,), dtype=int
        The index of the most likely cluster for each value.
        Set to ``-1`` for values that contain *NaN*.
    """
    diff = centers[:, np.newaxis, :] - values[np.newaxis, :, :]
    if periodic_dim is not None:
        # Wrap the difference along periodic dimensions to [-pi, pi], so that values
        # near +-pi do not get a spurious large distance
        wrapped_diff = (diff + np.pi) % (2 * np.pi) - np.pi
        diff = np.where(periodic_dim, wrapped_diff, diff)
    # The log-density of a cluster is `weights - 0.5 * D`,
    # with `D` being the squared Mahalanobis distance to the cluster center
    mahalanobis = np.einsum("kni,kij,knj->kn", diff, inv_covariances, diff)
    log_likelihood = weights[:, np.newaxis] - 0.5 * mahalanobis
    # A value containing NaN yields a NaN log-likelihood for every cluster
    cluster_indices = np.full(len(values), -1, dtype=int)
    available_mask = ~np.isnan(log_likelihood).any(axis=0)
    cluster_indices[available_mask] = np.argmax(
        log_likelihood[:, available_mask], axis=0
    )
    return cluster_indices  # pyright: ignore[reportReturnType]


@functools.cache
def _inverse_covariances() -> NDArray3[K, M, M, np.floating]:
    """
    Get the inverse covariance matrices (``Sigma^-1``) of the clusters as symmetric
    3x3 matrices, expanded from the six unique entries listed in the paper.
    """
    th_th, ta_th, ta_ta, thp_th, thp_ta, thp_thp = _CLEPAPS_INV_COVARIANCES.T
    return np.array(
        [
            [th_th, ta_th, thp_th],
            [ta_th, ta_ta, thp_ta],
            [thp_th, thp_ta, thp_thp],
        ]
    ).transpose(2, 0, 1)


@functools.cache
def _log_weights() -> NDArray1[K, np.floating]:
    """
    Get the cluster-dependent additive term of the log-density of the Gaussian mixture
    model, i.e. ``log(pi) + log(sqrt(|Sigma^-1|))``.

    The remaining term of the log-density that is constant across clusters is omitted,
    as it does not affect the choice of the most likely cluster.
    """
    return np.log(CLEPAPS_WEIGHTS) + 0.5 * np.log(np.linalg.det(_inverse_covariances()))
