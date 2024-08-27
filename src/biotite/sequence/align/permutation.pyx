# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["Permutation", "RandomPermutation", "FrequencyPermutation"]

cimport cython
cimport numpy as np

import abc
import numpy as np


ctypedef np.int64_t int64


class Permutation(metaclass=abc.ABCMeta):
    """
    Provides an order for *k-mers*, usually used by *k-mer* subset
    selectors such as :class:`MinimizerSelector`.
    The method how such order is computed depends on the concrete
    subclass of this abstract base class.

    Without a :class:`Permutation` subset selectors usually resort to
    the symbol order in the :class:`KmerAlphabet`.
    That order is often the lexicographical order, which is known to
    yield suboptimal *k-mer* selection many cases
    :footcite:`Roberts2004`.

    Attributes
    ----------
    min, max: int
        The minimum and maximum value, the permutated value
        (i.e. the return value of :meth:`permute()`)
        can take.
        Must be overriden by subclasses.
    """


    @property
    @abc.abstractmethod
    def min(self):
        pass

    @property
    @abc.abstractmethod
    def max(self):
        pass


    @abc.abstractmethod
    def permute(self, kmers):
        """
        permute(kmers)

        Give the given *k-mers* a new order.

        Parameters
        ----------
        kmers : ndarray, dtype=np.int64
            The *k-mers* to reorder given as *k-mer* code.

        Returns
        -------
        order : ndarray, dtype=np.int64
            The sort key for the new order, i.e. a *k-mer* ``A`` is
            smaller than *k-mer* ``B``, if ``order[A] < order[B]``
            The order value may not only contain positive but also
            negative integers.
            The order is unambiguous:
            If ``A != B``, then ``order[A] != order[B]``.
        """
        pass


class RandomPermutation(Permutation):
    r"""
    Provide a pseudo-randomized order for *k-mers*.

    Notes
    -----

    This class uses a simple full-period *linear congruential generator*
    (LCG) to provide pseudo-randomized values:

    .. math:: \text{order} = (a \, c_\text{k-mer} + 1) \mod 2^{64}.

    The factor :math:`a` is taken from :footcite:`Steele2021` to ensure
    full periodicity and good random behavior.
    However, note that LCGs in general do not provide perfect random
    behavior, but only *good-enough* values for this purpose.

    Attributes
    ----------
    min, max: int
        The minimum and maximum value, the permutated value
        (i.e. the return value of :meth:`permute()`)
        can take.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> kmer_alph = KmerAlphabet(NucleotideSequence.alphabet_unamb, k=2)
    >>> permutation = RandomPermutation()
    >>> # k-mer codes representing the k-mers from 'AA' to 'TT'
    >>> # in lexicographic order
    >>> kmer_codes = np.arange(len(kmer_alph))
    >>> print(kmer_codes)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
    >>> print(["".join(kmer_alph.decode(c)) for c in kmer_codes])
    ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    >>> # Shuffle order of these k-mer codes using the permutation
    >>> order = permutation.permute(kmer_codes)
    >>> print(order)
    [                   1 -3372029247567499370 -6744058495134998741
      8330656331007053504  4958627083439554133  1586597835872054762
     -1785431411695444609 -5157460659262943980 -8529489906830443351
      6545224919311608894  3173195671744109523  -198833575823389848
     -3570862823390889219 -6942892070958388590  8131822755183663655
      4759793507616164284]
    >>> # The order is not lexicographic anymore
    >>> kmer_codes = kmer_codes[np.argsort(order)]
    >>> print(["".join(kmer_alph.decode(c)) for c in kmer_codes])
    ['GA', 'TC', 'AG', 'CT', 'TA', 'AC', 'CG', 'GT', 'AA', 'CC', 'GG', 'TT', 'CA', 'GC', 'TG', 'AT']
    """

    LCG_A = 0xd1342543de82ef95
    LCG_C = 1


    @property
    def min(self):
        return np.iinfo(np.int64).min

    @property
    def max(self):
        return np.iinfo(np.int64).max


    def permute(self, kmers):
        kmers = kmers.astype(np.int64, copy=False)
        # Cast to unsigned int to harness the m=2^64 LCG
        kmers = kmers.view(np.uint64)
        # Apply LCG
        # Applying the modulo operator is not necessary
        # is the corresponding bits are truncated automatically
        permutation = RandomPermutation.LCG_A * kmers + RandomPermutation.LCG_C
        # Convert back to required signed int64
        # The resulting integer overflow changes the order, but this is
        # no problem since the order is pseudo-random anyway
        return permutation.view(np.int64)


class FrequencyPermutation(Permutation):
    """
    __init__(kmer_alphabet, counts)

    Provide an order for *k-mers* from a given
    :class:`KmerAlphabet`, such that less frequent *k-mers* are smaller
    than more frequent *k-mers*.
    The frequency of each *k-mer* can either be given directly via the
    constructor or can be computed from a :class:`KmerTable` via
    :meth:`from_table()`.

    Parameters
    ----------
    kmer_alphabet : KmerAlphabet, length=n
        The *k-mer* alphabet that defines the range of possible *k-mers*
        that should be permuted.
    counts : ndarray, shape=(n,), dtype=np.int64
        The absolute frequency, i.e. the number of occurrences, of each
        *k-mer* in `kmer_alphabet` in the sequence database of interest.
        ``counts[c] = f``, where ``c`` is the *k-mer* code and ``f`` is
        the corresponding frequency.

    Attributes
    ----------
    min, max: int
        The minimum and maximum value, the permutated value
        (i.e. the return value of :meth:`permute()`)
        can take.
    kmer_alphabet : KmerAlphabet
        The *k-mer* alphabet that defines the range of possible *k-mers*
        that should be permuted.

    Notes
    -----

    In actual sequences some sequence patterns appear in high quantity.
    When selecting a subset of *k-mers*, e.g. via
    :class:`MinimizerSelector`, it is desireable to select the
    low-frequency *informative* *k-mers* to avoid spurious matches.
    To achieve such selection this class can be used.

    This class uses a table to look up the order.
    Hence, the memory consumption is :math:`8 n^k` bytes,
    where :math:`n` is the size of the base alphabet and :math:`k` is
    the *k-mer* size.

    Examples
    --------

    >>> alphabet = LetterAlphabet("abcdr")
    >>> sequence = GeneralSequence(alphabet, "abracadabra")
    >>> kmer_table = KmerTable.from_sequences(k=2, sequences=[sequence])
    >>> print(kmer_table)
    ab: (0, 0), (0, 7)
    ac: (0, 3)
    ad: (0, 5)
    br: (0, 1), (0, 8)
    ca: (0, 4)
    da: (0, 6)
    ra: (0, 2), (0, 9)
    >>> # Create all k-mers in lexicographic order
    >>> kmer_alph = kmer_table.kmer_alphabet
    >>> kmer_codes = np.arange(0, len(kmer_alph))
    >>> print(["..."] + ["".join(kmer_alph.decode(c)) for c in kmer_codes[-10:]])
    ['...', 'da', 'db', 'dc', 'dd', 'dr', 'ra', 'rb', 'rc', 'rd', 'rr']
    >>> # After applying the permutation the k-mers are ordered
    >>> # by their frequency in the table
    >>> # -> the most frequent k-mers have low rank
    >>> permutation = FrequencyPermutation.from_table(kmer_table)
    >>> order = permutation.permute(kmer_codes)
    >>> print(order)
    [ 0 22 18 19  1  2  3  4  5 23 20  6  7  8  9 21 10 11 12 13 24 14 15 16
     17]
    >>> kmer_codes = kmer_codes[np.argsort(order)]
    >>> print(["..."] + ["".join(kmer_alph.decode(c)) for c in kmer_codes[-10:]])
    ['...', 'rc', 'rd', 'rr', 'ac', 'ad', 'ca', 'da', 'ab', 'br', 'ra']
    """

    def __init__(self, kmer_alphabet, counts):
        if len(kmer_alphabet) != len(counts):
            raise IndexError(
                f"The k-mer alphabet has {len(kmer_alphabet)} k-mers, "
                f"but {len(counts)} counts were given"
            )
        # 'order' maps a permutation to a k-mer
        # Stability is important to get the same k-mer subset selection
        # on different architectures
        order = np.argsort(counts, kind="stable")
        # '_permutation_table' should perform the reverse mapping
        self._permutation_table = _invert_mapping(order)
        self._kmer_alph = kmer_alphabet


    @property
    def min(self):
        return 0

    @property
    def max(self):
        return len(self._permutation_table) - 1

    @property
    def kmer_alphabet(self):
        return self._kmer_alph


    @staticmethod
    def from_table(kmer_table):
        """
        from_table(kmer_table)

        Create a :class:`FrequencyPermutation` from the *k-mer* counts
        of a :class:`KmerTable`.

        Parameters
        ----------
        kmer_table : KmerTable
            The *k-mer* counts are taken from this table.

        Returns
        -------
        permutation : FrequencyPermutation
            The permutation is based on the counts.
        """
        return FrequencyPermutation(
            kmer_table.kmer_alphabet, kmer_table.count()
        )


    def permute(self, kmers):
        return self._permutation_table[kmers]


@cython.boundscheck(False)
@cython.wraparound(False)
def _invert_mapping(int64[:] mapping):
    """
    If `mapping` maps an unqiue integer ``A`` to an unique integer
    ``B``, i.e. ``B = mapping[A]``, this function inverts the mapping
    so that ``A = inverted[B]``.

    Note that it is necessary that the mapping must be bijective and in
    the range ``0..n``.
    """
    cdef int64 i
    cdef int64 value

    cdef int64[:] inverted = np.empty(mapping.shape[0], dtype=np.int64)
    for i in range(mapping.shape[0]):
        value = mapping[i]
        inverted[value] = i

    return np.asarray(inverted)