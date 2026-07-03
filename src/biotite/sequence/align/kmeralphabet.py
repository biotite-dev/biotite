# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["KmerAlphabet"]

from collections.abc import Iterator
from collections.abc import Sequence as SequenceABC
from typing import Any, overload
import numpy as np
from biotite.rust.sequence.align import create_kmers as rust_create_kmers
from biotite.rust.sequence.align import split_kmers as rust_split_kmers
from biotite.sequence.alphabet import Alphabet, AlphabetError, LetterAlphabet
from biotite.typing import K, N, NDArray1, NDArray2, S


class KmerAlphabet(Alphabet[tuple[S, ...]]):
    """
    __init__(base_alphabet, k, spacing=None)

    This type of alphabet uses *k-mers* as symbols, i.e. all
    combinations of *k* symbols from its *base alphabet*.

    It's primary use is its :meth:`create_kmers()` method, that iterates
    over all overlapping *k-mers* in a :class:`Sequence` and encodes
    each one into its corresponding *k-mer* symbol code
    (*k-mer* code in short).
    This functionality is prominently used by a :class:`KmerTable` to
    find *k-mer* matches between two sequences.

    A :class:`KmerAlphabet` has :math:`n^k` different symbols, where
    :math:`n` is the number of symbols in the base alphabet.

    Parameters
    ----------
    base_alphabet : Alphabet
        The base alphabet.
        The created :class:`KmerAlphabet` contains all combinations of
        *k* symbols from this alphabet.
    k : int
        An integer greater than 1 that defines the length of the
        *k-mers*.
    spacing : None or str or list or ndarray, dtype=int, shape=(k,)
        If provided, spaced *k-mers* are used instead of continuous
        ones :footcite:`Ma2002`.
        The value contains the *informative* positions relative to the
        start of the *k-mer*, also called the *model*.
        The number of *informative* positions must equal *k*.

        If a string is given, each ``'1'`` in the string indicates an
        *informative* position.
        For a continuous *k-mer* the `spacing` would be ``'111...'``.

        If a list or array is given, it must contain unique non-negative
        integers, that indicate the *informative* positions.
        For a continuous *k-mer* the `spacing` would be
        ``[0, 1, 2,...]``.

    Attributes
    ----------
    base_alphabet : Alphabet
        The base alphabet, from which the :class:`KmerAlphabet` was
        created.
    k : int
        The length of the *k-mers*.
    spacing : None or ndarray, dtype=int
        The *k-mer* model in array form, if spaced *k-mers* are used,
        ``None`` otherwise.

    Notes
    -----
    The symbol code for a *k-mer* :math:`s` calculates as

    .. math:: RMSD = \\sum_{i=0}^{k-1} n^{k-i-1} s_i

    where :math:`n` is the length of the base alphabet.

    Hence the :class:`KmerAlphabet` sorts *k-mers* in the order of the
    base alphabet, where leading positions within the *k-mer* take
    precedence.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    Create an alphabet of nucleobase *2-mers*:

    >>> base_alphabet = NucleotideSequence.unambiguous_alphabet()
    >>> print(base_alphabet.get_symbols())
    ('A', 'C', 'G', 'T')
    >>> kmer_alphabet = KmerAlphabet(base_alphabet, 2)
    >>> print(kmer_alphabet.get_symbols())
    ('AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT')

    Encode and decode *k-mers*:

    >>> print(kmer_alphabet.encode("TC"))
    13
    >>> print(kmer_alphabet.decode(13))
    ['T' 'C']

    Fuse symbol codes from the base alphabet into a *k-mer* code
    and split the *k-mer* code back into the original symbol codes:

    >>> symbol_codes = base_alphabet.encode_multiple("TC")
    >>> print(symbol_codes)
    [3 1]
    >>> print(kmer_alphabet.fuse(symbol_codes))
    13
    >>> print(kmer_alphabet.split(13))
    [3 1]

    Encode all overlapping continuous k-mers of a sequence:

    >>> sequence = NucleotideSequence("ATTGCT")
    >>> kmer_codes = kmer_alphabet.create_kmers(sequence.code)
    >>> print(kmer_codes)
    [ 3 15 14  9  7]
    >>> print(["".join(kmer) for kmer in kmer_alphabet.decode_multiple(kmer_codes)])
    ['AT', 'TT', 'TG', 'GC', 'CT']

    Encode all overlapping k-mers using spacing:

    >>> base_alphabet = ProteinSequence.alphabet
    >>> kmer_alphabet = KmerAlphabet(base_alphabet, 3, spacing="1101")
    >>> sequence = ProteinSequence("BIQTITE")
    >>> kmer_codes = kmer_alphabet.create_kmers(sequence.code)
    >>> # Pretty print k-mers
    >>> strings = ["".join(kmer) for kmer in kmer_alphabet.decode_multiple(kmer_codes)]
    >>> print([s[0] + s[1] + "_" + s[2] for s in strings])
    ['BI_T', 'IQ_I', 'QT_T', 'TI_E']
    """

    def __init__(
        self,
        base_alphabet: Alphabet[S],
        k: int,
        spacing: str | list[int] | NDArray1[K, np.integer] | None = None,
    ) -> None:
        if not isinstance(base_alphabet, Alphabet):
            raise TypeError(
                f"Got {type(base_alphabet).__name__}, but Alphabet was expected"
            )
        if k < 2:
            raise ValueError("k must be at least 2")
        self._base_alph = base_alphabet
        self._k = k

        base_alph_len = len(self._base_alph)
        self._radix_multiplier = np.array(
            [base_alph_len**n for n in reversed(range(0, self._k))], dtype=np.int64
        )

        if spacing is None:
            self._spacing = None

        elif isinstance(spacing, str):
            self._spacing = _to_array_form(spacing)

        else:
            self._spacing = np.array(spacing, dtype=np.int64)
            self._spacing.sort()
            if (self._spacing < 0).any():
                raise ValueError("Only non-negative integers are allowed for spacing")
            if len(np.unique(self._spacing)) != len(self._spacing):
                raise ValueError("Spacing model contains duplicate values")

        if self._spacing is not None and len(self._spacing) != self._k:
            raise ValueError(
                f"Expected {self._k} informative positions, "
                f"but got {len(self._spacing)} positions in spacing"
            )

    @property
    def base_alphabet(self) -> Alphabet[S]:
        return self._base_alph

    @property
    def k(self) -> int:
        return self._k

    @property
    def spacing(self) -> NDArray1[K, np.int64] | None:
        return None if self._spacing is None else self._spacing.copy()

    def get_symbols(self) -> tuple:
        """
        get_symbols()

        Get the symbols in the alphabet.

        Returns
        -------
        symbols : tuple
            A tuple of all *k-mer* symbols, i.e. all possible
            combinations of *k* symbols from its *base alphabet*.

        Notes
        -----
        In contrast the base :class:`Alphabet` and
        :class:`LetterAlphabet` class, :class:`KmerAlphabet` does not
        hold a list of its symbols internally for performance reasons.
        Hence calling :meth:`get_symbols()` may be quite time consuming
        for large base alphabets or large *k* values, as the list needs
        to be created first.
        """
        if isinstance(self._base_alph, LetterAlphabet):
            return tuple(
                # For a `LetterAlphabet` the decoded symbols are single characters
                ["".join(self.decode(code)) for code in range(len(self))]  # pyright: ignore[reportCallIssue, reportArgumentType]
            )
        else:
            return tuple([list(self.decode(code)) for code in range(len(self))])

    def extends(self, alphabet: Alphabet[Any]) -> bool:
        # A KmerAlphabet cannot really extend another KmerAlphabet:
        # If k is not equal, all symbols are not equal
        # If the base alphabet has additional symbols, the correct
        # order is not preserved
        # A KmerAlphabet can only 'extend' another KmerAlphabet,
        # if the two alphabets are equal
        return alphabet == self

    def encode(self, symbol: SequenceABC[S]) -> int:
        return self.fuse(self._base_alph.encode_multiple(symbol))

    def decode(self, code: int) -> SequenceABC[S]:
        return self._base_alph.decode_multiple(self.split(code))

    @overload
    def fuse(self, codes: NDArray1[K, np.integer]) -> int: ...
    @overload
    def fuse(self, codes: NDArray2[N, K, np.integer]) -> NDArray1[N, np.int64]: ...
    def fuse(
        self, codes: NDArray1[K, np.integer] | NDArray2[N, K, np.integer]
    ) -> int | NDArray1[N, np.int64]:
        """
        fuse(codes)

        Get the *k-mer* code for *k* symbol codes from the base
        alphabet.

        This method can be used in a vectorized manner to obtain
        *n* *k-mer* codes from an *(n,k)* integer array.

        Parameters
        ----------
        codes : ndarray, dtype=int, shape=(k,) or shape=(n,k)
            The symbol codes from the base alphabet to be fused.

        Returns
        -------
        kmer_codes : int or ndarray, dtype=np.int64, shape=(n,)
            The fused *k-mer* code(s).

        See Also
        --------
        split
            The reverse operation.

        Examples
        --------

        >>> base_alphabet = NucleotideSequence.unambiguous_alphabet()
        >>> kmer_alphabet = KmerAlphabet(base_alphabet, 2)
        >>> symbol_codes = base_alphabet.encode_multiple("TC")
        >>> print(symbol_codes)
        [3 1]
        >>> print(kmer_alphabet.fuse(symbol_codes))
        13
        >>> print(kmer_alphabet.split(13))
        [3 1]
        """
        if codes.shape[-1] != self._k:
            raise AlphabetError(
                f"Given k-mer(s) has {codes.shape[-1]} symbols, "
                f"but alphabet expects {self._k}-mers"
            )
        if np.any(codes > len(self._base_alph)):
            raise AlphabetError("Given k-mer(s) contains invalid symbol code")

        orig_shape = codes.shape
        codes = np.atleast_2d(codes)
        kmer_code = np.sum(self._radix_multiplier * codes, axis=-1)
        # The last dimension is removed since it collpased in np.sum
        return kmer_code.reshape(orig_shape[:-1])

    @overload
    def split(self, kmer_code: int) -> NDArray1[K, np.uint64]: ...
    @overload
    def split(
        self, kmer_code: NDArray1[N, np.integer]
    ) -> NDArray2[N, K, np.uint64]: ...
    def split(
        self, kmer_code: int | NDArray1[N, np.integer]
    ) -> NDArray1[K, np.uint64] | NDArray2[N, K, np.uint64]:
        """
        split(kmer_code)

        Convert a *k-mer* code back into *k* symbol codes from the base
        alphabet.

        This method can be used in a vectorized manner to split
        *n* *k-mer* codes into an *(n,k)* integer array.

        Parameters
        ----------
        kmer_code : int or ndarray, dtype=int, shape=(n,)
            The *k-mer* code(s).

        Returns
        -------
        codes : ndarray, dtype=np.uint64, shape=(k,) or shape=(n,k)
            The split symbol codes from the base alphabet.

        See Also
        --------
        fuse
            The reverse operation.

        Examples
        --------

        >>> base_alphabet = NucleotideSequence.unambiguous_alphabet()
        >>> kmer_alphabet = KmerAlphabet(base_alphabet, 2)
        >>> symbol_codes = base_alphabet.encode_multiple("TC")
        >>> print(symbol_codes)
        [3 1]
        >>> print(kmer_alphabet.fuse(symbol_codes))
        13
        >>> print(kmer_alphabet.split(13))
        [3 1]
        """
        if np.any(kmer_code >= len(self)) or np.any(kmer_code < 0):
            raise AlphabetError("Given k-mer symbol code is invalid for this alphabet")

        orig_shape = np.shape(kmer_code)
        split_codes = rust_split_kmers(
            np.atleast_1d(kmer_code).astype(np.int64, copy=False),
            self._k,
            len(self._base_alph),
        )
        return split_codes.reshape(orig_shape + (self._k,))

    def kmer_array_length(self, length: int) -> int:
        """
        kmer_array_length(length)

        Get the length of the *k-mer* array, created by
        :meth:`create_kmers()`, if a sequence of size `length` would be
        given.

        Parameters
        ----------
        length : int
            The length of the hypothetical sequence.

        Returns
        -------
        kmer_length : int
            The length of created *k-mer* array.
        """
        if self._spacing is None:
            return length - self._k + 1
        else:
            max_offset = int(self._spacing[-1]) + 1
            return length - max_offset + 1

    def create_kmers(self, seq_code: NDArray1[N, np.integer]) -> NDArray1[K, np.int64]:
        """
        create_kmers(seq_code)

        Create *k-mer* codes for all overlapping *k-mers* in the given
        sequence code.

        Parameters
        ----------
        seq_code : ndarray, dtype={np.uint8, np.uint16, np.uint32, np.uint64}
            The sequence code to be converted into *k-mers*.

        Returns
        -------
        kmer_codes : ndarray, dtype=int64
            The symbol codes for the *k-mers*.

        Examples
        --------

        >>> base_alphabet = NucleotideSequence.unambiguous_alphabet()
        >>> kmer_alphabet = KmerAlphabet(base_alphabet, 2)
        >>> sequence = NucleotideSequence("ATTGCT")
        >>> kmer_codes = kmer_alphabet.create_kmers(sequence.code)
        >>> print(kmer_codes)
        [ 3 15 14  9  7]
        >>> print(["".join(kmer) for kmer in kmer_alphabet.decode_multiple(kmer_codes)])
        ['AT', 'TT', 'TG', 'GC', 'CT']
        """
        return rust_create_kmers(seq_code, self._k, len(self._base_alph), self._spacing)

    def __str__(self) -> str:
        return str(self.get_symbols())

    def __repr__(self) -> str:
        return (
            f"KmerAlphabet({repr(self._base_alph)}, {self._k}, {repr(self._spacing)})"
        )

    def __eq__(self, item: object) -> bool:
        if item is self:
            return True
        if not isinstance(item, KmerAlphabet):
            return False
        if self._base_alph != item._base_alph:
            return False
        if self._k != item._k:
            return False

        if self._spacing is None:
            if item._spacing is not None:
                return False
        elif np.any(self._spacing != item._spacing):
            return False

        return True

    def __hash__(self) -> int:
        spacing = None if self._spacing is None else tuple(self._spacing.tolist())
        return hash((self._base_alph, self._k, spacing))

    def __len__(self) -> int:
        return int(len(self._base_alph) ** self._k)

    def __iter__(self) -> Iterator[SequenceABC[S]]:
        # Creating all symbols is expensive
        # -> Use a generator instead
        if isinstance(self._base_alph, LetterAlphabet):
            return (
                "".join(self.decode(code))  # pyright: ignore[reportCallIssue, reportArgumentType]
                for code in range(len(self))
            )
        else:
            return (tuple(self.decode(code)) for code in range(len(self)))

    def __contains__(self, symbol: object) -> bool:
        try:
            self.fuse(self._base_alph.encode_multiple(symbol))  # pyright: ignore[reportArgumentType]
            return True
        except AlphabetError:
            return False


def _to_array_form(model_string: str) -> NDArray1[K, np.int64]:
    """
    Convert the the common string representation of a *k-mer* spacing
    model into an array, e.g. ``'1*11'`` into ``[0, 2, 3]``.
    """
    return np.array(
        [i for i in range(len(model_string)) if model_string[i] == "1"], dtype=np.int64
    )
