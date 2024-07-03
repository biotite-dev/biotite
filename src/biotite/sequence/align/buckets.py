# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["bucket_number"]

from os.path import dirname, join, realpath
import numpy as np

_primes = None


def bucket_number(n_kmers, load_factor=0.8):
    """
    Find an appropriate number of buckets for a :class:`BucketKmerTable`
    based on the number of elements (i.e. *k-mers*) that should be
    stored in the table.

    Parameters
    ----------
    n_kmers : int
        The expected number of *k-mers* that will be stored in the
        :class:`BucketKmerTable`.
        If this number deviates from the actual number of *k-mers* that
        will be stored, the load factor of the table will deviate
        by the same percentage.
    load_factor : float, optional
        The ratio of bucket number to *k-mer* number.
        The actual load factor will be lower, as the closest greater
        prime is returned (see *Notes*).

    Returns
    -------
    n_buckets : int
        The recommended number of buckets to use for a
        :class:`BucketKmerTable`, that stores `n_kmers` at the given
        `load_factor`.

    Notes
    -----
    The function returns the closest greater prime number from a
    precomputed list of primes to use as the number of buckets.
    The reason is that primer numbers have proven to be good hash table
    sizes, if the hash function is not randomized.

    Let's take unambiguous nucleotide *k-mers* as example.
    If powers of two would be used as table size (another common scheme),
    taking the modulo operation on the *k-mer* code would simply erase
    the upper bits corresponding to the first nucleotide(s) in a
    *k-mer*.
    Hence, all *k-mers* with the same suffix would be stored in the same
    bin.
    """
    global _primes
    if _primes is None:
        with open(join(dirname(realpath(__file__)), "primes.txt")) as file:
            _primes = np.array(
                [
                    int(line)
                    for line in file.read().splitlines()
                    if len(line) != 0 and line[0] != "#"
                ]
            )

    number = int(n_kmers / load_factor)
    index = np.searchsorted(_primes, number, side="left")
    if index == len(_primes):
        raise ValueError("Number of buckets too large")
    return _primes[index]
