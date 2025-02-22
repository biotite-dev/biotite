# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides functionality for sequence alignments.

The two central classes involved are :class:`SubstitutionMatrix` and
:class:`Alignment`:

Every function that performs an alignment requires a
:class:`SubstitutionMatrix` that provides similarity scores for each
symbol combination of two alphabets (usually both alphabets are equal).
The alphabets in the :class:`SubstitutionMatrix` must match or extend
the alphabets of the sequences to be aligned.

An alignment cannot be directly represented as list of :class:`Sequence`
objects, since a gap indicates the absence of any symbol.
Instead, the aligning functions return one or more :class:`Alignment`
instances.
These objects contain the original sequences and a trace, that describe
which positions (indices) in the sequences are aligned.
Optionally they also contain the similarity score.

The aligning functions :func:`align_optimal()` and
:func:`align_multiple()` cover most use cases for pairwise and multiple
sequence alignments respectively.

However, *Biotite* provides also a modular system to build performant
heuristic alignment search methods, e.g. for finding homologies in a sequence
database or map reads to a genome.
The table below summarizes those provided functionalities.
The typical stages in alignment search, where those functionalities are used,
are arranged from top to bottom.

.. grid::
    :gutter: 0
    :class-container: sd-text-center

    .. grid-item::
        :padding: 2
        :outline:
        :columns: 3

        **Entire k-mer set**

    .. grid-item::
        :padding: 2
        :outline:
        :columns: 9

        .. grid::
            :margin: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                **k-mer subset selection**

            .. grid-item::
                :padding: 2
                :columns: 4

                Minimizers

                :class:`MinimizerSelector`

            .. grid-item::
                :padding: 2
                :columns: 4

                Syncmers

                :class:`SyncmerSelector`

                :class:`CachedSyncmerSelector`

            .. grid-item::
                :padding: 2
                :columns: 4

                Mincode

                :class:`MincodeSelector`

    .. grid-item::
        :padding: 2
        :outline:
        :columns: 12

        .. grid::
            :margin: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                **k-mer indexing and matching**

            .. grid-item::
                :padding: 2
                :columns: 6

                Perfect hashing

                :class:`KmerTable`

            .. grid-item::
                :padding: 2
                :columns: 6

                Space-efficient hashing

                :class:`BucketKmerTable`

                :func:`bucket_number()`

    .. grid-item::
        :padding: 2
        :outline:
        :columns: 12

        .. grid::
            :margin: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                **Ungapped seed extension**

                :class:`align_local_ungapped()`

    .. grid-item::
        :padding: 2
        :outline:
        :columns: 12

        .. grid::
            :margin: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                **Gapped alignment**

            .. grid-item::
                :padding: 2
                :columns: 6

                Banded local/semiglobal alignment

                :class:`align_banded()`

            .. grid-item::
                :padding: 2
                :columns: 6

                Local alignment (*X-drop*)

                :class:`align_local_gapped()`

    .. grid-item::
        :padding: 2
        :outline:
        :columns: 12

        .. grid::
            :margin: 0

            .. grid-item::
                :padding: 2
                :columns: 12

                **Significance evaluation**

                :class:`EValueEstimator`
"""

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"

from .alignment import *
from .banded import *
from .buckets import *
from .cigar import *
from .kmeralphabet import *
from .kmersimilarity import *
from .kmertable import *
from .localgapped import *
from .localungapped import *
from .matrix import *
from .multiple import *
from .pairwise import *
from .permutation import *
from .selector import *
from .statistics import *
