# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["KmerTable", "BucketKmerTable"]

from biotite.rust.sequence.align import BucketKmerTable, KmerTable

# The classes are implemented in Rust and therefore are not generic at runtime;
# allow ``KmerTable[S]`` subscription so the annotated stub works
KmerTable.__class_getitem__ = classmethod(lambda cls, params: cls)
BucketKmerTable.__class_getitem__ = classmethod(lambda cls, params: cls)
