# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.structure as struc
import biopython.structure.io.npz as npz
import numpy as np
from os.path import join
from .util import data_dir
import pytest


def test_sse():
    file = npz.NpzFile()
    file.read(join(data_dir, "3o5r.npz"))
    array = file.get_structure()[0]
    sse = struc.annotate_sse(array, "A")
    sse_str = "".join(sse.tolist())
    print("".join(sse.tolist()))
    assert sse_str == ("caaaaaacccccccccccccbbbbbccccccbbbbccccccccccccccc"
                       "ccccccccccccbbbbbbcccccccaaaaaaaaaccccccbbbbbccccc"
                       "ccccccccccccbbbbbbbccccccccc")