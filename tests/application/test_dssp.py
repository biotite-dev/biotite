# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure.io as strucio
from biotite.application.dssp import DsspApp
from ..structure.util import data_dir
from os.path import join
import numpy as np
import pytest

def test_dssp():
    stack = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    array = stack[0]
    print(DsspApp.annotate_sse(array).tolist())
    assert DsspApp.annotate_sse(array).tolist() == \
        ["C","H","H","H","H","H","H","H","T","T",
         "G","G","G","G","T","C","C","C","C","C"]