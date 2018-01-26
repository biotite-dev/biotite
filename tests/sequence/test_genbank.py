# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import numpy as np
import os
import os.path
from .util import data_dir
import pytest

def test_access():
    