# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import numpy as np
from os.path import join
from .util import data_dir
import pytest


def test_conversion():
    gb_file = gb.GenBankFile()
    gb_file.read(join(data_dir, "ec_bl21.gb"))
    annotation = gb_file.get_annotation(include_only=["CDS"])
    feature = annotation[5]
    assert feature.key == "CDS"
    assert feature.qual["gene"] == "yaaA"
    assert str(feature.locs[0]) == "< 5681-6457"