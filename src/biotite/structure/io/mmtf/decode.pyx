# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import numpy as np

__all__ = ["decode_array"]


def decode_array(codec, array):
    pass


def _delta_decode(array):
    return array.cumsum()


def _run_length_decode(array):
    values = array[::2]
    lengths = array[1::2]
    output = np.zeros(np.sum(lengths), dtype=np.int32)