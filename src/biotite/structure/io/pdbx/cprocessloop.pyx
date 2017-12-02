# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import numpy as np
import shlex

def c_process_looped(list lines not None, bint whitepace_values):
    cdef str line = ""
    cdef list values = []
    cdef str value = ""
    cdef dict category_dict = {}
    cdef list keys = []
    # Array index
    cdef int i = 0
    # Dictionary key index
    cdef int j = 0
    # Line index
    cdef int k = 0
    for k in range(len(lines)):
        line = lines[k]
        if line[0] == "_":
            # Key line
            key = line.split(".")[1]
            keys.append(key)
            # Pessimistic array allocation
            # numpy array filled with strings
            category_dict[key] = np.zeros(len(lines), dtype=object)
            keys_length = len(keys)
        else:
            # If whitespace is expected in quote protected values,
            # use standard shlex split
            # Otherwise use much more faster whitespace split
            # and quote removal if applicable,
            # bypassing the slow shlex module 
            if whitepace_values:
                values = shlex.split(line)
            else:
                values = _split(line)
            for value in values:
                category_dict[keys[j]][i] = value
                j += 1
                if j == keys_length:
                    # If all keys have been filled with a value,
                    # restart with first key with incremented index
                    j = 0
                    i += 1
    for key in category_dict.keys():
        # Trim to correct size
        category_dict[key] = category_dict[key][:i]
    return category_dict


cdef list _split(str line):
    cdef list values = line.split()
    cdef int i
    for i in range(len(values)):
        # Remove quotes
        if ((values[i][0] == '"' and values[i][-1] == '"') or
            (values[i][0] == "'" and values[i][-1] == "'")):
                values[i] = values[i][1:-1]
    return values