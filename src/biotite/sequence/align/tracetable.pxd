# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

cimport cython
cimport numpy as np


cdef int follow_trace(np.uint8_t[:,:] trace_table,
                      bint banded,
                      int i, int j, int pos,
                      np.int64_t[:,:] trace,
                      list trace_list,
                      int state,
                      int* curr_trace_count,
                      int max_trace_count,
                      int lower_diag, int upper_diag) except -1