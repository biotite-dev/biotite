# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

cimport cython
cimport numpy as np


cdef np.uint8_t get_trace_linear(np.int32_t match_score,
                                 np.int32_t gap_left_score,
                                 np.int32_t gap_top_score,
                                 np.int32_t *max_score)

cdef np.uint8_t get_trace_affine(np.int32_t match_to_match_score,
                                 np.int32_t gap_left_to_match_score,
                                 np.int32_t gap_top_to_match_score,
                                 np.int32_t match_to_gap_left_score,
                                 np.int32_t gap_left_to_gap_left_score,
                                 np.int32_t match_to_gap_top_score,
                                 np.int32_t gap_top_to_gap_top_score,
                                 np.int32_t *max_match_score,
                                 np.int32_t *max_gap_left_score,
                                 np.int32_t *max_gap_top_score)

cdef int follow_trace(np.uint8_t[:,:] trace_table,
                      bint banded,
                      int i, int j, int pos,
                      np.int64_t[:,:] trace,
                      list trace_list,
                      int state,
                      int* curr_trace_count,
                      int max_trace_count,
                      int lower_diag, int upper_diag) except -1