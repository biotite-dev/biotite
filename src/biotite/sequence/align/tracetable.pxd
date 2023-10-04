# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

cimport cython
cimport numpy as np


# A trace table saves the directions a cell came from
# A "1" in the corresponding bit in the trace table means
# the cell came from this direction

cdef enum TraceDirectionLinear:
    # Values for linear gap penalty (one score table)
    MATCH    = 1    # bit 1 -> diagonal -> alignment of symbols
    GAP_LEFT = 2    # bit 2 -> left     -> gap in first sequence
    GAP_TOP  = 4    # bit 3 -> top      -> gap in second sequence

cdef enum TraceDirectionAffine:
    # Values for affine gap penalty (three score tables)
    MATCH_TO_MATCH       = 1    # bit 1 -> match - match transition
    GAP_LEFT_TO_MATCH    = 2    # bit 2 -> seq 1 gap - match transition
    GAP_TOP_TO_MATCH     = 4    # bit 3 -> seq 2 gap - match transition
    MATCH_TO_GAP_LEFT    = 8    # bit 4 -> match - seq 1 gap transition
    GAP_LEFT_TO_GAP_LEFT = 16   # bit 5 -> seq 1 gap - seq 1 gap transition
    MATCH_TO_GAP_TOP     = 32   # bit 6 -> match - seq 2 gap transition
    GAP_TOP_TO_GAP_TOP   = 64   # bit 7 -> seq 2 gap - seq 2 gap transition


cdef enum TraceState:
    # The state specifies the table the traceback is currently in
    # For linear gap penalty (only one table/state exists):
    NO_STATE = 0
    # For affine gap penalty (three tables/states exists): 
    MATCH_STATE    = 1
    GAP_LEFT_STATE = 2
    GAP_TOP_STATE  = 3


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