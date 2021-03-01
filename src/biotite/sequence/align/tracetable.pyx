# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A module for Biotite's internal use only.
Contains C-functions for handling trace tables in a reuasable way for
# different alignment functions.
"""

__author__ = "Patrick Kunzmann"

cimport cython
cimport numpy as np

import numpy as np

# A trace table saves the directions a cell came from
# A "1" in the corresponding bit in the trace table means
# the cell came from this direction
# Values for linear gap penalty (one score table)
#     bit 1 -> 1  -> diagonal -> alignment of symbols
#     bit 2 -> 2  -> left     -> gap in first sequence
#     bit 3 -> 4  -> top      -> gap in second sequence
# Values for affine gap penalty (three score tables)
#     bit 1 -> 1  -> match - match transition
#     bit 2 -> 2  -> seq 1 gap - match transition
#     bit 3 -> 4  -> seq 2 gap - match transition
#     bit 4 -> 8  -> match - seq 1 gap transition
#     bit 5 -> 16 -> seq 1 gap - seq 1 gap transition
#     bit 6 -> 32 -> match - seq 2 gap transition
#     bit 7 -> 64 -> seq 2 gap - seq 2 gap transition
DEF MATCH    = 1
DEF GAP_LEFT = 2
DEF GAP_TOP  = 4
DEF MATCH_TO_MATCH       = 1
DEF GAP_LEFT_TO_MATCH    = 2
DEF GAP_TOP_TO_MATCH     = 4
DEF MATCH_TO_GAP_LEFT    = 8
DEF GAP_LEFT_TO_GAP_LEFT = 16
DEF MATCH_TO_GAP_TOP     = 32
DEF GAP_TOP_TO_GAP_TOP   = 64


# The state specifies the table the traceback is currently in
DEF NO_STATE = 0 # For linear gap penalty
DEF MATCH_STATE = 1
DEF GAP_LEFT_STATE = 2
DEF GAP_TOP_STATE = 3


cdef int follow_trace(np.uint8_t[:,:] trace_table,
                      bint banded,
                      int i, int j, int pos,
                      np.int64_t[:,:] trace,
                      list trace_list,
                      int state,
                      int* curr_trace_count,
                      int max_trace_count,
                      int lower_diag, int upper_diag) except -1:
    """
    Follow and return traces from a trace table.

    Parameters
    ----------
    trace_table
        A matrix containing values indicating the direction for the
        traceback.
    i, j
        The current position in the trace table.
        For the first branch, this is the start of the traceback.
        For additional branches this is the start of the respective
        branch.
    pos
        The index in the alignment trace to be created.
        For the first branch, this is 0.
        For additional branches the value of the parent branch is taken.
    trace
        The alignment trace to be filled
    trace_list
        When a trace is finished, it is appened to this list
    state
        The current table (m, g1, g2) the traceback is in, taken from
        parent branch. Always 0 when a linear gap penalty is used.
    curr_trace_count
        The current number of branches. The value is a pointer, so that
        updating this value propagates the value to all other branches
    max_trace_count
        The maximum number of branches created. When the number of
        branches reaches this value, no new branches are created.
    """
    
    cdef list next_indices
    cdef list next_states
    cdef int trace_value
    cdef int k
    cdef int seq_i, seq_j
    cdef int i_match, i_gap_left, i_gap_top
    cdef int j_match, j_gap_left, j_gap_top
    
    if state == NO_STATE:
        # Linear gap penalty
        # Trace table has a 0 -> no trace direction -> break loop
        # The '0'-cell itself is also not included in the traceback
        while trace_table[i,j] != 0:
            if banded:
                seq_i = i - 1
                seq_j = j + seq_i + lower_diag - 1
                i_match, i_gap_left, i_gap_top = i-1, i,   i-1
                j_match, j_gap_left, j_gap_top = j  , j-1, j+1
            else:
                # -1 is necessary due to the shift of the sequences
                # to the bottom/right in the table
                seq_i = i - 1
                seq_j = j - 1
                i_match, i_gap_left, i_gap_top = i-1, i,   i-1
                j_match, j_gap_left, j_gap_top = j-1, j-1, j
            trace[pos, 0] = seq_i
            trace[pos, 1] = seq_j
            pos += 1
            # Traces may split
            next_indices = []
            trace_value = trace_table[i,j]
            if trace_value & MATCH:
                next_indices.append((i_match, j_match))
            if trace_value & GAP_LEFT:
                next_indices.append((i_gap_left, j_gap_left))
            if trace_value & GAP_TOP:
                next_indices.append((i_gap_top, j_gap_top))
            # Trace branching
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                if curr_trace_count[0] < max_trace_count:
                    curr_trace_count[0] += 1
                    new_i, new_j = next_indices[k]
                    follow_trace(
                        trace_table, banded, new_i, new_j, pos,
                        np.copy(trace), trace_list, 0,
                        curr_trace_count, max_trace_count,
                        lower_diag, upper_diag
                    )
            # Continue in this method with indices[0]
            i, j = next_indices[0]
    else:
        # Affine gap penalty
        # -> check only for the current state whether the trace ends
        while (
            (state == MATCH_STATE    and trace_table[i,j] & (MATCH_TO_MATCH | GAP_LEFT_TO_MATCH | GAP_TOP_TO_MATCH) != 0) or
            (state == GAP_LEFT_STATE and trace_table[i,j] & (MATCH_TO_GAP_LEFT | GAP_LEFT_TO_GAP_LEFT) != 0) or
            (state == GAP_TOP_STATE  and trace_table[i,j] & (MATCH_TO_GAP_TOP | GAP_TOP_TO_GAP_TOP) != 0)
        ):
            if banded:
                seq_i = i - 1
                seq_j = j + seq_i + lower_diag - 1
                i_match, i_gap_left, i_gap_top = i-1, i,   i-1
                j_match, j_gap_left, j_gap_top = j  , j-1, j+1
            else:
                # -1 is necessary due to the shift of the sequences
                # to the bottom/right in the table
                seq_i = i - 1
                seq_j = j - 1
                i_match, i_gap_left, i_gap_top = i-1, i,   i-1
                j_match, j_gap_left, j_gap_top = j-1, j-1, j
            trace[pos, 0] = seq_i
            trace[pos, 1] = seq_j
            pos += 1
            next_indices = []
            next_states = []
            # Get value of trace corresponding to current state
            # = table trace is currently in
            if state == MATCH_STATE:
                trace_value = trace_table[i,j] & (MATCH_TO_MATCH | GAP_LEFT_TO_MATCH | GAP_TOP_TO_MATCH)
            elif state == GAP_LEFT_STATE:
                trace_value = trace_table[i,j] & (MATCH_TO_GAP_LEFT | GAP_LEFT_TO_GAP_LEFT)
            else: # state == GAP_TOP_STATE:
                trace_value = trace_table[i,j] & (MATCH_TO_GAP_TOP | GAP_TOP_TO_GAP_TOP)
            # Determine indices and state of next trace step
            if trace_value & MATCH_TO_MATCH:
                next_indices.append((i_match, j_match))
                next_states.append(MATCH_STATE)
            if trace_value & GAP_LEFT_TO_MATCH:
                next_indices.append((i_match, j_match))
                next_states.append(GAP_LEFT_STATE)
            if trace_value & GAP_TOP_TO_MATCH:
                next_indices.append((i_match, j_match))
                next_states.append(GAP_TOP_STATE)
            if trace_value & MATCH_TO_GAP_LEFT:
                next_indices.append((i_gap_left, j_gap_left))
                next_states.append(MATCH_STATE)
            if trace_value & GAP_LEFT_TO_GAP_LEFT:
                next_indices.append((i_gap_left, j_gap_left))
                next_states.append(GAP_LEFT_STATE)
            if trace_value & MATCH_TO_GAP_TOP:
                next_indices.append((i_gap_top, j_gap_top))
                next_states.append(MATCH_STATE)
            if trace_value & GAP_TOP_TO_GAP_TOP:
                next_indices.append((i_gap_top, j_gap_top))
                next_states.append(GAP_TOP_STATE)
            # Trace branching
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                if curr_trace_count[0] < max_trace_count:
                    curr_trace_count[0] += 1
                    new_i, new_j = next_indices[k]
                    new_state = next_states[k]
                    follow_trace(
                        trace_table, banded, new_i, new_j, pos,
                        np.copy(trace), trace_list, new_state,
                        curr_trace_count, max_trace_count,
                        lower_diag, upper_diag
                    )
            # Continue in this method with indices[0] and states[0]
            i, j = next_indices[0]
            state = next_states[0]
    # Trim trace to correct size (delete all pure -1 entries)
    # and append to trace_list
    tr_arr = np.asarray(trace)
    trace_list.append(tr_arr[(tr_arr[:,0] != -1) | (tr_arr[:,1] != -1)])
    return 0