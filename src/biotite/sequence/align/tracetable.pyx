# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A module for Biotite's internal use only.
Contains C-functions for handling trace tables in a reuasable way for
different alignment functions.
"""

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = []

cimport cython
cimport numpy as np

import numpy as np


cdef inline np.uint8_t get_trace_linear(np.int32_t match_score,
                                        np.int32_t gap_left_score,
                                        np.int32_t gap_top_score,
                                        np.int32_t *max_score):
    """
    Find maximum score from the input scores and return corresponding
    trace direction for linear gap penalty.
    """
    if match_score > gap_left_score:
        if match_score > gap_top_score:
            trace = TraceDirectionLinear.MATCH
            max_score[0] = match_score
        elif match_score == gap_top_score:
            trace = (
                TraceDirectionLinear.MATCH |
                TraceDirectionLinear.GAP_TOP
            )
            max_score[0] = match_score
        else:
            trace = TraceDirectionLinear.GAP_TOP
            max_score[0] = gap_top_score
    elif match_score == gap_left_score:
        if match_score > gap_top_score:
            trace = (
                TraceDirectionLinear.MATCH |
                TraceDirectionLinear.GAP_LEFT
            )
            max_score[0] = match_score
        elif match_score == gap_top_score:
            trace = (
                TraceDirectionLinear.MATCH |
                TraceDirectionLinear.GAP_LEFT |
                TraceDirectionLinear.GAP_TOP
            )
            max_score[0] = match_score
        else:
            trace = TraceDirectionLinear.GAP_TOP
            max_score[0] = gap_top_score
    else:
        if gap_left_score > gap_top_score:
            trace = TraceDirectionLinear.GAP_LEFT
            max_score[0] = gap_left_score
        elif gap_left_score == gap_top_score:
            trace = (
                TraceDirectionLinear.GAP_LEFT |
                TraceDirectionLinear.GAP_TOP
            )
            max_score[0] = gap_left_score
        else:
            trace = TraceDirectionLinear.GAP_TOP
            max_score[0] = gap_top_score
    
    return trace


cdef inline np.uint8_t get_trace_affine(np.int32_t match_to_match_score,
                                        np.int32_t gap_left_to_match_score,
                                        np.int32_t gap_top_to_match_score,
                                        np.int32_t match_to_gap_left_score,
                                        np.int32_t gap_left_to_gap_left_score,
                                        np.int32_t match_to_gap_top_score,
                                        np.int32_t gap_top_to_gap_top_score,
                                        np.int32_t *max_match_score,
                                        np.int32_t *max_gap_left_score,
                                        np.int32_t *max_gap_top_score):
    """
    Find maximum scores from the input scores and return corresponding
    trace direction for affine gap penalty.
    """
    # Match Table
    if match_to_match_score > gap_left_to_match_score:
        if match_to_match_score > gap_top_to_match_score:
            trace = TraceDirectionAffine.MATCH_TO_MATCH
            max_match_score[0] = match_to_match_score
        elif match_to_match_score == gap_top_to_match_score:
            trace = (
                TraceDirectionAffine.MATCH_TO_MATCH |
                TraceDirectionAffine.GAP_TOP_TO_MATCH
            )
            max_match_score[0] = match_to_match_score
        else:
            trace = TraceDirectionAffine.GAP_TOP_TO_MATCH
            max_match_score[0] = gap_top_to_match_score
    elif match_to_match_score == gap_left_to_match_score:
        if match_to_match_score > gap_top_to_match_score:
            trace = (
                TraceDirectionAffine.MATCH_TO_MATCH | 
                TraceDirectionAffine.GAP_LEFT_TO_MATCH
            )
            max_match_score[0] = match_to_match_score
        elif match_to_match_score == gap_top_to_match_score:
            trace = (
                TraceDirectionAffine.MATCH_TO_MATCH |
                TraceDirectionAffine.GAP_LEFT_TO_MATCH |
                TraceDirectionAffine.GAP_TOP_TO_MATCH
            )
            max_match_score[0] = match_to_match_score
        else:
            trace = TraceDirectionAffine.GAP_TOP_TO_MATCH
            max_match_score[0] = gap_top_to_match_score
    else:
        if gap_left_to_match_score > gap_top_to_match_score:
            trace = TraceDirectionAffine.GAP_LEFT_TO_MATCH
            max_match_score[0] = gap_left_to_match_score
        elif gap_left_to_match_score == gap_top_to_match_score:
            trace = (
                TraceDirectionAffine.GAP_LEFT_TO_MATCH |
                TraceDirectionAffine.GAP_TOP_TO_MATCH
            )
            max_match_score[0] = gap_left_to_match_score
        else:
            trace = TraceDirectionAffine.GAP_TOP_TO_MATCH
            max_match_score[0] = gap_top_to_match_score
    
    # 'Gap left' table
    if match_to_gap_left_score > gap_left_to_gap_left_score:
        trace |= TraceDirectionAffine.MATCH_TO_GAP_LEFT
        max_gap_left_score[0] = match_to_gap_left_score
    elif match_to_gap_left_score < gap_left_to_gap_left_score:
        trace |= TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT
        max_gap_left_score[0] = gap_left_to_gap_left_score
    else:
        trace |= (
            TraceDirectionAffine.MATCH_TO_GAP_LEFT |
            TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT
        )
        max_gap_left_score[0] = match_to_gap_left_score
    
    # 'Gap right' table
    if match_to_gap_top_score > gap_top_to_gap_top_score:
        trace |= TraceDirectionAffine.MATCH_TO_GAP_TOP
        max_gap_top_score[0] = match_to_gap_top_score
    elif match_to_gap_top_score < gap_top_to_gap_top_score:
        trace |= TraceDirectionAffine.GAP_TOP_TO_GAP_TOP
        max_gap_top_score[0] = gap_top_to_gap_top_score
    else:
        trace |= (
            TraceDirectionAffine.MATCH_TO_GAP_TOP |
            TraceDirectionAffine.GAP_TOP_TO_GAP_TOP
        )
        max_gap_top_score[0] = gap_top_to_gap_top_score
    
    return trace


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
    banded
        Whether the trace table belongs to a banded alignment
    i, j
        The current position in the trace table.
        For the first branch, this is the start of the traceback.
        For additional branches this is the start of the respective
        branch.
    pos
        The current position inthe trace array to be created.
        For the first branch, this is 0.
        For additional branches the value of the parent branch is taken.
    trace
        The alignment trace array to be filled.
    trace_list
        When a trace is finished, it is appened to this list
    state
        The current score table (*match*, *gap left*, *gap top*)
        the traceback is in, taken from parent branch.
        Always 0 when a linear gap penalty is used.
    curr_trace_count
        The current number of branches. The value is a pointer, so that
        updating this value propagates the value to all other branches
    max_trace_count
        The maximum number of branches created. When the number of
        branches reaches this value, no new branches are created.
    lower_diag, upper_diag
        The lower and upper diagonal for a banded alignment.
        Unused, if `banded` is false.
    
    Returns
    -------
    int
        ``0`` if, no exception is raised, otherwisw ``-1``.
    """
    
    cdef list next_indices
    cdef list next_states
    cdef int trace_value
    cdef int k
    cdef int seq_i, seq_j
    cdef int i_match, i_gap_left, i_gap_top
    cdef int j_match, j_gap_left, j_gap_top
    
    if state == TraceState.NO_STATE:
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
            if trace_value & TraceDirectionLinear.MATCH:
                next_indices.append((i_match, j_match))
            if trace_value & TraceDirectionLinear.GAP_LEFT:
                next_indices.append((i_gap_left, j_gap_left))
            if trace_value & TraceDirectionLinear.GAP_TOP:
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
            (
                state == TraceState.MATCH_STATE and trace_table[i,j] & (
                    TraceDirectionAffine.MATCH_TO_MATCH |
                    TraceDirectionAffine.GAP_LEFT_TO_MATCH |
                    TraceDirectionAffine.GAP_TOP_TO_MATCH
                ) != 0
            ) or (
                state == TraceState.GAP_LEFT_STATE and trace_table[i,j] & (
                    TraceDirectionAffine.MATCH_TO_GAP_LEFT |
                    TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT
                ) != 0
            ) or (
                state == TraceState.GAP_TOP_STATE and trace_table[i,j] & (
                    TraceDirectionAffine.MATCH_TO_GAP_TOP |
                    TraceDirectionAffine.GAP_TOP_TO_GAP_TOP
                ) != 0
            )
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
            if state == TraceState.MATCH_STATE:
                trace_value = trace_table[i,j] & (
                    TraceDirectionAffine.MATCH_TO_MATCH |
                    TraceDirectionAffine.GAP_LEFT_TO_MATCH |
                    TraceDirectionAffine.GAP_TOP_TO_MATCH
                )
            elif state == TraceState.GAP_LEFT_STATE:
                trace_value = trace_table[i,j] & (
                    TraceDirectionAffine.MATCH_TO_GAP_LEFT |
                    TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT
                )
            else: # state == TraceState.GAP_TOP_STATE:
                trace_value = trace_table[i,j] & (
                    TraceDirectionAffine.MATCH_TO_GAP_TOP |
                    TraceDirectionAffine.GAP_TOP_TO_GAP_TOP
                )
            
            # Determine indices and state of next trace step
            if trace_value & TraceDirectionAffine.MATCH_TO_MATCH:
                next_indices.append((i_match, j_match))
                next_states.append(TraceState.MATCH_STATE)
            if trace_value & TraceDirectionAffine.GAP_LEFT_TO_MATCH:
                next_indices.append((i_match, j_match))
                next_states.append(TraceState.GAP_LEFT_STATE)
            if trace_value & TraceDirectionAffine.GAP_TOP_TO_MATCH:
                next_indices.append((i_match, j_match))
                next_states.append(TraceState.GAP_TOP_STATE)
            if trace_value & TraceDirectionAffine.MATCH_TO_GAP_LEFT:
                next_indices.append((i_gap_left, j_gap_left))
                next_states.append(TraceState.MATCH_STATE)
            if trace_value & TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT:
                next_indices.append((i_gap_left, j_gap_left))
                next_states.append(TraceState.GAP_LEFT_STATE)
            if trace_value & TraceDirectionAffine.MATCH_TO_GAP_TOP:
                next_indices.append((i_gap_top, j_gap_top))
                next_states.append(TraceState.MATCH_STATE)
            if trace_value & TraceDirectionAffine.GAP_TOP_TO_GAP_TOP:
                next_indices.append((i_gap_top, j_gap_top))
                next_states.append(TraceState.GAP_TOP_STATE)
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