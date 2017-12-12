# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

cimport cython
import numpy as np
cimport numpy as np

ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.int64_t int64


cdef inline int32 int_max(int32 a, int32 b): return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def c_fill_align_table(uint8[:] code1 not None,
                       uint8[:] code2 not None,
                       int32[:,:] matrix not None,
                       uint8[:,:] trace_table not None,
                       int32[:,:] score_table not None,
                       int gap_penalty,
                       bint local):
    cdef int i, j
    cdef int32 from_diag, from_left, from_top
    cdef uint8 trace
    cdef int32 score
    
    # Starts at 1 since the first row and column are already filled
    for i in range(1, score_table.shape[0]):
        for j in range(1, score_table.shape[1]):
            # Evaluate score from diagonal direction
            # -1 is in sequence index is necessary
            # due to the shift of the sequences
            # to the bottom/right in the table
            from_diag = score_table[i-1, j-1] + matrix[code1[i-1], code2[j-1]]
            # Evaluate score from left direction
            from_left = score_table[i, j-1] + gap_penalty
            # Evaluate score from top direction
            from_top = score_table[i-1, j] + gap_penalty
            
            # Find maximum
            if from_diag > from_left:
                if from_diag > from_top:
                    trace, score = 1, from_diag
                elif from_diag == from_top:
                    trace, score = 5, from_diag
                else:
                    trace, score = 4, from_top
            elif from_diag == from_left:
                if from_diag > from_top:
                    trace, score = 3, from_diag
                elif from_diag == from_top:
                    trace, score = 7, from_diag
                else:
                    trace, score =  4, from_top
            else:
                if from_left > from_top:
                    trace, score = 2, from_left
                elif from_left == from_top:
                    trace, score = 6, from_diag
                else:
                    trace, score = 4, from_top
            
            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then 0 is saved on the field and the trace ends here
            if local == True and score <= 0:
                score_table[i,j] = 0
            else:
                score_table[i,j] = score
                trace_table[i,j] = trace


@cython.boundscheck(False)
@cython.wraparound(False)
def c_fill_align_table_affine(uint8[:] code1 not None,
                              uint8[:] code2 not None,
                              int32[:,:] matrix not None,
                              uint8[:,:] trace_table not None,
                              int32[:,:] m_table not None,
                              int32[:,:] g1_table not None,
                              int32[:,:] g2_table not None,
                              int gap_open,
                              int gap_ext,
                              bint local):
    cdef int i, j
    cdef int32 mm_score, g1m_score, g2m_score
    cdef int32 mg1_score, g1g1_score
    cdef int32 mg2_score, g2g2_score
    cdef uint8 trace
    cdef int32 m_score, g1_score, g2_score
    cdef int32 similarity
    
    for i in range(1, trace_table.shape[0]):
        for j in range(1, trace_table.shape[1]):
            # Calculate the scores for possible transitions
            # into the current cell
            similarity = matrix[code1[i-1], code2[j-1]]
            mm_score = m_table[i-1,j-1] + similarity
            g1m_score = g1_table[i-1,j-1] + similarity
            g2m_score = g2_table[i-1,j-1] + similarity
            # No transition from g1_table to g2_table and vice versa
            # Since this would mean adjacent gaps in both sequences
            # A substitution makes more sense in this case
            mg1_score = m_table[i,j-1]  + gap_open
            g1g1_score = g1_table[i,j-1] + gap_ext
            mg2_score = m_table[i-1,j]  + gap_open
            g2g2_score = g2_table[i-1,j] + gap_ext
            
            # Find maximum score and trace
            # (similar to general gap method)
            # At first for match table (m_table)
            if mm_score > g1m_score:
                if mm_score > g2m_score:
                    trace, m_score = 1, mm_score
                elif mm_score == g2m_score:
                    trace, m_score = 5, mm_score
                else:
                    trace, m_score = 4, g2m_score
            elif mm_score == g1m_score:
                if mm_score > g2m_score:
                    trace, m_score = 3, mm_score
                elif mm_score == g2m_score:
                    trace, m_score = 7, mm_score
                else:
                    trace, m_score =  4, g2m_score
            else:
                if g1m_score > g2m_score:
                    trace, m_score = 2, g1m_score
                elif g1m_score == g2m_score:
                    trace, m_score = 6, mm_score
                else:
                    trace, m_score = 4, g2m_score
            #Secondly for gap tables (g1_table and g2_table)
            if mg1_score > g1g1_score:
                trace |= 8
                g1_score = mg1_score
            elif mg1_score < g1g1_score:
                trace |= 16
                g1_score = g1g1_score
            else:
                trace |= 24
                g1_score = mg1_score
            if mg2_score > g2g2_score:
                trace |= 32
                g2_score = mg2_score
            elif mg2_score < g2g2_score:
                trace |= 64
                g2_score = g2g2_score
            else:
                trace |= 96
                g2_score = g2g2_score
            # Fill values into tables
            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then 0 is saved on the field and the trace ends here
            if local == True:
                if m_score <= 0:
                    m_table[i,j] = 0
                    # End trace in specific table
                    # by filtering the the bits of other tables  
                    trace &= ~7
                else:
                    m_table[i,j] = m_score
                if g1_score <= 0:
                    g1_table[i,j] = 0
                    trace &= ~24
                else:
                    g1_table[i,j] = g1_score
                if g2_score <= 0:
                    g2_table[i,j] = 0
                    trace &= ~96
                else:
                    g2_table[i,j] = g2_score
            else:
                m_table[i,j] = m_score
                g1_table[i,j] = g1_score
                g2_table[i,j] = g2_score
            trace_table[i,j] = trace


cpdef c_follow_trace(uint8[:,:] trace_table,
                     int i, int j, int pos,
                     int64[:,:] trace,
                     list trace_list,
                     int state):
    cdef list next_indices
    cdef list next_states
    cdef int trace_value
    cdef int k
    
    if state == 0:
        # General gap penalty
        while trace_table[i,j] != 0:
            # -1 is necessary due to the shift of the sequences
            # to the bottom/right in the table
            trace[pos, 0] = i-1
            trace[pos, 1] = j-1
            pos += 1
            # Traces may split
            next_indices = []
            trace_value = trace_table[i,j]
            if trace_value & 1:
                next_indices.append((i-1, j-1))
            if trace_value & 2:
                next_indices.append((i, j-1))
            if trace_value & 4:
                next_indices.append((i-1, j))
            # Trace split
            # -> Recursive call of c_follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                new_i, new_j = next_indices[k]
                c_follow_trace(trace_table, new_i, new_j, pos,
                              np.copy(trace), trace_list, 0)
            # Continue in this method with indices[0]
            i, j = next_indices[0]
    else:
        # Affine gap penalty -> state specifies the table
        # we are currently in
        # The states are
        # 1 -> m
        # 2 -> g1
        # 3 -> g2
        while True:
            # Check for loop break condition
            # -> trace for current table (state) is 0
            if (   (state == 1 and trace_table[i,j] & 7 == 0)
                or (state == 2 and trace_table[i,j] & 24 == 0)
                or (state == 3 and trace_table[i,j] & 96 == 0)):
                    break
            # If no break occurred, continue as usual
            trace[pos, 0] = i-1
            trace[pos, 1] = j-1
            pos += 1
            next_indices = []
            next_states = []
            # Get value of trace respective of current state
            # = table trace is currently in
            if state == 1:
                trace_value = trace_table[i,j] & 7
            elif state == 2:
                trace_value = trace_table[i,j] & 24
            else: # state == 3:
                trace_value = trace_table[i,j] & 96
            # Determine indices and state of next trace step
            if trace_value & 1:
                next_indices.append((i-1, j-1))
                next_states.append(1)
            if trace_value & 2:
                next_indices.append((i-1, j-1))
                next_states.append(2)
            if trace_value & 4:
                next_indices.append((i-1, j-1))
                next_states.append(3)
            if trace_value & 8:
                next_indices.append((i, j-1))
                next_states.append(1)
            if trace_value & 16:
                next_indices.append((i, j-1))
                next_states.append(2)
            if trace_value & 32:
                next_indices.append((i-1, j))
                next_states.append(1)
            if trace_value & 64:
                next_indices.append((i-1, j))
                next_states.append(3)
            # Trace split
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                new_i, new_j = next_indices[k]
                new_state = next_states[k]
                c_follow_trace(trace_table, new_i, new_j, pos,
                              np.copy(trace), trace_list, new_state)
            # Continue in this method with indices[0] and states[0]
            i, j = next_indices[0]
            state = next_states[0]
    # Trim trace to correct size (delete all pure -1 entries)
    # and append to trace_list
    tr_arr = np.asarray(trace)
    trace_list.append(tr_arr[(tr_arr[:,0] != -1) | (tr_arr[:,1] != -1)])

