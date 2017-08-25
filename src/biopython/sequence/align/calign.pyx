# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

cimport cython
import numpy as np
cimport numpy as np


ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64

@cython.boundscheck(False)
@cython.wraparound(False)
def c_fill_align_table(uint8[:] code1 not None,
                     uint8[:] code2 not None,
                     int32[:,:] matrix not None,
                     int32[:,:] score_table not None,
                     uint8[:,:] trace_table not None,
                     int gap_opening,
                     int gap_extension,
                     bint local):    
    cdef int i, j
    cdef int max_i = score_table.shape[0]
    cdef int max_j = score_table.shape[1]
    
    cdef int32 from_diag, from_left, from_top
    
    cdef uint8 trace
    cdef int32 score
    
    # Starts at 1 since the first row and column are already filled
    i = 1
    while i < max_i:
        j = 1
        while j < max_j:
            # Evaluate score from diagonal direction
            from_diag = score_table[i-1, j-1]
            # -1 is necessary due to the shift of the sequences
            # to the bottom/right in the table
            from_diag += matrix[code1[i-1], code2[j-1]]
            # Evaluate score from left direction
            from_left = score_table[i, j-1]
            if trace_table[i, j-1] & 2:
                from_left += gap_extension
            else:
                from_left += gap_opening
            # Evaluate score from top direction
            from_top = score_table[i-1, j]
            if trace_table[i-1, j] & 4:
                from_top += gap_extension
            else:
                from_top += gap_opening
            
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
            j +=1
        i += 1
    