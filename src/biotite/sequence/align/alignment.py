# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import numpy as np
import copy
import textwrap
    

__all__ = ["Alignment"]


class Alignment(object):
    """
    An `Alignment` object stores information about which symbols of
    *n* sequences are aligned to each other and it stores the
    corresponding alignment score.
    
    Rather than saving a list of aligned symbols, this class saves the
    original *n* sequences, that were aligned, and a so called *trace*,
    which indicate the aligned symbols of these sequences.
    The trace is a *(m x n)* `ndarray` with alignment length *m*, where
    the first dimension is the position in the alignment and the second
    dimension represents the sequence.
    Each element of the trace is the index in the corresponding
    sequence. A gap is represented by the value -1.
    
    Furthermore this class provides multiple utility functions for
    conversion into strings in order to make the alignment human
    readable.
    
    Unless an `Alignment` object is the result of an multiple sequence
    alignment, the object will contain only two sequences.
    
    All attributes of this class are publicly accessible.
    
    Parameters
    ----------
    sequences : list
        A list of aligned sequences.
    trace : ndarray, dtype=int, shape=(n,m)
        The alignment trace.
    score : int
        Alignment score.
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("CGTT")
    >>> seq2 = NucleotideSequence("TTGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_global(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTT--
    --TTGC
    >>> print(ali.trace)
    [[ 0 -1]
     [ 1 -1]
     [ 2  0]
     [ 3  1]
     [-1  2]
     [-1  3]]
    >>> print(ali[1:4].trace)
    [[ 1 -1]
     [ 2  0]
     [ 3  1]]
    >>> print(ali[1:4,0].trace)
    [1 2 3]
    """
    
    
    def __init__(self, sequences, trace, score):
        self.sequences = copy.copy(sequences)
        self.trace = trace
        self.score = score
    
    def _gapped_str(self, seq_index):
        seq_str = ""
        for i in range(len(self.trace)):
            j = self.trace[i][seq_index]
            if j != -1:
                seq_str += self.sequences[seq_index][j]
            else:
                seq_str += "-"
        return seq_str
    
    def __str__(self):
        # Check if any of the sequences
        # has an non-single letter alphabet
        all_single_letter = True
        for seq in self.sequences:
            if not seq.get_alphabet().is_letter_alphabet():
                all_single_letter = False
        if all_single_letter:
            # First dimension: sequence number,
            # second dimension: line number
            seq_str_lines_list = []
            wrapper = textwrap.TextWrapper(break_on_hyphens=False)
            for i in range(len(self.sequences)):
                seq_str_lines_list.append(wrapper.wrap(self._gapped_str(i)))
            ali_str = ""
            for row_i in range(len(seq_str_lines_list[0])):
                for seq_j in range(len(seq_str_lines_list)):
                    ali_str += seq_str_lines_list[seq_j][row_i] + "\n"
                ali_str += "\n"
            # Remove final line breaks
            return ali_str[:-2]
        else:
            super().__str__()
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Alignment(self.sequences[index[1]], self.trace[index],
                             self.score)
        elif isinstance(index, slice):
            return Alignment(self.sequences[:], self.trace[index], self.score)
        else:
            raise IndexError("Invalid alignment index")
    
    @staticmethod
    def trace_from_strings(seq_str_list):
        """
        Create a trace from strings that represent aligned sequences.
        
        Parameters
        ----------
        seq_str_list : list
            The strings, where each each one represents a sequence
            in an alignment.
        
        Returns
        -------
        trace : ndarray, dtype=int, shype=(n,2)
            The created trace.
        """
        seq_i = np.zeros(len(seq_str_list))
        trace = np.full(( len(seq_str_list[0]), len(seq_str_list) ),
                        -1, dtype=int)
        # Get length of string (same length for all strings)
        # rather than length of list
        for pos_i in range(len(seq_str_list[0])):
            for str_j in range(len(seq_str_list)):
                if seq_str_list[str_j][pos_i] == "-":
                    trace[pos_i, str_j] = -1
                else:
                    trace[pos_i, str_j] = seq_i[str_j]
                    seq_i[str_j] += 1
        return trace