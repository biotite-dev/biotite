# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ....file import TextFile
from ...sequence import Sequence
from ...alphabet import AlphabetError
from ...seqtypes import NucleotideSequence, ProteinSequence
import textwrap
import copy

__all__ = ["FastaFile"]


class FastaFile(TextFile):
    
    def __init__(self):
        super().__init__()
        self._header_i = []
    
    def read(self, file_name):
        super().read(file_name)
        # Filter out empty and comment lines
        self._lines = [line for line in self._lines
                       if len(line) != 0 and line[0] != ";"]
        self._find_header_lines()
    
    def get_header(self, index=0):
        # Remove '>' in the header
        return self._lines[self._header_i[index]][1:]
    
    def get_sequence(self, index=0):
        # Get lines belonging to the header
        # Check if index is last index in order to prevent IndexError
        if index+1 == len(self._header_i):
            lines = self._lines[self._header_i[index] +1 :]
        else:
            lines = self._lines[self._header_i[index] +1 :
                                self._header_i[index+1]]
        # Fill the sequence string
        seq_string = ""
        for line in lines:
            seq_string += line.strip()
        # Determine the sequence type:
        # If NucleotideSequence can be created it is a DNA sequence,
        # otherwise protein sequence
        try:
            return NucleotideSequence(seq_string)
        except AlphabetError:
            pass
        try:
            return ProteinSequence(seq_string)
        except AlphabetError:
            raise ValueError("FASTA data cannot be converted either to "
                             "NucleotideSequence nor to Protein Sequence")
            
    def set_header(self, index, string):
        processed_string = ">" + string.replace(">","").replace("\n","")
        self._lines[self._header_i[index]] = processed_string
    
    def set_sequence(self, index, sequence):
        seq_string_list = textwrap.wrap(str(sequence))
        del self._lines[self._header_i[index] +1 : self._header_i[index+1]]
        # Insert entry
        insert_index = self._header_i[index] +1
        self._lines[insert_index : insert_index] = seq_string_list
        # Amount of lines changed
        # -> look for new position of header lines
        self._find_header_lines()
    
    def add(self, header, sequence):
        header_string = ">" + header.replace(">","").replace("\n","")
        seq_string_list = textwrap.wrap(str(sequence))
        self._lines += [header_string] + seq_string_list
        # Amount of lines changed
        # -> look for new position of header lines
        self._find_header_lines()
        
    def __setitem__(self, index, value):
        if not isinstance(index, int):
            raise IndexError("FastaFile only supports integer indexing")
        self.set_header(index, value[0])
        self.set_sequence(index, value[1])
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            raise IndexError("FastaFile only supports integer indexing")
        return self.get_header(index), self.get_sequence(index)
    
    def __delitem__(self, index):
        del self._lines[self._header_i[index] : self._header_i[index+1]]
        # Amount of lines changed
        # -> look for new position of header lines
        self._find_header_lines()
    
    def __len__(self):
        return len(self._header_i)
    
    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1
    
    def _find_header_lines(self):
        self._header_i = []
        for i, line in enumerate(self._lines):
            if line[0] == ">":
                self._header_i.append(i)
    
    