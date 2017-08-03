# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from .sequence import Sequence
from .alphabet import Alphabet, AlphabetError
import numpy as np
import abc

class _NucleotideSequence(Sequence, metaclass=abc.ABCMeta):
    
    def __init__(self, sequence=[], ambiguous=False):
        if isinstance(sequence, str):
            sequence = sequence.upper()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        if ambiguous == False:
            try:
                self._alphabet = self.unambiguous_alphabet()
                seq_code = Sequence.encode(sequence, self._alphabet)
            except AlphabetError:
                self._alphabet = self.ambiguous_alphabet()
                seq_code = Sequence.encode(sequence, self._alphabet)
        else:
            self._alphabet = self.ambiguous_alphabet()
            seq_code = Sequence.encode(sequence, self._alphabet)
        super().__init__()
        self.set_seq_code(seq_code)
        
    def copy(self, new_seq_code=None):
        if self._alphabet == type(self).alph_ambiguous:
            seq_copy = type(self)(ambiguous=True)
        else:
            seq_copy = type(self)(ambiguous=False)
        self._copy_code(seq_copy, new_seq_code)
        return seq_copy
    
    def get_alphabet(self):
        return self._alphabet
    
    @abc.abstractmethod
    def complement(self):
        compl_code = self._complement_func(self.get_seq_code())
        return self.copy(compl_code)
    
    @abc.abstractstaticmethod
    def unambiguous_alphabet():
        pass
    
    @abc.abstractstaticmethod
    def ambiguous_alphabet():
        pass
    
    


class DNASequence(_NucleotideSequence):
    
    alph_unambiguous = Alphabet("UnambiguousDNA", ["A","C","G","T"])
    alph_ambiguous = Alphabet("AmbiguousDNA",
                              ["A","C","G","T","R","Y","W","S",
                               "M","K","H","B","V","D","N","X"],
                              parents=["UnambiguousDNA"])
    
    compl_symbol_dict = {"A" : "T",
                         "C" : "G",
                         "G" : "C",
                         "T" : "A",
                         "M" : "K",
                         "R" : "Y",
                         "W" : "W",
                         "S" : "S",
                         "Y" : "R",
                         "K" : "M",
                         "V" : "B",
                         "H" : "D",
                         "D" : "H",
                         "B" : "V",
                         "X" : "X",
                         "N" : "N"}
    compl_dict = {}
    for key, value in compl_symbol_dict.items():
        key_code = alph_ambiguous.encode(key)
        val_code = alph_ambiguous.encode(value)
        compl_dict[key_code] = val_code
    # Vectorized function that returns a complement code
    _complement_func = np.vectorize(compl_dict.__getitem__)
    
    def transcribe(self):
        if self.get_alphabet() == DNASequence.alph_unambiguous:
            ambiguous = False
        else:
            ambiguous = True
        rna_seq = RNASequence(ambiguous=ambiguous)
        # Alphabets of RNA and DNA are completely identical,
        # only the symbol 'T' is substituted by 'U'
        rna_seq.set_seq_code(self.get_seq_code())
        return rna_seq
    
    def complement(self):
        compl_code = DNASequence._complement_func(self.get_seq_code())
        return self.copy(compl_code)
    
    @staticmethod
    def unambiguous_alphabet():
        return DNASequence.alph_unambiguous
    
    @staticmethod
    def ambiguous_alphabet():
        return DNASequence.alph_ambiguous
        


class RNASequence(_NucleotideSequence):
    
    alph_unambiguous = Alphabet("UnambiguousRNA", ["A","C","G","U"])
    alph_ambiguous = Alphabet("AmbiguousRNA",
                              ["A","C","G","U","R","Y","W","S",
                               "M","K","H","B","V","D","N","X"],
                              parents=["UnambiguousRNA"])
    
    compl_symbol_dict = {"A" : "U",
                         "C" : "G",
                         "G" : "C",
                         "U" : "A",
                         "M" : "K",
                         "R" : "Y",
                         "W" : "W",
                         "S" : "S",
                         "Y" : "R",
                         "K" : "M",
                         "V" : "B",
                         "H" : "D",
                         "D" : "H",
                         "B" : "V",
                         "X" : "X",
                         "N" : "N"}
    compl_dict = {}
    for key, value in compl_symbol_dict.items():
        key_code = alph_ambiguous.encode(key)
        val_code = alph_ambiguous.encode(value)
        compl_dict[key_code] = val_code
    # Vectorized function that returns a complement code sequence
    _complement_func = np.vectorize(compl_dict.__getitem__)
    
    def reverse_transcribe(self):
        if self.get_alphabet() == DNASequence.alph_unambiguous:
            ambiguous = False
        else:
            ambiguous = True
        dna_seq = DNASequence(ambiguous=ambiguous)
        # Alphabets of RNA and DNA are completely identical,
        # only the symbol 'T' is substituted by 'U'
        dna_seq.set_seq_code(self.get_seq_code())
        return dna_seq
    
    def translate(self, **kwargs):
        # start_codons
        # entire
        # codon_table
        if self._alphabet == RNASequence.alph_ambiguous:
            raise AlphabetError("Translation requires unambiguous alphabet")
        raise NotImplementedError()
    
    def complement(self):
        compl_code = RNASequence._complement_func(self.get_seq_code())
        return self.copy(compl_code)
    
    @staticmethod
    def unambiguous_alphabet():
        return RNASequence.alph_unambiguous
    
    @staticmethod
    def ambiguous_alphabet():
        return RNASequence.alph_ambiguous


class ProteinSequence(Sequence):
    
    alphabet = Alphabet("Protein", ["A","C","D","E","F","G","H","I","K","L",
                                    "M","N","P","Q","R","S","T","V","W","Y"])
    dict_3to1 = {"ALA" : "A",
                 "CYS" : "C",
                 "ASP" : "D",
                 "GLU" : "E",
                 "PHE" : "F",
                 "GLY" : "G",
                 "HIS" : "H",
                 "ILE" : "I",
                 "LYS" : "K",
                 "LEU" : "L",
                 "MET" : "M",
                 "ASN" : "N",
                 "PRO" : "P",
                 "GLN" : "Q",
                 "ARG" : "R",
                 "SER" : "S",
                 "THR" : "T",
                 "VAL" : "V",
                 "TRP" : "W",
                 "TYR" : "Y"}
    
    def __init__(self, sequence=[]):
        dict_3to1 = ProteinSequence.dict_3to1
        alph = ProteinSequence.alphabet
        sequence = [alph.encode(dict_3to1[symbol]) if len(symbol) == 3
                    else alph.encode(symbol) for symbol in sequence]
        super().__init__(sequence)
    
    def get_alphabet(self):
        return ProteinSequence.alphabet
    