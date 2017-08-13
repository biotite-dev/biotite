# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from .sequence import Sequence
from .alphabet import Alphabet, AlphabetError
import numpy as np
import abc
import copy

__all__ = ["DNASequence", "RNASequence", "ProteinSequence"]


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
        if self._alphabet == self.ambiguous_alphabet:
            seq_copy = type(self)(ambiguous=True)
        else:
            seq_copy = type(self)(ambiguous=False)
        self._copy_code(seq_copy, new_seq_code)
        return seq_copy
    
    def get_alphabet(self):
        return self._alphabet
    
    @abc.abstractmethod
    def complement(self):
        pass
    
    @abc.abstractstaticmethod
    def unambiguous_alphabet():
        pass
    
    @abc.abstractstaticmethod
    def ambiguous_alphabet():
        pass


class DNASequence(_NucleotideSequence):
    
    alphabet     = Alphabet(["A","C","G","T"])
    alphabet_amb = Alphabet(["A","C","G","T","R","Y","W","S",
                             "M","K","H","B","V","D","N","X"],
                            parents=[alphabet])
    
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
        key_code = alphabet_amb.encode(key)
        val_code = alphabet_amb.encode(value)
        compl_dict[key_code] = val_code
    # Vectorized function that returns a complement code
    _complement_func = np.vectorize(compl_dict.__getitem__)
    
    def transcribe(self):
        if self.get_alphabet() == DNASequence.alphabet:
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
        return DNASequence.alphabet
    
    @staticmethod
    def ambiguous_alphabet():
        return DNASequence.alphabet_amb


class RNASequence(_NucleotideSequence):
    
    alphabet     = Alphabet(["A","C","G","U"])
    alphabet_amb = Alphabet(["A","C","G","U","R","Y","W","S",
                             "M","K","H","B","V","D","N","X"],
                            parents=[alphabet])
    
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
        key_code = alphabet_amb.encode(key)
        val_code = alphabet_amb.encode(value)
        compl_dict[key_code] = val_code
    # Vectorized function that returns a complement code sequence
    _complement_func = np.vectorize(compl_dict.__getitem__)
    
    def reverse_transcribe(self):
        if self.get_alphabet() == DNASequence.alphabet:
            ambiguous = False
        else:
            ambiguous = True
        dna_seq = DNASequence(ambiguous=ambiguous)
        # Alphabets of RNA and DNA are completely identical,
        # only the symbol 'T' is substituted by 'U'
        dna_seq.set_seq_code(self.get_seq_code())
        return dna_seq
    
    def translate(self, **kwargs):
        if self._alphabet == RNASequence.alphabet_amb:
            raise AlphabetError("Translation requires unambiguous alphabet")
        # Determine codon_table
        if "codon_table" in kwargs:
            codon_table = kwargs["codon_table"]
        else:
            codon_table = ProteinSequence.std_codon_table()
        stop_code = ProteinSequence.alphabet.encode("*")
        
        if "complete" in kwargs and kwargs["complete"] == True:
            if len(self) % 3 != 0:
                raise ValueError("Sequence needs to be a multiple of 3 "
                                 "for complete translation")
            # Pessimistic array allocation
            aa_code = np.zeros(len(self) // 3)
            aa_i = 0
            seq_code = self.get_seq_code()
            for i in range(0, len(seq_code), 3):
                codon = tuple(seq_code[i : i+3])
                aa_code[aa_i] = codon_table[codon]
                aa_i += 1
            # Trim to correct size
            aa_code = aa_code[:aa_i]
            protein_seq = ProteinSequence()
            protein_seq.set_seq_code(aa_code)
            return protein_seq
        
        else:
            if "start_codons" in kwargs:
                start_codons = kwargs["start_codons"]
            else:
                start_codons = ["AUG"]
            start_codons = [self.encode(codon, self.get_alphabet())
                            for codon in start_codons]
            seq_code = self.get_seq_code()
            protein_seqs = []
            for i in range(len(seq_code) - 3 + 1):
                sub_seq = seq_code[i : i + 3]
                # sub_seq equals all nucleotides
                # in any of the start codons
                if (sub_seq == start_codons).all(axis=1).any(axis=0):
                    j = i
                    # Pessimistic array allocation
                    aa_code = np.zeros(len(self) // 3)
                    aa_i = 0
                    stop_found = False
                    while j < len(seq_code) - 3 + 1 and not stop_found:
                        codon = tuple(seq_code[j : j+3])
                        code = codon_table[codon]
                        aa_code[aa_i] = code
                        aa_i += 1
                        j += 3
                        if code == stop_code:
                            stop_found = True
                    # Trim to correct size
                    aa_code = aa_code[:aa_i]
                    protein_seq = ProteinSequence()
                    protein_seq.set_seq_code(aa_code)
                    protein_seqs.append(protein_seq)
            return protein_seqs
                
    
    def complement(self):
        compl_code = RNASequence._complement_func(self.get_seq_code())
        return self.copy(compl_code)
    
    @staticmethod
    def unambiguous_alphabet():
        return RNASequence.alphabet
    
    @staticmethod
    def ambiguous_alphabet():
        return RNASequence.alphabet_amb


class ProteinSequence(Sequence):
    
    _codon_symbol_table = {
     "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
     "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
     "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
     "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
     "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
     "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
     "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
     "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
     "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
     "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
     "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
     "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
     "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
     "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
     "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
     "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G"}
    
    _codon_table = None
    
    alphabet = Alphabet(["A","C","D","E","F","G","H","I","K","L",
                         "M","N","P","Q","R","S","T","V","W","Y",
                         "B","Z","X","*"])
    
    _dict_3to1 = {"ALA" : "A",
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
                  "TYR" : "Y",
                  "ASX" : "B",
                  "GLX" : "Z",
                  "UNK" : "X",
                  " * "  : "*"}
    
    _dict_1to3 = {}
    for key, value in _dict_3to1.items():
        _dict_1to3[value] = key
    
    def __init__(self, sequence=[]):
        dict_3to1 = ProteinSequence._dict_3to1
        alph = ProteinSequence.alphabet
        sequence = [dict_3to1[symbol] if len(symbol) == 3
                    else symbol for symbol in sequence]
        super().__init__(sequence)
    
    def get_alphabet(self):
        return ProteinSequence.alphabet
    
    def remove_stops(self):
        stop_code = ProteinSequence.alphabet.encode("*")
        no_stop = self.copy()
        seq_code = no_stop.get_seq_code()
        no_stop.set_seq_code(seq_code[seq_code != stop_code])
        return no_stop
    
    @staticmethod
    def convert_letters_3to1(symbol):
        return ProteinSequence._dict_3to1[symbol]
    
    @staticmethod
    def convert_letters_1to3(symbol):
        return ProteinSequence._dict_3to1[symbol]
    
    @staticmethod
    def convert_codon_table(symbol_table):
        code_table = {}
        for key, value in symbol_table.items():
            key_code = tuple(Sequence.encode(key, RNASequence.alphabet))
            val_code = ProteinSequence.alphabet.encode(value)
            code_table[key_code] = val_code
        return code_table
    
    @staticmethod
    def std_codon_table():
        if ProteinSequence._codon_table is None:
            ProteinSequence._codon_table = ProteinSequence.convert_codon_table(
                                           ProteinSequence._codon_symbol_table)
        return ProteinSequence._codon_table
    
    @staticmethod
    def std_codon_symbol_table():
        return copy.copy(ProteinSequence._codon_symbol_table)
    