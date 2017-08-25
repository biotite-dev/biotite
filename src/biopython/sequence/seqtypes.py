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
        self.code = seq_code
        
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
    """
    Representation of a DNA sequence.
    
    This class may one of two different alphabets:
    `alphabet` contains only the unambiguous DNA letters 'A', 'C', 'G'
    and 'T'.
    `alphabet_amb` uses an extended alphabet for ambiguous letters.
    
    Parameters
    ----------
    sequence : iterable object, optional
        The initial DNA sequence. This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.
    ambiguous : bool, optional
        If true, the ambiguous DNA alphabet is used. By default the
        object tries to use the unambiguous alphabet. If this fails due
        ambiguous letters in the sequence, the ambiguous DNA alphabet
        is used.
    """
    
    alphabet     = Alphabet(["A","C","G","T"])
    alphabet_amb = Alphabet(["A","C","G","T","R","Y","W","S",
                             "M","K","H","B","V","D","N","X"])
    
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
        """
        Transcribe the DNA sequence into a RNA sequence.
        
        Returns
        -------
        transcript : RNASequence
            The trancribed sequence.
        
        Examples
        --------
        
            >>> dna_seq = DNASequence("ACGCTT")
            >>> rna_seq = dna_seq.transcribe()
            >>> print(rna_seq)
            ACGCUU

        """
        if self.get_alphabet() == DNASequence.alphabet:
            ambiguous = False
        else:
            ambiguous = True
        rna_seq = RNASequence(ambiguous=ambiguous)
        # Alphabets of RNA and DNA are completely identical,
        # only the symbol 'T' is substituted by 'U'
        rna_seq.code = self.code
        return rna_seq
    
    def complement(self):
        """
        Get the complement DNA sequence.
        
        Returns
        -------
        complement : DNASequence
            The complement sequence.
        
        Examples
        --------
        
            >>> dna_seq = DNASequence("ACGCTT")
            >>> print(dna_seq.complement())
            TGCGAA
            >>> print(dna_seq.reverse().complement())
            AAGCGT
        
        """
        compl_code = DNASequence._complement_func(self.code)
        return self.copy(compl_code)
    
    @staticmethod
    def unambiguous_alphabet():
        return DNASequence.alphabet
    
    @staticmethod
    def ambiguous_alphabet():
        return DNASequence.alphabet_amb


class RNASequence(_NucleotideSequence):
    """
    Representation of a RNA sequence.
    
    This class may one of two different alphabets:
    `alphabet` contains only the unambiguous RNA letters 'A', 'C', 'G'
    and 'U'.
    `alphabet_amb` uses an extended alphabet for ambiguous letters.
    
    Parameters
    ----------
    sequence : iterable object, optional
        The initial RNA sequence. This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.
    ambiguous : bool, optional
        If true, the ambiguous RNA alphabet is used. By default the
        object tries to use the unambiguous alphabet. If this fails due
        ambiguous letters in the sequence, the ambiguous DNA alphabet
        is used.
    """
    
    alphabet     = Alphabet(["A","C","G","U"])
    alphabet_amb = Alphabet(["A","C","G","U","R","Y","W","S",
                             "M","K","H","B","V","D","N","X"])
    
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
        """
        Reverse-transcribe the RNA sequence into a DNA sequence.
        
        Returns
        -------
        transcript : DNASequence
            The reverse-trancribed sequence.
        
        Examples
        --------
        
            >>> rna_seq = RNASequence("UGACCU")
            >>> dna_seq = rna_seq.reverse_transcribe()
            >>> print(dna_seq)
            TGACCT
        
        """
        if self.get_alphabet() == DNASequence.alphabet:
            ambiguous = False
        else:
            ambiguous = True
        dna_seq = DNASequence(ambiguous=ambiguous)
        # Alphabets of RNA and DNA are completely identical,
        # only the symbol 'T' is substituted by 'U'
        dna_seq.code = self.code
        return dna_seq
    
    def translate(self, **kwargs):
        """
        Translate the RNA sequence into a protein sequence.
        
        If `complete` is true, the entire sequence is translated,
        beginning with the first codon and ending with the last codon,
        even if stop codons occur during the translation.
        
        Otherwise this method returns possible ORFs in the
        sequence, even if not stop codon occurs in an ORF.
        
        Parameters
        ----------
        complete : bool, optional
            If true, the complete sequence is translated, otherwise all
            ORFs are translated. (Default: False)
        codon_table : dict, optional
            The codon table to be used. A codon table maps triplett
            sequence codes to amino acid single letter sequence codes.
            The table can be generated from a dictionary, containing
            strings as keys and values, via
            `ProteinSequence.convert_codon_table()`. By default an
            codon table from *E. coli* is used.
        start_codons : list of strings. optional
            A list of codons to be used as starting point for
            translation. By default the list contains only "ATG".
        
        Returns
        -------
        protein : ProteinSequence or list of ProteinSequence
            The translated protein sequence. If `complete` is true,
            only a single `ProteinSequence` is returned. Otherwise
            a list of `ProteinSequence` is returned, which contains
            every ORF.
        pos : list of tuple (int, int)
            Is only returned if `complete` is false. The list contains
            a tuple for each ORF.
            The first element of the tuple is the index of the 
            RNASequence`, where the translation starts.
            The second element is the exclusive stop index, therefore
            it represents the first nucleotide in the RNASequence after
            a stop codon.
        
        Examples
        --------
        
            >>> rna_seq = RNASequence("AAUGAUGCUAUAGAU")
            >>> prot_seq = rna_seq.translate(complete=True)
            >>> print(prot_seq)
            NDAID
            >>> prot_seqs, pos = rna_seq.translate(complete=False)
            >>> for seq in prot_seqs:
            ...    print(seq)
            MML*
            ML*
        
        """
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
            seq_code = self.code
            for i in range(0, len(seq_code), 3):
                codon = tuple(seq_code[i : i+3])
                aa_code[aa_i] = codon_table[codon]
                aa_i += 1
            # Trim to correct size
            aa_code = aa_code[:aa_i]
            protein_seq = ProteinSequence()
            protein_seq.code = aa_code
            return protein_seq
        
        else:
            if "start_codons" in kwargs:
                start_codons = kwargs["start_codons"]
            else:
                start_codons = ["AUG"]
            start_codons = [self.encode(codon, self.get_alphabet())
                            for codon in start_codons]
            seq_code = self.code
            protein_seqs = []
            pos = []
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
                    protein_seq.code = aa_code
                    protein_seqs.append(protein_seq)
                    pos.append((i,j))
            return protein_seqs, pos
                
    
    def complement(self):
        """
        Get the complement RNA sequence.
        
        Returns
        -------
        complement : RNASequence
            The complement sequence.
        
        Examples
        --------
        
            >>> rna_seq = RNASequence("UGACCU")
            >>> print(rna_seq.complement())
            ACUGGA
            >>> print(rna_seq.reverse().complement())
            AGGUCA
        
        """
        compl_code = RNASequence._complement_func(self.code)
        return self.copy(compl_code)
    
    @staticmethod
    def unambiguous_alphabet():
        return RNASequence.alphabet
    
    @staticmethod
    def ambiguous_alphabet():
        return RNASequence.alphabet_amb


class ProteinSequence(Sequence):
    """
    Representation of a protein sequence.
    
    Furthermore this class offers a codon table for conversion of
    nucleotide tripletts into amino acids. A *codon symbol table* holds
    triplett strings as keys (e.g. 'AUG') and 1-letter amino acid
    representations as values (e.g. 'M'). A *codon table* is a *codon
    symbol table* that is converted into symbol codes. Therefore a
    *codon table* holds tuples of 3 integers as keys and amino acid
    symbol codes as values.
    
    Parameters
    ----------
    sequence : iterable object, optional
        The initial protein sequence. This may either be a list or a
        string. May take upper or lower case letters. If a list is
        given, the list elements can be 1-letter or 3-letter amino acid
        representations. By default the sequence is empty.
    """
    
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
        # Convert 3-letter codes to single letter codes,
        # if list contains 3-letter codes
        sequence = [dict_3to1[symbol.upper()] if len(symbol) == 3
                    else symbol for symbol in sequence]
        super().__init__(sequence)
    
    def get_alphabet(self):
        return ProteinSequence.alphabet
    
    def remove_stops(self):
        """
        Remove *stop signals* from the sequence.
        
        Returns
        -------
        no_stop : ProteinSequence
            A copy of this sequence without stop signals.
        """
        stop_code = ProteinSequence.alphabet.encode("*")
        no_stop = self.copy()
        seq_code = no_stop.code
        no_stop.code = seq_code[seq_code != stop_code]
        return no_stop
    
    @staticmethod
    def convert_letter_3to1(symbol):
        """
        Convert a 3-letter to a 1-letter amino acid representation.
        
        Parameters
        ----------
        symbol : string
            3-letter amino acid representation.
        
        Returns
        -------
        convert : string
            1-letter amino acid representation.
        """
        return ProteinSequence._dict_3to1[symbol]
    
    @staticmethod
    def convert_letter_1to3(symbol):
        """
        Convert a 1-letter to a 3-letter amino acid representation.
        
        Parameters
        ----------
        symbol : string
            1-letter amino acid representation.
        
        Returns
        -------
        convert : string
            3-letter amino acid representation.
        """
        return ProteinSequence._dict_3to1[symbol]
    
    @staticmethod
    def convert_codon_table(symbol_table):
        """
        Convert a symbol codon table into a codon table
        
        Parameters
        ----------
        symbol_table : dict
            The codon symbol table.
        
        Returns
        -------
        code_table : dict
            The codon table.
        """
        code_table = {}
        for key, value in symbol_table.items():
            key_code = tuple(Sequence.encode(key, RNASequence.alphabet))
            val_code = ProteinSequence.alphabet.encode(value)
            code_table[key_code] = val_code
        return code_table
    
    @staticmethod
    def std_codon_table():
        """
        Get the standard codon table, which is the *E. coli* codon
        table.
        
        Returns
        -------
        code_table : dict
            The standard codon table.
        """
        if ProteinSequence._codon_table is None:
            ProteinSequence._codon_table = ProteinSequence.convert_codon_table(
                                           ProteinSequence._codon_symbol_table)
        return ProteinSequence._codon_table
    
    @staticmethod
    def std_codon_symbol_table():
        """
        Get the standard codon symbol table, which is the *E. coli*
        codon table.
        
        Returns
        -------
        symbol_table : dict
            The standard codon symbol table.
        """
        return copy.copy(ProteinSequence._codon_symbol_table)
    