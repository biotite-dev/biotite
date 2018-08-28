# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["GeneralSequence", "NucleotideSequence", "ProteinSequence"]

from .sequence import Sequence
from .alphabet import LetterAlphabet, AlphabetError
import numpy as np
import copy


class GeneralSequence(Sequence):
    """
    This class allows the creation of a sequence with custom
    `Alphabet` without the need to subclass `Sequence`.
        
    Parameters
    ----------
    alphabet : Alphabet
        The alphabet of this sequence.
    sequence : iterable object, optional
        The symbol sequence, the `Sequence` is initialized with. For
        alphabets containing single letter strings, this parameter
        may also be a `str` object. By default the sequence is empty.
    """
        
    def __init__(self, alphabet, sequence=()):
        self._alphabet = alphabet
        super().__init__(sequence)
    
    def __copy_create__(self):
        return GeneralSequence(self._alphabet)
    
    def get_alphabet(self):
        return self._alphabet

class NucleotideSequence(Sequence):
    """
    Representation of a nucleotide sequence (DNA or RNA).
    
    This class may one of two different alphabets:
    `unambiguous_alphabet()` contains only the unambiguous DNA
    letters 'A', 'C', 'G' and 'T'.
    `ambiguous_alphabet()` uses an extended alphabet for ambiguous 
    letters.
    
    Parameters
    ----------
    sequence : iterable object, optional
        The initial DNA sequence. This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.
    ambiguous : bool, optional
        If true, the ambiguous alphabet is used. By default the
        object tries to use the unambiguous alphabet. If this fails due
        ambiguous letters in the sequence, the ambiguous alphabet
        is used.
    """
    
    alphabet     = LetterAlphabet(["A","C","G","T"])
    alphabet_amb = LetterAlphabet(["A","C","G","T","R","Y","W","S",
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
    _compl_dict = {}
    for _key, _value in compl_symbol_dict.items():
        _key_code = alphabet_amb.encode(_key)
        _val_code = alphabet_amb.encode(_value)
        _compl_dict[_key_code] = _val_code
    # Vectorized function that returns a complement code
    _complement_func = np.vectorize(_compl_dict.__getitem__)
    
    def __init__(self, sequence=[], ambiguous=False):
        if isinstance(sequence, str):
            sequence = sequence.upper()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        if ambiguous == False:
            try:
                self._alphabet = NucleotideSequence.alphabet
                seq_code = self._alphabet.encode_multiple(sequence)
            except AlphabetError:
                self._alphabet = NucleotideSequence.alphabet_amb
                seq_code = self._alphabet.encode_multiple(sequence)
        else:
            self._alphabet = NucleotideSequence.alphabet_amb
            seq_code = self._alphabet.encode_multiple(sequence)
        super().__init__()
        self.code = seq_code
        
    def __copy_create__(self):
        if self._alphabet == NucleotideSequence.alphabet_amb:
            seq_copy = NucleotideSequence(ambiguous=True)
        else:
            seq_copy = NucleotideSequence(ambiguous=False)
        return seq_copy
    
    def get_alphabet(self):
        return self._alphabet
    
    def complement(self):
        """
        Get the complement nucleotide sequence.
        
        Returns
        -------
        complement : NucleotideSequence
            The complement sequence.
        
        Examples
        --------
        
        >>> dna_seq = NucleotideSequence("ACGCTT")
        >>> print(dna_seq.complement())
        TGCGAA
        >>> print(dna_seq.reverse().complement())
        AAGCGT
        
        """
        compl_code = NucleotideSequence._complement_func(self.code)
        return self.copy(compl_code)
    
    def translate(self, complete=False, codon_table=None, met_start=False):
        """
        Translate the nucleotide sequence into a protein sequence.
        
        If `complete` is true, the entire sequence is translated,
        beginning with the first codon and ending with the last codon,
        even if stop codons occur during the translation.
        
        Otherwise this method returns possible ORFs in the
        sequence, even if not stop codon occurs in an ORF.
        
        Parameters
        ----------
        complete : bool, optional
            If true, the complete sequence is translated. In this case
            the sequence length must be a multiple of 3.
            Otherwise all ORFs are translated. (Default: False)
        codon_table : CodonTable, optional
            The codon table to be used. By default the default table
            will be used
            (NCBI "Standard" table with "ATG" as single start codon).
        met_start : bool, optional
            If true, the translation starts always with a 'methionine',
            even if the start codon codes for another amino acid.
            Otherwise the translation starts with the amino acid
            the codon codes for. Only applies, if `complete` is false.
            (Default: False)
        
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
        
        >>> dna_seq = NucleotideSequence("AATGATGCTATAGAT")
        >>> prot_seq = dna_seq.translate(complete=True)
        >>> print(prot_seq)
        NDAID
        >>> prot_seqs, pos = dna_seq.translate(complete=False)
        >>> for seq in prot_seqs:
        ...    print(seq)
        MML*
        ML*
        
        """
        if self._alphabet == NucleotideSequence.alphabet_amb:
            raise AlphabetError("Translation requires unambiguous alphabet")
        # Determine codon_table
        if codon_table is None:
            # Import at this position to avoid circular import
            from .codon import CodonTable
            codon_table = CodonTable.default_table()
        stop_code = ProteinSequence.alphabet.encode("*")
        met_code =  ProteinSequence.alphabet.encode("M")
        
        if complete:
            if len(self) % 3 != 0:
                raise ValueError("Sequence length needs to be a multiple of 3 "
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
            start_codons = np.array(codon_table.start_codons(True))
            seq_code = self.code
            protein_seqs = []
            pos = []
            for i in range(len(seq_code) - 3 + 1):
                sub_seq = seq_code[i : i + 3]
                # sub_seq equals all nucleotides
                # in any of the start codons
                if (sub_seq == start_codons).all(axis=1).any(axis=0):
                    #Translation start
                    j = i
                    # Pessimistic array allocation
                    aa_code = np.zeros(len(self) // 3)
                    # Index for protein sequence
                    aa_i = 0
                    stop_found = False
                    if met_start:
                        aa_code[aa_i] = met_code
                        aa_i += 1
                        j += 3
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
    
    @staticmethod
    def unambiguous_alphabet():
        return NucleotideSequence.alphabet
    
    @staticmethod
    def ambiguous_alphabet():
        return NucleotideSequence.alphabet_amb


class ProteinSequence(Sequence):
    """
    Representation of a protein sequence.
    
    Furthermore this class offers a conversion of amino acids from
    3-letter code into 1-letter code and vice versa.
    
    Parameters
    ----------
    sequence : iterable object, optional
        The initial protein sequence. This may either be a list or a
        string. May take upper or lower case letters. If a list is
        given, the list elements can be 1-letter or 3-letter amino acid
        representations. By default the sequence is empty.
    """
    
    _codon_table = None
    
    alphabet = LetterAlphabet(["A","C","D","E","F","G","H","I","K","L",
                               "M","N","P","Q","R","S","T","V","W","Y",
                               "B","Z","X","*"])
    
    _dict_1to3 = {"A" : "ALA",
                  "C" : "CYS",
                  "D" : "ASP",
                  "E" : "GLU",
                  "F" : "PHE",
                  "G" : "GLY",
                  "H" : "HIS",
                  "I" : "ILE",
                  "K" : "LYS",
                  "L" : "LEU",
                  "M" : "MET",
                  "N" : "ASN",
                  "P" : "PRO",
                  "Q" : "GLN",
                  "R" : "ARG",
                  "S" : "SER",
                  "T" : "THR",
                  "V" : "VAL",
                  "W" : "TRP",
                  "Y" : "TYR",
                  "B" : "ASX",
                  "Z" : "GLX",
                  "X" : "UNK",
                  "*" : " * "}
    
    _dict_3to1 = {}
    for _key, _value in _dict_1to3.items():
        _dict_3to1[_value] = _key
    _dict_3to1["SEC"] = "C"
    _dict_3to1["MSE"] = "M"
    
    def __init__(self, sequence=()):
        dict_3to1 = ProteinSequence._dict_3to1
        alph = ProteinSequence.alphabet
        # Convert 3-letter codes to single letter codes,
        # if list contains 3-letter codes
        sequence = [dict_3to1[symbol.upper()] if len(symbol) == 3
                    else symbol.upper() for symbol in sequence]
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
        return ProteinSequence._dict_3to1[symbol.upper()]
    
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
        return ProteinSequence._dict_1to3[symbol.upper()]
    