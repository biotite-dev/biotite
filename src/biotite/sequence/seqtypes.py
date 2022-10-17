# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence"
__author__ = "Patrick Kunzmann", "Thomas Nevolianis"
__all__ = ["GeneralSequence", "NucleotideSequence", "ProteinSequence"]

from .sequence import Sequence
from .alphabet import LetterAlphabet, AlphabetError, AlphabetMapper
import numpy as np
import copy


class GeneralSequence(Sequence):
    """
    This class allows the creation of a sequence with custom
    :class:`Alphabet` without the need to subclass :class:`Sequence`.
        
    Parameters
    ----------
    alphabet : Alphabet
        The alphabet of this sequence.
    sequence : iterable object, optional
        The symbol sequence, the :class:`Sequence` is initialized with.
        For alphabets containing single letter strings, this parameter
        may also be a :class:`str` object.
        By default the sequence is empty.
    """
        
    def __init__(self, alphabet, sequence=()):
        self._alphabet = alphabet
        super().__init__(sequence)

    def __repr__(self):
        """Represent GeneralSequence as a string for debugging."""
        return f"GeneralSequence(Alphabet({self._alphabet}), " \
               f"[{', '.join([repr(symbol) for symbol in self.symbols])}])"

    def __copy_create__(self):
        return GeneralSequence(self._alphabet)
    
    def get_alphabet(self):
        return self._alphabet
    
    def as_type(self, sequence):
        """
        Convert the :class:`GeneralSequence` into a sequence of another
        :class:`Sequence` type.

        This function simply replaces the sequence code of the given
        sequence with the sequence code of this object.

        Parameters
        ----------
        sequence : Sequence
            The `Sequence` whose sequence code is replaced with the one
            of this object.
            The alphabet must equal or extend the alphabet of this
            object.
        
        Returns
        -------
        sequence : Sequence
            The input `sequence` with replaced sequence code.
        
        Raises
        ------
        AlphabetError
            If the the :class:`Alphabet` of the input `sequence` does
            not extend the :class:`Alphabet` of this sequence.
        """
        if not sequence.get_alphabet().extends(self._alphabet):
            raise AlphabetError(
                f"The alphabet of '{type(sequence).__name__}' "
                f"is not compatible with the alphabet of this sequence"
            )
        sequence.code = self.code
        return sequence

class NucleotideSequence(Sequence):
    """
    Representation of a nucleotide sequence (DNA or RNA).
    
    This class may have one of two different alphabets:
    :attr:`unambiguous_alphabet()` contains only the unambiguous DNA
    letters 'A', 'C', 'G' and 'T'.
    :attr:`ambiguous_alphabet()` uses an extended alphabet for ambiguous 
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
    
    alphabet_unamb = LetterAlphabet(["A","C","G","T"])
    alphabet_amb   = LetterAlphabet(
        ["A","C","G","T","R","Y","W","S",
         "M","K","H","B","V","D","N"]
    )
    
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
                         "N" : "N"}
    # List comprehension does not work in this scope
    _compl_symbols = []
    for _symbol in alphabet_amb.get_symbols():
        _compl_symbols.append(compl_symbol_dict[_symbol])
    _compl_alphabet_unamb = LetterAlphabet(_compl_symbols)
    _compl_mapper = AlphabetMapper(_compl_alphabet_unamb, alphabet_amb)
    
    def __init__(self, sequence=[], ambiguous=None):
        if isinstance(sequence, str):
            sequence = sequence.upper()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        if ambiguous is None:
            try:
                self._alphabet = NucleotideSequence.alphabet_unamb
                seq_code = self._alphabet.encode_multiple(sequence)
            except AlphabetError:
                self._alphabet = NucleotideSequence.alphabet_amb
                seq_code = self._alphabet.encode_multiple(sequence)
        elif not ambiguous:
            self._alphabet = NucleotideSequence.alphabet_unamb
            seq_code = self._alphabet.encode_multiple(sequence)
        else:
            self._alphabet = NucleotideSequence.alphabet_amb
            seq_code = self._alphabet.encode_multiple(sequence)
        super().__init__()
        self.code = seq_code

    def __repr__(self):
        """Represent NucleotideSequence as a string for debugging."""
        if self._alphabet == NucleotideSequence.alphabet_amb:
            ambiguous = True
        else:
            ambiguous = False
        return f'NucleotideSequence("{"".join(self.symbols)}", ambiguous={ambiguous})'

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
        # Interpreting the sequence code of this object in the
        # complementary alphabet gives the complementary symbols
        # In order to get the complementary symbols in the original
        # alphabet, the sequence code is mapped from the complementary
        # alphabet into the original alphabet
        compl_code = NucleotideSequence._compl_mapper[self.code]
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
            only a single :class:`ProteinSequence` is returned. Otherwise
            a list of :class:`ProteinSequence` is returned, which contains
            every ORF.
        pos : list of tuple (int, int)
            Is only returned if `complete` is false. The list contains
            a tuple for each ORF.
            The first element of the tuple is the index of the 
            :class:`NucleotideSequence`, where the translation starts.
            The second element is the exclusive stop index, it
            represents the first nucleotide in the
            :class:`NucleotideSequence` after a stop codon.
        
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
        met_code  = ProteinSequence.alphabet.encode("M")
        
        if complete:
            if len(self) % 3 != 0:
                raise ValueError("Sequence length needs to be a multiple of 3 "
                                 "for complete translation")
            # Reshape code into (n,3), with n being the amount of codons
            codons = self.code.reshape(-1, 3)
            protein_seq = ProteinSequence()
            protein_seq.code = codon_table.map_codon_codes(codons)
            return protein_seq
        
        else:
            protein_seqs = []
            pos = []
            code = self.code
            # Create all three frames
            for shift in range(3):
                # The frame length is always a multiple of 3
                # If there is a trailing partial codon, remove it
                frame_length = ((len(code) - shift) // 3) * 3
                frame = code[shift : shift+frame_length]
                # Reshape frame into (n,3), with n being the amount of codons
                frame_codons = frame.reshape(-1, 3)
                # At first, translate frame completely
                protein_code = codon_table.map_codon_codes(frame_codons)
                # Iterate over all start codons in this frame
                starts = np.where(codon_table.is_start_codon(frame_codons))[0]
                for start_i in starts:
                    # Protein sequence beginning from start codon
                    code_from_start = protein_code[start_i:]
                    # Get all stop codon positions
                    # relative to 'code_from_start'
                    stops = np.where(code_from_start == stop_code)[0]
                    # Find first stop codon after start codon
                    # Include stop -> stops[0] + 1
                    stop_i = stops[0] + 1 if len(stops) > 0 \
                             else len(code_from_start)
                    code_from_start_to_stop = code_from_start[:stop_i]
                    prot_seq = ProteinSequence()
                    if met_start:
                        # Copy as the slice is edited
                        prot_seq.code = code_from_start_to_stop.copy()
                        prot_seq.code[0] = met_code
                    else:
                        prot_seq.code = code_from_start_to_stop
                    protein_seqs.append(prot_seq)
                    # Codon indices are transformed
                    # to nucleotide sequence indices
                    pos.append((shift + start_i*3, shift + (start_i+stop_i)*3))
            # Sort by start position
            order = np.argsort([start for start, stop in pos])
            pos = [pos[i] for i in order]
            protein_seqs = [protein_seqs[i] for i in order]
            return protein_seqs, pos
    
    @staticmethod
    def unambiguous_alphabet():
        """
        Get the unambiguous nucleotide alphabet containing the symbols
        ``A``,  ``C``,  ``G`` and  ``T``.

        Returns
        -------
        alphabet : LetterAlphabet
            The unambiguous nucleotide alphabet.
        """
        return NucleotideSequence.alphabet_unamb
    
    @staticmethod
    def ambiguous_alphabet():
        """
        Get the ambiguous nucleotide alphabet containing the symbols
        ``A``,  ``C``,  ``G`` and  ``T`` and symbols describing
        ambiguous combinations of these.

        Returns
        -------
        alphabet : LetterAlphabet
            The ambiguous nucleotide alphabet.
        """
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
    
    Notes
    -----
    The :class:`Alphabet` of this :class:`Sequence` class does not
    support selenocysteine.
    Please convert selenocysteine (``U``) into cysteine (``C``)
    or use a custom :class:`Sequence` class, if the differentiation is
    necessary.
    """

    _codon_table = None
    
    alphabet = LetterAlphabet(["A","C","D","E","F","G","H","I","K","L",
                               "M","N","P","Q","R","S","T","V","W","Y",
                               "B","Z","X","*"])

    # Masses are taken from
    # https://web.expasy.org/findmod/findmod_masses.html#AA

    _mol_weight_average = np.array([
         71.0788,  # A
        103.1388,  # C
        115.0886,  # D
        129.1155,  # E
        147.1766,  # F
         57.0519,  # G
        137.1411,  # H
        113.1594,  # I
        128.1741,  # K
        113.1594,  # L
        131.1926,  # M
        114.1038,  # N
         97.1167,  # P
        128.1307,  # Q
        156.1875,  # R
         87.0782,  # S
        101.1051,  # T
         99.1326,  # V
        186.2132,  # W
        163.1760,  # Y
          np.nan,  # B
          np.nan,  # Z
          np.nan,  # X
          np.nan,  # *
    ])

    _mol_weight_monoisotopic = np.array([
         71.03711,  # A
        103.00919,  # C
        115.02694,  # D
        129.04259,  # E
        147.06841,  # F
         57.02146,  # G
        137.05891,  # H
        113.08406,  # I
        128.09496,  # K
        113.08406,  # L
        131.04049,  # M
        114.04293,  # N
         97.05276,  # P
        128.05858,  # Q
        156.10111,  # R
         87.03203,  # S
        101.04768,  # T
         99.06841,  # V
        186.07931,  # W
        163.06333,  # Y
        np.nan,  # B
        np.nan,  # Z
        np.nan,  # X
        np.nan,  # *
    ])

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

    def __repr__(self):
        """Represent ProteinSequence as a string for debugging."""
        return f'ProteinSequence("{"".join(self.symbols)}")'

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

    def get_molecular_weight(self, monoisotopic=False):
        """
        Calculate the molecular weight of this protein.
        
        Average protein molecular weight is calculated by the addition
        of average isotopic masses of the amino acids
        in the protein and the average isotopic mass of one water
        molecule.

        Returns
        -------
        weight : float
            Molecular weight of the protein represented by the sequence.
            Molecular weight values are given in Dalton (Da).
        """
        if monoisotopic:
            weight = np.sum(self._mol_weight_monoisotopic[self.code]) + 18.015
        else:
            weight = np.sum(self._mol_weight_average[self.code]) + 18.015

        if np.isnan(weight):
            raise ValueError(
                "Sequence contains ambiguous amino acids, "
                "cannot calculate weight"
            )
        return weight
