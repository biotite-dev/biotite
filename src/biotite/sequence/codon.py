# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence"
__author__ = "Patrick Kunzmann"
__all__ = ["CodonTable"]

import copy
from os.path import join, dirname, realpath
import numpy as np
from numbers import Integral
from .seqtypes import NucleotideSequence, ProteinSequence


# Abbreviations
_NUC_ALPH = NucleotideSequence.alphabet_unamb
_PROT_ALPH = ProteinSequence.alphabet

# Multiplier array that converts a codon in code representation
# into a unique integer
_radix = len(_NUC_ALPH)
_radix_multiplier = np.array([_radix**n for n in (2,1,0)], dtype=int)


class CodonTable(object):
    """
    A :class:`CodonTable` maps a codon (sequence of 3 nucleotides) to an
    amino acid.
    It also defines start codons. A :class:`CodonTable`
    takes/outputs either the symbols or code of the codon/amino acid.
    
    Furthermore, this class is able to give a list of codons that
    corresponds to a given amino acid.
    
    The :func:`load()` method allows loading of NCBI codon tables.
    
    Objects of this class are immutable.
    
    Parameters
    ----------
    codon_dict : dict of (str -> str)
        A dictionary that maps codons to amino acids. The keys must be
        strings of length 3 and the values strings of length 1
        (all upper case).
        The dictionary must provide entries for all 64 possible codons.
    starts : iterable object of str
        The start codons. Each entry must be a string of length 3
        (all upper case).
    
    Examples
    --------
    
    Get the amino acid coded by a given codon (symbol and code):
        
    >>> table = CodonTable.default_table()
    >>> print(table["ATG"])
    M
    >>> print(table[(1,2,3)])
    14
        
    Get the codons coding for a given amino acid (symbol and code):
        
    >>> table = CodonTable.default_table()
    >>> print(table["M"])
    ('ATG',)
    >>> print(table[14])
    ((0, 2, 0), (0, 2, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3))
    """
    
    # For efficient mapping of codon codes to amino acid codes,
    # especially in in the 'map_codon_codes()' function, the class
    # maps each possible codon into a unique number using a radix based
    # approach.
    # For example the codon (3,1,2) would be represented as
    # 3*16 + 1*4 + 2**1 = 53

    # file for builtin codon tables from NCBI
    _table_file = join(dirname(realpath(__file__)), "codon_tables.txt")
    
    def __init__(self, codon_dict, starts):
        # Check if 'starts' is iterable object of length 3 string
        for start in starts:
            if not isinstance(start, str) or len(start) != 3:
                raise ValueError(f"Invalid codon '{start}' as start codon")
        # Internally store codons as single unique numbers
        start_codon_codes = np.array(
            [_NUC_ALPH.encode_multiple(start) for start in starts], dtype=int
        )
        self._starts = CodonTable._to_number(start_codon_codes)
        # Use -1 as error code
        # The array uses the number representation of codons as index
        # and stores the corresponding symbol codes for amino acids
        self._codons = np.full(_radix**3, -1, dtype=int)
        for key, value in codon_dict.items():
            codon_code = _NUC_ALPH.encode_multiple(key)
            codon_number = CodonTable._to_number(codon_code)
            aa_code = _PROT_ALPH.encode(value)
            self._codons[codon_number] = aa_code
        if (self._codons == -1).any():
            # Find the missing codon
            missing_index = np.where(self._codons == -1)[0][0]
            codon_code = CodonTable._to_codon(missing_index) 
            codon = _NUC_ALPH.decode_multiple(codon_code)
            codon_str = "".join(codon)
            raise ValueError(
                f"Codon dictionary does not contain codon '{codon_str}'"
            )

    def __repr__(self):
        """Represent CodonTable as a string for debugging."""
        return f"CodonTable({self.codon_dict()}, {self.start_codons()})"

    def __eq__(self, item):
        if not isinstance(item, CodonTable):
            return False
        if self.codon_dict() != item.codon_dict():
            return False
        if self.start_codons() != item.start_codons():
            return False
        return True

    def __ne__(self, item):
        return not self == item

    def __getitem__(self, item):
        if isinstance(item, str):
            if len(item) == 1:
                # Amino acid -> return possible codons
                aa_code = _PROT_ALPH.encode(item)
                codon_numbers = np.where(self._codons == aa_code)[0]
                codon_codes = CodonTable._to_codon(codon_numbers)
                codons = tuple(
                    ["".join(_NUC_ALPH.decode_multiple(codon_code))
                    for codon_code in codon_codes]
                )
                return codons
            elif len(item) == 3:
                # Codon -> return corresponding amino acid
                codon_code = _NUC_ALPH.encode_multiple(item)
                codon_number = CodonTable._to_number(codon_code)
                aa_code = self._codons[codon_number]
                aa = _PROT_ALPH.decode(aa_code)
                return aa
            else:
                raise ValueError(f"'{item}' is an invalid index")
        elif isinstance(item, int):
            # Code for amino acid -> return possible codon codes
            codon_numbers = np.where(self._codons == item)[0]
            codon_codes = tuple(CodonTable._to_codon(codon_numbers))
            codon_codes = tuple([tuple(code) for code in codon_codes])
            return codon_codes
        else:
            # Code for codon as any iterable object
            # Code for codon -> return corresponding amino acid codes
            if len(item) != 3:
                raise ValueError(
                    f"{item} is an invalid sequence code for a codon"
                )
            codon_number = CodonTable._to_number(item)
            aa_code = self._codons[codon_number]
            return aa_code
    
    def map_codon_codes(self, codon_codes):
        """
        Efficiently map multiple codons to the corresponding amino
        acids.
        
        Parameters
        ----------
        codon_codes : ndarray, dtype=int, shape=(n,3)
            The codons to be translated into amino acids.
            The codons are given as symbol codes.
            *n* is the amount of codons.
        
        Returns
        -------
        aa_codes : ndarray, dtype=int, shape=(n,)
            The amino acids as symbol codes.
        
        Examples
        --------
        >>> dna = NucleotideSequence("ATGGTTTAA")
        >>> sequence_code = dna.code
        >>> print(sequence_code)
        [0 3 2 2 3 3 3 0 0]
        >>> # Reshape to get codons
        >>> codon_codes = sequence_code.reshape(-1, 3)
        >>> print(codon_codes)
        [[0 3 2]
         [2 3 3]
         [3 0 0]]
        >>> # Map to amino acids
        >>> aa_codes = CodonTable.default_table().map_codon_codes(codon_codes)
        >>> print(aa_codes)
        [10 17 23]
        >>> # Put into a protein sequence
        >>> protein = ProteinSequence()
        >>> protein.code = aa_codes
        >>> print(protein)
        MV*
        """
        if codon_codes.shape[-1] != 3:
            raise ValueError(
                f"Codons must be length 3, "
                f"but size of last dimension is {codon_codes.shape[-1]}"
            )
        codon_numbers = CodonTable._to_number(codon_codes)
        aa_codes = self._codons[codon_numbers]
        return aa_codes
        
    def codon_dict(self, code=False):
        """
        Get the codon to amino acid mappings dictionary.
        
        Parameters
        ----------
        code : bool
            If true, the dictionary contains keys and values as code.
            Otherwise, the dictionary contains strings for codons and
            amino acid. (Default: False)
        
        Returns
        -------
        codon_dict : dict
            The dictionary mapping codons to amino acids.
        """
        if code:
            return {tuple(CodonTable._to_codon(codon_number)): aa_code
                    for codon_number, aa_code in enumerate(self._codons)}
        else:
            return {"".join(_NUC_ALPH.decode_multiple(codon_code)):
                        _PROT_ALPH.decode(aa_code)
                    for codon_code, aa_code
                    in self.codon_dict(code=True).items()}
    
    def is_start_codon(self, codon_codes):
        codon_numbers = CodonTable._to_number(codon_codes)
        return np.isin(codon_numbers, self._starts)
    
    def start_codons(self, code=False):
        """
        Get the start codons of the codon table.
        
        Parameters
        ----------
        code : bool
            If true, the code will be returned instead of strings.
            (Default: False)
        
        Returns
        -------
        start_codons : tuple
            The start codons. Contains strings or tuples, depending on
            the `code` parameter.
        """
        if code:
            return tuple(
                [tuple(CodonTable._to_codon(codon_number))
                 for codon_number in self._starts]
            )
        else:
            return tuple(
                ["".join(_NUC_ALPH.decode_multiple(codon_code))
                 for codon_code in self.start_codons(code=True)]
            )
    
    def with_start_codons(self, starts):
        """
        Create an new :class:`CodonTable` with the same codon mappings,
        but changed start codons.
        
        Parameters
        ----------
        starts : iterable object of str
            The new start codons.
        
        Returns
        -------
        new_table : CodonTable
            The codon table with the new start codons.
        """
        # Copy this table and replace the start codons
        new_table = copy.deepcopy(self)
        start_codon_codes = np.array(
            [_NUC_ALPH.encode_multiple(start) for start in starts], dtype=int
        )
        new_table._starts = CodonTable._to_number(start_codon_codes)
        return new_table
    
    def with_codon_mappings(self, codon_dict):
        """
        Create an new :class:`CodonTable` with partially changed codon
        mappings.
        
        Parameters
        ----------
        codon_dict : dict of (str -> str)
            The changed codon mappings.
        
        Returns
        -------
        new_table : CodonTable
            The codon table with changed codon mappings.
        """
        # Copy this table and replace the codon
        new_table = copy.deepcopy(self)
        for key, value in codon_dict.items():
            codon_code = _NUC_ALPH.encode_multiple(key)
            codon_number = CodonTable._to_number(codon_code)
            aa_code = _PROT_ALPH.encode(value)
            new_table._codons[codon_number] = aa_code
        return new_table

    def __str__(self):
        string = ""
        # ['A', 'C', 'G', 'T']
        bases = _NUC_ALPH.get_symbols()
        for b1 in bases:
            for b2 in bases:
                for b3 in bases:
                    codon = b1 + b2 + b3
                    string += codon + " " + self[codon]
                    # Indicator for start codon
                    codon_code = _NUC_ALPH.encode_multiple(codon)
                    if CodonTable._to_number(codon_code) in self._starts:
                        string += " i "
                    else:
                        string += "   "
                    # Add space for next codon
                    string += " "*3
                # Remove terminal space
                string = string [:-6]
                # Jump to next line
                string += "\n"
            # Add empty line
            string += "\n"
        # Remove the two terminal new lines
        string = string[:-2]
        return string

    @staticmethod
    def _to_number(codons):
        if not isinstance(codons, np.ndarray):
            codons = np.array(list(codons), dtype=int)
        return np.sum(_radix_multiplier * codons, axis=-1)

    @staticmethod
    def _to_codon(numbers):
        if isinstance(numbers, Integral):
            # Only a single number
            return CodonTable._to_codon(np.array([numbers]))[0]
        if not isinstance(numbers, np.ndarray):
            numbers = np.array(list(numbers), dtype=int)
        codons = np.zeros(numbers.shape + (3,), dtype=int)
        for n in (2,1,0):
            val = _radix**n
            digit = numbers // val
            codons[..., -(n+1)] = digit
            numbers = numbers - digit * val
        return codons

    @staticmethod
    def load(table_name):
        """
        Load a NCBI codon table.
        
        Parameters
        ----------
        table_name : str or int
            If a string is given, it is interpreted as official NCBI
            codon table name (e.g. "Vertebrate Mitochondrial").
            An integer is interpreted as NCBI codon table ID.
        
        Returns
        -------
        table : CodonTable
            The NCBI codon table.
        """
        # Loads codon tables from codon_tables.txt
        with open(CodonTable._table_file, "r") as f:
            lines = f.read().split("\n")
        
        # Extract data for codon table from file
        table_found = False
        aa = None
        init = None
        base1 = None
        base2 = None
        base3 = None
        for line in lines:
            if not line:
                table_found = False
            if type(table_name) == int and line.startswith("id"):
                # remove identifier 'id'
                if table_name == int(line[2:]):
                    table_found = True
            elif type(table_name) == str and line.startswith("name"):
                # Get list of table names from lines
                # (separated with ';')
                # remove identifier 'name'
                names = [name.strip() for name in line[4:].split(";")]
                if table_name in names:
                    table_found = True
            if table_found:
                if line.startswith("AA"):
                    #Remove identifier
                    aa = line[5:].strip()
                elif line.startswith("Init"):
                    init = line[5:].strip()
                elif line.startswith("Base1"):
                    base1 = line[5:].strip()
                elif line.startswith("Base2"):
                    base2 = line[5:].strip()
                elif line.startswith("Base3"):
                    base3 = line[5:].strip()
        
        # Create codon table from data
        if aa is not None and init is not None \
            and base1 is not None and base2 is not None and base3 is not None:
                symbol_dict = {}
                starts = []
                # aa, init and baseX all have the same length
                for i in range(len(aa)):
                    codon = base1[i] + base2[i] + base3[i]
                    if init[i] == "i":
                        starts.append(codon)
                    symbol_dict[codon] = aa[i]
                return CodonTable(symbol_dict, starts)
        else:
            raise ValueError(f"Codon table '{table_name}' was not found")

    @staticmethod
    def table_names():
        """
        The possible codon table names for :func:`load()`.
        
        Returns
        -------
        names : list of str
            List of valid codon table names.
        """
        with open(CodonTable._table_file, "r") as f:
            lines = f.read().split("\n")
        names = []
        for line in lines:
            if line.startswith("name"):
                names.extend([name.strip() for name in line[4:].split(";")])
        return names
    
    @staticmethod
    def default_table():
        """
        The default codon table.
        The table is equal to the NCBI "Standard" codon table,
        with the difference that only "ATG" is a start codon.
        
        Returns
        -------
        table : CodonTable
            The default codon table.
        """
        return _default_table


_default_table = CodonTable.load("Standard").with_start_codons(["ATG"])
