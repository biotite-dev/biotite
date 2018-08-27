# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["CodonTable"]

import copy
from os.path import join, dirname, realpath
from .seqtypes import NucleotideSequence, ProteinSequence


class CodonTable(object):
    """
    A `CodonTable` maps a codon (sequence of 3 nucleotides) to an amino
    acid. It also defines start codons. A `CodonTable`
    takes/outputs either the symbols or code of the codon/amino acid.
    
    Furthermore, this class is able to give a list of codons that
    corresponds to a given amino acid.
    
    The `load()` method allows loading of NCBI codon tables.
    
    Objects of this class are immutable.
    
    Parameters
    ----------
    codon_dict : dict
        A dictionary that maps codons to amino acids. The keys must be
        strings of length 3 and the values strings of length 1
        (all upper case). The dictionary must provide entries for all
        64 possible codons.
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
    ((1, 2, 3), (1, 2, 1), (1, 2, 0), (1, 2, 2), (0, 2, 0), (0, 2, 2))
    """
    
    # file for codon tables
    _table_file = join(dirname(realpath(__file__)), "codon_tables.txt")
    
    def __init__(self, codon_dict, starts):
        # Check if 'starts' is iterable objectof length 3 strings
        for start in starts:
            if not isinstance(start, str) or len(start) != 3:
                raise ValueError(f"Invalid codon '{start}' as start codon")
        self._symbol_dict = dict(codon_dict.items())
        self._code_dict = {}
        for key, value in self._symbol_dict.items():
            key_code = tuple(NucleotideSequence.alphabet.encode_multiple(key))
            val_code = ProteinSequence.alphabet.encode(value)
            self._code_dict[key_code] = val_code
        self._start_symbols = tuple((starts))
        self._start_codes = tuple(
            [tuple(NucleotideSequence.alphabet.encode_multiple(start))
             for start in self._start_symbols]
        )
    
    def __getitem__(self, item):
        if isinstance(item, str):
            if len(item) == 1:
                # Amino acid -> return possible codons
                codons = []
                for key, val in self._symbol_dict.items():
                    if val == item:
                        codons.append(key)
                return tuple(codons)
            elif len(item) == 3:
                # Codon -> return corresponding amino acid
                return self._symbol_dict[item]
            else:
                raise ValueError(f"'{item}' is an invalid index")
        elif isinstance(item, int):
            # Code for amino acid -> return possible codon codes
            codons = []
            for key, val in self._code_dict.items():
                if val == item:
                    codons.append(key)
            return tuple(codons)
        elif isinstance(item, tuple):
            # Code for codon -> return corresponding amino acid code
            return self._code_dict[item]
        else:
            raise TypeError(
                f"'{type(item).__name__}' objects are invalid indices"
            )
    
    def __str__(self):
        string = ""
        bases = ["A","C","G","T"]
        for b1 in bases:
            for b2 in bases:
                for b3 in bases:
                    codon = b1 + b2 + b3
                    string += codon + " " + self._symbol_dict[codon]
                    # Indicator for start codon
                    if codon in self._start_symbols:
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
            return copy.copy(self._code_dict)
        else:
            return copy.copy(self._symbol_dict)
    
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
            return self._start_codes
        else:
            return self._start_symbols
    
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
        
        # Create codon tbale from data
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
        The possible codon table names for `load()`.
        
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

_default_table = CodonTable(CodonTable.load("Standard").codon_dict(), ["ATG"])
    
    