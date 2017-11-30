# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import copy
from os.path import join, dirname, realpath
from .sequence import Sequence
from .seqtypes import NucleotideSequence, ProteinSequence

__all__ = ["CodonTable"]


class CodonTable(object):
    
    # file for codon datbles
    _table_file = join(dirname(realpath(__file__)), "codon_tables.txt")
    
    def __init__(self, codon_dict, starts):
        self._symbol_dict = copy.copy(codon_dict)
        self._code_dict = {}
        for key, value in self._symbol_dict.items():
            key_code = tuple(Sequence.encode(key, NucleotideSequence.alphabet))
            val_code = ProteinSequence.alphabet.encode(value)
            self._code_dict[key_code] = val_code
        self._start_symbols = tuple((starts))
        self._start_codes = tuple(
            [tuple(Sequence.encode(start, NucleotideSequence.alphabet))
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
                raise TypeError("'{:}' is an invalid index".format(str(item)))
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
            raise TypeError("'{:}' is an invalid index".format(str(item)))
    
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
        if code:
            return copy.copy(self._code_dict)
        else:
            return copy.copy(self._symbol_dict)
    
    def start_codons(self, code=False):
        if code:
            return self._start_codes
        else:
            return self._start_symbols
    
    @staticmethod
    def load(table_name):
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
            raise ValueError("Codon table '{:}' was not found"
                             .format(str(table_name)))
    
    @staticmethod
    def table_names():
        with open(CodonTable._table_file, "r") as f:
            lines = f.read().split("\n")
        names = []
        for line in lines:
            if line.startswith("name"):
                names.extend([name.strip() for name in line[4:].split(";")])
        return names
    
    @staticmethod
    def default_table():
        return _default_table

_default_table = CodonTable(CodonTable.load("Standard").codon_dict(), ["ATG"])
    
    