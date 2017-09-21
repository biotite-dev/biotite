# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..sequence import Sequence
from ..seqtypes import NucleotideSequence, ProteinSequence
from ..alphabet import Alphabet
import numpy as np
import os.path

__all__ = ["SubstitutionMatrix"]


class SubstitutionMatrix(object):
    """
    A `SubstitutionMatrix` is the base for scoring in sequence
    alignments. A `SubstitutionMatrix` maps each possible pairing
    of a symbol of a first alphabet with a symbol of a second alphabet
    to a score (integer).
    
    The class uses a 2-D (m x n) `ndarray` (dtype=`np.int32`),
    where each element stores the score for a symbol pairing, indexed by
    the symbol codes of the respective symbols in an m-length alphabet 1
    and an n-length alphabet 2.
    
    Instances can be created by either providing directly the
    2-D `ndarray`, or by providing a dictionary, where the
    keys are pairing tuples and values are the corresponding scores.
    The pairing tuples consist of a symbol of alphabet 1 as first
    element and a symbol of alphabet 2 as second element. Parings have
    to be provided for each possible combination.
    
    Since this class can handle two different alphabets, it is possible
    to align two different types of sequences
    
    Objects of this class are immutable.
    
    Parameters
    ----------
    alphabet1 : Alphabet, length=m
        The first alphabet of the substitution matrix.
    alphabet2 : Alphabet, length=n
        The second alphabet of the substitution matrix.
    matrix : ndarray, shape=(m,n) or dict
        Either a symbol code indexed `ndarray` containing the scores
        or a dictionary mapping the symbol pairing to scores.
    
    Examples
    --------
    
    >>> alph_1 = Alphabet(["foo","bar"])
    >>> alph_2 = Alphabet([1,2,3])
    >>> matrix_dict = {("foo",1):5,  ("foo",2):10, ("foo",3):15,
    ...                ("bar",1):42, ("bar",2):42, ("bar",3):42}
    >>> matrix = SubstitutionMatrix(alph_1, alph_2, matrix_dict)
    >>> print(matrix.get_matrix())
    [[ 5 10 15]
     [42 42 42]]
    >>> print(matrix.get_score("foo", 2))
    10
    >>> print(matrix.get_score_by_code(0, 1))
    10
    
    See also
    --------
    biopython.database.matrix.fetch
    """
    
    def __init__(self, alphabet1, alphabet2, matrix):
        self._alph1 = alphabet1
        self._alph2 = alphabet2
        if isinstance(matrix, dict):
            matrix_dict = matrix
            self._matrix = np.zeros(( len(alphabet1), len(alphabet2) ),
                                    dtype=np.int32)
            for i in range(len(alphabet1)):
                for j in range(len(alphabet2)):
                    sym1 = alphabet1.decode(i)
                    sym2 = alphabet2.decode(j)
                    self._matrix[i,j] = int(matrix_dict[sym1, sym2])
        elif isinstance(matrix, np.ndarray):
            alph_shape = (len(alphabet1), len(alphabet2))
            if matrix.shape != alph_shape:
                raise ValueError("Matrix has shape {:}, "
                                 "but {:} is required"
                                 .format(matrix.shape, alph_shape))
            self._matrix = np.copy(matrix.astype(np.int32))
        else:
            raise TypeError("Matrix must be either a dictionary "
                            "or an 2-D ndarray")
    
    def get_alphabet1(self):
        """
        Get the first alphabet. 
        
        Returns
        -------
        alphabet : Alphabet
            The first alphabet.
        """
        return self._alph1
    
    def get_alphabet2(self):
        """
        Get the second alphabet. 
        
        Returns
        -------
        alphabet : Alphabet
            The second alphabet.
        """
        return self._alph2
    
    def get_matrix(self):
        """
        Get a copy of the 2-D `ndarray` containing the score values. 
        
        Returns
        -------
        matrix : ndarray
            The symbol code indexed score matrix.
        """
        return np.copy(self._matrix)
    
    def transpose(self):
        """
        Get a copy of this instance, where the alphabets are swapped
        with other.
        
        Returns
        -------
        transposed : SubstitutionMatrix
            The transposed substitution matrix.
        """
        new_alph1 = self._alph2
        new_alph2 = self._alph1
        new_matrix = np.transpose(self._matrix)
        return SubstitutionMatrix(new_alph1, new_alph2, new_matrix)
    
    def get_score_by_code(self, code1, code2):
        """
        Get the substitution score of two symbols,
        represented by their code.
        
        Parameters
        ----------
        code1, code2 : int
            Symbol codes of the two symbols to be aligned.
        
        Returns
        -------
        score : int
            The substitution / alignment score.
        """
        return self._matrix[code1, code2]
    
    def get_score(self, symbol1, symbol2):
        """
        Get the substitution score of two symbols.
        
        Parameters
        ----------
        symbol1, symbol2 : object
            Symbols to be aligned.
        
        Returns
        -------
        score : int
            The substitution / alignment score.
        """
        code1 = self._alph1.encode(symbol1)
        code2 = self._alph2.encode(symbol2)
        return self._matrix[code1, code2]
    
    def shape(self):
        """
        Get the shape (i.e. the length of both alphabets)
        of the subsitution matrix.
        
        Returns
        -------
        shape : tuple
            Matrix shape.
        """
        return (len(self._alph1), len(self._alph2))
    
    def __str__(self):
        string = "{:>3}".format("")
        for symbol in self._alph2:
            string += " {:>3}".format(str(symbol))
        string += "\n"
        for i, symbol in enumerate(self._alph1):
            string += "{:>3}".format(str(symbol))
            for j in range(len(self._alph2)):
                string += " {:>3}".format(int(self._matrix[i,j]))
            string += "\n"
        return string
    
    @staticmethod
    def std_protein_matrix():
        """
        Get the default `SubstitutionMatrix` for protein sequence
        alignments, which is BLOSUM62.
        
        Returns
        -------
        matrix : SubstitutionMatrix
            Default matrix.
        """
        return _matrix_blosum62
    
    @staticmethod
    def std_nucleotide_matrix():
        """
        Get the default `SubstitutionMatrix` for DNA sequence
        alignments.
        
        Returns
        -------
        matrix : SubstitutionMatrix
            Default matrix.
        """
        return _matrix_nuc


# Preformatted BLOSUM62 and NUC substitution matrix from NCBI

_matrix_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "matrix_data")

_matrix = np.load(os.path.join(_matrix_dir, "blosum62.npy"))
_alph = ProteinSequence.alphabet 
_matrix_blosum62 = SubstitutionMatrix(_alph, _alph, _matrix)

_matrix = np.load(os.path.join(_matrix_dir, "nuc.npy"))
_alph = NucleotideSequence.alphabet 
_matrix_nuc = SubstitutionMatrix(_alph, _alph, _matrix)

