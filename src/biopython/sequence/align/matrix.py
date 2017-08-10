# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..sequence import Sequence
from ..alphabet import Alphabet
import numpy as np


class SubstitutionMatrix(object):
    
    def __init__(self, alphabet1, alphabet2, matrix):
        self._alph1 = alphabet1
        self._alph2 = alphabet2
        if isinstance(matrix, dict):
            self._matrix = np.full(( len(alphabet1), len(alphabet2) ), np.nan)
            for key, value in matrix.items():
                i = alphabet1.encode(key[0])
                j = alphabet2.encode(key[1])
                self._matrix[i,j] = value
        elif isinstance(matrix, np.ndarray):
            self._matrix = np.copy(matrix)
        else:
            raise TypeError("Matrix must be either a dictionary "
                            "or an 2-D ndarray")
    
    def get_alphabet1(self):
        return self._alph1
    
    def get_alphabet2(self):
        return self._alph2
    
    def get_matrix(self):
        return np.copy(self._matrix)
    
    def get_score_by_code(code1, code2):
        return self._matrix[code1, code2]
    
    def get_score(symbol1, symbol2):
        code1 = self._alph1.encode(symbol1)
        code2 = self._alph1.encode(symbol2)
        return self._matrix[code1, code2]