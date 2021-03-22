# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from biotite.sequence import NucleotideSequence
from biotite.sequence import ProteinSequence
from biotite.sequence import Alphabet
from biotite.sequence import GeneralSequence
from biotite.sequence import LetterAlphabet
import pytest

__author__ = "Maximilian Greil"


@pytest.mark.parametrize("repr_object",
                         [NucleotideSequence("AACTGCTA"),
                          NucleotideSequence("AACTGCTA", ambiguous=True),
                          ProteinSequence("BIQTITE"),
                          Alphabet(["X", "Y", "Z"]),
                          GeneralSequence(Alphabet(["X", "Y", "Z"]), "XYZ"),
                          LetterAlphabet(["X", "Y", "Z"])])
def test_repr(repr_object):
    assert eval(repr(repr_object)) == repr_object
