# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from biotite.sequence import NucleotideSequence
from biotite.sequence import ProteinSequence
from biotite.sequence import Alphabet
from biotite.sequence import GeneralSequence
from biotite.sequence import LetterAlphabet
from biotite.sequence import Location
from biotite.sequence import Feature
from biotite.sequence import Annotation
from biotite.sequence import AnnotatedSequence
from biotite.sequence.align import Alignment
from biotite.structure import Atom
import numpy as np
from numpy import float32, int32
from biotite.sequence import CodonTable
from biotite.sequence.align import SubstitutionMatrix
from biotite.sequence import SequenceProfile
import pytest

__author__ = "Maximilian Greil"


@pytest.mark.parametrize("repr_object",
                         [NucleotideSequence("AACTGCTA"),
                          NucleotideSequence("AACTGCTA", ambiguous=True),
                          ProteinSequence("BIQTITE"),
                          Alphabet(["X", "Y", "Z"]),
                          GeneralSequence(Alphabet(["X", 42, False]), ["X", 42, "X"]),
                          LetterAlphabet(["X", "Y", "Z"]),
                          Location(98, 178),
                          Feature("CDS", [Location(98, 178)], qual={"gene": "test1"}),
                          Annotation([Feature("CDS", [Location(98, 178)], qual={"gene": "test1"})]),
                          AnnotatedSequence(Annotation([Feature("CDS", [Location(98, 178)], qual={"gene": "test1"})]),
                                            NucleotideSequence("AACTGCTA")),
                          Alignment([NucleotideSequence("CGTCAT", ambiguous=False),
                                     NucleotideSequence("TCATGC", ambiguous=False)],
                                    np.array([[0, -1], [1, -1], [2, 0], [3, 1], [4, 2], [5, 3], [-1, 4], [-1, 5]]),
                                    score=-20),
                          Atom([1, 2, 3], chain_id="A"),
                          CodonTable.default_table(),
                          SubstitutionMatrix(Alphabet(["foo", "bar"]), Alphabet([1, 2, 3]),
                                             {("foo", 1): 5, ("foo", 2): 10, ("foo", 3): 15, ("bar", 1): 42,
                                              ("bar", 2): 42, ("bar", 3): 42}),
                          SequenceProfile(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2], [0, 2, 0, 0],
                                                    [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 0]]),
                                          np.array([1, 1, 0, 0, 0, 0, 1, 1]),
                                          Alphabet(["A", "C", "G", "T"]))])
def test_repr(repr_object):
    assert eval(repr(repr_object)) == repr_object
