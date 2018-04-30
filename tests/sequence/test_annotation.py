# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
from biotite.sequence import Location, Feature, Annotation, AnnotatedSequence
import biotite.sequence.io.genbank as gb
import numpy as np
from os.path import join
from .util import data_dir
import pytest


def test_annotation_creation():
    feature1 = Feature("CDS", [], qual={"gene" : "test1"})
    feature2 = Feature("CDS", [], qual={"gene" : "test2"})
    feature_list = [feature1, feature2]
    annotation = Annotation(feature_list)
    for i, f in enumerate(annotation):
        assert f.key == feature_list[i].key
        assert f.qual["gene"] == feature_list[i].qual["gene"]

def test_annotation_concatenation():
    feature1 = Feature("CDS", [], qual={"gene" : "test1"})
    feature2 = Feature("CDS", [], qual={"gene" : "test2"})
    annot1 = Annotation([feature1, feature2])
    feature3 = Feature("CDS", [], qual={"gene" : "test3"})
    feature4 = Feature("CDS", [], qual={"gene" : "test4"})
    annot2 = Annotation([feature3, feature4])
    feature5 = Feature("CDS", [], qual={"gene" : "test5"})
    concat = annot1 + annot2 + feature5
    assert [f.qual["gene"] for f in concat] == ["test1", "test2", "test3",
                                                "test4", "test5"]

def test_annotation_indexing():
    feature1 = Feature("CDS", [Location(-10,30 )], qual={"gene" : "test1"})
    feature2 = Feature("CDS", [Location(20, 50 )], qual={"gene" : "test2"})
    feature3 = Feature("CDS", [Location(100,130)], qual={"gene" : "test3"})
    feature4 = Feature("CDS", [Location(150,250)], qual={"gene" : "test4"})
    feature5 = Feature("CDS", [Location(-50,200)], qual={"gene" : "test5"})
    annotation = Annotation([feature1,feature2,feature3,feature4,feature5])
    sub_annot = annotation[40:150]
    assert [f.locs[0].defect for f in sub_annot] \
                == [Location.Defect.MISS_LEFT, Location.Defect.NONE,
                    (Location.Defect.MISS_LEFT | Location.Defect.MISS_RIGHT)]
    assert [f.qual["gene"] for f in sub_annot] == ["test2", "test3", "test5"]

def test_annotated_sequence():
    sequence = seq.NucleotideSequence("ATGGCGTACGATTAGAAAAAAA")
    feature1 = Feature("misc_feature", [Location(1,2), Location(11,12)],
                       {"note" : "walker"})
    feature2 = Feature("misc_feature", [Location(16,22)], {"note" : "poly-A"})
    annotation = Annotation([feature1, feature2])
    annot_seq = AnnotatedSequence(annotation, sequence)
    assert annot_seq[2] == "T"
    assert annot_seq.sequence[2] == "G"
    annot_seq2 = annot_seq[:16]
    assert annot_seq2.sequence == seq.NucleotideSequence("ATGGCGTACGATTAG")
    assert annot_seq[feature1] == seq.NucleotideSequence("ATAT")
    assert annot_seq[feature2] == seq.NucleotideSequence("AAAAAAA")
    annot_seq[feature1] = seq.NucleotideSequence("CCCC")
    assert annot_seq.sequence == seq.NucleotideSequence("CCGGCGTACGCCTAGAAAAAAA")