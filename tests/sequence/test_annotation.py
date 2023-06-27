# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
from biotite.sequence import Location, Feature, Annotation, AnnotatedSequence
import biotite.sequence.io.genbank as gb
import numpy as np
from os.path import join
from ..util import data_dir
import pytest


def test_annotation_creation():
    feature1 = Feature("CDS", [seq.Location(1,2)], qual={"gene" : "test1"})
    feature2 = Feature("CDS", [seq.Location(3,4)], qual={"gene" : "test2"})
    feature_list = [feature1, feature2]
    annotation = Annotation(feature_list)
    for feature in annotation:
        assert feature.key in [f.key for f in feature_list]
        assert feature.qual["gene"] in [
            f.qual["gene"] for f in feature_list
        ]

def test_annotation_concatenation():
    feature1 = Feature("CDS", [seq.Location(1,1)], qual={"gene" : "test1"})
    feature2 = Feature("CDS", [seq.Location(2,2)], qual={"gene" : "test2"})
    annot1 = Annotation([feature1, feature2])
    feature3 = Feature("CDS", [seq.Location(3,3)], qual={"gene" : "test3"})
    feature4 = Feature("CDS", [seq.Location(4,4)], qual={"gene" : "test4"})
    annot2 = Annotation([feature3, feature4])
    feature5 = Feature("CDS", [seq.Location(5,5)], qual={"gene" : "test5"})
    concat = annot1 + annot2 + feature5
    assert set([f.qual["gene"] for f in concat]) \
        == set(["test1", "test2", "test3", "test4", "test5"])

def test_annotation_indexing():
    feature1 = Feature("CDS", [Location(-10,30 )], qual={"gene" : "test1"})
    feature2 = Feature("CDS", [Location(20, 50 )], qual={"gene" : "test2"})
    feature3 = Feature("CDS", [Location(100,130)], qual={"gene" : "test3"})
    feature4 = Feature("CDS", [Location(150,250)], qual={"gene" : "test4"})
    feature5 = Feature("CDS", [Location(-50,200)], qual={"gene" : "test5"})
    annotation = Annotation([feature1,feature2,feature3,feature4,feature5])
    sub_annot = annotation[40:150]
    # Only one location per feature
    assert set([list(f.locs)[0].defect for f in sub_annot]) \
        == set([Location.Defect.MISS_LEFT, Location.Defect.NONE,
                (Location.Defect.MISS_LEFT | Location.Defect.MISS_RIGHT)])
    assert set([f.qual["gene"] for f in sub_annot]) \
        == set(["test2", "test3", "test5"])

def test_annotated_sequence():
    sequence = seq.NucleotideSequence("ATGGCGTACGATTAGAAAAAAA")
    feature1 = Feature("misc_feature", [Location(1,2), Location(11,12)],
                       {"note" : "walker"})
    feature2 = Feature("misc_feature", [Location(16,22)], {"note" : "poly-A"})
    annotation = Annotation([feature1, feature2])
    annot_seq = AnnotatedSequence(annotation, sequence)
    assert annot_seq[2] == "T"
    assert annot_seq.sequence[2] == "G"
    
    # test slicing with only stop
    annot_seq2 = annot_seq[:16]
    assert annot_seq2.sequence == seq.NucleotideSequence("ATGGCGTACGATTAG")
    
    # test slicing with only start
    annot_seq3 = annot_seq[16:]
    assert annot_seq3.sequence == seq.NucleotideSequence("AAAAAAA")
    
    assert annot_seq[feature1] == seq.NucleotideSequence("ATAT")
    assert annot_seq[feature2] == seq.NucleotideSequence("AAAAAAA")
    annot_seq[feature1] = seq.NucleotideSequence("CCCC")
    assert annot_seq.sequence == seq.NucleotideSequence("CCGGCGTACGCCTAGAAAAAAA")

def test_reverse_complement():
    gb_file = gb.GenBankFile.read(join(data_dir("sequence"), "ec_bl21.gb"))
    annot_seq = gb.get_annotated_sequence(gb_file)
    assert annot_seq == annot_seq.reverse_complement().reverse_complement()