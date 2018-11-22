# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
import biotite.sequence.io.genbank as gb
import numpy as np
from os.path import join
from .util import data_dir
import pytest


def test_genbank_conversion():
    gb_file = gb.GenBankFile()
    gb_file.read(join(data_dir, "ec_bl21.gb"))
    assert gb_file.get_locus()["length"] == "4558953"
    assert gb_file.get_locus()["type"] == "DNA circular"
    assert gb_file.get_definition() == ("Escherichia coli BL21(DE3), "
                                        "complete genome.")
    assert gb_file.get_version() == "CP001509.3"
    assert gb_file.get_gi() == "296142109"
    assert gb_file.get_db_link() == {"BioProject" : "PRJNA20713",
                                     "BioSample" : "SAMN02603478"}
    assert len(gb_file.get_references()) == 5
    for ref in gb_file.get_references()[1:]:
        assert ref["location"] == (1,4558953)
        assert ref["journal"].endswith("Republic of Korea")
    assert gb_file.get_comment() == ("On May 17, 2010 this sequence version "
                                     "replaced CP001509.2. Bacteria available "
                                     "from F. William Studier "
                                     "(studier\x40bnl.gov).")
    annotation = gb_file.get_annotation(include_only=["CDS"])
    feature = annotation.get_features()[5]
    assert feature.key == "CDS"
    assert feature.qual["gene"] == "yaaA"
    assert feature.qual["transl_table"] == "11"
    # Get first loc
    for loc in feature.locs:
        break
    assert str(loc) == "< 5681-6457"

def test_genpept_conversion():
    gp_file = gb.GenPeptFile()
    gp_file.read(join(data_dir, "bt_lysozyme.gp"))
    assert gp_file.get_locus()["length"] == "147"
    assert gp_file.get_definition() == "lysozyme [Bos taurus]."
    assert gp_file.get_version() == "AAC37312.1"
    assert gp_file.get_gi() == "163334"
    assert gp_file.get_db_source() == "locus BOVLYSOZMC accession M95099.1"
    ref = gp_file.get_references()[0]
    assert ref["title"] == ("Characterization of the cow stomach lysozyme "
                            "genes: repetitive DNA and concerted evolution")
    assert ref["pubmed"] == "8308905"
    assert ref["location"] == (1,147)
    assert gp_file.get_comment() == "Method: conceptual translation."
    annotation = gp_file.get_annotation()
    feature = annotation.get_features()[3]
    assert feature.key == "Site"
    assert feature.qual["note"] == "lysozyme catalytic cleft [active]"
    assert feature.qual["site_type"] == "active"
    firsts = [loc.first for loc in feature.locs]
    lasts = [loc.last for loc in feature.locs]
    assert firsts == [52,55,62,76,78,81,117,120,125]
    assert lasts  == [53,55,62,76,78,81,117,120,126]

def test_multi_file():
    multi_file = gb.MultiFile(file_type="gp")
    multi_file.read(join(data_dir, "multifile.gp"))
    accessions = [f.get_accession() for f in multi_file]
    assert accessions == ["1L2Y_A", "3O5R_A", "5UGO_A"]