# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
import biotite.sequence.phylo as phylo
import biotite.sequence.align as align
from biotite.application.muscle import MuscleApp
from biotite.application.mafft import MafftApp
from biotite.application.clustalo import ClustalOmegaApp
import numpy as np
import pytest
import shutil


@pytest.fixture
def sequences():
    return [seq.ProteinSequence(string) for string in [
        "BIQTITE",
        "TITANITE",
        "BISMITE",
        "IQLITE"
]]


@pytest.mark.skipif(shutil.which("muscle")   is None or
                    shutil.which("mafft")    is None or
                    shutil.which("clustalo") is None,
                    reason="At least one MSA application is not installed")
@pytest.mark.parametrize("app_cls, exp_ali, exp_order",
    [(MuscleApp,
      "BIQT-ITE\n"
      "TITANITE\n"
      "BISM-ITE\n"
      "-IQL-ITE",
      [1,2,0,3]),                
     (MafftApp,
      "-BIQTITE\n"
      "TITANITE\n"
      "-BISMITE\n"
      "--IQLITE",
      [0,3,2,1]),
     (ClustalOmegaApp, 
      "-BIQTITE\n"
      "TITANITE\n"
      "-BISMITE\n"
      "--IQLITE",
     [1,2,0,3])]
)
def test_msa(sequences, app_cls, exp_ali, exp_order):
    app = app_cls(sequences)
    app.start()
    app.join()
    alignment = app.get_alignment()
    order = app.get_alignment_order()
    assert str(alignment) == exp_ali
    assert order.tolist() == exp_order


def test_additional_options(sequences):
    app1 = ClustalOmegaApp(sequences)
    app1.start()
    
    app2 = ClustalOmegaApp(sequences)
    app2.add_additional_options(["--full"])
    app2.start()
    
    app1.join()
    app2.join()
    assert "--full" not in app1.get_command()
    assert "--full"     in app2.get_command()
    assert app1.get_alignment() == app2.get_alignment()


@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp])
def test_custom_substitution_matrix(sequences, app_cls):
    alph = seq.ProteinSequence.alphabet
    # Strong identity matrix
    score_matrix = np.identity(len(alph)) * 1000
    matrix = align.SubstitutionMatrix(alph, alph, score_matrix)
    exp_ali = (
        "BI-QTITE\n"
        "TITANITE\n"
        "BI-SMITE\n"
        "-I-QLITE"
    )
    app = app_cls(sequences, matrix=matrix)
    app.start()
    app.join()
    alignment = app.get_alignment()
    assert str(alignment) == exp_ali


@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp])
def test_custom_sequence_type(app_cls):
    alph = seq.Alphabet(("foo", "bar", 42))
    sequences = [seq.GeneralSequence(alph, sequence) for sequence in [
        ["foo", "bar", 42, "foo",        "foo", 42, 42],
        ["foo",        42, "foo", "bar", "foo", 42, 42],
    ]]
    exp_trace = [
        [ 0,  0],
        [ 1, -1],
        [ 2,  1],
        [ 3,  2],
        [-1,  3],
        [ 4,  4],
        [ 5,  5],
        [ 6,  6],
    ]
    # Strong identity matrix
    score_matrix = np.identity(len(alph))
    score_matrix[score_matrix == 0] = -1000
    score_matrix[score_matrix == 1] = 1000
    matrix = align.SubstitutionMatrix(alph, alph, score_matrix)
    app = app_cls(sequences, matrix=matrix)
    app.start()
    app.join()
    alignment = app.get_alignment()
    assert alignment.sequences == sequences
    assert alignment.trace.tolist() == exp_trace


@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp, ClustalOmegaApp])
def test_invalid_sequence_type(app_cls):
    pass


def test_clustalo_matrix(sequences):
    ref_matrix = [
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0]
    ]
    app = ClustalOmegaApp(sequences)
    app.full_matrix_calculation()
    app.set_distance_matrix(np.array(ref_matrix))
    app.start()
    app.join()
    test_matrix = app.get_distance_matrix()
    assert np.allclose(ref_matrix, test_matrix)


def test_clustalo_tree(sequences):
    leaves = [phylo.TreeNode(index=i) for i in range(len(sequences))]
    inter1 = phylo.TreeNode(leaves[0], leaves[1], 1.0, 1.0)
    inter2 = phylo.TreeNode(leaves[2], leaves[3], 2.5, 2.5)
    root = phylo.TreeNode(inter1, inter2, 3.5, 2)
    tree = phylo.Tree(root)
    # You cannot simultaneously set and get a tree in ClustalOmega
    # -> Test whether both is possible in separate calls
    app = ClustalOmegaApp(sequences)
    app.set_guide_tree(tree)
    app.start()
    app.join()

    app = ClustalOmegaApp(sequences)
    app.start()
    app.join()
    assert app.get_guide_tree() is not None


def test_mafft_tree(sequences):
    app = MafftApp(sequences)
    app.start()
    app.join()
    tree = app.get_guide_tree()
    assert tree is not None