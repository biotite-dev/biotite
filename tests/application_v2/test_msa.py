# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.phylo as phylo
from biotite.application_v2 import VersionError
from biotite.application_v2.clustalo import ClustalOmegaApp
from biotite.application_v2.mafft import MafftApp
from biotite.application_v2.muscle import Muscle3App, Muscle5App
from tests.util import is_not_installed

# Each implemented MSA `LocalApp` together with the name of its executable
BIN_PATH = {
    Muscle3App: "muscle",
    Muscle5App: "muscle",
    MafftApp: "mafft",
    ClustalOmegaApp: "clustalo",
}


def _app_or_skip(app_class):
    """
    Return an MSA app instance or skip the test if it is not available.
    """
    bin_path = BIN_PATH[app_class]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    try:
        return app_class(bin_path)
    except VersionError:
        pytest.skip("Invalid software version")


@pytest.fixture
def sequences():
    return [
        seq.ProteinSequence(string)
        for string in ["BIQTITE", "TITANITE", "BISMITE", "IQLITE"]
    ]


@pytest.mark.parametrize(
    ["app_class", "exp_ali", "exp_order"],
    [
        (
            Muscle3App,
            "BIQT-ITE\n"
            "TITANITE\n"
            "BISM-ITE\n"
            "-IQL-ITE",
            [1, 2, 0, 3],
        ),
        (
            Muscle5App,
            "BI-QTITE\n"
            "TITANITE\n"
            "BI-SMITE\n"
            "-I-QLITE",
            [1, 3, 0, 2],
        ),
        (
            MafftApp,
            "-BIQTITE\n"
            "TITANITE\n"
            "-BISMITE\n"
            "--IQLITE",
            [0, 3, 2, 1],
        ),
        (
            ClustalOmegaApp,
            "-BIQTITE\n"
            "TITANITE\n"
            "-BISMITE\n"
            "--IQLITE",
            [1, 2, 0, 3],
        ),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)  # fmt: skip
def test_msa(sequences, app_class, exp_ali, exp_order):
    """
    Test the MSA software on short toy sequences with known alignment
    result.
    """
    app = _app_or_skip(app_class)

    result = app.run(sequences).result()
    assert str(result.alignment) == exp_ali
    assert result.order.tolist() == exp_order


@pytest.mark.parametrize("app_class", [Muscle3App, MafftApp, ClustalOmegaApp])
def test_large_sequence_number(app_class):
    """
    Test MSA software on a large number of (identical) sequences.
    The quality of the MSA is not evaluated here, therefore identical
    sequences are used.
    """
    SEQ_LENGTH = 50
    SEQ_NUMBER = 100

    app = _app_or_skip(app_class)

    sequence = seq.ProteinSequence()
    sequence.code = np.random.default_rng().integers(20, size=SEQ_LENGTH)
    sequences = [sequence] * SEQ_NUMBER

    result = app.run(sequences).result()
    # Expect completely matching sequences
    assert result.alignment.trace.tolist() == [
        [i] * SEQ_NUMBER for i in range(SEQ_LENGTH)
    ]


@pytest.mark.parametrize("app_class", [Muscle3App, MafftApp])
def test_custom_substitution_matrix(sequences, app_class):
    app = _app_or_skip(app_class)

    alph = seq.ProteinSequence.alphabet
    # Strong identity matrix
    score_matrix = np.identity(len(alph), dtype=int) * 1000
    matrix = align.SubstitutionMatrix(alph, alph, score_matrix)
    exp_ali = (
        "BI-QTITE\n"
        "TITANITE\n"
        "BI-SMITE\n"
        "-I-QLITE"
    )  # fmt: skip
    result = app.run(sequences, matrix=matrix).result()
    assert str(result.alignment) == exp_ali


# Ignore warnings about missing tree output in MUSCLE
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("app_class", [Muscle3App, MafftApp])
def test_custom_sequence_type(app_class):
    app = _app_or_skip(app_class)

    alph = seq.Alphabet(("foo", "bar", 42))
    sequences = [seq.GeneralSequence(alph, sequence) for sequence in [
        ["foo", "bar", 42, "foo",        "foo", 42, 42],
        ["foo",        42, "foo", "bar", "foo", 42, 42],
    ]]  # fmt: skip
    exp_trace = [
        [0, 0],
        [1, -1],
        [2, 1],
        [3, 2],
        [-1, 3],
        [4, 4],
        [5, 5],
        [6, 6],
    ]
    # Strong identity matrix
    score_matrix = np.identity(len(alph), dtype=int)
    score_matrix[score_matrix == 0] = -1000
    score_matrix[score_matrix == 1] = 1000
    matrix = align.SubstitutionMatrix(alph, alph, score_matrix)

    result = app.run(sequences, matrix=matrix).result()
    assert result.alignment.sequences == sequences
    assert result.alignment.trace.tolist() == exp_trace


@pytest.mark.parametrize("app_class", [Muscle3App, MafftApp, ClustalOmegaApp])
def test_invalid_sequence_type_no_matrix(app_class):
    """
    A custom substitution matrix is required for normally unsupported
    sequence types.
    """
    app = _app_or_skip(app_class)

    alph = seq.Alphabet(("foo", "bar", 42))
    sequences = [seq.GeneralSequence(alph, sequence) for sequence in [
        ["foo", "bar", 42, "foo",        "foo", 42, 42],
        ["foo",        42, "foo", "bar", "foo", 42, 42],
    ]]  # fmt: skip
    with pytest.raises(TypeError):
        app.run(sequences)


@pytest.mark.parametrize("app_class", [Muscle3App, MafftApp, ClustalOmegaApp])
def test_invalid_sequence_type_unsuitable_alphabet(app_class):
    """
    The alphabet of the custom sequence type cannot be longer than the
    amino acid alphabet.
    """
    app = _app_or_skip(app_class)

    alph = seq.Alphabet(range(50))
    sequences = [
        seq.GeneralSequence(alph, sequence)
        for sequence in [
            [1, 2, 3],
            [1, 2, 3],
        ]
    ]
    with pytest.raises(TypeError):
        app.run(sequences)


def test_invalid_muscle_version():
    """
    One of `Muscle3App` and `Muscle5App` raises an error, since one is
    incompatible with the installed version.
    """
    if is_not_installed("muscle"):
        pytest.skip("'muscle' is not installed")

    with pytest.raises(VersionError):
        Muscle3App("muscle")
        Muscle5App("muscle")


def test_muscle3_tree(sequences):
    """
    MUSCLE 3 provides both guide trees.
    """
    app = _app_or_skip(Muscle3App)

    result = app.run(sequences).result()
    assert result.guide_tree_kmer is not None
    assert result.guide_tree_identity is not None


def test_muscle3_gap_penalty(sequences):
    """
    Setting a gap penalty is reflected in the executed command and still
    yields a valid alignment.
    """
    app = _app_or_skip(Muscle3App)

    future = app.run(sequences, gap_penalty=(-12, -1))
    assert "-gapopen" in future.command
    assert "-gapextend" in future.command
    alignment = future.result().alignment
    assert len(alignment.sequences) == len(sequences)


def test_mafft_tree(sequences):
    """
    MAFFT provides a guide tree.
    """
    app = _app_or_skip(MafftApp)

    result = app.run(sequences).result()
    assert result.guide_tree is not None


def test_clustalo_matrix(sequences):
    """
    Clustal-Omega returns the distance matrix when the full matrix
    calculation is used.
    """
    app = _app_or_skip(ClustalOmegaApp)

    ref_matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ], dtype=float)  # fmt: skip
    result = app.run(
        sequences, distance_matrix=ref_matrix, use_full_matrix=True
    ).result()
    assert np.allclose(ref_matrix, result.distance_matrix)


def test_clustalo_tree(sequences):
    """
    Clustal-Omega accepts a guide tree as input and provides one as
    output.
    """
    app = _app_or_skip(ClustalOmegaApp)

    leaves = [phylo.TreeNode(index=i) for i in range(len(sequences))]
    inter1 = phylo.TreeNode([leaves[0], leaves[1]], [1.0, 1.0])
    inter2 = phylo.TreeNode([leaves[2], leaves[3]], [2.5, 2.5])
    root = phylo.TreeNode([inter1, inter2], [3.5, 2])
    tree = phylo.Tree(root)

    # A guide tree can be provided as input
    result_with_tree = app.run(sequences, guide_tree=tree).result()
    assert result_with_tree.guide_tree is tree

    # A guide tree is provided as output
    result_without_tree = app.run(sequences).result()
    assert result_without_tree.guide_tree is not None
