# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from distutils.version import Version
import biotite.sequence as seq
import biotite.sequence.phylo as phylo
import biotite.sequence.align as align
from biotite.application import VersionError
from biotite.application.muscle import MuscleApp, Muscle5App
from biotite.application.mafft import MafftApp
from biotite.application.clustalo import ClustalOmegaApp
import numpy as np
import pytest
import shutil
from ..util import is_not_installed


BIN_PATH = {
    MuscleApp : "muscle",
    Muscle5App : "muscle",
    MafftApp : "mafft",
    ClustalOmegaApp: "clustalo"
}


@pytest.fixture
def sequences():
    return [seq.ProteinSequence(string) for string in [
        "BIQTITE",
        "TITANITE",
        "BISMITE",
        "IQLITE"
]]


@pytest.mark.parametrize("app_cls, exp_ali, exp_order",
    [(MuscleApp,
      "BIQT-ITE\n"
      "TITANITE\n"
      "BISM-ITE\n"
      "-IQL-ITE",
      [1, 2, 0, 3]),
     (Muscle5App,
      "BI-QTITE\n"
      "TITANITE\n"
      "BI-SMITE\n"
      "-I-QLITE",
      [0, 3, 1, 2]),
     (MafftApp,
      "-BIQTITE\n"
      "TITANITE\n"
      "-BISMITE\n"
      "--IQLITE",
      [0, 3, 2, 1]),
     (ClustalOmegaApp,
      "-BIQTITE\n"
      "TITANITE\n"
      "-BISMITE\n"
      "--IQLITE",
     [1, 2, 0, 3])]
)
def test_msa(sequences, app_cls, exp_ali, exp_order):
    """
    Test MSA software on short toy sequences with known alignment
    result.
    """
    bin_path = BIN_PATH[app_cls]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")

    try:
        app = app_cls(sequences)
    except VersionError:
        pytest.skip(f"Invalid software version")
    app.start()
    app.join()
    alignment = app.get_alignment()
    order = app.get_alignment_order()
    assert str(alignment) == exp_ali
    assert order.tolist() == exp_order


@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp, ClustalOmegaApp])
def test_large_sequence_number(app_cls):
    """
    Test MSA software on large number of sequences.
    The quality of the MSA is not evaluated here, therefore identical
    sequences are used
    """
    SEQ_LENGTH = 50
    SEQ_NUMBER = 100

    bin_path = BIN_PATH[app_cls]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")

    # Create random sequence
    sequence = seq.ProteinSequence()
    sequence.code = np.random.randint(20, size=SEQ_LENGTH)
    # Use identical sequences
    sequences = [sequence] * SEQ_NUMBER

    try:
        app = app_cls(sequences)
    except VersionError:
        pytest.skip(f"Invalid software version")
    app.start()
    app.join()
    alignment = app.get_alignment()
    # Expect completely matching sequences
    assert alignment.trace.tolist() == [
        [i]*SEQ_NUMBER for i in range(SEQ_LENGTH)
    ]

def test_additional_options(sequences):
    bin_path = BIN_PATH[ClustalOmegaApp]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")

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
    bin_path = BIN_PATH[app_cls]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
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
    try:
        app = app_cls(sequences, matrix=matrix)
    except VersionError:
        pytest.skip(f"Invalid software version")
    app.start()
    app.join()
    alignment = app.get_alignment()
    assert str(alignment) == exp_ali


# Ignore warnings about missing tree output in MUSCLE
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp])
def test_custom_sequence_type(app_cls):
    bin_path = BIN_PATH[app_cls]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
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
    try:
        app = app_cls(sequences, matrix=matrix)
    except VersionError:
        pytest.skip(f"Invalid software version")
    app.start()
    app.join()
    alignment = app.get_alignment()
    assert alignment.sequences == sequences
    assert alignment.trace.tolist() == exp_trace


@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp, ClustalOmegaApp])
def test_invalid_sequence_type_no_matrix(app_cls):
    """
    A custom substitution matrix is required for normally unsupported
    sequence types.
    """
    bin_path = BIN_PATH[app_cls]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    alph = seq.Alphabet(("foo", "bar", 42))
    sequences = [seq.GeneralSequence(alph, sequence) for sequence in [
        ["foo", "bar", 42, "foo",        "foo", 42, 42],
        ["foo",        42, "foo", "bar", "foo", 42, 42],
    ]]
    with pytest.raises(TypeError):
        try:
            app_cls(sequences)
        except VersionError:
            pytest.skip(f"Invalid software version")


@pytest.mark.parametrize("app_cls", [MuscleApp, MafftApp, ClustalOmegaApp])
def test_invalid_sequence_type_unsuitable_alphabet(app_cls):
    """
    The alphabet of the custom sequence type cannot be longer than the
    amino acid alphabet.
    """
    bin_path = BIN_PATH[app_cls]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    alph = seq.Alphabet(range(50))
    sequences = [seq.GeneralSequence(alph, sequence) for sequence in [
        [1,2,3],
        [1,2,3],
    ]]
    with pytest.raises(TypeError):
        try:
            app_cls(sequences)
        except VersionError:
            pytest.skip(f"Invalid software version")


def test_invalid_muscle_version(sequences):
    """
    One of `MuscleApp` and `Muscle5App` should raise an error, since one
    is incompatible with the installed version
    """
    bin_path = BIN_PATH[MuscleApp]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    if is_not_installed("muscle"):
        pytest.skip(f"'muscle' is not installed")

    with pytest.raises(VersionError):
        MuscleApp(sequences)
        Muscle5App(sequences)


def test_clustalo_matrix(sequences):
    bin_path = BIN_PATH[ClustalOmegaApp]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
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
    bin_path = BIN_PATH[ClustalOmegaApp]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    leaves = [phylo.TreeNode(index=i) for i in range(len(sequences))]
    inter1 = phylo.TreeNode([leaves[0], leaves[1]], [1.0, 1.0])
    inter2 = phylo.TreeNode([leaves[2], leaves[3]], [2.5, 2.5])
    root = phylo.TreeNode([inter1, inter2], [3.5, 2])
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
    bin_path = BIN_PATH[MafftApp]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    app = MafftApp(sequences)
    app.start()
    app.join()
    tree = app.get_guide_tree()
    assert tree is not None


def test_muscle_tree(sequences):
    bin_path = BIN_PATH[MuscleApp]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    try:
        app = MuscleApp(sequences)
    except VersionError:
        pytest.skip(f"Invalid software version")
    app.start()
    app.join()
    tree1 = app.get_guide_tree(iteration="kmer")
    tree2 = app.get_guide_tree(iteration="identity")
    assert tree1 is not None
    assert tree2 is not None


def test_muscle5_options(sequences):
    bin_path = BIN_PATH[Muscle5App]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    
    try:
        app = Muscle5App(sequences)
    except VersionError:
        pytest.skip(f"Invalid software version")
    app.use_super5()
    app.set_iterations(2, 100)
    app.set_thread_number(2)
    app.start()

    assert "-super5" in app.get_command()
    assert "-consiters" in app.get_command()
    assert "-refineiters" in app.get_command()
    assert "-threads" in app.get_command()

    app.join()
    assert str(app.get_alignment()) == "BI-QTITE\n" \
                                       "TITANITE\n" \
                                       "BI-SMITE\n" \
                                       "-I-QLITE"