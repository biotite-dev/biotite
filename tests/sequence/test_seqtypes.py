import numpy as np
import pytest
import biotite.sequence as seq


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("seq_type", [seq.NucleotideSequence, seq.ProteinSequence])
def test_construction(seq_type, seed):
    """
    Check whether the unambiguous alphabet is chosen for a random sequence string
    without ambiguous symbols and the ambiguous alphabet otherwise.
    Furthermore, check that the alphabet is retained when the sequence is copied
    or recreated from its ``repr()`` and that the alphabet can be enforced.
    """
    SEQ_LENGTH = 50

    rng = np.random.default_rng(seed)
    unamb_alph = seq_type.unambiguous_alphabet()
    amb_alph = seq_type.ambiguous_alphabet()
    assert amb_alph.extends(unamb_alph)
    amb_only_symbols = amb_alph.get_symbols()[len(unamb_alph) :]
    unamb_string = "".join(rng.choice(unamb_alph.get_symbols(), size=SEQ_LENGTH))
    # Replace a random position with an ambiguous symbol
    pos = rng.integers(SEQ_LENGTH)
    amb_string = (
        unamb_string[:pos] + rng.choice(amb_only_symbols) + unamb_string[pos + 1 :]
    )

    for string, ref_alph in [(unamb_string, unamb_alph), (amb_string, amb_alph)]:
        for sequence in [
            seq_type(string),
            seq_type(string.lower()),
            seq_type(list(string)),
        ]:
            assert sequence.get_alphabet() == ref_alph
            assert str(sequence) == string
            assert sequence.copy().get_alphabet() == ref_alph
            assert eval(repr(sequence), vars(seq)) == sequence

    assert seq_type(unamb_string, ambiguous=True).get_alphabet() == amb_alph
    with pytest.raises(seq.AlphabetError):
        seq_type(amb_string, ambiguous=False)


def test_reverse_complement():
    string = "AATGCGTTANY"
    dna = seq.NucleotideSequence(string)
    assert str(dna.reverse().complement()) == "RNTAACGCATT"


def test_stop_removal():
    string = "LYG*GR*"
    protein = seq.ProteinSequence(string)
    assert str(protein.remove_stops()) == string.replace("*", "")


@pytest.mark.parametrize(
    ["dna_str", "protein_str"],
    [("CACATAGCATGA", "HIA*"), ("ATGTAGCTA", "M*L"), ("CACATAGCA", "HIA")],
)
def test_full_translation(dna_str, protein_str):
    """
    Check the translated sequence against a reference.
    This includes the alphabet, which should be unambiguous, if there is no stop
    signal.
    """
    dna = seq.NucleotideSequence(dna_str)
    protein = dna.translate(complete=True)
    assert protein == seq.ProteinSequence(protein_str)


@pytest.mark.parametrize(
    ["dna_str", "protein_str_list"],
    [
        ("CA", []),
        ("GAATGCACTGAGATGCAATAG", ["MH*", "MQ*"]),
        ("ATGCACATGTAGGG", ["MHM*", "M*"]),
        ("GATGCATGTGAAAA", ["MHVK", "M*"]),
    ],
)
def test_frame_translation(dna_str, protein_str_list):
    dna = seq.NucleotideSequence(dna_str)
    proteins, pos = dna.translate(complete=False)
    assert len(proteins) == len(protein_str_list)
    assert set([str(protein) for protein in proteins]) == set(protein_str_list)
    for protein in proteins:
        # The alphabet should be unambiguous, if there is no stop signal
        assert protein == seq.ProteinSequence(str(protein))
    # Test if the positions are also right
    # -> Get sequence slice and translate completely
    assert set(
        [str(dna[start:stop].translate(complete=True)) for start, stop in pos]
    ) == set(protein_str_list)


def test_translation_met_start():
    """
    Test whether the start amino acid is replaced by methionine,
    i.e. the correct function of the 'met_start' parameter.
    """
    codon_table = seq.CodonTable.default_table().with_start_codons("AAA")
    dna = seq.NucleotideSequence("GAAACTGAAATAAGAAC")
    proteins, _ = dna.translate(codon_table=codon_table, met_start=True)
    assert [str(protein) for protein in proteins] == ["MLK*", "M*"]


def test_letter_conversion():
    for symbol in seq.ProteinSequence.alphabet_ambiguous:
        three_letters = seq.ProteinSequence.convert_letter_1to3(symbol)
        single_letter = seq.ProteinSequence.convert_letter_3to1(three_letters)
        assert symbol == single_letter


@pytest.mark.parametrize(
    ["monoisotopic", "expected_mol_weight_protein"],
    # Reference values taken from https://web.expasy.org/compute_pi/
    [(True, 2231.06), (False, 2232.56)],
)
def test_get_molecular_weight(monoisotopic, expected_mol_weight_protein):
    """
    Test whether the molecular weight of a protein is calculated
    correctly.
    """
    protein = seq.ProteinSequence("ACDEFGHIKLMNPQRSTVW")
    mol_weight_protein = protein.get_molecular_weight(monoisotopic=monoisotopic)
    assert mol_weight_protein == pytest.approx(expected_mol_weight_protein, abs=1e-2)


def test_positional_sequence():
    """
    Check whether the original sequence symbols can be reconstructed from a
    :class:`PositionalSequence`.
    """
    SEQ_LENGTH = 100

    rng = np.random.default_rng(0)
    orig_sequence = seq.ProteinSequence()
    orig_sequence.code = rng.integers(0, len(orig_sequence.alphabet), size=SEQ_LENGTH)
    positional_sequence = seq.PositionalSequence(orig_sequence)

    assert str(positional_sequence) == str(orig_sequence)
    reconstructed_sequence = positional_sequence.reconstruct()
    assert np.all(reconstructed_sequence.code == orig_sequence.code)
    assert reconstructed_sequence.alphabet == orig_sequence.alphabet
