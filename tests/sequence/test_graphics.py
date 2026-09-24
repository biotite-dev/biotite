from pathlib import Path
import pytest
import biotite.sequence as seq
import biotite.structure.alphabet as struc_alph


@pytest.mark.parametrize(
    "scheme_path",
    sorted((Path(seq.__file__).parent / "graphics" / "color_schemes").glob("*.json")),
    ids=lambda path: path.name,
)
def test_load_color_scheme(scheme_path):
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgb
    import biotite.sequence.graphics as graphics

    supported_alphabets = [
        seq.NucleotideSequence.alphabet_amb,
        seq.ProteinSequence.alphabet,
        struc_alph.I3DSequence.alphabet,
        struc_alph.ProteinBlocksSequence.alphabet,
    ]

    test_scheme = graphics.load_color_scheme(scheme_path)

    assert isinstance(test_scheme, graphics.ColorScheme)
    assert test_scheme.alphabet in supported_alphabets
    assert len(test_scheme.colors) == len(test_scheme.alphabet)
    for color in test_scheme.colors:
        if color is not None:
            # Should not raise error
            to_rgb(color)
    # The fitted scheme replaces undefined colors by the default
    fitted = test_scheme.fit(test_scheme.alphabet, default="black")
    assert len(fitted) == len(test_scheme.alphabet)
    assert None not in fitted


def test_color_scheme_fit():
    """
    Fitting a scheme to a smaller alphabet gives the colors of the
    respective symbols, while an incompatible alphabet is rejected.
    """
    pytest.importorskip("matplotlib")
    import biotite.sequence.graphics as graphics

    alphabet = seq.NucleotideSequence.alphabet_amb
    scheme = graphics.ColorScheme(
        "test", alphabet, [f"C{i}" for i in range(len(alphabet))]
    )
    unamb_alphabet = seq.NucleotideSequence.alphabet_unamb
    assert scheme.fit(unamb_alphabet) == ["C0", "C1", "C2", "C3"]
    assert scheme.fit(alphabet) == scheme.colors
    with pytest.raises(ValueError):
        scheme.fit(seq.ProteinSequence.alphabet)
    with pytest.raises(ValueError):
        graphics.ColorScheme("test", alphabet, ["C0"])
