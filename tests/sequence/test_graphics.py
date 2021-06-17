# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import dirname, abspath, join
import glob
import pytest
import biotite.sequence as seq
from ..util import cannot_import


@pytest.mark.skipif(
    cannot_import("matplotlib"), reason="Matplotlib is not installed"
)
@pytest.mark.parametrize(
    "scheme_path", glob.glob(
        join(
            dirname(abspath(seq.__file__)),
            "graphics", "color_schemes", "*.json"
        )
    )
)
def test_load_color_scheme(scheme_path):
    from matplotlib.colors import to_rgb
    import biotite.sequence.graphics as graphics

    supported_alphabets = [
        seq.NucleotideSequence.alphabet_amb,
        seq.ProteinSequence.alphabet,
        seq.LetterAlphabet("abcdefghijklmnop") # Protein block alphabet
    ]
    
    test_scheme = graphics.load_color_scheme(scheme_path)

    assert test_scheme["alphabet"] in supported_alphabets
    assert len(test_scheme["colors"]) == len(test_scheme["alphabet"])
    for color in test_scheme["colors"]:
        if color is not None:
            # Should not raise error
            to_rgb(color)