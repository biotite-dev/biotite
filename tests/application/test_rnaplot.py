# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
from biotite.application.viennarna.rnaplot import RNAplotApp
from ..util import is_not_installed


@pytest.fixture
def sample_app():
    """
    Provide a `RNAplotApp` object, where RNAplot has been executed for a
    sample structure.
    """
    app = RNAplotApp('((..))')
    app.start()
    app.join()
    return app


@pytest.mark.skipif(
    is_not_installed("RNAplot"), reason="RNAplot is not installed"
)
def test_get_cooordinates(sample_app):
    assert (
        np.all(
            sample_app.get_coordinates() == np.array([[ -92.5 ,   92.5 ],
                                                      [ -92.5 ,   77.5 ],
                                                      [ -90.31,   58.24],
                                                      [-109.69,   58.24],
                                                      [-107.5 ,   77.5 ],
                                                      [-107.5 ,   92.5 ]])
        )
    )
