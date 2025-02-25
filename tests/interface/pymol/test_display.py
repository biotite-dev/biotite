from os.path import join
import pytest
import biotite.interface.pymol as pymol_interface
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir, is_not_installed


@pytest.mark.skipif(
    is_not_installed("magick") or is_not_installed("ffmpeg"),
    reason="Rendering software is not installed",
)
@pytest.mark.parametrize(
    "function_name, kwargs",
    [
        ("show", {"use_ray": False}),
        ("show", {"use_ray": True}),
        ("play", {"format": "gif"}),
        ("play", {"format": "mp4"}),
    ],
)
def test_display(function_name, kwargs):
    """
    Simply check if the :func:`show()` and :func:`play()` function creates image data.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    structure = pdbx.get_structure(pdbx_file, include_bonds=True)
    _ = pymol_interface.PyMOLObject.from_structure(structure)
    pymol_interface.cmd.mset()
    function = getattr(pymol_interface, function_name)
    image_or_video = function(**kwargs)
    assert len(image_or_video.data) > 0
