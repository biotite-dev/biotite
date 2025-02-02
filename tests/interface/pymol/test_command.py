from os.path import join
import pytest
import biotite.interface.pymol as pymol_interface
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


def _get_mask():
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    structure = pdbx.get_structure(pdbx_file)
    return structure.res_id < 10


MASK = _get_mask()


@pytest.mark.parametrize(
    "command_name, kwargs",
    [
        (
            "alter",
            {
                "selection": MASK,
                "expression": "chain='B'",
            },
        ),
        (
            "cartoon",
            {
                "type": "tube",
            },
        ),
        (
            "cartoon",
            {
                "type": "tube",
                "selection": "resi 1-10",
            },
        ),
        (
            "cartoon",
            {
                "type": "tube",
                "selection": MASK,
            },
        ),
        ("center", {}),
        (
            "center",
            {
                "selection": MASK,
                "state": 1,
                "origin": True,
            },
        ),
        (
            "clip",
            {
                "mode": "near",
                "distance": 1.0,
                "state": 1,
            },
        ),
        (
            "color",
            {
                "color": "green",
            },
        ),
        (
            "color",
            {
                "color": (0.0, 1.0, 1.0),
            },
        ),
        (
            "color",
            {
                "color": [0.0, 1.0, 1.0],
            },
        ),
        (
            "color",
            {
                "color": [0.0, 1.0, 1.0],
                "representation": "surface",
            },
        ),
        # Not available in Open Source PyMOL
        # ("desaturate", {
        # }),
        ("disable", {}),
        (
            "distance",
            {"name": "dist1", "selection1": MASK, "selection2": MASK, "mode": 4},
        ),
        ("dss", {}),
        ("dss", {"state": 1}),
        (
            "hide",
            {
                "representation": "cartoon",
            },
        ),
        ("indicate", {}),
        ("label", {"selection": MASK, "text": "Some text"}),
        ("orient", {}),
        (
            "orient",
            {
                "state": 1,
            },
        ),
        ("origin", {}),
        (
            "origin",
            {
                "state": 1,
            },
        ),
        (
            "select",
            {
                "name": "selection1",
            },
        ),
        (
            "set",
            {
                "name": "sphere_color",
                "value": "green",
            },
        ),
        (
            "set_bond",
            {
                "name": "stick_color",
                "value": "green",
            },
        ),
        (
            "show",
            {
                "representation": "sticks",
            },
        ),
        (
            "show_as",
            {
                "representation": "sticks",
            },
        ),
        ("smooth", {}),
        (
            "unset",
            {
                "name": "sphere_color",
            },
        ),
        (
            "unset",
            {
                "name": "line_color",
            },
        ),
        ("zoom", {}),
    ],
)
def test_command(command_name, kwargs):
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    structure = pdbx.get_structure(pdbx_file, include_bonds=True)
    pymol_obj = pymol_interface.PyMOLObject.from_structure(structure)
    command = getattr(pymol_interface.PyMOLObject, command_name)
    command(pymol_obj, **kwargs)
