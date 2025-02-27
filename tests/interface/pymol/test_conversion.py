from os.path import join
import numpy as np
import pytest
import biotite.interface.pymol as pymol_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture(
    params=[
        "1aki",
        "1dix",
        "1f2n",
        "1gya",
        "1igy",
        "1l2y",
        "1o1z",
        "3o5r",
        "5h73",
        "5ugo",
    ]
)
def path(request):
    pdb_id = request.param
    return join(data_dir("structure"), f"{pdb_id}.cif")


@pytest.mark.parametrize("state", [1, None])
@pytest.mark.parametrize("altloc", ["first", "occupancy", "all"])
def test_to_biotite(path, altloc, state):
    pdbx_file = pdbx.CIFFile.read(path)
    ref_array = pdbx.get_structure(pdbx_file, model=state, altloc=altloc)

    pymol_interface.cmd.load(path, "test")
    test_array = pymol_interface.PyMOLObject("test").to_structure(
        state=state, altloc=altloc
    )

    for cat in [c for c in ref_array.get_annotation_categories() if c != "altloc_id"]:
        assert (test_array.get_annotation(cat) == ref_array.get_annotation(cat)).all()
    assert np.allclose(test_array.coord, ref_array.coord)
    # Do not test bonds,
    # as PyMOL determines bonds in another way than Biotite


@pytest.mark.filterwarnings("ignore::biotite.interface.LossyConversionWarning")
def test_to_pymol(path):
    pymol_interface.cmd.load(path, "test")
    ref_model = pymol_interface.cmd.get_model("test", 1)

    pdbx_file = pdbx.CIFFile.read(path)
    atom_array = pdbx.get_structure(
        pdbx_file, model=1, extra_fields=["b_factor", "occupancy", "charge"]
    )
    test_model = pymol_interface.to_model(atom_array)

    test_atoms = test_model.atom
    ref_atoms = [atom for atom in ref_model.atom if atom.alt in ("", "A")]
    assert len(test_atoms) == len(ref_atoms)
    for test_atom, ref_atom in zip(test_atoms, ref_atoms):
        assert test_atom.symbol == ref_atom.symbol
        assert test_atom.name == ref_atom.name
        assert test_atom.resn == ref_atom.resn
        assert test_atom.ins_code == ref_atom.ins_code
        assert test_atom.resi_number == ref_atom.resi_number
        assert test_atom.b == pytest.approx(ref_atom.b)
        assert test_atom.q == pytest.approx(ref_atom.q)
        assert test_atom.hetatm == ref_atom.hetatm
        assert test_atom.chain == ref_atom.chain
        assert test_atom.coord == pytest.approx(ref_atom.coord)
        # Proper charge information is not included in the CIF files
        # -> Do not compare formal charge


@pytest.mark.filterwarnings("ignore::biotite.interface.LossyConversionWarning")
@pytest.mark.parametrize("state", [1, None])
def test_both_directions(path, state):
    pdbx_file = pdbx.CIFFile.read(path)
    ref_array = pdbx.get_structure(pdbx_file, model=state, include_bonds=True)

    test_array = pymol_interface.PyMOLObject.from_structure(ref_array).to_structure(
        state=state, include_bonds=True
    )

    for cat in ref_array.get_annotation_categories():
        assert (test_array.get_annotation(cat) == ref_array.get_annotation(cat)).all()
    assert np.allclose(test_array.coord, ref_array.coord)

    # By default PyMOL uses formal bond orders
    test_array.bonds.remove_aromaticity()
    ref_array.bonds.remove_aromaticity()
    # PyMOL is not able to represent coordination bonds
    ref_bonds = ref_array.bonds.as_array()
    ref_bonds[ref_bonds[:, 2] == struc.BondType.COORDINATION, 2] = struc.BondType.SINGLE
    assert test_array.bonds.as_set() == set([tuple(bond) for bond in ref_bonds])
