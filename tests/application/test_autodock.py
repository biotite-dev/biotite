# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
from biotite.application.autodock import VinaApp
from ..util import data_dir, is_not_installed


@pytest.mark.skipif(
    is_not_installed("vina"), reason="Autodock Vina is not installed"
)
@pytest.mark.parametrize("flexible", [False, True])
def test_docking(flexible):
    """
    Test :class:`VinaApp` for the case of docking biotin to
    streptavidin.
    The output binding pose should be very similar to the pose in the
    PDB structure.
    """
    # A structure of a straptavidin-biotin complex
    mmtf_file = mmtf.MMTFFile.read(join(data_dir("application"), "2rtg.mmtf"))
    structure = mmtf.get_structure(
        mmtf_file, model=1, extra_fields=["charge"], include_bonds=True
    )
    structure = structure[structure.chain_id == "B"]
    receptor = structure[struc.filter_amino_acids(structure)]
    ref_ligand = structure[structure.res_name == "BTN"]
    ref_ligand_coord = ref_ligand.coord

    ligand = info.residue("BTN")
    # Remove hydrogen atom that is missing in ref_ligand
    ligand = ligand[ligand.atom_name != "HO2"]

    if flexible:
        # Two residues within the binding pocket: ASN23, SER88
        flexible_mask = np.isin(receptor.res_id, (23, 88))
    else:
        flexible_mask = None
    
    app = VinaApp(
        ligand, receptor, struc.centroid(ref_ligand), [20, 20, 20],
        flexible=flexible_mask
    )
    app.set_seed(0)
    app.start()
    app.join()
    
    test_ligand_coord = app.get_ligand_coord()
    test_receptor_coord = app.get_receptor_coord()
    energies = app.get_energies()
    # One energy value per model
    assert len(test_ligand_coord) == len(energies)
    assert len(test_receptor_coord) == len(energies)

    assert np.all(energies < 0)

    # Select best binding pose
    test_ligand_coord = test_ligand_coord[0]
    not_nan_mask = ~np.isnan(test_ligand_coord).any(axis=-1)
    ref_ligand_coord  =  ref_ligand_coord[not_nan_mask]
    test_ligand_coord = test_ligand_coord[not_nan_mask]
    # Check if it least one atom is preserved
    assert test_ligand_coord.shape[1] > 0
    rmsd = struc.rmsd(ref_ligand_coord, test_ligand_coord)
    # The deviation of the best pose from the real conformation
    # should be less than 1 Å
    assert rmsd < 1.0

    if flexible:
        # Select best binding pose
        test_receptor_coord = test_receptor_coord[0]
        not_nan_mask = ~np.isnan(test_receptor_coord).any(axis=-1)
        ref_receptor_coord  =  receptor[not_nan_mask]
        test_receptor_coord = test_receptor_coord[not_nan_mask]
        # Check if it least one atom is preserved
        assert test_receptor_coord.shape[1] > 0
        # The flexible residues should have a maximum deviation of 1 Å
        # from the original conformation
        assert np.max(
            struc.distance(test_receptor_coord, ref_receptor_coord)
        ) < 1.0
    else:
        ref_receptor_coord = receptor.coord
        for model_coord in test_receptor_coord:
            assert np.array_equal(model_coord, ref_receptor_coord)
            