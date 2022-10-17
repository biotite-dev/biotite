# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import biotite.structure as struc
from ..util import data_dir


@pytest.mark.parametrize(
    "path, coord_only", itertools.product(
        glob.glob(join(data_dir("structure"), "*.mmtf")),
        [False, True]
    )
)
def test_superimposition_array(path, coord_only):
    """
    Take a structure and rotate and translate a copy of it, so that they
    are not superimposed anymore.
    Then superimpose these structure onto each other and expect an
    almost perfect match.
    """
    fixed = strucio.load_structure(path, model=1)
    
    mobile = fixed.copy()
    mobile = struc.rotate(mobile, (1,2,3))
    mobile = struc.translate(mobile, (1,2,3))
    
    if coord_only:
        fixed = fixed.coord
        mobile = mobile.coord

    fitted, transformation = struc.superimpose(
        fixed, mobile
    )
    
    if coord_only:
        assert isinstance(fitted, np.ndarray)
    assert struc.rmsd(fixed, fitted) == pytest.approx(0, abs=6e-4)
    
    fitted = struc.superimpose_apply(mobile, transformation)
    
    if coord_only:
        assert isinstance(fitted, np.ndarray)
    assert struc.rmsd(fixed, fitted) == pytest.approx(0, abs=6e-4)


@pytest.mark.parametrize("ca_only", (True, False))
def test_superimposition_stack(ca_only):
    """
    Take a structure with multiple models where each model is not
    (optimally) superimposed onto each other.
    Then superimpose and expect an improved RMSD.
    """
    path = join(data_dir("structure"), "1l2y.mmtf")
    stack = strucio.load_structure(path)
    fixed = stack[0]
    mobile = stack[1:]
    if ca_only:
        mask = (mobile.atom_name == "CA")
    else:
        mask = None
    
    fitted, _ = struc.superimpose(fixed, mobile, mask)
    
    if ca_only:
        # The superimpositions are better for most cases than the
        # superimpositions in the structure file
        # -> Use average
        assert np.mean(struc.rmsd(fixed, fitted)) \
             < np.mean(struc.rmsd(fixed, mobile))
    else:
        # The superimpositions are better than the superimpositions
        # in the structure file
        assert (struc.rmsd(fixed, fitted) < struc.rmsd(fixed, mobile)).all()



@pytest.mark.parametrize("seed", range(5))
def test_masked_superimposition(seed):
    """
    Take two models of the same structure and superimpose based on a
    single, randomly chosen atom.
    Since two atoms can be superimposed perfectly, the distance between
    the atom in both models should be 0.
    """

    path = join(data_dir("structure"), "1l2y.mmtf")
    fixed = strucio.load_structure(path, model=1)
    mobile = strucio.load_structure(path, model=2)

    # Create random mask for a single atom
    np.random.seed(seed)
    mask = np.full(fixed.array_length(), False)
    mask[np.random.randint(fixed.array_length())] = True
    
    # The distance between the atom in both models should not be
    # already 0 prior to superimposition
    assert struc.distance(fixed[mask], mobile[mask])[0] \
        != pytest.approx(0, abs=5e-4)

    fitted, transformation = struc.superimpose(
        fixed, mobile, mask
    )
    
    assert struc.distance(fixed[mask], fitted[mask])[0] \
        == pytest.approx(0, abs=5e-4)
    
    fitted = struc.superimpose_apply(mobile, transformation)
    
    struc.distance(fixed[mask], fitted[mask])[0] \
        == pytest.approx(0, abs=5e-4)


@pytest.mark.parametrize(
    "single_model, single_atom", itertools.product([False, True], [False, True])
)
def test_input_shapes(single_model, single_atom):
    """
    Test whether :func:`superimpose()` infers the correct output shape,
    even if the input :class:`AtomArrayStack` contains only a single
    model or a single atom.
    """
    path = join(data_dir("structure"), "1l2y.mmtf")
    stack = strucio.load_structure(path)
    fixed = stack[0]
    
    mobile = stack
    if single_model:
        mobile = mobile[:1, :]
    if single_atom:
        mobile = mobile[:, :1]
        fixed = fixed[:1]
    
    fitted, _ = struc.superimpose(fixed, mobile)

    assert type(fitted) == type(mobile)
    assert fitted.coord.shape == mobile.coord.shape
