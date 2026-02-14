# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Use this module to calculate the Solvent Accessible Surface Area (SASA) of
a protein or single atoms.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["sasa"]

import numpy as np
from biotite.rust.structure import sasa as rust_sasa
from biotite.structure.filter import (
    filter_heavy,
    filter_monoatomic_ions,
    filter_solvent,
)
from biotite.structure.info.radii import vdw_radius_protor, vdw_radius_single


def sasa(
    array,
    probe_radius=1.4,
    atom_filter=None,
    ignore_ions=True,
    point_number=1000,
    point_distr="Fibonacci",
    vdw_radii="ProtOr",
):
    """
    sasa(array, probe_radius=1.4, atom_filter=None, ignore_ions=True,
         point_number=1000, point_distr="Fibonacci", vdw_radii="ProtOr")

    Calculate the Solvent Accessible Surface Area (SASA) of a protein.

    This function uses the Shrake-Rupley ("rolling probe")
    algorithm :footcite:`Shrake1973`:
    Every atom is occupied by a evenly distributed point mesh. The
    points that can be reached by the "rolling probe", are surface
    accessible.

    Parameters
    ----------
    array : AtomArray
        The protein model to calculate the SASA for.
    probe_radius : float, optional
        The VdW-radius of the solvent molecules.
    atom_filter : ndarray, dtype=bool, optional
        If this parameter is given, SASA is only calculated for the
        filtered atoms.
    ignore_ions : bool, optional
        If true, all monoatomic ions are removed before SASA calculation.
    point_number : int, optional
        The number of points in the mesh occupying each atom for SASA
        calculation.
        The SASA calculation time is proportional to the amount of sphere points.
    point_distr : str or function, optional
        If a function is given, the function is used to calculate the
        point distribution for the mesh.
        The function must take `float` *n* as parameter and return a
        *(n x 3)* :class:`ndarray` containing points on the surface of a unit sphere.
        Alternatively a string can be given to choose a built-in
        distribution:

            - **Fibonacci** - Distribute points using a golden section spiral.

        By default *Fibonacci* is used.
    vdw_radii : str or ndarray, dtype=float, optional
        Indicates the set of VdW radii to be used. If an `array`-length
        :class:`ndarray` is given, each atom gets the radius at the
        corresponding index. Radii given for atoms that are not used in
        SASA calculation (e.g. solvent atoms) can have arbitrary values
        (e.g. `NaN`). If instead a string is given, one of the
        built-in sets is used:

            - **ProtOr** - A set, which does not require hydrogen atoms
              in the model. Suitable for crystal structures.
              :footcite:`Tsai1999`
            - **Single** - A set, which uses a defined VdW radius for
              every single atom, therefore hydrogen atoms are required
              in the model (e.g. NMR elucidated structures).
              Values for main group elements are taken from :footcite:`Mantina2009`,
              and for relevant transition metals from the :footcite:`RDKit`.

        By default *ProtOr* is used.

    Returns
    -------
    sasa : ndarray, dtype=bool, shape=(n,)
        Atom-wise SASA. `NaN` for atoms where SASA has not been
        calculated
        (solvent atoms, hydrogen atoms (ProtOr), atoms not in `filter`).

    References
    ----------

    .. footbibliography::
    """
    if atom_filter is not None:
        # Filter for all atoms to calculate SASA for
        sasa_filter = np.array(atom_filter, dtype=bool)
    else:
        sasa_filter = np.ones(len(array), dtype=bool)
    # Only include atoms within finite coordinates
    sasa_filter &= np.isfinite(array.coord).all(axis=-1)
    # Filter for all atoms that are considered for occlusion calculation
    # sasa_filter is subfilter of occlusion_filter
    occlusion_filter = np.ones(len(array), dtype=bool)
    # Remove water residues, since it is the solvent
    filter = ~filter_solvent(array)
    sasa_filter = sasa_filter & filter
    occlusion_filter = occlusion_filter & filter
    if ignore_ions:
        filter = ~filter_monoatomic_ions(array)
        sasa_filter = sasa_filter & filter
        occlusion_filter = occlusion_filter & filter

    if callable(point_distr):
        sphere_points = point_distr(point_number)
    elif point_distr == "Fibonacci":
        sphere_points = _create_fibonacci_points(point_number)
    else:
        raise ValueError(f"'{point_distr}' is not a valid point distribution")
    sphere_points = sphere_points.astype(np.float32, copy=False)

    if isinstance(vdw_radii, np.ndarray):
        radii = vdw_radii.astype(np.float32)
        if len(radii) != array.array_length():
            raise ValueError(
                f"Amount VdW radii ({len(radii)}) and "
                f"amount of atoms ({array.array_length()}) are not equal"
            )
    elif vdw_radii == "ProtOr":
        filter = filter_heavy(array)
        sasa_filter = sasa_filter & filter
        occlusion_filter = occlusion_filter & filter
        radii = np.full(len(array), np.nan, dtype=np.float32)
        for i in np.arange(len(radii))[occlusion_filter]:
            rad = vdw_radius_protor(array.res_name[i], array.atom_name[i])
            # 1.8 is default radius
            radii[i] = rad if rad is not None else 1.8
    elif vdw_radii == "Single":
        radii = np.full(len(array), np.nan, dtype=np.float32)
        for i in np.arange(len(radii))[occlusion_filter]:
            rad = vdw_radius_single(array.element[i])
            # 1.5 is default radius
            radii[i] = rad if rad is not None else 1.8
    else:
        raise KeyError(f"'{vdw_radii}' is not a valid radii set")
    # Increase atom radii by probe size ("rolling probe")
    radii += probe_radius

    return rust_sasa(array.coord, radii, sphere_points, sasa_filter, occlusion_filter)


def _create_fibonacci_points(n):
    """
    Get an array of approximately equidistant points on a unit sphere surface
    using a golden section spiral.
    """
    phi = (3 - np.sqrt(5)) * np.pi * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)
    coords = np.zeros((n, 3))
    coords[:, 0] = radius * np.cos(phi)
    coords[:, 1] = radius * np.sin(phi)
    coords[:, 2] = z
    return coords
