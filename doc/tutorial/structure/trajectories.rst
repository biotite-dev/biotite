.. include:: /tutorial/preamble.rst

Working with molecular trajectories
===================================
As :mod:`biotite.structure` provides efficient tools for analyzing multi-model
structures, the package is predestined for handling trajectories from
*molecular dynamics* (MD) simulations.
If you like, you can even use the :mod:`biotite.interface.openmm` subpackage with the
`OpenMM <https://openmm.org/>`_ molecular simulation toolkit to integrate MD
simulations.

Reading trajectory files
------------------------

.. currentmodule:: biotite.structure.io.xtc

*Biotite* provides a read/write interface for different trajectory file
formats.
All supported trajectory formats have in common, that they store
only coordinates.
These can be extracted as :class:`ndarray` with the
:meth:`XTCFile.get_coord()` method.

.. jupyter-execute::

    from tempfile import NamedTemporaryFile, gettempdir
    import requests
    import biotite.structure.io.xtc as xtc

    # Download 1L2Y as XTC file for demonstration purposes
    temp_xtc_file = NamedTemporaryFile("wb", suffix=".xtc", delete=False)
    response = requests.get(
        "https://raw.githubusercontent.com/biotite-dev/biotite/master/"
        "tests/structure/data/1l2y.xtc"
    )
    temp_xtc_file.write(response.content)

    traj_file = xtc.XTCFile.read(temp_xtc_file.name)
    coord = traj_file.get_coord()
    print(coord.shape)

If only an excerpt of frames is desired, the behavior of the
:meth:`read()` function can be customized with the `start`, `stop` and
`step` parameters.

.. jupyter-execute::

    # Read only every second frame
    traj_file = xtc.XTCFile.read(temp_xtc_file.name, step=2)
    coord = traj_file.get_coord()
    print(coord.shape)

In order to extract an entire structure, i.e. an
:class:`AtomArrayStack`, from a trajectory file, a *template*
structure must be given, since the trajectory file contains only
coordinate information.

.. jupyter-execute::

    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1l2y", "bcif", gettempdir()))
    template = pdbx.get_structure(pdbx_file, model=1)

    traj_file = xtc.XTCFile.read(temp_xtc_file.name)
    trajectory = traj_file.get_structure(template)
    temp_xtc_file.close()

Comparing frames with each other
--------------------------------

.. currentmodule:: biotite.structure

In the analysis of MD simulations, the frames are seldom only observed in
isolation, but also compared with each other.
For example, the *root mean square deviation* (RMSD) over the course of the
simulation gives a hint about the convergence of the simulation.
On the other hand, the *root mean square fluctuation* (RMSF) describes the
structural flexibility of each atom.

.. jupyter-execute::

    import biotite.structure as struc

    # The comparison requires that the frames are properly superimposed
    # Arbitrarily choose the first frame as reference
    trajectory, _ = struc.superimpose(trajectory[0], trajectory)
    # Compute RMSD/RMSF only for CA atoms
    trajectory = trajectory[:, trajectory.atom_name == "CA"]
    rmsd = struc.rmsd(trajectory[0], trajectory)
    print("RMSD:")
    print(rmsd)
    # Compute RMSF for each residue relative to the average over the trajectory
    rmsf = struc.rmsf(struc.average(trajectory), trajectory)
    print("RMSF:")
    print(rmsf)

Simulation boxes and unit cells
-------------------------------
Usually each frame in a trajectory has an associated simulation box, that
describes the box boundaries in terms of vectors.
*Biotite* represents these boxes (and unit cells from crystal structures alike),
as *(3,3)*-shaped :class:`ndarray` objects.
Each element in the array is one of the three vectors spanning the box or unit
cell.
Let's create an orthorhombic box from the vector lengths and the angles between
the vectors.

.. jupyter-execute::

    import numpy as np

    # The function uses angles in radians
    box = struc.vectors_from_unitcell(10, 20, 30, np.pi/2, np.pi/2, np.pi/2)
    print("Box:")
    print(box)
    print("Box volume:", struc.box_volume(box))
    print("Is the box orthogonal?", struc.is_orthogonal(box))
    len_a, len_b, len_c, alpha, beta, gamma = struc.unitcell_from_vectors(box)
    print("Cell lengths:")
    print(len_a, len_b, len_c)
    print("Cell angles:")
    print(np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma))

An :class:`AtomArray` may have an associated box,
which is used in functions, that consider periodic boundary conditions.
:class:`AtomArrayStack` objects require a *(m,3,3)*-shaped :class:`ndarray`,
that contains the box vectors for each frame.
The box is accessed via the ``box`` attribute, which is ``None`` by
default.

.. jupyter-execute::

    array = struc.AtomArray(length=100)
    print(array.box)
    array.box = struc.vectors_from_unitcell(10, 20, 30, np.pi/2, np.pi/2, np.pi/2)
    print(array.box)

When loaded from a structure file, the box described in the file is
automatically used.

.. jupyter-execute::

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1aki", "bcif", gettempdir()))
    stack = pdbx.get_structure(pdbx_file)
    print(stack.box)

Measurements over periodic boundaries
-------------------------------------
A common feature (and issue) in MD simulations are
*periodic boundary conditions* (PBC).
This means the simulation box is virtually repeated in all directions, an
infinite number of times, or in other words, an atom leaving the box on the
right side reenters on the left side.
Many measurement functions in :mod:`biotite.structure` accept a `periodic`
or `box` parameter to deal with PBCs.
If the respective parameter are set, the function considers periodic boundary
conditions for the measurement, i.e. the nearest periodic copies of the atoms
are considered.

.. note::

    The measurements in the following snippet are nonsensical, as the
    associated `box` of the used structure is a placeholder from the PDB entry.
    However, the code works also for actual trajectories.

.. jupyter-execute::

    # PBC-aware distance between first and second CA atoms
    distances = struc.distance(
        trajectory[:, 0], trajectory[:, 1], box=trajectory.box
    )
    # PBC-aware radial distribution function of each CA atom to all CA atoms
    bins, rdf = struc.rdf(
        center=trajectory, atoms=trajectory, interval=(0, 10), periodic=True
    )

As PBC-aware measurements are computationally more expensive, it can be
reasonable to remove the segmentation of a structure over the periodic
boundary via :func:`remove_pbc()` to be able to use the faster non-PBC
measurements.

.. jupyter-execute::

    trajectory = struc.remove_pbc(trajectory)
    # The same distance calculation but PBC-awareness is not necessary anymore
    distances = struc.distance(trajectory[:, 0], trajectory[:, 1])