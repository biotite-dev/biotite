.. include:: /tutorial/preamble.rst

Interface to OpenMM
===================

.. currentmodule:: biotite.interface.openmm

This subpackage provides an interface to the `OpenMM <https://openmm.org/>`_
molecular simulation toolkit, expanding the structure capabilities of
*Biotite* into the realm of molecular dynamics.

Like other :mod:`biotite.interface` subpackages, *OpenMM* objects are created with
``to_<x>()`` functions (:func:`to_topology()`, :func:`to_system()`) and converted back
to an :class:`.AtomArray` with ``from_<x>()`` functions
(:func:`from_topology()`, :func:`from_system()`, :func:`from_states()`, etc.).
Like in the :doc:`RDKit interface tutorial <rdkit>`, explaining all the functionalities
of *OpenMM* would exceed the scope of this tutorial.
Please refer to the `OpenMM documentation <http://docs.openmm.org/latest/api-python/>`_
for further information.
This tutorial will only give a few examples of how *OpenMM* and *Biotite* work in
tandem.

First example: MD simulation of lysozyme
----------------------------------------
Here the aim is to run the prominent
`'Lysozyme in water' <http://www.mdtutorials.com/gmx/lysozyme/>`_ example in *OpenMM*.
For structure preparation before and analysis afterwards, we will use *Biotite*.

.. jupyter-execute::

    import numpy as np
    import openmm
    import openmm.app as app
    from openmm.unit import angstrom, kelvin, nanometer, picosecond
    import biotite.database.rcsb as rcsb
    import biotite.interface.openmm as openmm_interface
    import biotite.interface.pymol as pymol_interface
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure as struc

    BUFFER = 10

    atoms = pdbx.get_structure(
        pdbx.BinaryCIFFile.read(rcsb.fetch("1aki", "bcif")), model=1, include_bonds=True
    )
    # Remove solvent, as proper solvent addition is handled by OpenMM
    molecule = atoms[struc.filter_amino_acids(atoms)]
    # Create a box with some buffer around the protein
    # This box will be used as starting simulation box
    box_dim = np.max(molecule.coord, axis=0) - np.min(molecule.coord, axis=0)
    molecule.box = np.diag(box_dim + BUFFER)

The :class:`.AtomArray` is converted to an :class:`openmm.topology.Topology` simply
by calling :func:`to_topology()`.

.. jupyter-execute::

    # Create an OpenMM Topology from the AtomArray
    topology = openmm_interface.to_topology(molecule)

The lysozyme structure we use does not have hydrogen atoms, but we require them for the
MD simulation.
Hence, we use functionality from *OpenMM* to model the missing hydrogen atoms.

.. jupyter-execute::

    force_field = openmm.app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    # Add hydrogen atoms and water
    modeller = app.Modeller(topology, molecule.coord * angstrom)
    modeller.addHydrogens(force_field)
    modeller.addSolvent(force_field)
    topology = modeller.topology

The last prerequisite for the MD simulation is the energy minimization to get a
proper starting conformation.
After that, we can finally run the simulation for the given number of steps.
After a certain number of steps, we record the current state (i.e. conformation) of the
system for later analysis.

.. jupyter-execute::

    TIME_STEP = 0.004
    FRAME_STEP = 2.0
    N_FRAMES = 60

    system = force_field.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1 * nanometer,
        constraints=app.HBonds,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, TIME_STEP * picosecond
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    # Run simulation and record the current state (the coordinates)
    # every FRAME_STEP picoseconds
    states = []
    for i in range(N_FRAMES):
        simulation.step(FRAME_STEP // TIME_STEP)
        states.append(simulation.context.getState(getPositions=True))

While each state contains information about the current atom coordinates, box dimensions
etc., it misses the topology of the system, i.e. residues, atoms and bonds.
Hence, to get the trajectory as :class:`.AtomArrayStack` via :func:`from_states()`,
we need to provide an :class:`.AtomArray` as template.
However, we cannot use the original input lysozyme structure from above, as hydrogen
atoms were added meanwhile and thus the topology has changed.
To solve this problem we obtain an updated template first using :func:`from_topology()`.

.. jupyter-execute::

    template = openmm_interface.from_topology(topology)
    trajectory = openmm_interface.from_states(template, states)
    # Center protein in box
    trajectory.coord -= struc.centroid(
        trajectory.coord[:, struc.filter_amino_acids(trajectory)]
    )[:, np.newaxis, :]
    trajectory.coord += np.sum(trajectory.box / 2, axis=-2)[:, np.newaxis, :]
    # Remove segmentation over periodic boundary
    trajectory = struc.remove_pbc(trajectory)

Now we could analyze the trajectory using functionality from :mod:`biotite.structure`,
as exemplified in the :doc:`trajectory tutorial <../structure/trajectories>`, or use
some custom analysis.
For this tutorial we will simply visualize the trajectory using
:mod:`biotite.interface.pymol`.

.. jupyter-execute::

    pymol_object = pymol_interface.PyMOLObject.from_structure(trajectory)
    pymol_object.color("biotite_lightorange", struc.filter_amino_acids(trajectory))
    # Draw simulation box
    # As the box does not extend much during the simulation, simply draw the mean size
    box = pymol_interface.draw_box(
        np.mean(trajectory.box, axis=0), color=(0, 0, 0), width=1
    )

    # Create an isometric view
    pymol_interface.cmd.turn("y", 45)
    pymol_interface.cmd.turn("x", 45)
    pymol_object.zoom(buffer=15)
    pymol_interface.cmd.set("orthoscopic", 1)

    pymol_interface.cmd.mset()
    pymol_interface.play(
        (1400, 1400), format="mp4", html_attributes="loop autoplay width=700"
    )

Second example: Simulating water in vacuum
------------------------------------------
Although simulating a single molecule of water is a frankly boring application,
it demonstrates the usage of custom forces in *OpenMM*:
Instead of creating a :class:`openmm.openmm.System` from a
:class:`openmm.app.topology.Topology`, we could alternatively creating it manually,
allowing us to customize the forces. And because ``Force`` objects use atom indices to
define interacting atoms and :func:`to_system()` creates a system with the same
atom ordering as the input :class:`.AtomArray`, we can directly use atom indices
obtained in *Biotite* to create forces in *OpenMM*.

We begin by constructing the molecule from the *Chemical Component Dictionary*.

.. jupyter-execute::

    import biotite.structure.info as info

    TIME_STEP = 0.001
    FRAME_STEP = 0.1
    N_FRAMES = 120

    water = info.residue("HOH")
    system = openmm_interface.to_system(water)

For the sake of brevity the force field will only contain bonded forces for bond
length and angle stretching. This is also the reason a box was not set in the system:
For bonded forces, periodic boundaries simply do not matter.

.. jupyter-execute::

    # Adopted from TIP3P
    # (https://github.com/openmm/openmmforcefields/blob/main/amber/files/tip3p.xml)
    BOND_LENGTH_IDEAL = 0.09572
    BOND_LENGTH_STIFFNESS = 462750.4
    BOND_ANGLE_IDEAL = 1.82421813418
    BOND_ANGLE_STIFFNESS = 836.8

    length_force = openmm.HarmonicBondForce()
    for i, j, _ in water.bonds.as_array():
        length_force.addBond(i, j, BOND_LENGTH_IDEAL, BOND_LENGTH_STIFFNESS)
    system.addForce(length_force)

    angle_force = openmm.HarmonicAngleForce()
    angle_force.addAngle(
        np.where(water.atom_name == "H1")[0][0],
        np.where(water.atom_name == "O")[0][0],
        np.where(water.atom_name == "H2")[0][0],
        BOND_ANGLE_IDEAL,
        BOND_ANGLE_STIFFNESS,
    )
    _ = system.addForce(angle_force)

Now we are prepared to run the simulation.
Here we cannot use a ``Simulation`` object as convenient wrapper, as we have bypassed
the creation of a ``Topology`` this time.
Hence, we use the ``Context`` and ``Integrator`` directly.

.. jupyter-execute::

    integrator = openmm.LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, TIME_STEP * picosecond
    )
    context = openmm.Context(system, integrator)
    context.setPositions(water.coord * angstrom)

    states = []
    for i in range(N_FRAMES):
        integrator.step(FRAME_STEP // TIME_STEP)
        states.append(context.getState(getPositions=True))

Finally we can obtain the trajectory as :class:`.AtomArrayStack` and visualize the
trajectory, analogous to the snippets shown above.

.. jupyter-execute::

    # In this case we can reuse the input structure as template,
    # as not atoms have been added or removed
    trajectory = openmm_interface.from_states(water, states)
    # Superimpose the frames
    trajectory, _ = struc.superimpose(trajectory[0], trajectory)

    # Remove the box from the previous visualization
    del box
    pymol_object = pymol_interface.PyMOLObject.from_structure(trajectory)
    # Visualize as stick model
    pymol_interface.cmd.set("stick_radius", 0.15)
    pymol_interface.cmd.set("sphere_scale", 0.25)
    pymol_interface.cmd.set("sphere_quality", 4)
    # Visualize docked model
    pymol_object.show("spheres")
    pymol_object.show("sticks")
    pymol_object.set("stick_color", "black")
    pymol_object.orient()

    pymol_interface.cmd.mset()
    # Play as looping GIF
    pymol_interface.play((300, 300))
