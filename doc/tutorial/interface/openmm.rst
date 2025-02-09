# Molmarbles


![Molmarbles](https://raw.githubusercontent.com/biotite-dev/molmarbles/master/logo.svg)

This small package allows conversion between structures objects from
[Biotite](https://www.biotite-python.org/)
(`AtomArray` and `AtomArrayStack`) and `System`, `Topology`,
`Context` or `State` objects from [OpenMM](https://openmm.org/).
This allows for data preparation, MD simulation and trajectory analysis from
the same script without intermediate files.

> **Note**
> The development of this package is work in progress.
> The API will probably change in the future.


## Installation

*Molmarbles* can be installed via *pip*, either from PyPI...

```shell
$ pip install molmarbles
```

or a local repository clone.

```shell
$ pip install .
```

Note that *OpenMM* must also be installed for *Molmarbles* to work.
*OpenMM* is not distributed via *PyPI* and must installed via *Conda*.

```shell
$ conda install -c conda-forge openmm
```


## Usage

`AtomArray` and `AtomArrayStack` objects can be converted to the respective
*OpenMM* objects with

- `to_system()`
- `to_topology()`

and vice versa with

- `from_topology()`
- `from_context()`
- `from_states()`
- `from_state()`

Detailed description of parameters and return values is provided by the
respective
[docstring](https://github.com/biotite-dev/molmarbles/blob/master/molmarbles/__init__.py).


## Example

A short MD simulation of lysozyme

```python
import os
from os.path import isdir, join
import numpy as np
import matplotlib.pyplot as plt
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import molmarbles
import openmm
import openmm.app as app
from openmm.unit import nanometer, angstrom, kelvin, picosecond


BUFFER = 10
TIME_STEP = 0.004
FRAME_STEP = 0.1
N_FRAMES = 100


molecule = mmtf.get_structure(
    mmtf.MMTFFile.read(rcsb.fetch("1aki", "mmtf")),
    model=1,
    include_bonds=True
)
# Remove solvent, as proper solvent addition is handled by OpenMM
molecule = molecule[struc.filter_amino_acids(molecule)]
# Create a box with some buffer around the protein
# This box represents the
box_dim = np.max(molecule.coord, axis=0) - np.min(molecule.coord, axis=0)
molecule.box = np.diag(box_dim + BUFFER)

# Create an OpenMM Topology from the AtomArray
topology = molmarbles.to_topology(molecule)


force_field = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
# Add hydrogen atoms and water
modeller = app.Modeller(topology, molecule.coord * angstrom)
modeller.addHydrogens(force_field)
modeller.addSolvent(force_field)
topology = modeller.topology

system = force_field.createSystem(
    topology, nonbondedMethod=app.PME,
    nonbondedCutoff=1*nanometer, constraints=app.HBonds
)
integrator = openmm.LangevinMiddleIntegrator(300*kelvin, 1/picosecond, TIME_STEP*picosecond)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()

# Run simulation and record the current state (the coordinates)
# every FRAME_STEP picoseconds
states = []
states.append(simulation.context.getState(getPositions=True))
for i in range(N_FRAMES):
    simulation.step(FRAME_STEP // TIME_STEP)
    states.append(simulation.context.getState(getPositions=True))


# Transfer the trajectory back to Biotite
# The topology was changed in the process of adding hydrogen and solvent
# -> Update the structure
template = molmarbles.from_topology(topology)
trajectory = molmarbles.from_states(template, states)
# Center protein in box
trajectory.coord -= struc.centroid(
    trajectory.coord[:, struc.filter_amino_acids(trajectory)]
)[:, np.newaxis, :]
trajectory.coord += np.sum(trajectory.box / 2, axis=-2)[:, np.newaxis, :]
# Remove segmentation over periodic boundary
trajectory = struc.remove_pbc(trajectory)
```

Visualization with [Ammolite](https://ammolite.biotite-python.org/) and
[PyMOL](https://pymol.org/)


https://user-images.githubusercontent.com/28051833/220104497-ecf2cdb2-5e1d-4c22-ae27-ae4ca59ebb8e.mp4


## Testing

*Molmarbles* uses *pytest* for running its tests.
