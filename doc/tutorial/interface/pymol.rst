.. include:: /tutorial/preamble.rst

Interface to PyMOL
==================

.. currentmodule:: biotite.interface.pymol

`PyMOL <https://pymol.org/>`_ is a prominent molecular visualization software.
Arguably most users interact with it via it graphical user interface, but it also offers
an extensive Python API.
This API is interfaced by the :mod:`biotite.interface.pymol` subpackage to transfer
structures to PyMOL and back again for convenient visualization.

Launching PyMOL
---------------
By default :mod:`biotite.interface.pymol` starts *PyMOL* in object-oriented
*library mode*.
This means that there is no *graphical user interface* (GUI) available, by default:
The interaction with *PyMOL* purely happens via its Python API and the resulting image
can be inspected by saving it as PNG file.
There is also the option to start an interactive *PyMOL* GUI by running

.. code-block:: python

    import biotite.interface.pymol as pymol_interface
    pymol_interface.launch_pymol()

but for the purpose of this tutorial we will stick to the library mode.
For now, just keep in mind that there are different options to launch *PyMOL*, which
are more thoroughly described in the API reference of :mod:`biotite.interface.pymol`.

Transferring structures from Biotite to PyMOL and vice versa
------------------------------------------------------------
An :class:`.AtomArray` or :class:`.AtomArrayStack` can be converted into
a *PyMOL* object via :meth:`PyMOLObject.from_structure()`.
This static method returns a :class:`PyMOLObject` - a wrapper around a
*PyMOL* object (alias *PyMOL* model).

.. jupyter-execute::

    import numpy as np
    from matplotlib.colors import to_rgb
    import biotite
    import biotite.database.rcsb as rcsb
    import biotite.interface.pymol as pymol_interface
    import biotite.structure.io.pdbx as pdbx

    # Fetch and load cytochrome C structure and remove water
    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1C75", "bcif"))
    structure = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    cyt_c = structure[structure.res_name != "HOH"]

    # Aromatic bonds can be either displayed as formal single/double bonds
    # or as delocalized (dashed) bonds
    pymol_cyt_c = pymol_interface.PyMOLObject.from_structure(cyt_c, delocalize_bonds=True)
    # By default the name of the PyMOL object is auto-generated
    print(pymol_cyt_c.name)

.. note::
    A :class:`PyMOLObject` becomes invalid when atoms are added to or are deleted from
    the wrapped *PyMOL* object or if the underlying *PyMOL* object does not exist
    anymore.

Conversely, :meth:`PyMOLObject.to_structure()` would convert a :class:`PyMOLObject`
object back into an :class:`.AtomArray` or :class:`.AtomArrayStack`.

From now on we can run any usual *PyMOL* commands on this *PyMOL* object by referring
to its name and eventually render the image and save it as PNG file.
For convenience, :mod:`biotite.interface.pymol` allows displaying the current canvas in
a Jupyter notebook.

.. jupyter-execute::

    PNG_SIZE = (1500, 1500)

    # Do not confuse with `PyMOLObject.show()` method we will see later
    pymol_interface.show(PNG_SIZE, use_ray=True)

Atom selections
---------------
*PyMOL* uses selection expressions (strings) to select atoms for its commands.
On the other side, *Biotite* uses *NumPy*-compatible indices to select atoms:
boolean masks, integer arrays and single indices.
To bring these two worlds together, *NumPy*-compatible indices can be converted to
These boolean masks can be converted into selection expressions via the
:meth:`PyMOLObject.where()` method.

.. jupyter-execute::

  # Select heme group
  selection_expression = pymol_cyt_c.where(cyt_c.res_name == "HEC")
  print(selection_expression)

Invoking commands
^^^^^^^^^^^^^^^^^
As mentioned above, *PyMOL* commands can be called as usual:
The current *PyMOL* session is obtained via the ``pymol`` attribute and commands
can be invoked using ``pymol.cmd``.

.. jupyter-execute::

    pymol_interface.cmd.set("sphere_scale", 1.5)

To add syntactic sugar, most object-specific commands are available as
:class:`PyMOLObject` methods.
These methods accept *NumPy*-compatible indices directly, without the need to call
:meth:`PyMOLObject.where()`.

.. jupyter-execute::

    # Style protein, use PyMOL-style selection string
    pymol_cyt_c.show_as("cartoon", "polymer")
    # PyMOLObject.color() command directly allows RGB values
    pymol_cyt_c.color(to_rgb(biotite.colors["lightgreen"]), "polymer and name CA")

    # Style heme group, use NumPy-style fancy indexing
    heme_mask = cyt_c.res_name == "HEC"
    pymol_cyt_c.show_as("sticks", heme_mask)
    pymol_cyt_c.color(
        to_rgb(biotite.colors["lightorange"]), heme_mask & (cyt_c.element == "C")
    )

    # Style Fe2+ ion, use single index
    fe_index = np.where(cyt_c.element == "FE")[0]
    pymol_cyt_c.show_as("spheres", fe_index)
    pymol_cyt_c.color(to_rgb(biotite.colors["darkorange"]), fe_index)
    pymol_cyt_c.set("sphere_scale", 0.75, fe_index)

    pymol_interface.show(PNG_SIZE, use_ray=True)

Have a look at the reference page of :class:`PyMOLObject` to see all available
commands.

Resetting the canvas
--------------------
To remove all objects from the canvas, and reset all parameters to defaults,
call ``pymol_interface.reset()``.

.. warning::

    Do not call the *PyMOL* ``reinitialize`` command directly.
    :func:`reset()` does this internally, but additionally sets some *PyMOL* parameters
    so *Biotite* and *PyMOL* interact properly.

However, we want to keep the styling here, so we only remove the existing *PyMOL* object
from the canvas simply by deleting the variable.

.. jupyter-execute::

    del pymol_cyt_c

Drawing custom shapes
---------------------
In addition to visualization of molecules *PyMOL* is capable of drawing
arbitrary geometric shapes using *compiled graphics objects* (CGOs).
A CGO (for example a sphere) is represented by a list of floating point values.
:mod:`biotite.interface.pymol` supports a range of CGOs, that can be created
conveniently using dedicated functions, such as :func:`get_cylinder_cgo()`.
Calling this function does not draw anything, yet.
Instead, one or multiple combined CGOs can be drawn using :func:`draw_cgo()` creating
a single :class:`PyMOLObject` object.
For example, to draw two spheres connected by a line (a cylinder) and
a color gradient from red to blue, you can call

.. jupyter-execute::

    PNG_SIZE = (1500, 750)
    RED = to_rgb("#db3a35")
    BLUE = to_rgb("#1772f0")

    gradient_bond = pymol_interface.draw_cgo(
        [
            pymol_interface.get_sphere_cgo(pos=(0, 0, 0), radius=1.0, color=RED),
            pymol_interface.get_cylinder_cgo(
                start=(0, 0, 0), end=(5, 0, 0), radius=0.5, start_color=RED, end_color=BLUE
            ),
            pymol_interface.get_sphere_cgo(pos=(5, 0, 0), radius=1.0, color=BLUE),
        ],
    )
    # Zomm a little bit out
    gradient_bond.zoom(buffer=2)
    pymol_interface.show(PNG_SIZE, use_ray=True)

For convenience, some predefined shapes can be drawn, that rely on a combination
of CGOs.

.. jupyter-execute::

    box = pymol_interface.draw_box(
        np.diag([10, 10, 10]),
        origin=[-2.5, -5.0, -5.0],
        width=5,
        # white
        color=(1, 1, 1),
    )
    box.zoom(buffer=2)
    pymol_interface.show(PNG_SIZE, use_ray=True)