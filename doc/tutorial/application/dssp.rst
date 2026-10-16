.. include:: /tutorial/preamble.rst

Secondary structure annotation
==============================

.. currentmodule:: biotite.application.dssp

Although :mod:`biotite.structure` implements the DSSP algorithm in
:func:`biotite.structure.annotate_dssp()`, the original DSSP software can also
be used via the :mod:`biotite.application.dssp` subpackage.
Let us demonstrate this on the example of the good old miniprotein *TC5b*.

.. Do not run the following Jupyter cells, as DSSP is currently not in build environment

.. jupyter-input::

    from tempfile import gettempdir
    import biotite.database.rcsb as rcsb
    import biotite.application.dssp as dssp
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1l2y", "bcif", gettempdir()))
    atom_array = pdbx.get_structure(pdbx_file, model=1)
    sse = dssp.DsspApp().run(atom_array).result()
    print("".join(struc.DsspElement.to_symbols(sse)))

.. jupyter-output::

    CHHHHHHHTTGGGGTCPPPC
