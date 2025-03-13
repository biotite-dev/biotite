.. include:: /tutorial/preamble.rst

Secondary structure annotation
==============================

.. currentmodule:: biotite.application.dssp

Althogh :mod:`biotite.structure` offers the function :func:`annotate_sse()` to
assign secondary structure elements based on the P-SEA algorithm, DSSP can also
be used via the :mod:`biotite.application.dssp` subpackage.
Let us demonstrate this on the example of the good old miniprotein *TC5b*.

.. Do not run the following Jupyter cells, as DSSP is currently not in build environment

.. jupyter-input::

    from tempfile import gettempdir
    import biotite.database.rcsb as rcsb
    import biotite.application.dssp as dssp
    import biotite.structure.io.pdbx as pdbx

    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("1l2y", "bcif", gettempdir()))
    atom_array = pdbx.get_structure(pdbx_file, model=1)
    app = dssp.DsspApp(atom_array)
    app.start()
    app.join()
    sse = app.get_sse()
    print("".join(sse))

.. jupyter-output::

    CHHHHHHHTTGGGGTCCCCC

Similar to the MSA examples, :class:`DsspApp` has the convenience
method :func:`DsspApp.annotate_sse()` as shortcut.
