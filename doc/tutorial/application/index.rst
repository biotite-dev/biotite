:sd_hide_title: true

.. include:: /tutorial/preamble.rst

#############################
``application_v2`` subpackage
#############################

Beyond Biotite - The ``application_v2`` subpackage
==================================================

.. currentmodule:: biotite.application_v2

Although you can achieve a lot with *Biotite*, there are still a lot of things
which are not implemented in this *Python* package.
But wait, this is what the :mod:`biotite.application_v2` package is for:
It contains interfaces for popular external software.
The usage of these interfaces is seamless: Rather than writing input files and
reading output files, you simply put in your *Python*
objects (e.g. instances of :class:`Sequence` or :class:`AtomArray`) and the
interface returns *Python* objects (e.g. an :class:`Alignment` object).

.. note::

    Note that in order to use an interface in :mod:`biotite.application_v2`
    the corresponding software must be installed.
    These programs are not shipped with the *Biotite* package.

Running an application
----------------------
Each interface is a subclass of :class:`Application`.
An :class:`Application` object is merely a handle to the installed software.
For example, creating a :class:`.Muscle3App` only looks up the ``muscle``
executable, but does not run it yet.
A run is launched by calling the respective method of the handle, in this case
:meth:`.Muscle3App.run()`.
It immediately returns a :class:`Future`, while the software is still running
in the background.

.. jupyter-execute::

    import biotite.sequence as seq
    import biotite.application_v2.muscle as muscle

    sequences = [
        seq.ProteinSequence("BIQTITE"),
        seq.ProteinSequence("TITANITE"),
        seq.ProteinSequence("BISMITE"),
        seq.ProteinSequence("IQLITE"),
    ]
    app = muscle.Muscle3App()
    future = app.run(sequences)
    # The application is running in the background,
    # so we can run other code in the meantime
    # Wait for the application to finish and get the result
    result = future.result()
    print(result.alignment)

The lines between the method call and :meth:`Future.result()` can be used to
run any other code while the application is running,
including starting another :class:`Application` in parallel.
:meth:`Future.result()` blocks until the run has finished and returns the
result object.
The :class:`Future` mirrors :class:`concurrent.futures.Future`, hence you can
also check via :meth:`Future.done()` whether the run has finished or abort it
via :meth:`Future.cancel()`.
As the handle does not keep any state of the run, the same
:class:`Application` object can be used for multiple runs.

Command line options
--------------------
Under the hood the method call is converted into a command line invocation of
the software.
The input objects are written into temporary files and the parameters of the
method are translated into the corresponding command line options.
The command is available via :attr:`.LocalProcessFuture.command`.

.. jupyter-execute::

    print(future.command)

Most programs have more options than the method parameters cover.
These can be passed as additional keyword arguments, that are named like the
respective command line option, with underscores instead of dashes.
The value is formatted automatically:
For example, a number or string is passed as the option value, ``None`` turns
the option into a flag without value and a list repeats the option for each
element.

.. jupyter-execute::

    future = app.run(sequences, maxiters=2, diags=None)
    print(future.command)
    print(future.result().alignment)

However, not any option is accepted:
Options that conflict with the ones the interface sets by itself (e.g.
the input and output files) or that would change the output format are
rejected.

.. jupyter-execute::
    :raises: ValueError

    app.run(sequences, quiet=None)

The following chapters will give you an overview of the different applications
interfaced in the :mod:`biotite.application_v2` subpackage.

.. toctree::
    :maxdepth: 1
    :hidden:

    msa
    mmseqs
    viennarna
    dssp
    vina
