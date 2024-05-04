:sd_hide_title: true

.. include:: /tutorial/preamble.rst

##########################
``application`` subpackage
##########################

Beyond Biotite - The ``application`` subpackage
===============================================

.. currentmodule:: biotite.application

Although you can achieve a lot with *Biotite*, there are still a lot of things
which are not implemented in this *Python* package.
But wait, this is what the :mod:`biotite.application` package is for:
It contains interfaces for popular external software.
This ranges from locally installed software to external tools running on web
servers.
The usage of these interfaces is seamless: Rather than writing input files and
reading output files, you simply put in your *Python*
objects (e.g. instances of :class:`Sequence` or :class:`AtomArray`) and the
interface returns *Python* objects (e.g. an :class:`Alignment` object).

.. note::

    Note that in order to use an interface in :mod:`biotite.application` the
    corresponding software must be installed or the web server must be
    reachable, respectively.
    These programs are not shipped with the *Biotite* package.

The base class for all interfaces is the :class:`Application` class.
Each :class:`Application` can be *started* by calling
:meth:`Application.start()`.
The results are collected with :meth:`Application.join()`.

.. jupyter-execute::
    :hide-code:

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="MUSCLE did not write a tree file from the second iteration"
    )

.. jupyter-execute::

    from biotite.sequence import ProteinSequence
    from biotite.application.muscle import MuscleApp

    app = MuscleApp([ProteinSequence("BIQTITE"), ProteinSequence("IQLITE")])
    app.start()
    # The application is running in the background
    app.join()


The lines between :meth:`start()` and :meth:`join()` can be used to run any
other code while the application is running in the background,
including starting another :class:`Application` in parallel.

The following chapters will give you an overview of the different applications
interfaced in the :mod:`biotite.application` subpackage.

.. toctree::
    :maxdepth: 1
    :hidden:

    msa
    blast
    tantan
    sra
    dssp
    viennarna