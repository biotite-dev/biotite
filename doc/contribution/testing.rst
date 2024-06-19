Testing the package
===================

In-development tests
--------------------
While developing a new feature or fixing a bug, it is handy to run a test
script against the code you are working on.
To ensure that the imported package ``biotite`` points to the code you are
working on, you may want to install the local repository clone in *editable*
mode:

.. code-block:: console

    $ pip install -e .

If you are writing or using an extension module in *Cython*, consider using
`pyximport <https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiling-with-pyximport>`_
at the beginning of the script you use for testing.

.. code-block:: python

    import numpy as np
    import pyximport
    pyximport.install(
        build_in_temp=False,
        setup_args={"include_dirs":np.get_include()},
        language_level=3
    )

To enforce the recompilation of the changed *Cython* module, delete the
respective compiled module (``.dll`` or ``.so``) from the ``src/`` directory,
if already existing.

Unit tests
----------
The backbone of testing *Biotite* are the unit tests in the ``tests``
directory.
`Pytest <https://docs.pytest.org>`_ is used as the testing framework.
To run the tests, install the local repository clone (in editable mode) and
run the tests:

.. code-block:: console

   $ pip install -e .
   $ pytest

Doctests
--------
For simple tests checking that some code simply does not raise an exception
and produces some predefined output,
`doctests <https://docs.python.org/3/library/doctest.html>`_ are suitable.
They are part of the docstrings of the corresponding functions and classes.
The doctests fulfill two purposes:
They are automatically executed by ``pytest`` via the
``tests/test_doctests.py`` module and give users reading the API reference
easily understandable examples how a function/class works.

Testing visualizations
----------------------
Testing visualization functions (e.g. in :mod:`biotite.sequence.graphics`) is
difficult, because the output can hardly be checked against some reference
value.
To still have at least some confirmation that these functions produce the
expected output, it is mandatory to have at least one example using that
function in the :ref:`gallery <example_gallery>`.