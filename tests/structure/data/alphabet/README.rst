Structural alphabet sequences
==============================

This directory contains structural alphabet sequences for the test structure files
from the `tests/structure/data/` directory, generated with the respective reference
implementation.

3Di sequences
-------------

The 3Di sequences in `i3d.fasta` were generated with `foldseek` according to
`these instructions <https://github.com/steineggerlab/foldseek/issues/314#issuecomment-2283329286>`_:

.. code-block:: console

    $ foldseek createdb --chain-name-mode 1 tests/structure/data/*.cif /tmp/biotite_3di
    $ foldseek lndb /tmp/biotite_3di_h /tmp/biotite_3di_ss_h
    $ foldseek convert2fasta /tmp/biotite_3di_ss tests/structure/data/alphabet/i3d.fasta