Test structures
===============

1AKI: Very simple protein structure
1DIX: Structure contains insertion code
1F2N: Structure contains biological assembly
5ZNG: Structure contains biological assembly with multiple entries in _pdbx_struct_assembly_gen
4ZXB: Multiple entries in _pdbx_struct_assembly_gen for the same assembly_id
1NCB: Multiple entries in _pdbx_struct_assembly_gen for the same assembly_id and chains
1GYA: Large multi-model structure
2AXD: Multi-model structure with different number of atoms per model
1IGY: Multi-chain structure
1L2Y: Very small and very simple multi-model structure
1O1Z: Structure contains negative residue IDs
3O5R: Structure contains altlocs
5H73: Structure contains residue with "'" in atom name
1QXB: Structure contains complementary DNA-sequence (only used in `base_pairs`)
5UGO: Structure contains deoxynucleotide
4GXY: Structure contains ribonucleotide
2D0F: Structure contains oligosaccharides
5EIL: Structure contains a non-canonical amino acid on each chain
4P5J: Structure contains a non-canonical nucleotide
1CRR: Multi-model structure with the first model ID not being 1
7GSA: Contains 5-character residue name 'A1AA6'

Creation
--------

To create the test files for the above entries, run the following command:

.. code-block:: console

    $ python create_test_structures.py -f ids.txt