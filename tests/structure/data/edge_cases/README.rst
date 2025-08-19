# Collection of structure file edges cases

- ``hetatm.pdb``: A simple PDB file containing a custom ligand, whose name is already
  taken by the CCD.
  However, since it contains only ``HETATM`` records, the bonds should not be taken from
  the CCD but from the ``CONECT`` records.
- ``res_ids.cif``: Subsequent residues have the same ``label_xxx`` annotation, which
  makes it hard to determine where a new residue starts.
  However, using ``label_seq_id`` as fallback allows resolving the residue starts.
  Derived from PDB entry ``5HU8``.