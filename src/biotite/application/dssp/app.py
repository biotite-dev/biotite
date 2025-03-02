# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.dssp"
__author__ = "Patrick Kunzmann"
__all__ = ["DsspApp"]

from subprocess import SubprocessError
from tempfile import NamedTemporaryFile
import numpy as np
from biotite.application.application import AppState, requires_state
from biotite.application.localapp import LocalApp, cleanup_tempfile, get_version
from biotite.structure.error import BadStructureError
from biotite.structure.filter import filter_amino_acids
from biotite.structure.io.pdbx.cif import CIFCategory, CIFColumn, CIFFile
from biotite.structure.io.pdbx.component import MaskValue
from biotite.structure.io.pdbx.convert import set_structure
from biotite.structure.repair import create_continuous_res_ids
from biotite.structure.residues import get_residue_starts


class DsspApp(LocalApp):
    r"""
    Annotate the secondary structure of a protein structure using the
    *DSSP* software.

    Internally this creates a :class:`Popen` instance, which handles
    the execution.

    DSSP differentiates between 8 different types of secondary
    structure elements:

       - C: loop, coil or irregular
       - H: :math:`{\alpha}`-helix
       - B: :math:`{\beta}`-bridge
       - E: extended strand, participation in :math:`{\beta}`-ladder
       - G: 3 :sub:`10`-helix
       - I: :math:`{\pi}`-helix
       - T: hydrogen bonded turn
       - S: bend

    Parameters
    ----------
    atom_array : AtomArray
        The atom array to be annotated.
    bin_path : str, optional
        Path of the *DDSP* binary.

    Examples
    --------

    >>> app = DsspApp(atom_array)
    >>> app.start()
    >>> app.join()
    >>> print(app.get_sse())
    ['C' 'H' 'H' 'H' 'H' 'H' 'H' 'H' 'T' 'T' 'G' 'G' 'G' 'G' 'T' 'C' 'P' 'P'
     'P' 'C']
    """

    def __init__(self, atom_array, bin_path="mkdssp"):
        super().__init__(bin_path)

        if not np.all(filter_amino_acids(atom_array)):
            raise BadStructureError("The input structure must contain only amino acids")
        self._array = atom_array.copy()
        # DSSP requires also the
        # 'occupancy', 'b_factor' and 'charge' fields
        # -> Add these placeholder values
        categories = self._array.get_annotation_categories()
        if "charge" not in categories:
            self._array.set_annotation(
                "charge", np.zeros(self._array.array_length(), dtype=int)
            )
        if "b_factor" not in categories:
            self._array.set_annotation(
                "b_factor", np.zeros(self._array.array_length(), dtype=float)
            )
        if "occupancy" not in categories:
            self._array.set_annotation(
                "occupancy", np.ones(self._array.array_length(), dtype=float)
            )
        # DSSP>=4 complains about the `pdbx_poly_seq_scheme` category,
        # if `seq_id` does not start at 1
        self._array.res_id = create_continuous_res_ids(self._array)

        try:
            # The parameters have changed in version 4
            self._new_cli = get_version(bin_path)[0] >= 4
        except SubprocessError:
            # In older versions, the no version is returned with `--version`
            # -> a SubprocessError is raised
            self._new_cli = False
        self._in_file = NamedTemporaryFile("w", suffix=".cif", delete=False)
        self._out_file = NamedTemporaryFile("r", suffix=".dssp", delete=False)

    def run(self):
        in_file = CIFFile()
        set_structure(in_file, self._array)
        in_file.block["pdbx_poly_seq_scheme"] = _create_pdbx_poly_seq_scheme(
            self._array, in_file.block["atom_site"]["label_entity_id"].as_array(str)
        )
        in_file.write(self._in_file)
        self._in_file.flush()
        if self._new_cli:
            self.set_arguments([self._in_file.name, self._out_file.name])
        else:
            self.set_arguments(["-i", self._in_file.name, "-o", self._out_file.name])
        super().run()

    def evaluate(self):
        super().evaluate()
        lines = self._out_file.read().split("\n")
        # Index where SSE records start
        sse_start = None
        for i, line in enumerate(lines):
            if line.startswith("  #  RESIDUE AA STRUCTURE"):
                sse_start = i + 1
        if sse_start is None:
            raise ValueError("DSSP file does not contain SSE records")
        # Remove "!" for missing residues
        lines = [
            line for line in lines[sse_start:] if len(line) != 0 and line[13] != "!"
        ]
        self._sse = np.zeros(len(lines), dtype="U1")
        # Parse file for SSE letters
        for i, line in enumerate(lines):
            self._sse[i] = line[16]
        self._sse[self._sse == " "] = "C"

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        cleanup_tempfile(self._out_file)

    @requires_state(AppState.JOINED)
    def get_sse(self):
        """
        Get the resulting secondary structure assignment.

        Returns
        -------
        sse : ndarray, dtype="U1"
            An array containing DSSP secondary structure symbols
            corresponding to the residues in the input atom array.
        """
        return self._sse

    @staticmethod
    def annotate_sse(atom_array, bin_path="mkdssp"):
        """
        Perform a secondary structure assignment to an atom array.

        This is a convenience function, that wraps the :class:`DsspApp`
        execution.

        Parameters
        ----------
        atom_array : AtomArray
            The atom array to be annotated.
        bin_path : str, optional
            Path of the DDSP binary.

        Returns
        -------
        sse : ndarray, dtype="U1"
            An array containing DSSP secondary structure symbols
            corresponding to the residues in the input atom array.
        """
        app = DsspApp(atom_array, bin_path)
        app.start()
        app.join()
        return app.get_sse()


def _create_pdbx_poly_seq_scheme(atom_array, entity_ids):
    """
    Create the ``pdbx_poly_seq_scheme`` category, as required by DSSP.

    Parameters
    ----------
    atom_array : AtomArray
        The atom array to create the category from.
    entity_ids : ndarray, dtype=str
        The entity IDs for each atoms.

    Returns
    -------
    pdbx_poly_seq_scheme : CIFCategory
        The ``pdbx_poly_seq_scheme`` category.
    """
    res_start_indices = get_residue_starts(atom_array)
    chain_id = atom_array.chain_id[res_start_indices]
    res_name = atom_array.res_name[res_start_indices]
    res_id = atom_array.res_id[res_start_indices]
    ins_code = atom_array.ins_code[res_start_indices]
    hetero = atom_array.hetero[res_start_indices]
    entity_id = entity_ids[res_start_indices]

    poly_seq_scheme = CIFCategory()
    poly_seq_scheme["asym_id"] = chain_id
    poly_seq_scheme["entity_id"] = entity_id
    poly_seq_scheme["seq_id"] = res_id
    poly_seq_scheme["mon_id"] = res_name
    poly_seq_scheme["ndb_seq_num"] = res_id
    poly_seq_scheme["pdb_seq_num"] = res_id
    poly_seq_scheme["auth_seq_num"] = res_id
    poly_seq_scheme["pdb_mon_id"] = res_name
    poly_seq_scheme["auth_mon_id"] = res_name
    poly_seq_scheme["pdb_strand_id"] = chain_id
    poly_seq_scheme["pdb_ins_code"] = CIFColumn(
        ins_code, np.where(ins_code == "", MaskValue.MISSING, MaskValue.PRESENT)
    )
    poly_seq_scheme["hetero"] = np.where(hetero, "y", "n")

    return poly_seq_scheme
