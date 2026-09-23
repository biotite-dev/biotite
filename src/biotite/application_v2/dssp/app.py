from __future__ import annotations

__name__ = "biotite.application_v2.dssp"
__author__ = "Patrick Kunzmann"
__all__ = ["DsspApp", "DsspElement"]

import subprocess
from enum import IntEnum
from os import PathLike
from tempfile import NamedTemporaryFile
import numpy as np
from numpy.typing import ArrayLike
from biotite.application_v2.localapp import (
    CLIArgument,
    CLIOption,
    CLIParameter,
    CommandSetup,
    LocalApp,
    cleanup_tempfile,
    command,
)
from biotite.structure.atoms import AtomArray
from biotite.structure.error import BadStructureError
from biotite.structure.filter import filter_amino_acids
from biotite.structure.io.pdbx.cif import CIFCategory, CIFColumn, CIFFile
from biotite.structure.io.pdbx.component import MaskValue
from biotite.structure.io.pdbx.convert import set_structure
from biotite.structure.repair import create_continuous_res_ids
from biotite.structure.residues import get_residue_starts
from biotite.structure.sse import _symbols_to_values, _values_to_symbols
from biotite.typing import K, N, NDArray1


class DsspElement(IntEnum):
    r"""
    The secondary structure elements assigned by *DSSP*.

    Each element has a corresponding one-letter `symbol`, that is used
    in the *DSSP* output.

    - ``COIL`` (``'C'``): loop, coil or irregular
    - ``ALPHA_HELIX`` (``'H'``): :math:`{\alpha}`-helix
    - ``BRIDGE`` (``'B'``): isolated :math:`{\beta}`-bridge
    - ``STRAND`` (``'E'``): extended strand, participation in
      :math:`{\beta}`-ladder
    - ``HELIX_3_10`` (``'G'``): 3 :sub:`10`-helix
    - ``PI_HELIX`` (``'I'``): :math:`{\pi}`-helix
    - ``TURN`` (``'T'``): hydrogen bonded turn
    - ``BEND`` (``'S'``): bend
    - ``POLYPROLINE_HELIX`` (``'P'``): polyproline II helix
      (only assigned by *DSSP* 4 and later)

    Examples
    --------

    >>> sse = DsspElement.from_symbols(["C", "H", "H", "E", "P"])
    >>> print(sse)
    [0 1 1 3 8]
    >>> print(DsspElement.to_symbols(sse))
    ['C' 'H' 'H' 'E' 'P']
    >>> print(DsspElement.ALPHA_HELIX.symbol)
    H
    >>> print(DsspElement.from_symbol("E").name)
    STRAND
    """

    COIL = 0
    ALPHA_HELIX = 1
    BRIDGE = 2
    STRAND = 3
    HELIX_3_10 = 4
    PI_HELIX = 5
    TURN = 6
    BEND = 7
    POLYPROLINE_HELIX = 8

    @property
    def symbol(self) -> str:
        """
        The one-letter symbol of the secondary structure element.

        Returns
        -------
        symbol : str
            The one-letter symbol.
        """
        return _DSSP_SYMBOLS[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> DsspElement:
        """
        Get the secondary structure element from its one-letter symbol.

        Parameters
        ----------
        symbol : str
            The symbol.

        Returns
        -------
        sse : DsspElement
            The corresponding secondary structure element.
        """
        for sse, sse_symbol in _DSSP_SYMBOLS.items():
            if sse_symbol == symbol:
                return sse
        raise ValueError(f"Unknown secondary structure symbol '{symbol}'")

    @classmethod
    def from_symbols(cls, symbols: ArrayLike) -> NDArray1[K, np.int_]:
        """
        Convert an array of one-letter symbols into an array of
        secondary structure elements.

        Parameters
        ----------
        symbols : array-like, shape=(k,), dtype=str
            The symbols.

        Returns
        -------
        sse : ndarray, shape=(k,), dtype=int
            The corresponding secondary structure elements.
        """
        return _symbols_to_values(_DSSP_SYMBOLS, symbols)

    @classmethod
    def to_symbols(cls, sse: ArrayLike) -> NDArray1[K, np.str_]:
        """
        Convert an array of secondary structure elements into an array
        of one-letter symbols.

        Parameters
        ----------
        sse : array-like, shape=(k,), dtype=int
            The secondary structure elements.

        Returns
        -------
        symbols : ndarray, shape=(k,), dtype=str
            The corresponding symbols.
        """
        return _values_to_symbols(_DSSP_SYMBOLS, sse)


_DSSP_SYMBOLS = {
    DsspElement.COIL: "C",
    DsspElement.ALPHA_HELIX: "H",
    DsspElement.BRIDGE: "B",
    DsspElement.STRAND: "E",
    DsspElement.HELIX_3_10: "G",
    DsspElement.PI_HELIX: "I",
    DsspElement.TURN: "T",
    DsspElement.BEND: "S",
    DsspElement.POLYPROLINE_HELIX: "P",
}


class DsspApp(LocalApp):
    r"""
    A handle to the *DSSP* software.

    *DSSP* differentiates between the secondary structure elements
    listed in :class:`DsspElement`.

    Parameters
    ----------
    path : str, optional
        Path of the *DSSP* binary.

    Examples
    --------

    >>> sse = DsspApp().run(atom_array).result()
    >>> print(sse)
    [0 1 1 1 1 1 1 1 6 6 4 4 4 4 6 0 8 8 8 0]
    >>> print("".join(DsspElement.to_symbols(sse)))
    CHHHHHHHTTGGGGTCPPPC
    """

    def __init__(self, path: PathLike[str] | str = "mkdssp") -> None:
        super().__init__(path)

    @command
    def run(self, atom_array: AtomArray) -> CommandSetup[NDArray1[K, np.int_]]:
        """
        Assign the secondary structure of the given atom array.

        Parameters
        ----------
        atom_array : AtomArray
            The atom array to be annotated.

        Returns
        -------
        future : Future of ndarray, shape=(k,), dtype=int
            A handle to the running assignment.
            Call :meth:`Future.result()` to obtain the array of
            :class:`DsspElement` values, one per residue in the input
            atom array.
        """
        if not np.all(filter_amino_acids(atom_array)):
            raise BadStructureError("The input structure must contain only amino acids")
        array = atom_array.copy()
        # DSSP requires the 'occupancy', 'b_factor' and 'charge' fields
        # -> Add these placeholder values
        categories = array.get_annotation_categories()
        if "charge" not in categories:
            array.set_annotation("charge", np.zeros(array.array_length(), dtype=int))
        if "b_factor" not in categories:
            array.set_annotation(
                "b_factor", np.zeros(array.array_length(), dtype=float)
            )
        if "occupancy" not in categories:
            array.set_annotation(
                "occupancy", np.ones(array.array_length(), dtype=float)
            )
        # DSSP>=4 complains about the `pdbx_poly_seq_scheme` category,
        # if `seq_id` does not start at 1
        array.res_id = create_continuous_res_ids(array)

        try:
            # The command line parameters have changed in version 4;
            # older versions do not report a version with `--version`
            new_cli = self.version.major >= 4
        except subprocess.SubprocessError:
            new_cli = False

        in_file = NamedTemporaryFile("w", suffix=".cif", delete=False)
        out_file = NamedTemporaryFile("r", suffix=".dssp", delete=False)

        cif_file = CIFFile()
        set_structure(cif_file, array)
        cif_file.block["pdbx_poly_seq_scheme"] = _create_pdbx_poly_seq_scheme(
            array, cif_file.block["atom_site"]["label_entity_id"].as_array(str)
        )
        cif_file.write(in_file)
        in_file.flush()

        if new_cli:
            parameters: list[CLIParameter] = [
                CLIArgument(in_file.name),
                CLIArgument(out_file.name),
            ]
        else:
            parameters = [
                CLIOption("i", in_file.name),
                CLIOption("o", out_file.name),
            ]

        def evaluate(stdout: bytes, stderr: bytes) -> NDArray1[K, np.int_]:
            lines = out_file.read().split("\n")
            # Index where SSE records start
            sse_start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE AA STRUCTURE"):
                    sse_start = i + 1
            if sse_start is None:
                raise ValueError("DSSP file does not contain SSE records")
            # Remove "!" for missing residues
            filtered_lines = [
                line for line in lines[sse_start:] if len(line) != 0 and line[13] != "!"
            ]
            symbols = np.zeros(len(filtered_lines), dtype="U1")
            # Parse file for SSE letters
            for i, line in enumerate(filtered_lines):
                symbols[i] = line[16]
            symbols[symbols == " "] = "C"
            return DsspElement.from_symbols(symbols)

        def cleanup() -> None:
            cleanup_tempfile(in_file)
            cleanup_tempfile(out_file)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )


def _create_pdbx_poly_seq_scheme(
    atom_array: AtomArray, entity_ids: NDArray1[N, np.str_]
) -> CIFCategory:
    """
    Create the ``pdbx_poly_seq_scheme`` category, as required by DSSP.

    Parameters
    ----------
    atom_array : AtomArray
        The atom array to create the category from.
    entity_ids : ndarray, dtype=str
        The entity IDs for each atom.

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
