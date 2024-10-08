import pathlib
import pytest
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdb as pdb
from tests.util import data_dir


@pytest.mark.parametrize(
    "pdb_id, chain_id, ref_3di",
    [
        (
            "3bww",
            "A",
            "DKDFFEAAEDDLVCLVVLLPPPACPQRQAYEDALVVQVPDDPVSVVSVVNSLVHHAYAYEYEAQQL"
            "LDDPQGDVVSLVSVLVCCVVSVPQEYEYENDPPDADALDVVSLVSSLVSQLVSCVSSVGAYAYEDA"
            "ADQDHDPRHPDDVLVSRQSNCVSNVHAHAYELVRLVRCCVRPVPDDSLVSLVRHPLQRHQHYEYQV"
            "VSVVSVLVNLVDHQAHHYYYHDYPDDVVVNSVVRVVSRVSNVVSCVVVVHYIDMD",
        ),
        (
            "8crb",
            "A",
            "DWAKDKDWADEDAAQAKTKIKMATPPDLLQDFFKFKWFDAPPDDIDGQAPGACPSPPLADDVHHHH"
            "GKGWHDDSVRRMIMIMGGNDDQVVFGKMKMFTADDADPQVVVPDGDDTDDMHDIDTYGHPPDDFFA"
            "WDKDKDQDDPVPCPVQKPKIKMKTDDGDDDDKDKAWLVNPGDPQKDDFDWDADPVRGIIDMIIGMD"
            "GNVCFQVGFTKIWMAGVVVRDIDIDGGHD",
        ),
        (
            "8crb",
            "B",
            "DAAKDFDQQEEEAAQAKDKGWIFAADVPPVPDAFWKWWDAPPDDIDTAADPNQAGDPVDHSQKGWD"
            "ADHGITIIMGGRDDNSRQGFIWRAQPDDPDHNGHTDDTHGYYHCPDDQDDKDKDWDDAAVVVLVVL"
            "FGKTKIKIDDGDDPPKDKFKDLQNHTDDAQWDWDDWDLDPVRTIMTMIIRRDGVVSCVVSQKMKMW"
            "IDDDVHTDIDMDGNVVHD",
        ),
        (
            "8crb",
            "C",
            "DPCVLVVLVLQLVLVVLLLVVVVVVLVVCVVVLFKDWQDPVHDWQLACVSPDHDCPDCCSVPGSNN"
            "VQQCPKPLDDVTATNQSVQQIDDGDLDHDDDDDTIQGCPPPVRCSVVVVVVSVVSVVVSVVSCVVS"
            "VVVVVVD",
        ),
    ],
)
def test_to_3di(pdb_id, chain_id, ref_3di):
    """
    Check if the 3di sequence of a chain is correctly generated, by comparing the result
    to a reference sequence generated with *foldseek*.
    """
    path = pathlib.Path(data_dir("structure")) / f"{pdb_id}.pdb"
    file = pdb.PDBFile.read(path)
    atoms = file.get_structure(model=1)
    chain = atoms[atoms.chain_id == chain_id]

    test_3di, _ = strucalph.to_3di(chain)

    # We filtered one chain -> There should be only one 3di sequence
    assert len(test_3di) == 1
    assert str(test_3di[0]) == ref_3di


def test_missing():
    """
    Test if missing or non-peptide residues within a chain are correctly handled.
    """
    pass
