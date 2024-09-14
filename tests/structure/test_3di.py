import unittest
import pathlib

import numpy

from biotite.structure.io.pdb import PDBFile
from biotite.structure.atoms import AtomArray
from biotite.structure.alphabet import to_3di
from biotite.structure.alphabet.encoder import Encoder, PartnerIndexEncoder, VirtualCenterEncoder
from biotite.structure.chains import apply_chain_wise
from tests.util import data_dir


def _get_structure(name):
    path = pathlib.Path(data_dir("structure")).joinpath(f"{name}.pdb")
    file = PDBFile.read(path)
    structure = file.get_structure()
    return structure if isinstance(structure, AtomArray) else structure[0]


class TestEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder()

    def test_encode_3bww(self):
        structure = _get_structure("3bww")
        chain = structure[structure.chain_id == 'A']
        sequence = to_3di(chain)
        self.assertEqual(
            str(sequence),
            "DKDFFEAAEDDLVCLVVLLPPPACPQRQAYEDALVVQVPDDPVSVVSVVNSLVHHAYAYEYEAQQL"
            "LDDPQGDVVSLVSVLVCCVVSVPQEYEYENDPPDADALDVVSLVSSLVSQLVSCVSSVGAYAYEDA"
            "ADQDHDPRHPDDVLVSRQSNCVSNVHAHAYELVRLVRCCVRPVPDDSLVSLVRHPLQRHQHYEYQV"
            "VSVVSVLVNLVDHQAHHYYYHDYPDDVVVNSVVRVVSRVSNVVSCVVVVHYIDMD",
        )

    def test_encode_8crb(self):
        structure = _get_structure("8crb")
        sequences = apply_chain_wise(structure, structure, to_3di, axis=None)
        self.assertEqual(len(sequences), 3)

    def test_encode_8crb_A(self):
        structure = _get_structure("8crb")
        chain = structure[structure.chain_id == 'A']
        sequence = to_3di(chain)
        self.assertEqual(
            str(sequence),
            "DWAKDKDWADEDAAQAKTKIKMATPPDLLQDFFKFKWFDAPPDDIDGQAPGACPSPPLADDVHHHH"
            "GKGWHDDSVRRMIMIMGGNDDQVVFGKMKMFTADDADPQVVVPDGDDTDDMHDIDTYGHPPDDFFA"
            "WDKDKDQDDPVPCPVQKPKIKMKTDDGDDDDKDKAWLVNPGDPQKDDFDWDADPVRGIIDMIIGMD"
            "GNVCFQVGFTKIWMAGVVVRDIDIDGGHD",
        )

    def test_encode_8crb_B(self):
        structure = _get_structure("8crb")
        chain = structure[structure.chain_id == 'B']
        sequence = to_3di(chain)
        self.assertEqual(
            str(sequence),
            "DAAKDFDQQEEEAAQAKDKGWIFAADVPPVPDAFWKWWDAPPDDIDTAADPNQAGDPVDHSQKGWD"
            "ADHGITIIMGGRDDNSRQGFIWRAQPDDPDHNGHTDDTHGYYHCPDDQDDKDKDWDDAAVVVLVVL"
            "FGKTKIKIDDGDDPPKDKFKDLQNHTDDAQWDWDDWDLDPVRTIMTMIIRRDGVVSCVVSQKMKMW"
            "IDDDVHTDIDMDGNVVHD",
        )

    def test_encode_8crb_C(self):
        structure = _get_structure("8crb")
        chain = structure[structure.chain_id == 'C']
        sequence = to_3di(chain)
        self.assertEqual(
            str(sequence),
            "DPCVLVVLVLQLVLVVLLLVVVVVVLVVCVVVLFKDWQDPVHDWQLACVSPDHDCPDCCSVPGSNN"
            "VQQCPKPLDDVTATNQSVQQIDDGDLDHDDDDDTIQGCPPPVRCSVVVVVVSVVSVVVSVVSCVVS"
            "VVVVVVD",
        )
