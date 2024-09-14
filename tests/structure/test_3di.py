import unittest
import pathlib

import Bio.PDB
import numpy
from biotite.structure.alphabet.encoder import Encoder, PartnerIndexEncoder, VirtualCenterEncoder
from tests.util import data_dir


class TestVirtualCenterEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = VirtualCenterEncoder()
        cls.parser = Bio.PDB.PDBParser(QUIET=True)

    @classmethod
    def get_structure(cls, name):
        path = pathlib.Path(data_dir("structure")).joinpath(f"{name}.pdb")
        return cls.parser.get_structure(name, path)

    def test_calc_virtual_center(self):
        ca = numpy.array([[34.826, 19.254, 17.339]])
        cb = numpy.array([[35.285, 18.694, 15.994]])
        n_ = numpy.array([[35.805, 19.041, 18.426]])

        vc = self.encoder._compute_virtual_center(ca, cb, n_)
        self.assertAlmostEqual(vc[0, 0], 32.2276, places=4)
        self.assertAlmostEqual(vc[0, 1], 20.2157, places=4)
        self.assertAlmostEqual(vc[0, 2], 16.0518, places=4)

        ca = numpy.array([[21.056, 18.27, 0.063]])
        cb = numpy.array([[21.428, 19.604, 0.838]])
        n_ = numpy.array([[21.789, 17.734, -1.084]])

        vc = self.encoder._compute_virtual_center(ca, cb, n_)
        self.assertAlmostEqual(vc[0, 0], 18.5941, places=4)
        self.assertAlmostEqual(vc[0, 1], 17.8221, places=4)
        self.assertAlmostEqual(vc[0, 2], 2.01565, places=4)

    def test_encode_1xso_chainA(self):
        structure = self.get_structure("1xso")
        centers = self.encoder.encode_chain(structure[0]["A"], disordered_atom="last")

        self.assertAlmostEqual(centers[0, 0], 30.653, places=4)
        self.assertAlmostEqual(centers[0, 1], 12.7129, places=4)
        self.assertAlmostEqual(centers[0, 2], -3.9203, places=4)

        self.assertAlmostEqual(centers[4, 0], 18.5941, places=4)
        self.assertAlmostEqual(centers[4, 1], 17.8221, places=4)
        self.assertAlmostEqual(centers[4, 2], 2.01565, places=4)


class TestPartnerIndexEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = PartnerIndexEncoder()
        cls.parser = Bio.PDB.PDBParser(QUIET=True)

    @classmethod
    def get_structure(cls, name):
        path = pathlib.Path(data_dir("structure")).joinpath(f"{name}.pdb")
        return cls.parser.get_structure(name, path)

    def test_encode_1xso_chainA(self):
        structure = self.get_structure("1xso")
        partners = self.encoder.encode_chain(structure[0]["A"], disordered_atom="last")
        self.assertListEqual(
            list(partners[1:-1]),
            # fmt: off
            [
                17, 145, 15, 143, 13, 141, 6, 53, 8, 11, 12, 32, 5, 30, 3, 28,
                1, 26, 20, 19, 20, 21, 98, 98, 96, 18, 94, 16, 92, 14, 90, 12,
                89, 89, 117, 35, 85, 37, 83, 115, 80, 113, 79, 111, 57, 57,
                110, 143, 50, 53, 50, 50, 50, 55, 56, 55, 45, 45, 132, 43, 77,
                77, 62, 62, 66, 75, 131, 76, 122, 80, 97, 73, 74, 73, 66, 68,
                62, 99, 43, 41, 70, 119, 39, 91, 37, 87, 86, 87, 33, 31, 84,
                29, 94, 27, 94, 25, 71, 23, 78, 106, 23, 107, 104, 103, 104,
                100, 102, 144, 144, 47, 44, 142, 42, 139, 40, 136, 35, 136, 82,
                134, 134, 69, 130, 125, 124, 127, 128, 127, 132, 120, 67, 59,
                120, 120, 136, 116, 138, 137, 114, 7, 6, 112, 4, 109, 2, 109,
                148, 147
            ]
            # fmt: on
        )


class TestEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = Encoder()
        cls.parser = Bio.PDB.PDBParser(QUIET=True)

    @classmethod
    def get_structure(cls, name):
        path = pathlib.Path(data_dir("structure")).joinpath(f"{name}.pdb")
        return cls.parser.get_structure(name, path)

    def test_encode_1xso_disordered_last(self):
        structure = self.get_structure("1xso")
        states = self.encoder.encode_chain(structure[0]["A"], disordered_atom="last")
        sequence = self.encoder.build_sequence(states)
        self.assertEqual(
            sequence,
            "DKKKWWKDFPDPKTKIKIWDDDDLFKIKIWMKIFQADFDKKWKWWACAQDCPVHVVVSHFGAAPPD"
            "FWDFAQPDPRHGLTGDFIFGDDPRMTTDMDIHNSAGCDDPNRQQRIKMFIANAGQCGLPPPDPVSR"
            "GTSPRDDTRIMTGMHDDD",
        )

    def test_encode_3bww(self):
        structure = self.get_structure("3bww")
        states = self.encoder.encode_chain(structure[0]["A"])
        sequence = self.encoder.build_sequence(states)
        self.assertEqual(
            sequence,
            "DKDFFEAAEDDLVCLVVLLPPPACPQRQAYEDALVVQVPDDPVSVVSVVNSLVHHAYAYEYEAQQL"
            "LDDPQGDVVSLVSVLVCCVVSVPQEYEYENDPPDADALDVVSLVSSLVSQLVSCVSSVGAYAYEDA"
            "ADQDHDPRHPDDVLVSRQSNCVSNVHAHAYELVRLVRCCVRPVPDDSLVSLVRHPLQRHQHYEYQV"
            "VSVVSVLVNLVDHQAHHYYYHDYPDDVVVNSVVRVVSRVSNVVSCVVVVHYIDMD",
        )

    # def test_encode_3bww_masked(self):
    #     structure = self.get_structure("3bww.masked")
    #     states = self.encoder.encode_chain(structure[0]["A"])
    #     sequence = self.encoder.build_sequence(states)
    #     self.assertEqual(
    #         sequence,
    #         "DKDFFEAAEDDLVCLVVLLPPPACPQRQAYEDALVVQVPDDPVSVVSVVNSLVHHAYAYEYEAQQL"
    #         "DDDPQGDVVSLVSVLVCCVVSVPQEYEYENDPPDADALDPVDDDSSLVSQLVSCVSSVGAYAYEDA"
    #         "ADQDHDPRHPDDVLVSRQVSCVSNVHAHAYELVRLVRCCVRPVPDDSLVSLVRHPLQRHQHYEYQV"
    #         "VSVVSVLVNLVDHQAHHYYYHDYPDDVVVNSVVRVVSRVSNVVSCVVVVHYIDMD",
    #     )

    def test_encode_8crb(self):
        structure = self.get_structure("8crb")

        states = self.encoder.encode_chain(structure[0]["A"])
        sequence = self.encoder.build_sequence(states)
        self.assertEqual(
            sequence,
            "DWAKDKDWADEDAAQAKTKIKMATPPDLLQDFFKFKWFDAPPDDIDGQAPGACPSPPLADDVHHHH"
            "GKGWHDDSVRRMIMIMGGNDDQVVFGKMKMFTADDADPQVVVPDGDDTDDMHDIDTYGHPPDDFFA"
            "WDKDKDQDDPVPCPVQKPKIKMKTDDGDDDDKDKAWLVNPGDPQKDDFDWDADPVRGIIDMIIGMD"
            "GNVCFQVGFTKIWMAGVVVRDIDIDGGHD",
        )

        states = self.encoder.encode_chain(structure[0]["B"])
        sequence = self.encoder.build_sequence(states)
        self.assertEqual(
            sequence,
            "DAAKDFDQQEEEAAQAKDKGWIFAADVPPVPDAFWKWWDAPPDDIDTAADPNQAGDPVDHSQKGWD"
            "ADHGITIIMGGRDDNSRQGFIWRAQPDDPDHNGHTDDTHGYYHCPDDQDDKDKDWDDAAVVVLVVL"
            "FGKTKIKIDDGDDPPKDKFKDLQNHTDDAQWDWDDWDLDPVRTIMTMIIRRDGVVSCVVSQKMKMW"
            "IDDDVHTDIDMDGNVVHD",
        )

        states = self.encoder.encode_chain(structure[0]["C"])
        sequence = self.encoder.build_sequence(states)
        self.assertEqual(
            sequence,
            "DPCVLVVLVLQLVLVVLLLVVVVVVLVVCVVVLFKDWQDPVHDWQLACVSPDHDCPDCCSVPGSNN"
            "VQQCPKPLDDVTATNQSVQQIDDGDLDHDDDDDTIQGCPPPVRCSVVVVVVSVVSVVVSVVSCVVS"
            "VVVVVVD",
        )
