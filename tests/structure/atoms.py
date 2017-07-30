# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.structure as struc
import numpy as np

class AtomTypeTest(unittest.TestCase):
    
    def setUp(self):
        chain_id = ["A","A","B","B","B"]
        res_id = [1,1,1,1,2]
        res_name = ["ALA","ALA","PRO","PRO","MSE"]
        hetero = [False, False, False, False, True]
        atom_name = ["N", "CA", "O", "CA", "SE"]
        element = ["N","C","O","C","SE"]
        atom_list = []
        for i in range(5):
            atom_list.append(struc.Atom([i,i,i],
                                        chain_id = chain_id[i],
                                        res_id = res_id[i],
                                        res_name = res_name[i],
                                        hetero = hetero[i],
                                        atom_name = atom_name[i],
                                        element = element[i]))
        self.atom = atom_list[2]
        self.array = struc.array(atom_list)
        self.stack = struc.stack([self.array, self.array.copy(), self.array.copy()])
    
    def test_access(self):
        chain_id = ["A","A","B","B","B"]
        array = self.array.copy()
        self.assertEqual(array.coord.shape, (5,3))
        self.assertTrue(np.array_equal(array.chain_id, chain_id))
        self.assertTrue(np.array_equal(array.get_annotation("chain_id"), chain_id))
        array.add_annotation("test1", dtype=int)
        self.assertTrue(np.array_equal(array.test1, [0,0,0,0,0]))
        with self.assertRaises(IndexError):
            array.set_annotation("test2", np.array([0,1,2,3]))
    
    def test_array_indexing(self):
        array = self.array.copy()
        filtered_array = array[array.chain_id == "B"]
        self.assertTrue(np.array_equal(filtered_array.res_name, ["PRO","PRO","MSE"]))
        atom = filtered_array[0]
        self.assertEqual(atom, self.atom)
        filtered_array = array[[0,2,4]]
        self.assertTrue(np.array_equal(filtered_array.element, ["N","O","SE"]))
    
    def test_stack_indexing(self):
        stack = self.stack.copy()
        with self.assertRaises(IndexError):
            stack[5]
        filtered_stack = stack[0]
        self.assertEqual(type(filtered_stack), struc.AtomArray)
        filtered_stack = stack[0:2, stack.res_name == "PRO"]
        self.assertTrue(np.array_equal(filtered_stack.atom_name, ["O","CA"]))
        
    
    def test_concatenation(self):
        concat_array = self.array[2:] + self.array[:2]
        self.assertTrue(np.array_equal(concat_array.chain_id, ["B","B","B","A","A"]))
        self.assertEqual(concat_array.coord.shape, (5,3))
        concat_stack = self.stack[:,2:] + self.stack[:,:2]
        self.assertTrue(np.array_equal(concat_array.chain_id, ["B","B","B","A","A"]))
        self.assertEqual(concat_stack.coord.shape, (3,5,3))
    
    def test_comparison(self):
        mod_array = self.array.copy()
        self.assertEqual(mod_array, self.array)
        mod_array.coord += 1
        self.assertNotEqual(mod_array, self.array)
        mod_array = self.array.copy()
        mod_array.res_name[0] = "UNK"
        self.assertNotEqual(mod_array, self.array)