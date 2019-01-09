import os
import unittest

import collections

from mbtr import read_xyz_molecule, read_xyz_crystal


class TestMolecularXYZRead(unittest.TestCase):
    def test_read_one(self):
        aspirin_xyz_fn = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'aspirin.xyz'
        )

        molecules = read_xyz_molecule(aspirin_xyz_fn)
        self.assertEqual(len(molecules), 1)

        aspirin = molecules[0]
        self.assertFalse(aspirin['is_periodic'])
        self.assertListEqual(aspirin['properties'], ['0'])
        self.assertEqual(len(aspirin['atoms']), 21)
        atom_counter = collections.Counter(x[0] for x in aspirin['atoms'])
        self.assertDictEqual(atom_counter, {6: 9, 1: 8, 8: 4})

    def test_read_many(self):
        gdb13_xyz_fn = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'molecule_gdb13.xyz'
        )

        molecules = read_xyz_molecule(gdb13_xyz_fn)
        self.assertEqual(len(molecules), 7211)


class TestCrystalXYZRead(unittest.TestCase):
    def test_read_one(self):
        nacl_xyz_fn = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'nacl.xyz'
        )

        crystals = read_xyz_crystal(nacl_xyz_fn)
        self.assertEqual(len(crystals), 1)

        nacl = crystals[0]
        self.assertTrue(nacl['is_periodic'])
        self.assertListEqual(nacl['properties'], ['0.0'])
        self.assertEqual(len(nacl['atoms']), 8)
        atom_counter = collections.Counter(x[0] for x in nacl['atoms'])
        self.assertDictEqual(atom_counter, {11: 4, 17: 4})
        self.assertListEqual(
            list(nacl['basis_vector']),
            [2.820100, 2.820100, 0.000000,
             0.000000, 2.820100, 2.820100,
             2.820100, 0.000000, 2.820100]
        )
        self.assertListEqual(
            nacl['atoms'],
            [
                (11, 0.000000, 0.000000, 0.000000),
                (17, 2.820100, 2.820100, 2.820100),
            ])

    def test_read_many(self):
        oqmd_mgo_fn = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'periodic_oqmd_mgo.xyz'
        )

        crystals = read_xyz_crystal(oqmd_mgo_fn)
        self.assertEqual(len(crystals), 292)
