import os
import unittest

import numpy

from mbtr import MolsMBTR2D, read_xyz_crystal, PeriodicMBTR2D
from mbtr import read_xyz_molecule


class TestMolecularMBTR(unittest.TestCase):
    def load_xyz(self, filename, n_molecules):
        xyz_fn = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            filename,
        )
        molecules = read_xyz_molecule(xyz_fn)
        self.assertEqual(len(molecules), n_molecules)
        return molecules

    def test_atomic_numbers(self):
        mbtr = MolsMBTR2D(grid_size=10)
        aspirin = self.load_xyz('aspirin.xyz', 1)
        mbtr.fit(aspirin)
        arrays, ans = mbtr.transform(aspirin)

        self.assertEqual(ans, [1, 6, 8])

    def test_tensor_range(self):
        aspirin = self.load_xyz('aspirin.xyz', 1)

        atom_coordinates = numpy.array(list(x[1:] for x in aspirin[0]['atoms']))
        distance_matrix = numpy.linalg.norm(
            atom_coordinates.reshape((21, 1, 3)) - atom_coordinates.reshape((1, 21, 3)),
            axis=2
        )
        distances = distance_matrix[numpy.triu_indices(21, 1)]

        mbtr = MolsMBTR2D(grid_size=10, smearing_factor=0.001)
        mbtr.fit(aspirin)

        self.assertAlmostEqual(1 / distances.max(), mbtr.tensor_range[0], delta=0.01)
        self.assertAlmostEqual(1 / distances.min(), mbtr.tensor_range[1], delta=0.01)

    def test_mbtr_integral(self):
        aspirin = self.load_xyz('aspirin.xyz', 1)
        mbtr = MolsMBTR2D(grid_size=10, smearing_factor=0.001)
        mbtr.fit(aspirin)
        arrays, ans = mbtr.transform(aspirin)

        aspirin_array = arrays[0]
        aspirin_array_sum = numpy.sum(aspirin_array, axis=2)

        # We have 8 H, 9 C, 4 O.
        # Diagonals: N(N-1)
        # Off-diagonals: NxM
        expected = numpy.array([
            [56, 72, 32],
            [72, 72, 36],
            [32, 36, 12]
        ])
        self.assertTrue(numpy.all(
            abs(aspirin_array_sum - expected) < 0.01
        ), msg='Expected {0}, but got {1}'.format(expected, aspirin_array_sum))


class TestCrystalMBTR(unittest.TestCase):
    def load_xyz(self, filename, n_crystals):
        xyz_fn = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            filename,
        )
        crystals = read_xyz_crystal(xyz_fn)
        self.assertEqual(len(crystals), n_crystals)
        return crystals

    def test_atomic_numbers(self):
        mbtr = PeriodicMBTR2D(grid_size=10)
        nacl = self.load_xyz('nacl.xyz', 1)
        mbtr.fit(nacl)
        arrays, ans = mbtr.transform(nacl)

        self.assertEqual(ans, [11, 17])

    def test_mbtr_shape(self):
        nacl = self.load_xyz('nacl.xyz', 1)
        mbtr = PeriodicMBTR2D(grid_size=10, smearing_factor=0.1)
        mbtr.fit(nacl)
        arrays, ans = mbtr.transform(nacl)

        nacl_array = arrays[0]
        self.assertTupleEqual(nacl_array.shape, (2, 2, 10))
