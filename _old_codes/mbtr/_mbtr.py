import functools
import itertools
import operator

import numpy

from mbtr import mbtr_imp

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"

__all__ = [
    'MolsMBTR1D', 'MolsMBTR2D', 'MolsMBTR3D',
    'MolsMBTR2DQuadW', 'MolsMBTR2DIQuadW', 'MolsMBTR3DAngle',
    'PeriodicMBTR1D', 'PeriodicMBTR2D', 'PeriodicMBTR3D', 'PeriodicMBTR3DAngle',
]


class BaseMBTR(object):
    """
    Base class for all MBTR implementation. This class uses
    parameters to call an C++ python extension, and wraps the
    MBTR results.

    The internal of this implementation uses four functions, for
    formal definition, please see https://arxiv.org/pdf/1704.06439.pdf:

    Geometry function:  Determines how the relations among atoms
        are encoded in MBTR. Corresponds to g_k(i) in Eq. 3.
    Weighting function: Determines how important a tuple of atoms
        are encoded in MBTR. Corresponds to w_k(i) in Eq. 3.
    Density function:   Determines how the tuple of atoms is put, or
        broadened into MBTR. Corresponds to D(x, g) in Eq. 3.
    Correlation function: Determines how different atoms affect
        each other. Corresponds to C_zi in Eq. 3.
    """
    def __init__(self, grid_size, smearing_factor=0.01, d=2.0,
                 tensor_range=None):
        self.tensor_range = tensor_range
        self.grid_size = grid_size
        self.smearing_factor = float(smearing_factor)
        self.d = float(d)

    @property
    def is_periodic(self):
        """
        Returns if this MBTR handles periodic structures.

        :return: Boolean is periodic.
        :rtype: bool
        """
        raise NotImplementedError()

    @property
    def preset_id(self):
        """
        Returns the preset ID.

        :return: preset ID
        :rtype: int
        """
        raise NotImplementedError()

    @property
    def rank(self):
        """
        Returns MBTR rank.

        :return: MBTR rank
        :rtype: int
        """
        raise NotImplementedError()

    def _get_config(self):
        config = {
            'is_periodic': self.is_periodic,
            'preset': self.preset_id,
            'rank': self.rank,
            'sigma': self.smearing_factor,
            'D': self.d,
            'grid_size': self.grid_size,
        }

        if self.tensor_range is None:
            config.update({
                'is_range_set': False,
            })
        else:
            range_min, range_max = self.tensor_range
            config.update({
                'is_range_set': True,
                'tensor_range_min': range_min,
                'tensor_range_max': range_max,
            })

        return config

    @property
    def configured(self):
        return self.tensor_range is not None

    def fit(self, dataset):
        """
        Determine tensor_range (last dimension). If you specify
        tensor_range when creating this MBTR, this should not be
        called again.

        :param dataset: list of molecules.
        """
        if self.configured:
            raise ValueError('MBTR is alrady configured.')

        tensor_range = mbtr_imp.fit(dataset, self._get_config())
        self.tensor_range = tensor_range

    def _fill_not_non_atoms(self, atoms, all_atoms, values):
        """
        C++ Extension returns dense arrays without shape information.
        flatten(V[r0, r1, ..., rk, grid]), where k is rank, and
        0 <= ri <= elements.

        This function converts them into arrays of same shape:
        V[r0', r1', ..., rk, grid], where 0 <= ri' <= all_elements.

        :param atoms: Sorted array of unique elements in this array.
        :param all_atoms: Sorted array of unique elements for final array.
        :param values: 1D array containing all MBTR values.
        :return: New array of unified shape.
        :rtype: numpy.ndarray
        """
        shape = [len(all_atoms)] * self.rank + [self.grid_size]

        # Need to flatten indices because numpy only supports slicing in
        # one dimension.
        final_array = numpy.zeros(shape).reshape((-1, self.grid_size))
        strides = tuple(reversed([len(all_atoms) ** x for x in range(self.rank)]))

        # Tuples of indices of not null elements
        not_non_indices = []
        for index in itertools.product(
                [all_atoms.index(x) for x in atoms],
                repeat=self.rank):
            # Need to flatten indices because numpy only supports slicing in
            # one dimension.
            not_non_indices.append(functools.reduce(
                operator.add, (x * y for x, y in zip(strides, index))
            ))

        final_array[not_non_indices] = values.reshape((-1, self.grid_size))

        # back to the original shape.
        return final_array.reshape(shape)

    def transform(self, dataset, sparse=False):
        """
        Transform dataset into MBTR representation.

        :param dataset: List of structures.
        :param sparse:
        :return: Arrays and sorted atomic numbers.
        :rtype: tuple(list(numpy.ndarray), list(int))
        """
        if not self.configured:
            raise ValueError('MBTR is not configured, call fit() first.')

        systems = mbtr_imp.compute(dataset, self._get_config())

        all_atom_numbers = functools.reduce(operator.or_, (set(x[1]) for x in systems))
        all_atom_numbers = sorted(list(all_atom_numbers))

        arrays = []
        for buffer, atom_numbers in systems:
            values = numpy.frombuffer(buffer, dtype=numpy.float64)
            if not sparse:
                array = self._fill_not_non_atoms(
                    atom_numbers, all_atom_numbers, values
                )
            else:
                raise NotImplementedError()

            arrays.append(array)

        return arrays, all_atom_numbers


class MolsMBTR1D(BaseMBTR):
    """
    1D MBTR for molecules.

    Geometry function:      1
    Weighting function:     1
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 1
    preset_id = 101
    is_periodic = False


class MolsMBTR2D(BaseMBTR):
    """
    2D MBTR for molecules.

    Geometry function:      1/pair_distance
    Weighting function:     1
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 2
    preset_id = 102
    is_periodic = False


class MolsMBTR2DIQuadW(BaseMBTR):
    """
    2D MBTR for molecules.

    Geometry function:      1/pair_distance
    Weighting function:     1/pair_distance^2
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 2
    preset_id = 103
    is_periodic = False


class MolsMBTR2DQuadW(BaseMBTR):
    """
    2D MBTR for molecules.

    Geometry function:      1/pair_distance
    Weighting function:     pair_distance^2
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 2
    preset_id = 104
    is_periodic = False


class MolsMBTR3D(BaseMBTR):
    """
    3D MBTR for molecules.

    Geometry function:      cos(angle_ABC)
    Weighting function:     1
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 3
    preset_id = 105
    is_periodic = False


class MolsMBTR3DAngle(BaseMBTR):
    """
    3D MBTR for molecules.

    Geometry function:      angle_ABC (0 <= angle <= pi)
    Weighting function:     1/(rAB rAC rBC)
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 3
    preset_id = 106
    is_periodic = False


class PeriodicMBTR1D(BaseMBTR):
    """
    1D MBTR for crystals.

    Geometry function:      1
    Weighting function:     within unit cell
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 1
    preset_id = 151
    is_periodic = True


class PeriodicMBTR2D(BaseMBTR):
    """
    2D MBTR for crystals.

    Geometry function:      1/pair_distance
    Weighting function:     exp(-pair_distance/D)
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 2
    preset_id = 152
    is_periodic = True


class PeriodicMBTR3D(BaseMBTR):
    """
    3D MBTR for crystals.

    Geometry function:      cosine(angle_ABC)
    Weighting function:     exp(-(rAB + rAC + rBC)/D)
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 3
    preset_id = 154
    is_periodic = True


class PeriodicMBTR3DAngle(BaseMBTR):
    """
    3D MBTR for crystals.

    Geometry function:      angle_ABC (0 <= angle <= pi)
    Weighting function:     1/(rAB rAC rBC)
    Density function:       N(0, sigma)
    Correlation function:   delta(z1, z2)
    """
    rank = 3
    preset_id = 155
    is_periodic = True
