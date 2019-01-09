import os

import matplotlib.pyplot as plt
import numpy
from mbtr import read_xyz_molecule, MolsMBTR2DIQuadW, read_xyz_crystal, PeriodicMBTR3D


def visualize_mbtr(array, lines, x_range, title='Visualize'):
    fig = plt.figure()

    for label, index in lines:
        plt.plot(x_range, array[index], label=label)

    plt.xlabel('Geometry function')
    plt.ylabel('Density')
    fig.legend()
    plt.title(title)
    fig.show()


def visualize_aspirin():
    aspirin_fn = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'mbtr',
        'test',
        'aspirin.xyz'
    )
    molecules = read_xyz_molecule(aspirin_fn)

    mbtr = MolsMBTR2DIQuadW(grid_size=500, smearing_factor=0.023)
    mbtr.fit(molecules)
    tensor_range = numpy.linspace(mbtr.tensor_range[0], mbtr.tensor_range[1], 500)

    arrays, ans = mbtr.transform(molecules)
    for array in arrays:
        visualize_mbtr(
            array.reshape((-1, 500)),
            (
                ('HH', 0),
                ('HC, CH', 1),
                ('HO, OH', 2),
                ('CC', 4),
                ('CO, OC', 5),
                ('OO', 8),
            ),
            tensor_range,
            title='Aspirin'
        )


def visualize_nacl():
    nacl_fn = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'mbtr',
        'test',
        'nacl.xyz'
    )
    nacl = read_xyz_crystal(nacl_fn)

    mbtr = PeriodicMBTR3D(grid_size=500, smearing_factor=0.05, d=2.0)
    mbtr.fit(nacl)
    tensor_range = numpy.linspace(mbtr.tensor_range[0], mbtr.tensor_range[1], 500)

    arrays, ans = mbtr.transform(nacl)
    for array in arrays:
        visualize_mbtr(
            array.reshape((-1, 500)), (
                ('NaNaNa, ClClCl', 0),
                ('NaClCl, ClClNa, ClNaNa, NaNaCl', 1),
                ('NaClNa, ClNaCl', 2),
            ),
            tensor_range,
            title='NaCl'
        )


visualize_aspirin()
visualize_nacl()
plt.show()
