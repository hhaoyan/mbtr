import itertools
import os

import matplotlib.pyplot as plt
import numpy

from mbtr import read_xyz_molecule, MolsMBTR2DIQuadW, read_xyz_crystal, PeriodicMBTR3D

ELEMENTS = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
    'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
    'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


def visualize_mbtr(array, an, x_range, rank):
    fig = plt.figure()

    for atom_tuple in itertools.combinations_with_replacement(an, rank):
    # for atom_tuple in itertools.combinations(an, rank):
        index = list(an.index(x) for x in atom_tuple)
        index.append(slice(0, array.shape[-1]))
        values = array[index]

        atoms = tuple(ELEMENTS[i - 1] for i in atom_tuple)
        atoms_name = ''.join(atoms)
        plt.plot(x_range, values, label=atoms_name)

    plt.xlabel('Geometry function')
    plt.ylabel('Density')
    fig.legend()
    fig.show()


def visualize_aspirin():
    aspirin_fn = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'test',
        'aspirin.xyz'
    )
    molecules = read_xyz_molecule(aspirin_fn)

    mbtr = MolsMBTR2DIQuadW(grid_size=20, smearing_factor=0.023)
    mbtr.fit(molecules)
    tensor_range = numpy.linspace(mbtr.tensor_range[0], mbtr.tensor_range[1], 200)

    arrays, ans = mbtr.transform(molecules)
    for array in arrays:
        visualize_mbtr(array, ans, tensor_range, 2)


def visualize_nacl():
    nacl_fn = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'test',
        'nacl.xyz'
    )
    nacl = read_xyz_crystal(nacl_fn)

    mbtr = PeriodicMBTR3D(grid_size=100, smearing_factor=0.05, d=2.5)
    mbtr.fit(nacl)
    tensor_range = numpy.linspace(mbtr.tensor_range[0], mbtr.tensor_range[1], 100)

    arrays, ans = mbtr.transform(nacl)
    for array in arrays:
        visualize_mbtr(array, ans, tensor_range, 3)


# visualize_aspirin()
visualize_nacl()
plt.show()
