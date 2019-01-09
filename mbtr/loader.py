import re

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"

PT = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
    'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
    'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

ANGSTROM_TO_ATOMIC = 100.0 / 52.917720859


def _read_molecule(stream, transform_unit):
    """
    Reads molecule structures in XYZ file stream.

    Note: this function assumes XYZ file coordinates in Angstrom unit.

    :param stream: Stream of XYZ file.
    :param transform_unit: Transform values into different unit.
    :return: A molecular structure.
    """
    line = stream.readline()
    while not line.strip() and len(line):
        line = stream.readline()
    n_atoms = int(line)

    properties = re.split(r'\s+', stream.readline().strip())
    atoms = []
    for i in range(n_atoms):
        atom_info = stream.readline().strip()
        values = re.split(r'\s+', atom_info)
        if len(values) < 4:
            raise ValueError('Each atom line must have more '
                             'than four fields (an, px, py, pz)')
        atoms.append((PT[values[0]],
                      float(values[1]) * transform_unit,
                      float(values[2]) * transform_unit,
                      float(values[3]) * transform_unit))

    return {
        'is_periodic': False,
        'atoms': atoms,
        'properties': properties
    }


def _read_crystal(stream, transform_unit):
    """
    Reads crystal structures in modified XYZ file stream.
    The modified XYZ file format: there will be one more line
    consisting of basis vectors before atom definition lines.

    Note: this function assumes XYZ file coordinates in Angstrom unit.

    :param stream: Stream of XYZ file.
    :param transform_unit: Transform values into different unit.
    :return: A crystal structure.
    """
    line = stream.readline()
    while not line.strip() and len(line):
        line = stream.readline()
    n_atoms = int(line)

    properties = re.split(r'\s+', stream.readline().strip())
    basis_vectors = tuple(
        float(x) * transform_unit
        for x in re.split(r'\s+', stream.readline().strip())
    )
    atoms = []
    for i in range(n_atoms):
        atom_info = stream.readline().strip()
        values = re.split(r'\s+', atom_info)
        if len(values) < 4:
            raise ValueError('Each atom line must have more '
                             'than four fields (an, px, py, pz)')
        atoms.append((PT[values[0]],
                      float(values[1]) * transform_unit,
                      float(values[2]) * transform_unit,
                      float(values[3]) * transform_unit))

    return {
        'is_periodic': True,
        'atoms': atoms,
        'properties': properties,
        'basis_vector': basis_vectors,
    }


def read_xyz_molecule(filename, transform_unit=ANGSTROM_TO_ATOMIC):
    """
    Read molecule definitions in XYZ file into an array.
    Note: this function assumes XYZ file coordinates in Angstrom unit.

    :param filename: Filename of the XYZ file.
    :type filename: str
    :param transform_unit: Transform values into different unit.
    :type transform_unit: float
    :return: List of molecules
    :rtype: list(dict)
    """
    molecules = []
    with open(filename) as f:
        while True:
            try:
                molecules.append(_read_molecule(f, transform_unit))
            except EOFError:
                break
            except ValueError:
                break

    return molecules


def read_xyz_crystal(filename, transform_unit=1.0):
    """
    Read crystal definitions in XYZ file into an array.
    Note: this function assumes XYZ file coordinates in Angstrom unit.

    :param filename: Filename of the XYZ file.
    :type filename: str
    :param transform_unit: Transform values into different unit.
    :type transform_unit: float
    :return: List of crystals
    :rtype: list(dict)
    """
    crystals = []
    with open(filename) as f:
        while True:
            try:
                crystals.append(_read_crystal(f, transform_unit))
            except EOFError:
                break
            except ValueError:
                break

    return crystals
