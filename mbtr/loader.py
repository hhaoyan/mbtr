import logging
import re




def molecular_xyz_reader(filename):
    """
    Reads molecule structures in XYZ file into python data types.

    :param filename: Filename of the XYZ file.
    :return: List of molecular structures.
    """
    molecules = []
    with open(filename) as f:
        convert_atom_units = 100.0 / 52.917720859
        while True:
            try:
                molecule = {}
                n_atoms = int(f.readline())
                properties = [float(x) for x in re.split(r'\s+', f.readline().strip())]

                molecule['properties'] = properties
                molecule['atoms'] = []

                for i in range(n_atoms):
                    d = re.split(r'\s+', f.readline().strip())

                    if len(d) < 4:
                        raise RuntimeError('Not a valid atom definition.')

                    coordinates = (
                        float(d[1]) * convert_atom_units,
                        float(d[2]) * convert_atom_units,
                        float(d[3]) * convert_atom_units
                    )
                    molecule['atoms'].append((PT[d[0]], coordinates))

                molecules.append(molecule)

                f.readline()
            except EOFError:
                break
            except ValueError:
                break

    logging.debug('Loaded %d molecules', len(molecules))
    return molecules
