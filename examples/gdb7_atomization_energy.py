import io
import logging
import os
import random
import zipfile

import numpy
import requests
import sklearn.model_selection
from mbtr import read_xyz_molecule, MolsMBTR2DQuadW, MolsMBTR3DAngle

from KernelRidgeRegression import GaussianKernel, AutoTrainTwoParam, KernelRidgeRegression

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_data():
    """
    Load GDB7 data. If the file does not exist in the folder (same as
    this script), download from qmml.org.

    :return: List of molecules.
    """
    gdb7_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'dsgdb7njp.xyz')

    # File does not exist, download it.
    if not os.path.exists(gdb7_filename):
        gdb7_url = 'https://qmml.org/Datasets/gdb7-13.zip'
        zip_data = requests.get(gdb7_url).content
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as gdb7_zip:
            gdb7_zip.extract('dsgdb7njp.xyz', path=os.path.dirname(gdb7_filename))

    # Load the file and properties
    molecules = read_xyz_molecule(gdb7_filename, transform_unit=1.0)
    for molecule in molecules:
        molecule['atomization_energy'] = float(molecule['properties'][0])

    logging.info('Loaded %d molecules', len(molecules))
    return molecules


def split_dataset(molecules):
    """
    Split the dataset into two: training and validation.
    To find the best hyper-parameters, we first split the dataset into 5000 set
    and 2211 (test set), then, we further split the 5000 dataset into training/
    development set. We optimize the model hyper-parameters using the training/
    development set. After the best parameters are obtained, we train the model
    on training + development set, and evaluate it on the test set.

    Small molecules (less than 5 hydrogen atoms) will be included in training
    set to ensure the model knows how to predict small molecules.
    """
    indices_small_molecule = []
    indices_large_molecule = []
    for i, mol in enumerate(molecules):
        n_non_h_atom = len([atom for atom in mol['atoms'] if atom[0] != 1])
        if n_non_h_atom < 5:
            indices_small_molecule.append(i)
        else:
            indices_large_molecule.append(i)

    assert len(set(indices_large_molecule) & set(indices_small_molecule)) == 0

    (indices_production_train, indices_production_valid) = sklearn.model_selection.train_test_split(
        indices_large_molecule,
        train_size=5000 - len(indices_small_molecule),
        test_size=len(molecules) - 5000,
        random_state=RANDOM_SEED
    )

    (indices_model_selection_train, indices_model_selection_valid) = sklearn.model_selection.train_test_split(
        indices_production_train,
        train_size=5000 - len(indices_small_molecule) - 300,
        test_size=300,
        random_state=RANDOM_SEED
    )

    indices_model_selection_train.extend(indices_small_molecule)
    indices_production_train.extend(indices_small_molecule)

    logging.info('Model selection training/validation size: %d/%d, production training/validation size: %d/%d',
                 len(indices_model_selection_train), len(indices_model_selection_valid),
                 len(indices_production_train), len(indices_production_valid))

    return (indices_model_selection_train, indices_model_selection_valid), \
           (indices_production_train, indices_production_valid)


def compute_mbtr(molecules):
    """
    In this example, we concatenate 2D and 3D MBTR together as the feature vector.

    :param molecules: List of molecules
    :return: List of molecules with 'rep' as their representation.
    """
    sigma_2d = 0.06
    range_2d = -0.16, 1.30
    logging.info('Computing 2D MBTR, geometry function: 1/pairwise_distance, weighting function: pairwise_distance**2, '
                 'density function: N(0, %.4f), correlation function: delta(z1, z2), range: [%.3f, %.3f].',
                 sigma_2d, range_2d[0], range_2d[1])

    mbtr2 = MolsMBTR2DQuadW(grid_size=100, smearing_factor=sigma_2d,
                            tensor_range=range_2d)
    arrays_2d, ans = mbtr2.transform(molecules)

    sigma_3d = 0.18
    range_3d = -0.54, 3.6816
    logging.info('Computing 3D MBTR, geometry function: angle(A, B, C), weighting function: 1/r1r2r3, '
                 'density function: N(0, %.4f), correlation function: delta(z1, z2, z3), range: [%.3f, %.3f].',
                 sigma_3d, range_3d[0], range_3d[1])
    mbtr3 = MolsMBTR3DAngle(grid_size=100, smearing_factor=sigma_3d,
                            tensor_range=range_3d)
    arrays_3d, ans = mbtr3.transform(molecules)

    for mol, array_2d, array_3d in zip(molecules, arrays_2d, arrays_3d):
        # select only unique values
        array_2d = array_2d[numpy.triu_indices(array_2d.shape[0])].flatten()

        # for 3D MBTR, ABC is equal to CBA (since we have angle geometry)
        # here, the first and the third indices can be seen as 2D matrix.
        i0, i2 = numpy.triu_indices(array_3d.shape[0])
        i0, i1, i2 = (
            numpy.concatenate([i0] * array_3d.shape[0]),
            numpy.asarray(sum([[x] * len(i0) for x in range(array_3d.shape[0])], [])),
            numpy.concatenate([i2] * array_3d.shape[0])
        )
        array_3d = array_3d[(i0, i1, i2)].flatten()

        mol['rep'] = numpy.concatenate((array_2d, array_3d))


def get_data_tuple(molecules, indices):
    """
    Take list of molecules and indices, return (x, y) for training machine
    learning models.

    :param molecules: List of molecules.
    :param indices: Indices of molecules that should be in (x, y).
    :return:
    """
    data = [molecules[x] for x in indices]
    return (
        numpy.array(list(x['rep'] for x in data)),
        numpy.array(list(x['atomization_energy'] for x in data)),
    )


def run_experiment():
    molecules = load_data()
    compute_mbtr(molecules)
    # Calculating 3D MBTR is very time-consuming. Uncomment the following to cache
    # MBTR calculations.

    # import pickle
    # with open('mbtr.data', 'wb') as f:
    #     pickle.dump(molecules, f)
    # with open('mbtr.data', 'rb') as f:
    #     molecules = pickle.load(f)

    (indices_model_selection_train, indices_model_selection_valid), \
        (indices_production_train, indices_production_valid) = split_dataset(molecules)

    # Automatically try to find best kernel hyper-parameters. The following code
    # is not guaranteed to produce the best result! Rely on it by your discretion.
    # However, during my experiments, this class AutoTrainTwoParam works pretty well.
    auto_train = AutoTrainTwoParam(
        (
            get_data_tuple(molecules, indices_model_selection_train),
            get_data_tuple(molecules, indices_model_selection_valid)
        ),
        GaussianKernel,
        5.0e3, 1e-3
    )

    # Fire the optimization.
    auto_train.auto_train()
    logging.info('Final parameters: sigma: %.5e, %.5e', auto_train.sigma, auto_train.noise_level)

    # Now test against the unseen data.
    krr = KernelRidgeRegression(
        get_data_tuple(molecules, indices_production_train), GaussianKernel(auto_train.sigma),
        auto_train.sigma, auto_train.noise_level)
    rmse, mae, r2 = krr.validate(
        get_data_tuple(molecules, indices_production_valid)
    )
    logging.debug('Model performance on test set: RMSE: %.6f, MAE: %.6f, R2: %.6f',
                  rmse, mae, r2)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG)
    run_experiment()
