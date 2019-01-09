import io
import logging
import os
import random
import zipfile

import numpy
import requests
from mbtr import read_xyz_molecule, MolsMBTR2DIQuadW, MolsMBTR3D

from KernelRidgeRegression import GaussianKernel, AutoTrainTwoParam

random.seed(42)


def load_data():
    gdb7_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'dsgdb7njp.xyz'
    )

    if not os.path.exists(gdb7_filename):
        gdb7_url = 'https://qmml.org/Datasets/gdb7-13.zip'
        zip_data = requests.get(gdb7_url).content
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as gdb7_zip:
            gdb7_zip.extract('dsgdb7njp.xyz', path=os.path.dirname(gdb7_filename))

    molecules = read_xyz_molecule(gdb7_filename)
    for molecule in molecules:
        molecule['atomization_energy'] = float(molecule['properties'][0])

    logging.info('Loaded %d molecules', len(molecules))
    return molecules


def compute_mbtr(data):
    logging.info('Computing 2D MBTR')
    mbtr2 = MolsMBTR2DIQuadW(grid_size=100, smearing_factor=0.014,
                             tensor_range=(-0.010, 0.612))
    arrays2, ans = mbtr2.transform(data)
    #
    # logging.info('Computing 3D MBTR')
    # mbtr3 = MolsMBTR3D(grid_size=100, smearing_factor=0.05,
    #                    tensor_range=(-1.220, 1.220))
    # arrays3, ans = mbtr3.transform(data)

    # for mol, array2, array3 in zip(data, arrays2, arrays3):
    #     mol['rep'] = numpy.concatenate((array2.flatten(), array3.flatten()))

    for mol, array2 in zip(data, arrays2):
        mol['rep'] = array2.flatten()


def random_split(data):
    random.shuffle(data)

    train_size = int(len(data) * 0.6)
    dev_size = int(len(data) * 0.15)
    valid_size = len(data) - train_size - dev_size

    train = data[:train_size]
    dev = data[train_size:train_size + dev_size]
    valid = data[train_size + dev_size:]

    logging.info('Train/Dev/Validation size: %d/%d/%d',
                 train_size, dev_size, valid_size)
    return train, dev, valid


def get_data_tuple(data):
    return (
                numpy.array(list(x['rep'] for x in data)),
                numpy.array(list(x['atomization_energy'] for x in data)),
            )


def run_experiment():
    data = load_data()
    compute_mbtr(data)
    train, dev, valid = random_split(data)

    auto_train = AutoTrainTwoParam(
        (get_data_tuple(train), get_data_tuple(dev)),
        GaussianKernel,
        0.5, 1e-3
    )
    auto_train.auto_train()

    rmse, mae, r2 = auto_train.krr.validate(
        get_data_tuple(valid)
    )
    logging.info('Final parameters: sigma: %.5e, %.5e', auto_train.sigma, auto_train.noise_level)
    logging.debug('Model performance on validation set: RMSE: %.6f, MAE: %.6f, R2: %.6f',
                  rmse, mae, r2)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG)
    run_experiment()
