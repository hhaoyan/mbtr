import logging
import os
import random

import numpy

from mbtr import read_xyz_molecule, MolsMBTR2DIQuadW
from KernelRidgeRegression import GaussianKernel, AutoTrainTwoParam

random.seed(42)


def load_data():
    gdb7_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        'mbtr',
        'test',
        'molecule_gdb13.xyz'
    )

    molecules = read_xyz_molecule(gdb7_filename)
    for molecule in molecules:
        molecule['atomization_energy'] = float(molecule['properties'][0])

    logging.info('Loaded %d molecules', len(molecules))
    return molecules


def compute_mbtr(data):
    mbtr = MolsMBTR2DIQuadW(grid_size=20, smearing_factor=0.023)
    mbtr.fit(data)
    arrays, ans = mbtr.transform(data)
    for mol, array in zip(data, arrays):
        mol['rep'] = array.flatten()


def random_split(data):
    random.shuffle(data)

    train_size = int(len(data) * 0.6)
    dev_size = int(len(data) * 0.15)
    valid_size = len(data) - train_size - dev_size

    train = data[train_size:]
    dev = data[train_size:train_size + dev_size]
    valid = data[train_size + dev_size:]

    logging.info('Train/Dev/Validation size: %d/%d/%d',
                 train_size, dev_size, valid_size)
    return train, dev, valid


def run_experiment():
    data = load_data()
    compute_mbtr(data)
    train, dev, valid = random_split(data)

    auto_train = AutoTrainTwoParam(
        (
            (
                numpy.array(list(x['rep'] for x in train)),
                numpy.array(list(x['atomization_energy'] for x in train)),
            ),
            (
                numpy.array(list(x['rep'] for x in dev)),
                numpy.array(list(x['atomization_energy'] for x in dev)),
            )
        ),
        GaussianKernel,
        4.52616e-01, 2.27374e-16
    )
    auto_train.auto_train()

    rmse, mae, r2 = auto_train.krr.validate(
        (
            numpy.array(list(x['rep'] for x in valid)),
            numpy.array(list(x['atomization_energy'] for x in valid)),
        )
    )
    logging.info('Final parameters: sigma: %.5e, %.5e', auto_train.sigma, auto_train.noise_level)
    logging.debug('Model performance on validation set: RMSE: %.6f, MAE: %.6f, R2: %.6f',
                  rmse, mae, r2)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(name)s(PID-%(process)d) - %(levelname)s - %(message)s',
        level=logging.DEBUG)
    run_experiment()
