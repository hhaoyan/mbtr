# Note: you need to install sgdml from here:
# https://github.com/stefanch/sGDML
import os

import numpy as np
from sgdml.utils.desc import Desc
from sgdml.utils.perm import find_perms
from sklearn.model_selection import train_test_split


def compute_perm_and_tril_ind(data_r, data_z):
    perms = find_perms(data_r, data_z, max_processes=1)

    desc = Desc(
        train_r.shape[1],
        max_processes=1,
    )

    n_perms = perms.shape[0]
    tril_perms = np.array([desc.perm(p) for p in perms])

    dim_d = desc.dim

    perm_offsets = np.arange(n_perms)[:, None] * dim_d
    tril_perms_lin = (tril_perms + perm_offsets).flatten("F")

    return perms.tolist(), tril_perms_lin.tolist()


perms_and_tril = {}

for fn in [
    "notebooks/datasets/ethanol_dft.npz",
    "notebooks/datasets/aspirin_dft.npz",
    "notebooks/datasets/benzene2017_dft.npz",
    "notebooks/datasets/malonaldehyde_dft.npz",
    "notebooks/datasets/naphthalene_dft.npz",
    "notebooks/datasets/salicylic_dft.npz",
    "notebooks/datasets/toluene_dft.npz",
    "notebooks/datasets/uracil_dft.npz",
]:
    print("Loading", fn)
    data = np.load(fn)
    train_r, _ = train_test_split(data["R"], train_size=300)
    train_z = data["z"]

    name = os.path.basename(fn).split(".")[0]
    perms_and_tril[name] = compute_perm_and_tril_ind(train_r, train_z)

np.save("sgdml_perms.npy", perms_and_tril)
