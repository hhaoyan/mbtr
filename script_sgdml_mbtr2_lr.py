#!/usr/bin/env python
import argparse
import gzip
import logging
import os
import pickle

import coloredlogs
import numpy as np
from sklearn.model_selection import train_test_split

from mbtr_grad.adaptive import AdaptiveGridSearch
from mbtr_grad.constants import PRECOMPUTED_PERMS
from mbtr_grad.gdml import (
    perform_krr,
    predict_force_and_energy,
    eval_force,
    setup_einsum_engine,
)
from mbtr_grad.mbtr_python_torch import (
    GeomFunc2BodyInvDist,
    DistFuncGaussian,
    WeightFunc2BodyIdentity,
)
from mbtr_grad.utils import save_safe, load_md_data

logger = logging.getLogger("main")


class MBTRExperiment:
    def __init__(
        self,
        training_data_fn,
        train_size,
        val_size=50,
        mbtr_size=16,
        batch_size=100,
        seed=0,
    ):
        self.seed = seed
        self.mbtr_size = mbtr_size
        self.batch_size = batch_size

        self.training_data = load_md_data(training_data_fn)
        train_data_name = os.path.basename(args.training_data).split(".")[0]

        self.perms = np.array(PRECOMPUTED_PERMS[train_data_name][0])
        self.tril_perms_lin = np.array(PRECOMPUTED_PERMS[train_data_name][1])

        # Split data into train/valid/test
        energy_percentile = self.training_data["E_percentile"]

        data_ind = np.arange(self.training_data["R"].shape[0])
        train_val, self.test_ind = train_test_split(
            data_ind,
            train_size=train_size + val_size,
            stratify=energy_percentile,
            random_state=seed,
        )
        self.train_ind, self.val_ind = train_test_split(
            train_val, train_size=train_size, random_state=seed
        )

        logger.info(
            "Train size %d, val size %d, test size %d.",
            len(self.train_ind),
            len(self.val_ind),
            len(self.test_ind),
        )

    def train(self, sigma=1.0, lambda_=1e-5, mbtr_sig=0.015, include_train_error=True):
        n_atoms = self.training_data["R"].shape[1]

        mbtr_kwargs = {
            "order": 2,
            "grid": np.linspace(0, 1.28, self.mbtr_size),
            "weightf": WeightFunc2BodyIdentity(),
            "geomf": GeomFunc2BodyInvDist(),
            "distf": DistFuncGaussian(sigma=mbtr_sig),
            "flatten": True,
        }

        tril_perms_lin = self.tril_perms_lin.reshape((len(self.perms), -1), order="F")
        mbtr_tril_perms_lin = (
            (
                tril_perms_lin[:, :, None] * self.mbtr_size
                + np.arange(self.mbtr_size)[None, None, :]
            )
            .reshape((len(self.perms), -1))
            .flatten(order="F")
        )

        data_R = self.training_data["R"]
        data_E = self.training_data["E"]
        data_F = self.training_data["F"]

        logger.info(
            "Training KRR model. Data R shape: %r, data E shape: %r, "
            "data F shape: %r",
            data_R.shape,
            data_E.shape,
            data_F.shape,
        )

        krr = perform_krr(
            z=np.arange(n_atoms),
            r=data_R[self.train_ind],
            force=data_F[self.train_ind],
            energy=data_E[self.train_ind],
            n_perm=len(self.perms),
            tril_perms_lin=mbtr_tril_perms_lin,
            use_representation="MBTR",
            rep_kwargs=mbtr_kwargs,
            include_train_error=include_train_error,
            sigma=sigma,
            lambda_=lambda_,
        )
        val_forces, val_energies = predict_force_and_energy(
            z=np.arange(n_atoms),
            r=data_R[self.val_ind],
            sig=krr["sig"],
            std=krr["std"],
            c=krr["c"],
            use_representation="MBTR",
            rep_kwargs=mbtr_kwargs,
            R_d_desc_alpha=krr["R_d_desc_alpha"],
            R_desc=krr["R_desc"],
            tril_perms_lin=krr["tril_perms_lin"],
        )

        force_rmse, force_mae, energy_rmse, energy_mae = eval_force(
            val_forces,
            data_F[self.val_ind],
            val_energies,
            data_E[self.val_ind],
            name="Eval",
        )
        krr.update(
            {
                "seed": self.seed,
                "splits": {
                    "train": self.train_ind,
                    "val": self.val_ind,
                },
                "val_force_error": {
                    "rmse": force_rmse,
                    "mae": force_mae,
                },
                "val_energy_error": {
                    "rmse": energy_rmse,
                    "mae": energy_mae,
                },
            }
        )
        return krr

    def search_hyperparameter(
        self, initial_sig=1.75, initial_lam=-20, initial_mbtr_sig=-4.5
    ):
        opt = AdaptiveGridSearch(
            lambda sig, lam, mbtr_sig: self.train(
                sigma=sig, lambda_=lam, mbtr_sig=mbtr_sig, include_train_error=False
            )["val_force_error"]["mae"].mean(),
            [
                # val, pri, stp, min, max, dir, b
                (initial_sig, 2, 0.25, -2, 10, 0, 2.0),
                (initial_lam, 3, 1, -50, 1, +1, 2.0),
                (initial_mbtr_sig, 2, 0.25, -10, -1, 0, 2.0),
            ],
        )

        while not opt.done:
            opt.step()
            logger.info("Adaptive searching: %s", opt)

        return {
            "best_v": opt.best_v,
            "best_f": opt.best_f,
            "model": self.train(
                sigma=2 ** opt.best_v[0],
                lambda_=2 ** opt.best_v[1],
                mbtr_sig=2 ** opt.best_v[2],
            ),
        }

    def compute_test_error(self, krr_result, mbtr_sig):
        n_atoms = self.training_data["R"].shape[1]

        mbtr_kwargs = {
            "order": 2,
            "grid": np.linspace(0, 1.28, self.mbtr_size),
            "weightf": WeightFunc2BodyIdentity(),
            "geomf": GeomFunc2BodyInvDist(),
            "distf": DistFuncGaussian(sigma=mbtr_sig),
            "flatten": True,
        }

        val_forces, val_energies = predict_force_and_energy(
            z=np.arange(n_atoms),
            r=self.training_data["R"][self.test_ind],
            sig=krr_result["sig"],
            std=krr_result["std"],
            c=krr_result["c"],
            use_representation="MBTR",
            rep_kwargs=mbtr_kwargs,
            R_d_desc_alpha=krr_result["R_d_desc_alpha"],
            R_desc=krr_result["R_desc"],
            tril_perms_lin=krr_result["tril_perms_lin"],
            is_test=True,
            batch_size=self.batch_size,
        )

        force_rmse, force_mae, energy_rmse, energy_mae = eval_force(
            val_forces,
            self.training_data["F"][self.test_ind],
            val_energies,
            self.training_data["E"][self.test_ind],
            name="Test",
        )
        krr_result.update(
            {
                "test_force_error": {
                    "rmse": force_rmse,
                    "mae": force_mae,
                },
                "test_energy_error": {
                    "rmse": energy_rmse,
                    "mae": energy_mae,
                },
            }
        )
        return krr_result

    def run(self, run_dir, best_v, override=False):
        exp_fn = "MBTR_%d_%d_seed_%d.pypickle.gz" % (
            self.mbtr_size,
            len(self.train_ind),
            self.seed,
        )
        experiment_fn = os.path.join(run_dir, exp_fn)

        if not os.path.exists(experiment_fn) or override:
            model = self.search_hyperparameter(*best_v)
            del model["model"]["kernel"]
            save_safe(experiment_fn, model, use_gzip=True)
        else:
            with gzip.open(experiment_fn, "rb") as f:
                model = pickle.load(f)
                logger.info(
                    "Already ran this experiment and will reuse "
                    "old result: %r -> %.4f.",
                    model["best_v"],
                    model["best_f"],
                )

        if "test_force_error" not in model["model"] or override:
            mbtr_sig = 2 ** model["best_v"][2]
            self.compute_test_error(model["model"], mbtr_sig=mbtr_sig)

            save_safe(experiment_fn, model, use_gzip=True)

        return model


def main(data_fn, exp_dir, mbtr_size, batch_size, seed):
    train_name = os.path.basename(data_fn).split(".")[0]
    output_dir = os.path.join(exp_dir, train_name)
    os.makedirs(output_dir, exist_ok=True)

    best_v = (1.75, -10, -4.5)
    for train_size in [10, 20, 40, 50, 100, 150, 200, 300, 400, 500, 1000]:
        # Reset CUDA status
        setup_einsum_engine(engine="torch")
        experiment = MBTRExperiment(
            data_fn,
            train_size=train_size,
            val_size=50,
            mbtr_size=mbtr_size,
            seed=seed,
            batch_size=batch_size,
        )
        model = experiment.run(output_dir, best_v=best_v, override=False)
        best_v = (model["best_v"][0], -20, model["best_v"][2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", type=str, help="Training data filename")
    parser.add_argument("--mbtr_size", type=int, default=16, help="Size of MBTR")
    parser.add_argument(
        "--output",
        type=str,
        default="notebooks/results/sgdml_exps/mbtr",
        help="Output dir",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="More debugging"
    )
    parser.add_argument("--repeat", type=int, default=5, help="Repeat N experiments")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for testing"
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU(PyTorch) for calculation"
    )

    args = parser.parse_args()

    if args.verbose:
        coloredlogs.install(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.DEBUG
        )
    else:
        coloredlogs.install(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO
        )

    if args.use_gpu:
        setup_einsum_engine(engine="torch")
    else:
        setup_einsum_engine(engine="cpu")

    for i in range(args.repeat):
        main(args.training_data, args.output, args.mbtr_size, args.batch_size, i)
