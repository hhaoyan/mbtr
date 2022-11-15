#!/usr/bin/env python
import argparse
import gzip
import logging
import os.path
import pickle
import re

import coloredlogs
import numpy as np
from sklearn.model_selection import train_test_split

from mbtr_grad.utils import load_md_data, save_safe
from mbtr_grad.adaptive import AdaptiveGridSearch
from mbtr_grad.constants import PRECOMPUTED_PERMS
from mbtr_grad.gdml import (
    perform_krr,
    predict_force_and_energy,
    eval_force,
    setup_einsum_engine,
)

logger = logging.getLogger("main")


class CMExperiment:
    def __init__(self, training_data_fn, train_size, val_size=50, use_sym=True, seed=0):
        self.use_sym = use_sym
        self.seed = seed

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

    def train(self, sigma=1.0, lambda_=1e-5):
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

        if self.use_sym:
            krr = perform_krr(
                z=self.training_data["z"],
                r=data_R[self.train_ind],
                force=data_F[self.train_ind],
                energy=data_E[self.train_ind],
                n_perm=len(self.perms),
                tril_perms_lin=self.tril_perms_lin,
                use_representation="CM",
                sigma=sigma,
                lambda_=lambda_,
            )
        else:
            krr = perform_krr(
                z=self.training_data["z"],
                r=data_R[self.train_ind],
                force=data_F[self.train_ind],
                energy=data_E[self.train_ind],
                use_representation="CM",
                sigma=sigma,
                lambda_=lambda_,
            )

        val_forces, val_energies = predict_force_and_energy(
            z=self.training_data["z"],
            r=data_R[self.val_ind],
            sig=krr["sig"],
            std=krr["std"],
            c=krr["c"],
            use_representation="CM",
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

    def search_hyperparameter(self, initial_sig=1.75, initial_lam=-20):
        def search(sig, lam):
            try:
                return self.train(sigma=sig, lambda_=lam)["val_force_error"][
                    "mae"
                ].mean()
            except RuntimeError as e:
                if re.match(r"cholesky: U\(\d+,\d+\) is zero, singular U", str(e)):
                    logger.warning(
                        "Cholesky solve failed, fallback to 1e10 - %.3f", np.log2(lam)
                    )
                    return 1e10 - np.log2(lam)
                else:
                    raise

        opt = AdaptiveGridSearch(
            search,
            [
                # val, pri, stp, min, max, dir, b
                (initial_sig, 1, 0.25, -2, 10, 0, 2.0),
                (initial_lam, 2, 1, -50, 1, +1, 2.0),
            ],
        )

        while not opt.done:
            opt.step()
            logger.info("Adaptive searching: %s", opt)

        return {
            "best_v": opt.best_v,
            "best_f": opt.best_f,
            "model": self.train(sigma=2 ** opt.best_v[0], lambda_=2 ** opt.best_v[1]),
        }

    def compute_test_error(self, krr_result):
        n_atoms = self.training_data["R"].shape[1]

        val_forces, val_energies = predict_force_and_energy(
            z=np.arange(n_atoms),
            r=self.training_data["R"][self.test_ind],
            sig=krr_result["sig"],
            std=krr_result["std"],
            c=krr_result["c"],
            use_representation="CM",
            R_d_desc_alpha=krr_result["R_d_desc_alpha"],
            R_desc=krr_result["R_desc"],
            tril_perms_lin=krr_result["tril_perms_lin"],
            is_test=True,
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
        exp_fn = "CM_sym-%r_%d_seed_%d.pypickle.gz" % (
            self.use_sym,
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
            self.compute_test_error(model["model"])
            save_safe(experiment_fn, model, use_gzip=True)

        return model


def main(data_fn, exp_dir, use_sym, seed):
    train_name = os.path.basename(data_fn).split(".")[0]
    output_dir = os.path.join(exp_dir, train_name)
    os.makedirs(output_dir, exist_ok=True)

    best_v = (1.75, -20)
    for train_size in [10, 20, 40, 50, 100, 150, 200, 300, 400, 500, 1000]:
        experiment = CMExperiment(
            data_fn, train_size=train_size, val_size=50, use_sym=use_sym, seed=seed
        )
        model = experiment.run(output_dir, best_v=best_v, override=False)
        best_v = (model["best_v"][0], -20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", type=str, help="Training data filename")
    parser.add_argument(
        "--no_sym", action="store_true", default=False, help="Don't use symmetry"
    )
    parser.add_argument(
        "--output", type=str, default="./tmp/sgdml_exps/cm", help="Output dir"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="More debugging"
    )
    parser.add_argument("--repeat", type=int, default=5, help="Repeat N experiments")
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
        main(args.training_data, args.output, not args.no_sym, i)
