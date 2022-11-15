import logging
import time
from functools import partial
from typing import Optional, Dict, Tuple

import numpy as np
import scipy as sp
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# from mbtr_grad.cholesky import cholesky_inplace_upper
from mbtr_grad.coulomb_matrix import cm_rep
from mbtr_grad.matern import MaternKernel
from mbtr_grad.mbtr_python_torch import mbtr_python
from mbtr_grad.utils import np_lru_cache

__all__ = [
    "setup_einsum_engine",
    "compute_rep_div",
    "perform_krr",
    "predict_force_and_energy",
    "eval_force",
]

# This symbol is used to keep consistent with sGDML's data exchange format.
# They add a -1 to the alphas. Not sure why but let's keep it as is.
FLIP_SIGN = -1

gdml_status = {
    "einsum": np.einsum,
    "efficient_einsum": None,
    "cholesky_engine": "numpy",
    "matern_kernel": MaternKernel(None, np.einsum),
    "batch_size_fn": lambda _, __: 100,
    "rep_device": "cpu",
}
logger = logging.getLogger("GDML")


def setup_einsum_engine(engine: str = "cpu") -> None:
    """
    Setup the backend engine for fast einsum calculation.

    Currently, supported engines are:

    - cpu: Use numpy :func:`np.einsum`. This is very slow and not efficient.
    - torch: Use PyTorch :func:`torch.einsum` with cuda tensors. Requires
        `pytorch` package.

    :param engine: Value in `cpu`, and `torch`.
    """

    if engine == "cpu":
        import opt_einsum as oe
        'ij,ijkl,ijkl -> []'
        # three tensors (one matrix), want to calculate sum_{ijkl} ij*ijkl*ijkl with broadcasting.
        # 1. first: ij,ijkl -> ijkl
        # 2. second: ijkl,ijkl -> []
        # ijkl -> larger memory requirement.

        logger.info("Using cpu as the backend for einsum.")
        gdml_status.update(
            {
                "cholesky_engine": "numpy",
                "rep_device": "cpu",
                "batch_size_fn": lambda _, __: 100,
                "einsum": partial(
                    oe.contract, backend="numpy", use_blas=True, optimize="optimal"
                ),
            }
        )
    elif engine == "torch":
        logger.info("Using pytorch as the backend for einsum.")

        import torch
        from opt_einsum_torch.planner import EinsumPlanner

        dev_id = 0
        logger.info("Using device cuda device %d for PyTorch einsum.", dev_id)
        torch_device = torch.device("cuda:%d" % dev_id)

        def _torch_backend(formula, *arrs):
            return (
                gdml_status["efficient_einsum"]
                .einsum(formula, *arrs, async_computation=True)
                .numpy()
            )

        def _batch_size_fn(rep_div_shape, n_perm):
            batch_memory = (
                torch.cuda.get_device_properties(torch_device).total_memory * 0.4
            )

            n_total = rep_div_shape[0]
            n_rep = rep_div_shape[1]
            n_atom_coord = rep_div_shape[2] * rep_div_shape[3]

            return max(
                50,
                int(
                    (batch_memory / 8 - n_atom_coord * n_total * n_perm)
                    / n_perm
                    / ((n_rep + 2) * n_total + n_total * n_rep * n_atom_coord)
                ),
            )

        gdml_status.update(
            {
                "cholesky_engine": "torch",
                "einsum": _torch_backend,
                "batch_size_fn": _batch_size_fn,
                "efficient_einsum": EinsumPlanner(torch_device, 0.45),
                "matern_kernel": MaternKernel(torch_device, _torch_backend),
                "rep_device": torch_device,
            }
        )


def cholesky_solve(mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    if gdml_status["cholesky_engine"] == "torch":
        dev_id = 0

        logger.debug("Using GPU %d to perform PyTorch Cholesky decomposition.", dev_id)
        try:
            cuda_y = torch.tensor(y, device="cuda:%d" % dev_id)
            cuda_mat = torch.tensor(mat, device="cuda:%d" % dev_id)
            u_matrix = torch.cholesky(cuda_mat, upper=True)
            solution = torch.cholesky_solve(cuda_y[:, None], u_matrix, upper=True)
            return solution.cpu().numpy()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(
                    "CUDA is out of memory. Will switch to numpy " "for Cholesky solve."
                )
                gdml_status["cholesky_engine"] = "numpy"
            else:
                raise e

    if gdml_status["cholesky_engine"] == "numpy":
        return sp.linalg.cho_solve(sp.linalg.cho_factor(mat), y)
    else:
        raise ValueError("Unknown linalg backend: %s" % gdml_status["cholesky_engine"])


def eval_force(force_pred, force_true, energy_pred, energy_true, name=""):
    force_rmse = np.sqrt(
        mean_squared_error(
            force_true.reshape((-1, 3)),
            force_pred.reshape((-1, 3)),
            multioutput="raw_values",
        )
    )
    force_mae = mean_absolute_error(
        force_true.reshape((-1, 3)),
        force_pred.reshape((-1, 3)),
        multioutput="raw_values",
    )

    energy_rmse = np.sqrt(mean_squared_error(energy_true, energy_pred))
    energy_mae = mean_absolute_error(energy_true, energy_pred)

    if name:
        name = name + ""
    logging.info(
        "%sForce error: RMSE [%.3f, %.3f, %.3f] MAE [%.3f, %.3f, %.3f]",
        name,
        *(tuple(force_rmse) + tuple(force_mae))
    )
    logging.info("%sEnergy error: RMSE %.3f MAE %.3f", name, energy_rmse, energy_mae)

    return force_rmse, force_mae, energy_rmse, energy_mae


@np_lru_cache(maxsize=64)
def compute_rep_div(
    z: np.ndarray,
    r: np.ndarray,
    use_representation: str,
    rep_kwargs: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rep_kwargs = rep_kwargs or {}
    if use_representation == "CM":
        rep, rep_div = cm_rep(r, device=gdml_status["rep_device"], **rep_kwargs)
    elif use_representation == "MBTR":
        rep, rep_div = mbtr_python(
            z=z, r=r, device=gdml_status["rep_device"], **rep_kwargs
        )

        if rep_kwargs["order"] == 2:
            # Reduce memory size
            i, j = np.tril_indices(len(z), -1)

            rep = rep.reshape((r.shape[0], len(z), len(z), -1))
            # rep[:, np.arange(len(z)), np.arange(len(z)), :] /= 2
            rep = rep[:, i, j, :].reshape((len(rep), -1))  # * 2

            rep_div = rep_div.reshape(
                (r.shape[0], len(z), len(z), -1, r.shape[1], r.shape[2])
            )
            # rep_div[:, np.arange(len(z)), np.arange(len(z)), :, :, :] /= 2
            rep_div = rep_div[:, i, j, :, :, :].reshape(
                (r.shape[0], -1, r.shape[1], r.shape[2])
            )  # * 2
    else:
        raise ValueError("Unknown representation: %s" % use_representation)

    return rep, rep_div


def perform_krr(
    z: np.ndarray,
    r: np.ndarray,
    force: np.ndarray,
    energy: np.ndarray,
    use_representation: str = "CM",
    rep_kwargs: Optional[Dict] = None,
    sigma: float = 10,
    lambda_: float = 1e-15,
    include_train_error=True,
    n_perm: int = 1,
    tril_perms_lin: Optional[np.ndarray] = None,
):
    logger.debug("Calculating %s representation...", use_representation)
    rep, rep_div = compute_rep_div(z, r, use_representation, rep_kwargs)

    if tril_perms_lin is None:
        tril_perms_lin = np.arange(rep_div.shape[1])

    batch_size = gdml_status["batch_size_fn"](rep_div.shape, n_perm)
    logger.debug("Using %d as batch size", batch_size)
    logger.debug("Calculating kernel...")
    kernel = gdml_status["matern_kernel"].calculate_kernel(
        sigma,
        rep=rep,
        rep_div=rep_div,
        n_perm=n_perm,
        tril_perms_lin=tril_perms_lin,
        flatten_kernel=True,
        batch_size=batch_size,
    )
    logger.debug("Kernel shape [%d, %d].", *kernel.shape)

    y = force.flatten()
    y_std = np.std(y).item()
    y = y / y_std

    logger.debug(
        "Performing Cholesky solve [%d, %d] <- [%d].",
        kernel.shape[0],
        kernel.shape[1],
        y.shape[0],
    )
    diagonal = np.arange(kernel.shape[0])
    kernel[diagonal, diagonal] += lambda_
    alphas = FLIP_SIGN * cholesky_solve(kernel, y)
    kernel[diagonal, diagonal] -= lambda_

    logger.debug(
        "Calculating alphas of shape [batch=%d, dim=%d]...",
        rep_div.shape[0],
        rep_div.shape[1],
    )
    R_d_desc_alpha = gdml_status["einsum"](
        "ijk,ik->ij",
        rep_div.reshape((rep_div.shape[0], rep_div.shape[1], -1)),
        alphas.reshape((rep_div.shape[0], -1)),
    )

    result = {
        "kernel": kernel,
        "tril_perms_lin": tril_perms_lin,
        "std": y_std,
        "y": y,
        "R_desc": rep,
        # 'R_d_desc': rep_div,
        "alphas": alphas,
        "R_d_desc_alpha": R_d_desc_alpha,
        "sig": sigma,
        "z": z,
        "c": 0,
    }

    if include_train_error:
        # Fit energy integration constant
        force_pred, energy_pred = gdml_status["matern_kernel"].predict(
            sigma,
            rep,
            rep_div,
            R_d_desc_alpha,
            rep,
            tril_perms_lin,
            batch_size=batch_size,
        )
        force_pred *= FLIP_SIGN * y_std
        energy_pred *= FLIP_SIGN * y_std

        coef = np.corrcoef(energy_pred.flatten(), energy.flatten())[0, 1]
        if coef < -0.95:
            logger.warning(
                "The predicted energy has %.2f correlation, maybe you "
                "provided the negative energy values."
            )
        if coef < 0.95:
            logger.warning("The predicted energy has low correlation %.2f.", coef)
        energy_center = energy.flatten().mean() - energy_pred.flatten().mean()

        force_rmse, force_mae, energy_rmse, energy_mae = eval_force(
            force_pred, force, energy_pred + energy_center, energy, name="Train"
        )

        result.update(
            {
                "train_force_error": {
                    "rmse": force_rmse,
                    "mae": force_mae,
                },
                "train_energy_error": {
                    "rmse": energy_rmse,
                    "mae": energy_mae,
                },
                "c": energy_center,
            }
        )

    return result


def predict_force_and_energy(
    z,
    r,
    sig,
    std,
    c,
    R_d_desc_alpha,
    R_desc,
    tril_perms_lin,
    use_representation="CM",
    rep_kwargs=None,
    is_test=False,
    batch_size=500,
):
    """
    Predict forces based on trained model.

    :param z: List of chemical elements.
    :param r: Atom coordinates.
    :param sig: Sigma of kernel.
    :param std: Std of forces.
    :param c: Intercept for energy.
    :param R_d_desc_alpha:
    :param R_desc:
    :param use_representation:
    :param rep_kwargs:
    :param tril_perms_lin:
    :param is_test:
    :param batch_size: Batch size for prediction to avoid OOM error.
    :return:
    """
    logger.debug("Calculating %s representation...", use_representation)

    rep_kwargs = rep_kwargs or {}
    if is_test:
        rep_kwargs.update({"as_numpy": False})
        R_d_desc_alpha = torch.tensor(R_d_desc_alpha, device=gdml_status["rep_device"])
        R_desc = torch.tensor(R_desc, device=gdml_status["rep_device"])

    force_array, energy_array = [], []
    start_time = time.time()
    pbar = None
    for start in range(0, r.shape[0], batch_size):
        if time.time() - start_time > 30 and pbar is None:
            pbar = tqdm(desc="Predicting for test set", total=r.shape[0])
            pbar.update(start)

        end = min(start + batch_size, r.shape[0])
        if is_test:
            rep, rep_div = compute_rep_div.__original__(
                z, r[start:end], use_representation, rep_kwargs
            )
        else:
            rep, rep_div = compute_rep_div(
                z, r[start:end], use_representation, rep_kwargs
            )

        if tril_perms_lin is None:
            tril_perms_lin = np.arange(rep_div.shape[1])

        # During testing, it's not advised to use CPU since data transfer could
        # be very slow. We suggest using smaller batch sizes.
        force, energy = gdml_status["matern_kernel"].predict(
            sig, rep, rep_div, R_d_desc_alpha, R_desc, tril_perms_lin, batch_size
        )

        force_array.append(force)
        energy_array.append(energy)

        if pbar is not None:
            pbar.update(end - start)
    if pbar is not None:
        pbar.close()

    force = np.concatenate(force_array, axis=0)
    energy = np.concatenate(energy_array, axis=0)

    return FLIP_SIGN * std * force, FLIP_SIGN * std * energy + c
