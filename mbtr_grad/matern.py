import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch

from mbtr_grad.utils import np_lru_cache, to_gpu, to_cpu

__all__ = ["MaternKernel"]


class MaternKernel:
    _logger = logging.getLogger("MaternKernel")

    def __init__(
        self,
        cuda_device: Optional[torch.device],
        einsum_backend=torch.einsum,
        kernel_cache_size: int = 4,
    ):
        """
        Create a new kernel evaluator.

        :param cuda_device: The CUDA device for linear algebra. If set to None,
            CPU will be used instead.
        :param einsum_backend: einsum function to use. Defaults to
            :func:`torch.einsum`, which is not fully optimized.
        :param kernel_cache_size: Size of LRU cache for kernel evaluation.
        """
        self.cuda_device = cuda_device
        self.use_gpu = self.cuda_device is not None
        self.einsum_backend = einsum_backend

        if kernel_cache_size > 1:
            self.calculate_kernel = np_lru_cache(maxsize=kernel_cache_size)(
                self.calculate_kernel
            )

    def __hash__(self):
        """There is no difference between any two class instances."""
        return 1

    @torch.no_grad()
    def _compute_distance_cpu(
        self,
        sig: float,
        rep: np.ndarray,
        rep_ref: np.ndarray,
        rep_div: Optional[np.ndarray] = None,
        rep_div_ref: Optional[np.ndarray] = None,
        pin_memory=False,
    ):
        # (Batch, Batch, NRepeat, NRep)
        rep_distance_shape = (
            rep.shape[0],
            rep_ref.shape[0],
            rep_ref.shape[1],
            rep.shape[1],
        )
        self._logger.debug("Calculating rep distance: %r...", rep_distance_shape)
        rep = to_cpu(rep)
        rep_ref = to_cpu(rep_ref)
        rep_distance = torch.empty(
            rep_distance_shape, dtype=rep.dtype, pin_memory=pin_memory
        )
        torch.sub(rep[:, None, None, :], rep_ref[None, :, :, :], out=rep_distance)

        # (Batch, Batch, NRepeat)
        self._logger.debug(
            "Calculating r_norm_sq5: %r...", tuple(rep_distance.shape[:-1])
        )
        r_norm_sq5 = np.sqrt(5) * torch.norm(rep_distance, dim=-1)
        if pin_memory:
            r_norm_sq5 = r_norm_sq5.pin_memory()

        # (Batch, Batch, NRepeat)
        self._logger.debug("Calculating exponential: %r...", tuple(r_norm_sq5.shape))
        exponential = torch.exp(-r_norm_sq5 / sig)
        if pin_memory:
            exponential = exponential.pin_memory()

        self._logger.debug("Pinning memory for rep_div and repeated_div...")
        if rep_div is not None:
            rep_div = to_cpu(rep_div)
            if pin_memory:
                rep_div = rep_div.pin_memory()
        if rep_div_ref is not None:
            rep_div_ref = to_cpu(rep_div_ref)
            if pin_memory:
                rep_div_ref = rep_div_ref.pin_memory()

        return rep_div, rep_div_ref, rep_distance, r_norm_sq5, exponential

    @torch.no_grad()
    def _compute_distance_gpu(
        self,
        sig: float,
        rep: np.ndarray,
        rep_ref: np.ndarray,
        rep_div: Optional[np.ndarray] = None,
        rep_div_ref: Optional[np.ndarray] = None,
    ):
        rep_distance_shape = (
            rep.shape[0],
            rep_ref.shape[0],
            rep_ref.shape[1],
            rep.shape[1],
        )
        self._logger.debug("Calculating rep distance: %r...", rep_distance_shape)

        rep_cuda = to_gpu(rep, dev=self.cuda_device)
        rep_ref_cuda = to_gpu(rep_ref, dev=self.cuda_device)

        rep_distance = torch.empty(
            rep_distance_shape, dtype=rep_cuda.dtype, device=self.cuda_device
        )
        # (Batch, Batch, NRepeat, NRep)
        torch.sub(
            rep_cuda[:, None, None, :], rep_ref_cuda[None, :, :, :], out=rep_distance
        )

        # (Batch, Batch, NRepeat)
        self._logger.debug(
            "Calculating r_norm_sq5: %r...", tuple(rep_distance.shape[:-1])
        )
        r_norm_sq5 = np.sqrt(5) * torch.norm(rep_distance, dim=-1)

        # (Batch, Batch, NRepeat)
        self._logger.debug("Calculating exponential: %r...", tuple(r_norm_sq5.shape))
        exponential = torch.exp(-r_norm_sq5 / sig)

        self._logger.debug("Pre-allocating CUDA memory for rep_div and repeated_div...")
        if rep_div is not None:
            rep_div = to_gpu(rep_div, dev=self.cuda_device)
        if rep_div_ref is not None:
            rep_div_ref = to_gpu(rep_div_ref, dev=self.cuda_device)

        return rep_div, rep_div_ref, rep_distance, r_norm_sq5, exponential

    def _perform_einsum(
        self, sig, rep_div, repeated_div, rep_distance, r_norm_sq5, exponential
    ) -> np.ndarray:
        einsum1 = self.einsum_backend(
            "ilm,ijr,jrln->ijmn",
            rep_div,
            (1 + r_norm_sq5 / sig) * exponential,
            repeated_div,
        )
        einsum2 = self.einsum_backend(
            "ilm,ijrl,ijr,ijrk,jrkn->ijmn",
            rep_div,
            rep_distance,
            exponential,
            rep_distance,
            repeated_div,
        ) * (5 / sig**2)
        kernel = (5 / (3 * sig**2)) * (einsum1 - einsum2)
        if isinstance(kernel, torch.Tensor):
            kernel = kernel.numpy()

        return kernel

    def calculate_kernel(
        self,
        sig: float,
        rep: np.ndarray,
        rep_div: np.ndarray,
        tril_perms_lin: np.ndarray,
        n_perm: int,
        batch_size: int = 100,
        flatten_kernel: bool = True,
    ) -> np.ndarray:
        """
        Compute the kernel matrix. Denote the kernel as:

        .. math::
            K(R_i, R_j) = \\left(\\frac{5}{3}\\frac{d^2}{\\sigma^2} + \\frac{
            \\sqrt{5}d}{\\sigma}+1\\right) \\exp\\left(\\frac{\\sqrt{5}d}{
            \\sigma}\\right)

        This function calculates the Hessian matrix as kernel:

        .. math::
            H(R_i, R_j, x_m, x_n) = \\frac{\\partial^2 K(R_i, R_j)}{
            \\partial x_m \\partial x_n}

        The kernel will have 4 dimensions :math:`[batch, batch,
        natoms*cartesian, natoms*cartesian]`. However, if you specify
        ``flatten_kernel=True``, the kernel matrix will be flattened, resulting
        in only 4 dimensions :math:`[batch*natoms*cartesian,
        batch*natoms*cartesian]`.

        Args:
            sig: Sigma :math:`\\sigma`` of the kernel.
            rep: Representation tensor, should have shape :math:`[batch, dim]`.
            rep_div: Derivative tensor of representation, should have shape
                :math:`[batch, dim, natoms, cartesian]`.
            n_perm: Number of permutations.
            tril_perms_lin: Permutation index.
            batch_size: Batch size for computing kernel.
            flatten_kernel: Whether to flatten the kernel.
        """
        # Sanity checks
        assert len(rep.shape) == 2, "Representation should have shape [batch, dim]!"
        assert (
            len(rep_div.shape) == 4
        ), "Derivative should have shape [batch, dim, natoms, cartesian]!"
        assert (
            rep.shape[0] == rep_div.shape[0]
        ), "Batch dimension of representation and derivative differs!"
        assert (
            rep.shape[1] == rep_div.shape[1]
        ), "Representation dimension of representation and derivative differs!"

        rep_div = rep_div.reshape((rep_div.shape[0], rep_div.shape[1], -1))

        tril_perms_lin = tril_perms_lin.reshape((n_perm, -1), order="F")

        self._logger.debug(
            "Creating input tensor [batch=%d, sym=%d, rep=%d] and "
            "derivative [batch=%d, sym=%d, rep=%d, cartesian=%d]...",
            rep.shape[0],
            n_perm,
            rep.shape[1],
            rep_div.shape[0],
            n_perm,
            rep_div.shape[1],
            rep_div.shape[2],
        )
        # (Batch, NRepeat, NRep)
        repeated_rep = np.tile(rep, (1, n_perm))[:, tril_perms_lin]
        # (Batch, NRepeat, NRep, NAtom*NCoord)
        repeated_div = np.tile(rep_div, (1, n_perm, 1))[:, tril_perms_lin, :]

        def _batched(block_start, block_end):
            if self.use_gpu:
                (
                    _rep_div,
                    _repeated_div,
                    _rep_distance,
                    _r_norm_sq5,
                    _exponential,
                ) = self._compute_distance_gpu(
                    sig=sig,
                    rep=rep,
                    rep_div=rep_div,
                    rep_ref=repeated_rep[block_start:block_end],
                    rep_div_ref=repeated_div[block_start:block_end],
                )
            else:
                (
                    _rep_div,
                    _repeated_div,
                    _rep_distance,
                    _r_norm_sq5,
                    _exponential,
                ) = self._compute_distance_cpu(
                    sig=sig,
                    rep=rep,
                    rep_div=rep_div,
                    rep_ref=repeated_rep[block_start:block_end],
                    rep_div_ref=repeated_div[block_start:block_end],
                )

            self._logger.debug("Performing einsum...")
            return self._perform_einsum(
                sig, _rep_div, _repeated_div, _rep_distance, _r_norm_sq5, _exponential
            )

        kernel = np.empty(
            (rep.shape[0], rep.shape[0], rep_div.shape[2], rep_div.shape[2]),
            dtype=rep.dtype,
        )
        for start in range(0, rep.shape[0], batch_size):
            end = min(start + batch_size, rep.shape[0])
            kernel[:, start:end, :, :] = _batched(start, end)

        if flatten_kernel:
            kernel_dim = rep_div.shape[0] * rep_div.shape[-1]
            kernel = kernel.transpose((0, 2, 1, 3)).reshape((kernel_dim, kernel_dim))

        return kernel

    def calculate_hessian(
        self, sig: float, r_left: np.ndarray, r_right: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Hessian matrix. See the kernel definition in
        :meth:`calculate_kernel`.

        While :func:`calculate_kernel` calculates the Hessian with respect to
        cartesian coordinates, this function calculates the Hessian matrix with
        respect to representations:

        .. math::
            H(R_i^m, R_j^n) = \\frac{\\partial^2 K(R_i, R_j)}{\\partial R_i^m
            \\partial R_j^n}

        The kernel has 4 dimensions :math:`[batch, batch, repdim, repdim]`.

        :param sig: Sigma :math:`\\sigma`` of the kernel.
        :param r_left: Representation tensor 1, should have shape
            :math:`[batch, dim]`.
        :param r_right: Representation tensor 2, should have shape
            :math:`[batch, dim]`.
        :return: The Hessian matrix :math:`H(R_i^m, R_j^n)`.
        """
        assert len(r_left.shape) == 2, "r_left must be 2d (n_ref, dim)"
        assert len(r_right.shape) == 2, "r_right must be 2d (n_calc, dim)"
        assert (
            r_left.shape[-1] == r_right.shape[-1]
        ), "r_left/r_right must have same last dim"

        r_diff = np.sqrt(5) / sig * (r_left[:, None, :] - r_right[None, :, :])
        r_dist = np.linalg.norm(r_diff, axis=-1)
        hessian_ident = (5.0 / 3.0 / sig**2) * np.exp(-r_dist) * (r_dist + 1)

        hessian_mat = self.einsum_backend(
            "ij,ijk,ijl->ijkl",
            -(5.0 / 3.0 / sig**2) * np.exp(-r_dist),
            r_diff,
            r_diff,
        )
        if isinstance(hessian_mat, torch.Tensor):
            hessian_mat = hessian_mat.numpy()

        diagonals = np.arange(r_left.shape[-1])
        hessian_mat[:, :, diagonals, diagonals] += hessian_ident[:, :, None]
        return hessian_mat

    def _perform_einsum_force(
        self, sig, rep_div, rep_train_sym_alpha, rep_distance, r_norm_sq5, exponential
    ) -> Tuple[np.ndarray, np.ndarray]:
        exp_1_d = exponential * (r_norm_sq5 / sig + 1)

        energy = -self.einsum_backend(
            "ijr,ijrk,jrk->i", exp_1_d, rep_distance, rep_train_sym_alpha
        ) * (5.0 / (3.0 * sig**2))

        force = (
            self.einsum_backend(
                "jrk,ijr,ikmn->imn",
                rep_train_sym_alpha,
                exp_1_d,
                rep_div,
            )
            - self.einsum_backend(
                "jrl,ijr,ijrk,ijrl,ikmn->imn",
                rep_train_sym_alpha,
                exponential,
                rep_distance,
                rep_distance,
                rep_div,
            )
            * (np.sqrt(5) / sig) ** 2
        )
        force = force * (5.0 / (3.0 * sig**2))

        if isinstance(force, torch.Tensor):
            force = force.numpy()
        if isinstance(energy, torch.Tensor):
            energy = energy.numpy()

        return force, energy

    def _predict(
        self,
        sig: float,
        rep: np.ndarray,
        rep_div: np.ndarray,
        rep_train_sym: np.ndarray,
        rep_train_sym_alpha: np.ndarray,
    ):
        if self.use_gpu:
            results = self._compute_distance_gpu(
                sig=sig, rep=rep, rep_ref=rep_train_sym, rep_div=rep_div
            )
            rep_div, _, rep_distance, r_norm_sq5, exponential = results
        else:
            results = self._compute_distance_cpu(
                sig=sig, rep=rep, rep_ref=rep_train_sym, rep_div=rep_div
            )
            rep_div, _, rep_distance, r_norm_sq5, exponential = results

        self._logger.debug("Performing einsum for force...")
        force, energy = self._perform_einsum_force(
            sig, rep_div, rep_train_sym_alpha, rep_distance, r_norm_sq5, exponential
        )

        return force, energy

    def predict(
        self,
        sig: float,
        rep: np.ndarray,
        rep_div: np.ndarray,
        rep_train_alpha: Union[np.ndarray, torch.Tensor],
        rep_train: Union[np.ndarray, torch.Tensor],
        tril_perms_lin: np.ndarray = None,
        batch_size=200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict forces and energies based on trained kernel model.

        :param sig: Parameter :math:`\\sigma` of kernel.
        :param rep: Representation of the testing samples.
        :param rep_div: Derivative of the representation of the testing samples.
        :param rep_train_alpha: Alpha coefficients from a trained kernel model.
        :param rep_train: Representation of the training samples.
        :param tril_perms_lin: Permutation index.
        :param batch_size: Batch size for prediction to avoid OOM error.
        :return: (force prediction, energy prediction)
        """
        assert (
            len(rep_train.shape) == 2 and len(rep.shape) == 2
        ), "Representations must be 2D."
        assert (
            rep_train.shape[1] == rep.shape[1]
        ), "Training/testing samples have differing rep dimensions."

        if tril_perms_lin is None:
            tril_perms_lin = np.arange(rep_div.shape[1])

        n_perms = tril_perms_lin.size // rep_train.shape[1]
        perm_idx = tril_perms_lin.reshape((-1, n_perms)).T

        self._logger.debug("Reconstructing training rep/alpha with symmetry...")
        if isinstance(rep_train, np.ndarray):
            rep_train_sym = np.tile(rep_train, (1, n_perms))[:, perm_idx]
        else:
            rep_train_sym = torch.tile(rep_train, (1, n_perms))[:, perm_idx]
        if isinstance(rep_train_alpha, np.ndarray):
            rep_train_sym_alpha = np.tile(rep_train_alpha, (1, n_perms))[:, perm_idx]
        else:
            rep_train_sym_alpha = torch.tile(rep_train_alpha, (1, n_perms))[:, perm_idx]

        forces = []
        energies = []
        for start in range(0, rep.shape[0], batch_size):
            end = min(start + batch_size, rep.shape[0])
            force, energy = self._predict(
                sig,
                rep[start:end],
                rep_div[start:end],
                rep_train_sym,
                rep_train_sym_alpha,
            )
            forces.append(force)
            energies.append(energy)

        return np.concatenate(forces, axis=0), np.concatenate(energies, axis=0)
