from functools import lru_cache

import numpy as np
import torch

from mbtr_grad.utils import to_gpu, to_torch_dtype

__all__ = [
    "cm_rep",
]

tril_indices = lru_cache(maxsize=8)(np.tril_indices)


@torch.no_grad()
def cm_rep(r_calc, device="cpu", as_numpy=True):
    """
    This calculates the CM representation and its derivative.
    """
    i, j = tril_indices(r_calc.shape[1], k=-1)

    # Pre-allocate tensors to make memory compact.
    rep = torch.empty(
        (r_calc.shape[0], i.size), dtype=to_torch_dtype(r_calc.dtype), device=device
    )
    derivative = torch.zeros(
        (r_calc.shape[0], i.size, r_calc.shape[1], r_calc.shape[2]),
        dtype=to_torch_dtype(r_calc.dtype),
        device=device,
    )

    r_calc = to_gpu(r_calc, dev=torch.device(device))
    r_diff = torch.empty(
        (r_calc.shape[0], r_calc.shape[1], r_calc.shape[1], r_calc.shape[2]),
        dtype=to_torch_dtype(r_calc.dtype),
        device=device,
    )
    torch.subtract(r_calc.unsqueeze(2), r_calc.unsqueeze(1), out=r_diff)

    dist_mat = torch.linalg.norm(r_diff, dim=-1)
    inv_dist = 1.0 / dist_mat[:, i, j]
    rep.copy_(inv_dist)

    div_data = (r_diff[:, i, j, :]) * (inv_dist[:, :, None] ** 3)
    i_arange = np.arange(i.size)
    derivative[:, i_arange, i, :] = -div_data
    derivative[:, i_arange, j, :] = div_data

    if as_numpy:
        rep, derivative = rep.cpu().numpy(), derivative.cpu().numpy()

    return rep, derivative
