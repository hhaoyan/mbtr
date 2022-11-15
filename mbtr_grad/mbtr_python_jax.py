import itertools
import logging
from abc import abstractmethod
from functools import partial
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np
from jax import jit, jacfwd, vmap, config

from mbtr_grad.constants import JAX_GPU_INDEX
from mbtr_grad.utils import batch_mode

jax_logger = logging.getLogger("JAX_MBTR")

jax_x64 = True
config.update("jax_enable_x64", True)

if jax.default_backend() == "cpu":
    JAX_DEV = jax.devices("cpu")[0]
else:
    jax_gpu_dev = jax.devices("gpu")
    if len(jax_gpu_dev) > JAX_GPU_INDEX:
        JAX_DEV = jax_gpu_dev[JAX_GPU_INDEX]
    else:
        JAX_DEV = jax_gpu_dev[0]


def set_jax_device(dev):
    jax_logger.info("Using device %r.", dev)
    global JAX_DEV
    JAX_DEV = dev


__all__ = [
    "WeightFunc2BodyIdentity",
    "WeightFunc2BodyInvDist",
    "WeightFunc3BodyExpInvDist",
    "GeomFunc2BodyInvDistance",
    "GeomFunc3BodyCosAngle",
    "DistFuncGaussian",
    "mbtr_python",
    "mbtr_python_div",
    "set_jax_device",
]


def require_jit(static_argnums=None, argnums=None):
    def decorator(fn):
        setattr(fn, "_jax_static_argnums", static_argnums)
        setattr(fn, "_jax_argnums", argnums)
        return fn

    return decorator


class JaxFunc:
    """A scalar function with derivatives."""

    _jax_grad = None

    def __eq__(self, other):
        """Make this class hashable."""
        self_internals = self.__dict__
        other_internals = other.__dict__
        return type(other) is type(self) and all(
            self_internals[k] == other_internals[k]
            for k in self_internals
            if not k.startswith("_")
        )

    def __hash__(self):
        internals = ((k, v) for k, v in self.__dict__.items() if not k.startswith("_"))
        internals = tuple(sorted(internals))
        return hash(hash(type(self)) + hash(internals))

    @staticmethod
    def maybe_recompile(fn, src, use_jacfwd=False):
        need_recompile = (
            getattr(fn, "_jax_dev", None) != JAX_DEV
            or getattr(fn, "_jax_x64_enabled", None) is not jax_x64
        )
        if not need_recompile:
            return fn

        jax_logger.debug(
            "Recompiling function %r using device %r, x64: %r", src, JAX_DEV, jax_x64
        )
        static_argnums = getattr(fn, "_jax_static_argnums", None)
        argnums = getattr(fn, "_jax_argnums", None)
        if use_jacfwd:
            src = jacfwd(src, argnums)
        compiled = jit(src, static_argnums=static_argnums, device=JAX_DEV)
        setattr(compiled, "_jax_dev", JAX_DEV)
        setattr(compiled, "_jax_x64_enabled", jax_x64)
        return compiled

    @abstractmethod
    def __call__(self, *args, **kwargs):
        self._cached_forward = self.maybe_recompile(
            getattr(self, "_cached_forward", ()), self._forward
        )
        return self._cached_forward(*args, **kwargs)

    def div(self, *args, **kwargs):
        self._cached_div = self.maybe_recompile(
            getattr(self, "_cached_div", ()), self._forward
        )
        return self._cached_div(*args, **kwargs)

    def _forward(self, *args):
        raise NotImplemented()

    def _div(self, *args):
        raise NotImplemented()


class WeightFunc2BodyIdentity(JaxFunc):
    """
    2-body weighting function of constant 1.
    """

    @require_jit(static_argnums=(0,), argnums=(1, 2))
    def _forward(self, ri, rj):
        return 1.0


class WeightFunc2BodyInvDist(JaxFunc):
    """
    2-body weighting function that takes the form of:

    .. math::
        w(R_i, R_j) = \\frac{1}{|R_i - R_j|}
    """

    @require_jit(static_argnums=(0,), argnums=(1, 2))
    def _forward(self, ri, rj):
        return 1.0 / jnp.linalg.norm(ri - rj) ** 2


class WeightFunc3BodyExpInvDist(JaxFunc):
    """
    3-body weighting function that takes the form of:

    .. math::
        w(R_i, R_j, R_k) = exp\\left(-\\frac{|R_i-R_j|+|R_j-R_k|+|R_k-R_i|}{ls}
        \\right)
    """

    def __init__(self, ls):
        self.ls = ls

    @require_jit(static_argnums=(0,), argnums=(1, 2, 3))
    def _forward(self, ri, rj, rk):
        norms = (
            jnp.linalg.norm(ri - rj)
            + jnp.linalg.norm(ri - rk)
            + jnp.linalg.norm(rj - rk)
        )
        return jnp.exp(-norms / self.ls)


class GeomFunc2BodyInvDistance(JaxFunc):
    """
    Geometry function that takes the form of:

    .. math::
        G(R_i, R_j) = \\frac{1}{|R_i - R_j|}
    """

    @require_jit(static_argnums=(0,), argnums=(1, 2))
    def _forward(self, ri, rj):
        return 1.0 / jnp.linalg.norm(ri - rj)


class GeomFunc3BodyCosAngle(JaxFunc):
    """
    Geometry function of cosine function:

    .. math::
        G(R_i, R_j, R_k) = \\frac{(R_i-R_j)\\cdot(R_k-R_j)}{|R_i-R_j|
        \\cdot|R_k-R_j|}
    """

    @require_jit(static_argnums=(0,), argnums=(1, 2, 3))
    def _forward(self, ra, rb, rc):
        dotuv = jnp.dot(ra - rb, rc - rb)
        denominator = jnp.linalg.norm(ra - rb) * jnp.linalg.norm(rc - rb)
        return jnp.maximum(-1, jnp.minimum(1, dotuv / denominator))


class DistFuncGaussian(JaxFunc):
    """
    Gaussian distribution function.
    """

    def __init__(self, sigma):
        self.const = float(1.0 / (sigma * np.sqrt(2.0)))

    @require_jit(static_argnums=(0,), argnums=2)
    def _forward(self, val_range, geom_mean, *, dx):
        right = jsc.special.erf((val_range + dx - geom_mean) * self.const) / 2
        left = jsc.special.erf((val_range - geom_mean) * self.const) / 2
        return right - left


def _mbtr_python_one_system(
    r: "np.ndarray",
    z: "np.ndarray",
    grid: "np.ndarray",
    order: int,
    weightf: JaxFunc,
    distf: JaxFunc,
    geomf: JaxFunc,
):
    """
    Compute MBTR for a single system. See :func:`mbtr_python`.
    """
    elements = sorted(set(z))

    mbtr = jnp.zeros((len(elements),) * order + (grid.size,))

    dx = grid[1] - grid[0]
    for ats in itertools.product(*([range(z.size)] * order)):
        if len(set(ats)) != len(ats):
            continue

        zs = [elements.index(z[x]) for x in ats]
        rs = [r[x] for x in ats]

        indexer = tuple(zs)
        mbtr = mbtr.at[indexer].set(
            mbtr[indexer] + weightf(*rs) * distf(grid, geomf(*rs), dx=dx)
        )

    return mbtr


def mbtr_python(
    z: "np.ndarray",
    r: "np.ndarray",
    grid: Union["np.ndarray", Tuple],
    order: int,
    weightf: JaxFunc,
    distf: JaxFunc,
    geomf: JaxFunc,
    flatten=False,
    batch_size=None,
) -> "np.ndarray":
    """
    Compute MBTR using multiprocessing.

    :param z: Element types. (NAtom)
    :param r: Coordinates of atoms. (Batch, NAtom, 3)
    :param order: Order of MBTR.
    :param weightf: Weighting function.
    :param distf: Distribution function.
    :param geomf: Geometry function.
    :param grid: Grid definition.
    :param flatten: Whether to flatten representation.
    :param batch_size: Batch size.
    :return: MBTR tensor.
    """
    if isinstance(grid, tuple):
        grid = np.linspace(*grid)
    mapper = vmap(
        partial(
            _mbtr_python_one_system,
            z=z,
            order=order,
            grid=grid,
            weightf=weightf,
            distf=distf,
            geomf=geomf,
        )
    )

    if batch_size is not None:
        mapper = batch_mode(arg_id=0, batch_size=batch_size)(mapper)
    result = mapper(r)

    if flatten:
        result = np.array(result)
        return result.reshape((len(result), -1))
    return np.array(result)


def _mbtr_python_div_one_system(
    r: "np.ndarray",
    z: "np.ndarray",
    grid: "np.ndarray",
    order: int,
    weightf: JaxFunc,
    distf: JaxFunc,
    geomf: JaxFunc,
) -> "np.ndarray":
    """
    Compute MBTR derivative for one system. Also see :func:`mbtr_python_div`.
    """
    elements = sorted(set(z))

    mbtr_div = jnp.zeros((len(elements),) * order + (grid.size, z.size, r.shape[-1]))
    dx = grid[1] - grid[0]
    full = slice(None, None, None)
    for ats in itertools.product(*([range(z.size)] * order)):
        if len(set(ats)) != len(ats):
            continue

        zs = [elements.index(z[x]) for x in ats]
        rs = [r[x] for x in ats]

        geom_mean = geomf(*rs)
        wf = weightf(*rs)
        grid_values = distf(grid, geom_mean, dx=dx)
        grid_div = distf.div(grid, geom_mean, dx=dx)

        wf_div = weightf.div(*rs)
        gf_div = geomf.div(*rs)

        for at_id, wf_d, gf_d in zip(ats, wf_div, gf_div):
            indexer = tuple(zs) + (full, at_id, full)
            mbtr_div = mbtr_div.at[indexer].set(
                mbtr_div[indexer]
                + jnp.outer(grid_values, wf_d)
                + wf * jnp.outer(grid_div, gf_d)
            )

    return mbtr_div


def mbtr_python_div(
    z: "np.ndarray",
    r: "np.ndarray",
    grid: Union["np.ndarray", Tuple],
    order: int,
    weightf: JaxFunc,
    distf: JaxFunc,
    geomf: JaxFunc,
    flatten=False,
    batch_size=None,
) -> "np.ndarray":
    """
    Compute MBTR derivative.

    :param z: Element types. (NAtom)
    :param r: Coordinates of atoms. (Batch, NAtom, 3)
    :param order: MBTR Order
    :param weightf: Weighting function.
    :param distf: Distribution function.
    :param geomf: Geometry function.
    :param grid: Grid definition.
    :param flatten: Whether to flatten representation.
    :param batch_size: Batch size.
    :return: MBTR Derivative.
    """
    if isinstance(grid, tuple):
        grid = np.linspace(*grid)
    mapper = vmap(
        partial(
            _mbtr_python_div_one_system,
            z=z,
            order=order,
            grid=grid,
            weightf=weightf,
            distf=distf,
            geomf=geomf,
        )
    )

    if batch_size is not None:
        mapper = batch_mode(arg_id=0, batch_size=batch_size)(mapper)
    results = mapper(r)

    if flatten:
        result = np.array(results)
        return result.reshape((len(results), -1, r.shape[1], r.shape[2]))
    return np.array(results)
