# If you have multiple GPUs, we will distribute workloads to separate GPUs.
from pathlib import Path

import numpy

__all__ = [
    "PRECOMPUTED_PERMS",
]

JAX_GPU_INDEX = 0
EINSUM_GPU_INDEX = 1
CHOLESKY_GPU_INDEX = 2

PRECOMPUTED_PERMS = numpy.load(
    str(Path(__file__).resolve().parent / "sgdml_perms.npy"), allow_pickle=True
).item()
