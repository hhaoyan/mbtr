import ctypes
import time

import torch
import os

torch_cuda = ctypes.CDLL(
    os.path.join(os.path.dirname(torch.__file__), "lib/libtorch_cuda.so")
)
MagmaUpper = 121
MagmaLower = 122


def cholesky_inplace_upper(mat: torch.Tensor):
    """
    Perform upper Cholesky decomposition using inplace operation. In-place
    cholesky won't be faster than torch.cholesky but is able to handle
    larger matrices, since the default Cholesky creates duplicate matrices.

    This function uses the Magma linear algebra routine in PyTorch's binary.
    The routine is not documented and may change in the future, so use this
    function with caution.

    :param mat: The matrix to perform Cholesky on, must be a CUDA tensor.
    """
    # # Magma states rank should be multiples of 16, but it seems just fine
    # # with any rank values.
    # if mat.shape[0] % 16 != 0:
    #     raise ValueError(
    #         "The leading dimension of the array is not divisible by 16.")
    assert len(mat.shape) == 2 and mat.shape[1] == mat.shape[0]
    assert mat.is_cuda, "mat must be a CUDA tensor!"
    assert mat.is_contiguous(), "mat must be contiguous in memory!"

    rank = mat.shape[0]

    if mat.dtype == torch.float32:
        fn = torch_cuda.magma_spotrf_native
    elif mat.dtype == torch.float64:
        fn = torch_cuda.magma_dpotrf_native
    else:
        raise TypeError("Unsupported dtype %r" % mat.dtype)

    info = ctypes.c_int(0)
    ret_value = fn(
        # Magma uses Fortran order so we use MagmaLower
        ctypes.c_int(MagmaLower),
        ctypes.c_int(rank),
        ctypes.c_void_p(mat.data_ptr()),
        ctypes.c_int(rank),
        ctypes.byref(info),
    )
    if ret_value != 0:
        raise RuntimeError("magma_*potrf_gpu returned %d" % ret_value)
    if info.value < 0:
        raise ValueError(
            "magma_*potrf_gpu: " "the %d-th argument had an illegal value" % -info.value
        )
    if info.value > 0:
        raise ValueError(
            "magma_*potrf_gpu: "
            "the leading minor of order %d is not positive definite, "
            "and the factorization could not be completed." % info.value
        )
    mat.triu_()

    return mat


def gen_random_cuda_pd_mat(rank):
    a = torch.rand((rank,), dtype=torch.float64)
    a = torch.outer(a, a)
    a[list(range(a.shape[0])), list(range(a.shape[0]))] += 10
    return a


def test_cho():
    for r in [
        16,
        32,
        64,
        128,
        256,
        512,
        1000,
        1024,
        1044,
        1039,
        2048,
        4096,
        8192,
        8888,
        9999,
        10240,
        15000,
        25000,
        50000,
    ]:
        a = gen_random_cuda_pd_mat(r)
        b = a.cuda()
        torch.cuda.synchronize()
        start = time.time()
        cholesky_inplace_upper(b)
        torch.cuda.synchronize()
        b1_time = time.time() - start
        b1 = b.cpu()

        a = a.cuda()
        torch.cuda.synchronize()
        start = time.time()
        b2 = torch.cholesky(a, upper=True)
        torch.cuda.synchronize()
        b2_time = time.time() - start
        b2 = b2.cpu()
        print("Size", r, torch.allclose(b1, b2), "time: %.4f/%.4f" % (b1_time, b2_time))
