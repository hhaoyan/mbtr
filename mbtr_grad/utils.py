import gzip
import inspect
import logging
import pickle
import shutil
from collections import OrderedDict
from typing import Any, Iterable, Callable, Union, Dict

import numpy as np
import torch

__all__ = [
    "np_lru_cache",
    "batch_mode",
    "continue_checkpoint",
    "save_safe",
    "to_gpu",
    "to_cpu",
    "to_torch_dtype",
    "load_md_data",
]


def load_md_data(npz_fn: str) -> Dict[str, np.ndarray]:
    """Load molecular dynamic data in .npz format."""
    logging.info("Loading %s", npz_fn)
    data = np.load(npz_fn)
    data_R = data["R"]
    data_z = data["z"]
    data_F = data["F"]
    data_E = data["E"]
    data_E_percentile = np.minimum(
        9,
        ((data_E - data_E.min()) / (data_E.max() - data_E.min()) * 10)
        .astype(int)
        .flatten(),
    )
    logging.info("The dataset has %d atoms", data_R.shape[1])

    return {
        "R": data_R,
        "z": data_z,
        "F": data_F,
        "E": data_E,
        "E_percentile": data_E_percentile,
    }


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif np.issubdtype(dtype, np.floating):
        np_dtype = {2: torch.float16, 4: torch.float32, 8: torch.float64}
        return np_dtype[dtype.itemsize]
    elif np.issubdtype(dtype, np.integer):
        np_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}
        return np_dtype[dtype.itemsize]
    else:
        raise TypeError("Unknown dtype %r" % dtype)


def to_gpu(arr: Union[torch.Tensor, np.ndarray], dev: torch.device):
    """Convert an array into a CUDA array."""
    if isinstance(arr, torch.Tensor):
        if arr.device == dev:
            # Keep the same id... PyTorch returns the same tensor, but
            # we don't want to rely on this undocumented behavior.
            return arr
        else:
            return arr.to(dev, non_blocking=True)
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(dev)
    else:
        raise TypeError("Invalid array type %r" % type(arr))


def to_cpu(arr: Union[torch.Tensor, np.ndarray]):
    """Convert an array into a CPU array."""
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            return arr.cpu()
        else:
            return arr
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    else:
        raise TypeError("Invalid array type %r" % type(arr))


def np_lru_cache(maxsize: int = None, maxsize_bytes: int = 1024 * 1024 * 1024 * 8):
    """
    Decorator that helps cache functions whose arguments may be numpy arrays.

    :param maxsize: Maximal number of entries in cache.
    :param maxsize_bytes: Maximal number of numpy array bytes in cache.
    :return: Cached function.
    """
    logger = logging.getLogger("NumpyLRUCache")

    def np_to_immutable(arr):
        b = arr.tobytes()
        return (arr.shape, b), len(b)

    def dict_to_immutable(d):
        items = []
        n_bytes = 0
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                immutable, _bytes = np_to_immutable(v)
                n_bytes += _bytes
                items.append((k, immutable))
            else:
                items.append((k, v))
        return tuple(sorted(items)), n_bytes

    def decorator(func):
        _rep_cache = OrderedDict()
        _cache_info = {"hit": 0, "miss": 0, "size": 0, "cached_bytes": 0}

        def cached_function(*args, **kwargs):
            args_key = ()
            total_np_bytes = 0
            for arg in args:
                if isinstance(arg, np.ndarray):
                    immutable, n_bytes = np_to_immutable(arg)
                    total_np_bytes += n_bytes
                    args_key += (immutable,)
                elif isinstance(arg, dict):
                    immutable, n_bytes = dict_to_immutable(arg)
                    total_np_bytes += n_bytes
                    args_key += (immutable,)
                else:
                    args_key += (arg,)

            if kwargs is not None:
                immutable, n_bytes = dict_to_immutable(kwargs)
                total_np_bytes += n_bytes
                args_key += (immutable,)

            if args_key in _rep_cache:
                _cache_info["hit"] += 1
                _rep_cache.move_to_end(args_key, last=True)
                return _rep_cache[args_key][0]

            _cache_info["miss"] += 1
            result = func(*args, **kwargs)
            _rep_cache[args_key] = (result, total_np_bytes)
            _cache_info["size"] += 1
            _cache_info["cached_bytes"] += total_np_bytes
            logger.debug(
                "Current cached numpy bytes for function %s is %d",
                func.__name__,
                _cache_info["cached_bytes"],
            )

            if len(_rep_cache) > maxsize or _cache_info["cached_bytes"] > maxsize_bytes:
                _cache_info["size"] -= 1
                key, (val, removed_bytes) = _rep_cache.popitem(last=False)
                del key, val
                _cache_info["cached_bytes"] -= removed_bytes

            return result

        cached_function.__doc__ = (
            "Cached (max cache size %d, max numpy cache size %d) version of:\n"
            % (maxsize, maxsize_bytes)
            + (func.__doc__ or "No doc string provided.")
        )
        cached_function.__signature__ = inspect.signature(func)
        cached_function.__get_cache__ = lambda: _rep_cache
        cached_function.__cache_info__ = _cache_info
        cached_function.__original__ = func

        return cached_function

    return decorator


def batch_mode(arg_id: Union[int, str], batch_size=64):
    """
    Decorator that helps make functions run in batch mode.

    :param arg_id: Which parameter should be made into batch mode.
    :param batch_size: Batch size.
    :return: Batch-mode function.
    """

    def decorate(func):
        def batched_function(*args, **kwargs):
            if isinstance(arg_id, int):
                if len(args) <= arg_id:
                    raise NameError(
                        "Unable to make parameter args[%d] batch. "
                        "Did you call using kwargs?" % arg_id
                    )
                batch_arg = args[arg_id]
            elif isinstance(arg_id, str):
                if arg_id not in kwargs:
                    raise NameError("Unable to find kwargs '%s'" % arg_id)
                batch_arg = kwargs[arg_id]
            else:
                raise NameError("Unknown arg_id: %r" % arg_id)

            result = []
            for start in range(0, len(batch_arg), batch_size):
                end = min(start + batch_size, len(batch_arg))

                if isinstance(arg_id, int):
                    new_args = (
                        args[:arg_id] + (batch_arg[start:end],) + args[arg_id + 1 :]
                    )
                    new_kwargs = kwargs
                else:
                    new_args = args
                    new_kwargs = kwargs.copy()
                    new_kwargs[arg_id] = batch_arg[start:end]

                result.append(func(*new_args, **new_kwargs))

            return np.concatenate(result, axis=0)

        batched_function.__doc__ = "Batched version (batch size %d) of:\n" % (
            batch_size,
        ) + (func.__doc__ or "No doc string provided.\n")
        batched_function.__signature__ = inspect.signature(func)

        return batched_function

    return decorate


def save_safe(fn: str, data: Any, use_gzip: bool = False):
    """Safely pickle data to disk."""
    tmp_fn = fn + ".tmp"

    if use_gzip:
        with gzip.open(tmp_fn, "wb") as f:
            pickle.dump(data, f)
    else:
        with open(tmp_fn, "wb") as f:
            pickle.dump(data, f)

    shutil.move(tmp_fn, fn)


def continue_checkpoint(
    fn: Callable, best_v: Iterable[Any], save_fn: str, training_sizes: Iterable
):
    """
    Continue from a checkpoint.

    :param fn: Function to evaluate.
    :param best_v: Best parameter.
    :param save_fn: Checkpoint file.
    :param training_sizes: Training sizes
    :return:
    """
    try:
        with open(save_fn, "rb") as f:
            train_stats = pickle.load(f)
    except (IOError, FileNotFoundError):
        train_stats = []

    already_visited = set(x["results"]["train size"] for x in train_stats)
    for x in train_stats:
        best_v = x["best_v"]

    for size in training_sizes:
        if size in already_visited:
            continue
        train_stats.append(fn(size, best_v))
        best_v = train_stats[-1]["best_v"]

        with open(save_fn, "wb") as f:
            pickle.dump(train_stats, f)
