from typing import Any, Optional, TypeVar

import torch
from torch import nn


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_device(model: nn.Module) -> torch.device:
    """
    Returns the device on which the model is running.
    """
    if isinstance(model.device, torch.device):
        return model.device
    return next(model.parameters()).device


T = TypeVar("T", torch.Tensor, dict[Any, Any], list[Any], tuple[Any, ...])


def recursive_tensor_copy(
    x: T,
    clone: Optional[bool] = None,
    detach: Optional[bool] = None,
    retain_grad: Optional[bool] = None,
) -> T:
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_tensor_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_tensor_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


Svd = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def low_rank_pinv(
    *, matrix: torch.Tensor, rank: int, svd: Svd | None = None
) -> torch.Tensor:
    """Compute a low-rank pseudo-inverse of a matrix.

    Args:
        matrix: The matrix to invert.
        rank: The rank of the approximation.

    Returns:
        The pseudo-inverse.
    """
    if svd is None:
        svd = torch.svd(matrix.float())
    u, s, v = svd
    matrix_pinv = v[:, :rank] @ torch.diag(1 / s[:rank]) @ u[:, :rank].T
    return matrix_pinv.to(matrix.dtype)
