"""Kronecker product related functionality."""

from typing import Tuple

from einops import rearrange
from torch import Tensor


def best_kronecker(
    A: Tensor, B_shape: Tuple[int, int], C_shape: Tuple[int, int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Find the best approximation α (B ⊗ C) of A (in terms of Frobenius norm).

    See https://typeset.io/pdf/approximation-with-kronecker-products-24urjmqom7.pdf
    for details.

    Args:
        A: The matrix to approximate.
        B_shape: The shape of the first tensor in the Kronecker product.
        C_shape: The shape of the second tensor in the Kronecker product.

    Returns:
        The scalar α and the matrices B and C such that α (B ⊗ C) is the best Kronecker
        approximation of A.

    Raises:
        ValueError: If A is not a matrix.
        ValueErrror: If B_shape or C_shape are not 2-tuples.
        ValueError: If the shapes multiply to the incorrect total dimension.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be a matrix (2d). Got {A.ndim}d.")
    if len(B_shape) != 2 or len(C_shape) != 2:
        raise ValueError(
            f"B_shape and C_shape must be 2-tuples. Got {B_shape} and {C_shape}."
        )

    A_rows, A_cols = A.shape
    B_rows, B_cols = B_shape
    C_rows, C_cols = C_shape

    if B_rows * C_rows != A_rows or B_cols * C_cols != A_cols:
        raise ValueError(f"Invalid shapes: A: {A.shape}, B: {B_shape}, C: {C_shape}.")

    A_rearranged = rearrange(
        A,
        "(B_rows C_rows) (B_cols C_cols) -> (B_rows B_cols) (C_rows C_cols)",
        B_rows=B_rows,
        B_cols=B_cols,
        C_rows=C_rows,
        C_cols=C_cols,
    )

    # compute the leading singular vectors and values
    U, S, VT = A_rearranged.svd()
    u, s, v = U[:, 0], S[0], VT[:, 0]

    return s, u.reshape(B_shape), v.reshape(C_shape)
