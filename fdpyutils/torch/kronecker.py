"""Kronecker product related functionality."""

from typing import Tuple

from einops import rearrange
from scipy.sparse.linalg import svds
from torch import Tensor, from_numpy


def best_kronecker(
    A: Tensor,
    B_shape: Tuple[int, int],
    C_shape: Tuple[int, int],
    svd_backend: str = "torch",
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Find the best approximation $\alpha (B \otimes C)$ of $A$.

    'Best' means in terms of Frobenius norm.
    Requires computing the top singular vectors of a reshape of $A$.
    See [this paper](\
    https://typeset.io/pdf/approximation-with-kronecker-products-24urjmqom7.pdf)
    for details.

    Args:
        A: The matrix to approximate.
        B_shape: The shape of the first tensor in the Kronecker product.
        C_shape: The shape of the second tensor in the Kronecker product.
        svd_backend: The backend to use for SVD. Defaults to 'torch', which computes
            a full SVD. The alternative choice is 'scipy', which loads the matrix to
            CPU and uses SciPy's truncated SVD `scipy.sparse.linalg.svds` that may
            require fewer matrix-vector products.

    Returns:
        The scalar $\alpha$ and the matrices $B, C$ such that $\alpha (B \otimes C)$
            is the best Kronecker approximation of $A$.

    Raises:
        ValueError: If A is not a matrix.
        ValueError: If `B_shape` or `C_shape` are not 2-tuples.
        ValueError: If the shapes multiply to the incorrect total dimension.
        ValueError: If the value of `svd_backend` is unsupported.

    Examples:
        >>> from torch import kron, rand, manual_seed
        >>> _ = manual_seed(0) # make deterministic
        >>> # generate a random Kronecker product
        >>> shape_B, shape_C = (2, 3), (4, 5)
        >>> A = kron(rand(*shape_B), rand(*shape_C))
        >>> # find the best Kronecker approximation and check reconstruction matches
        >>> alpha, B, C = best_kronecker(A, shape_B, shape_C)
        >>> A.allclose(alpha * kron(B, C))
        True
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
    if svd_backend == "torch":
        U, S, VT = A_rearranged.svd()

    elif svd_backend == "scipy":
        A_rearranged_numpy = A_rearranged.detach().cpu().numpy()
        U, S, VT = svds(A_rearranged_numpy, k=1)
        # convert to PyTorch tensors (need to make a copy to avoid negative strides,
        # see https://discuss.pytorch.org/t/negative-strides-in-tensor-error/134287)
        U = from_numpy(U.copy()).to(A.device, A.dtype)
        S = from_numpy(S.copy()).to(A.device, A.dtype)
        VT = from_numpy(VT.T.copy()).to(A.device, A.dtype)

    else:
        raise ValueError(
            f"Unsupported svd_backend: {svd_backend}. Use 'torch' or 'scipy'."
        )

    u, s, v = U[:, 0], S[0], VT[:, 0]

    return s, u.reshape(B_shape), v.reshape(C_shape)
