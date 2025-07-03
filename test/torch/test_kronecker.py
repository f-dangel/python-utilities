"""Contains tests of `fdpyutils.torch.kronecker`."""

from test.torch.utils import DEVICE_IDS, DEVICES
from typing import Tuple

from pytest import mark
from torch import Tensor, allclose, device, dtype, float64, kron, manual_seed, rand
from torch.linalg import matrix_norm
from torch.optim import LBFGS

from fdpyutils.torch.kronecker import best_kronecker


def fit_best_kronecker(
    A: Tensor, B_shape: Tuple[int, int], C_shape: Tuple[int, int], num_steps: int = 200
) -> Tuple[Tensor, Tensor]:
    """Fit the best Kronecker approximation B ⊗ C of A using LBFGS.

    Args:
        A: The matrix to approximate.
        B_shape: The shape of the first tensor in the Kronecker product.
        C_shape: The shape of the second tensor in the Kronecker product.
        num_steps: The number of optimization steps to take. Default is `200`.

    Returns:
        The matrices B and C that best approximate A in terms of the Frobenius norm.
    """
    B = rand(B_shape, dtype=A.dtype, device=A.device, requires_grad=True)
    C = rand(C_shape, dtype=A.dtype, device=A.device, requires_grad=True)

    optimizer = LBFGS([B, C])

    def _closure() -> Tensor:
        """Closure for computing the Frobenius norm of the residual A - B ⊗ C."""
        optimizer.zero_grad()
        error = matrix_norm(A - kron(B, C))
        error.backward()
        return error

    for _ in range(num_steps):
        optimizer.step(closure=_closure)

    return B.detach_(), C.detach_()


B_C_SHAPES = [
    [(2, 3), (4, 5)],
    [(4, 25), (20, 5)],
    [(5, 5), (5, 5)],
    [(5, 6), (5, 6)],
]
B_C_IDS = [
    f"({'x'.join(str(s) for s in B_shape)})({'x'.join(str(s) for s in C_shape)})"
    for B_shape, C_shape in B_C_SHAPES
]

SVD_BACKENDS = ["torch", "scipy"]
SVD_BACKEND_IDS = [f"svd_backend={b}" for b in SVD_BACKENDS]


@mark.parametrize("svd_backend", SVD_BACKENDS, ids=SVD_BACKEND_IDS)
@mark.parametrize("B_shape, C_shape", B_C_SHAPES, ids=B_C_IDS)
@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
def test_best_kronecker(
    B_shape: Tuple[int, int],
    C_shape: Tuple[int, int],
    dev: device,
    svd_backend: str,
    dt: dtype = float64,
):
    """Test finding the best Kronecker approximation B ⊗ C of a matrix A.

    The matrix A will be randomly generated based on the shapes of B and C.

    Args:
        B_shape: The shape of the first tensor in the Kronecker product.
        C_shape: The shape of the second tensor in the Kronecker product.
        dev: The device to run the test on.
        svd_backend: The backend to use for SVD.
        dt: The data type to run the test in. Default is `float64`.
    """
    manual_seed(0)

    # generate random matrix A
    (B_rows, B_cols), (C_rows, C_cols) = B_shape, C_shape
    A = rand(B_rows * C_rows, B_cols * C_cols, device=dev, dtype=dt)

    # fit best Kronecker using LBFGS
    B_fit, C_fit = fit_best_kronecker(A, B_shape, C_shape)
    # compute best Kronecker using top singular vector
    alpha, B, C = best_kronecker(A, B_shape, C_shape, svd_backend=svd_backend)

    # compare
    best = alpha * kron(B, C)
    best_fit = kron(B_fit, C_fit)
    assert allclose(best, best_fit, atol=1e-7, rtol=1e-4)
