"""Tests for ``fdpyutils.einops.einsum``."""

from torch import allclose, manual_seed, rand

from fdpyutils.einops.einsum import einsum


def test_einsum():
    """Test einsum with support for index un-grouping syntax."""
    manual_seed(0)

    a, b, c = (3, 4, 5)
    A = rand(a, b, c)
    B = rand(a * b, c)

    # NOTE Need to specify dims ``a, b`` although they could be inferred
    C = einsum(A, B, "a b c, (a b) c -> (a b) c", a=a, b=b)
    C_truth = einsum(A, B.reshape(a, b, c), "a b c, a b c -> a b c").reshape(a * b, c)
    assert allclose(C, C_truth)
