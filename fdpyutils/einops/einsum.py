"""Einsum with support for ``einops``' index un-grouping syntax."""

from typing import Union

from einops import einsum as einops_einsum
from einops import rearrange
from einops.einops import Tensor


def einsum(*tensors_and_pattern: Union[Tensor, str], **axes_lengths: int) -> Tensor:
    """Same as ``einops.einsum`` but supports index un-grouping notation.

    For example, the following operation does not work (yet) in ``einops.einsum``, but
    works with this version: ``einsum(A, B, '(a b) c, a b c -> (a b) c', a=a, b=b)``.

    (More information [here](https://github.com/arogozhnikov/einops/blob/fcd36c9b017d52e452a301e75c1373b75ec23ee0/einops/einops.py#L833-L834)) # noqa: B950

    Args:
        tensors_and_pattern:
            tensors: tensors of any supported library (numpy, tensorflow, pytorch, jax).
            pattern: string, einsum pattern, with commas separating specifications for
                each tensor. Pattern should be provided after all tensors.
        axes_lengths: Length of axes that cannot be inferred.

    Returns:
        Tensor of the same type as input, after processing with einsum.

    Raises:
        NotImplementedError: If the pattern contains unsupported features.

    Examples:
        >>> from fdpyutils.einops.einsum import einsum
        >>> from torch import rand, allclose
        >>> a, b, c = 3, 4, 5 # dimensions
        >>> A, B = rand(a, b, c), rand(a * b, c) # operands
        >>> C = einsum(A, B, 'a b c, (a b) c -> (a b) c', a=a, b=b)
        >>> allclose(C, A.reshape(a * b, c) * B)
        True
    """
    try:
        return einops_einsum(*tensors_and_pattern)
    except NotImplementedError as e:
        tensors, pattern = tensors_and_pattern[:-1], tensors_and_pattern[-1]
        if "(" not in pattern or ")" not in pattern:
            raise NotImplementedError from e

        # un-group the operands
        lefts, right = pattern.split("->")
        lefts = lefts.split(",")
        lefts_ungrouped = [left.replace("(", "").replace(")", "") for left in lefts]
        tensors_ungrouped = [
            rearrange(t, " -> ".join([l, l_u]), **axes_lengths) if l != l_u else t
            for t, l, l_u in zip(tensors, lefts, lefts_ungrouped)
        ]

        # compute the result with un-grouped indices
        right_ungrouped = right.replace("(", "").replace(")", "")
        pattern_ungrouped = " -> ".join([",".join(lefts_ungrouped), right_ungrouped])
        result_ungrouped = einops_einsum(*tensors_ungrouped, pattern_ungrouped)

        # group the indices in the result tensor
        return (
            rearrange(
                result_ungrouped, " -> ".join([right_ungrouped, right]), **axes_lengths
            )
            if right_ungrouped != right
            else result_ungrouped
        )
