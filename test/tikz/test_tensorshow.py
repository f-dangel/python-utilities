"""Test visualization of tensors with TikZ."""

from os import path
from test.utils import DOC_ASSETS_DIR, RUNNING_IN_CI, convert_pdf_to_png

from torch import Size, linspace

from fdpyutils.tikz.tensorshow import TikzTensor


def test_TikzTensor():
    """Visualize a tensor with TikZ for the documentation."""
    shape = Size((1, 2, 3, 4, 10))
    tensor = linspace(0, 1, shape.numel()).reshape(shape)
    savepath = path.join(DOC_ASSETS_DIR, "TikzTensor.tex")
    tikz_tensor = TikzTensor(tensor)
    tikz_tensor.highlight((0, 0, 0, 3, 1), fill="green", fill_opacity=0.5)
    tikz_tensor.save(savepath, compile=not RUNNING_IN_CI)
    if not RUNNING_IN_CI:
        convert_pdf_to_png(savepath.replace(".tex", ".pdf"))
