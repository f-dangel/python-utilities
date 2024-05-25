"""Test visualization of 2d convolution."""

from os import path
from test.utils import DOC_ASSETS_DIR, RUNNING_IN_CI, convert_pdf_to_png

from torch import manual_seed, rand

from fdpyutils.tikz.conv2d import TikzConv2d

HEREDIR = path.dirname(path.abspath(__file__))


def test_TikzConv2d():
    """Visualize a 2d convolution with TikZ for the documentation."""
    manual_seed(0)
    N, C_in, I1, I2 = 2, 2, 4, 5
    G, C_out, K1, K2 = 1, 3, 2, 3
    P = (0, 1)
    weight = rand(C_out, C_in // G, K1, K2)
    x = rand(N, C_in, I1, I2)
    savedir = path.join(DOC_ASSETS_DIR, "TikzConv2d")
    TikzConv2d(weight, x, savedir, padding=P).save(compile=not RUNNING_IN_CI)
    if not RUNNING_IN_CI:
        convert_pdf_to_png(path.join(savedir, "example.pdf"))
