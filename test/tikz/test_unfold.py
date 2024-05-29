"""Test visualization of 2d convolution input unfolding."""

from os import path
from test.utils import DOC_ASSETS_DIR, RUNNING_IN_CI, convert_pdf_to_gif

from torch import manual_seed, rand

from fdpyutils.tikz.unfold import TikzUnfoldAnimated, TikzUnfoldWeightAnimated


def test_TikzUnfoldAnimated():
    """Visualize 2d convolution input unfolding with TikZ for the documentation."""
    manual_seed(0)
    N, C_in, I1, I2 = 1, 2, 5, 1
    G, C_out, K1, K2 = 1, 4, 3, 1
    P = (1, 0)
    weight = rand(C_out, C_in // G, K1, K2)
    x = rand(N, C_in, I1, I2)
    savedir = path.join(DOC_ASSETS_DIR, "TikzUnfoldAnimated")
    TikzUnfoldAnimated(weight, x, savedir, padding=P).save(
        compile=not RUNNING_IN_CI, max_frames=10
    )
    if not RUNNING_IN_CI:
        convert_pdf_to_gif(path.join(savedir, "example.pdf"))


def test_TikzUnfoldWeightAnimated():
    """Visualize 2d convolution weight unfolding with TikZ for the documentation."""
    manual_seed(0)
    N, C_in, I1, I2 = 1, 2, 5, 1
    G, C_out, K1, K2 = 1, 3, 4, 1
    P = (1, 0)
    weight = rand(C_out, C_in // G, K1, K2)
    x = rand(N, C_in, I1, I2)
    savedir = path.join(DOC_ASSETS_DIR, "TikzUnfoldWeightAnimated")
    TikzUnfoldWeightAnimated(weight, x, savedir, padding=P).save(
        compile=not RUNNING_IN_CI
    )
    if not RUNNING_IN_CI:
        convert_pdf_to_gif(path.join(savedir, "example.pdf"))
