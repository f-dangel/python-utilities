"""Test visualization of matrices with TikZ."""

from os import path
from test.utils import DOC_ASSETS_DIR, RUNNING_IN_CI, convert_pdf_to_png

from torch import linspace

from fdpyutils.tikz.matshow import TikzMatrix


def test_TikzMatrix():
    """Visualize a matrix with TikZ for the documentation."""
    mat = linspace(0, 1, 30).reshape(3, 10)
    savepath = path.join(DOC_ASSETS_DIR, "TikzMatrix.tex")
    tikz_mat = TikzMatrix(mat)
    tikz_mat.highlight(1, 0, fill="blue", fill_opacity=0.5)
    tikz_mat.save(savepath, compile=not RUNNING_IN_CI)
    if not RUNNING_IN_CI:
        convert_pdf_to_png(savepath.replace(".tex", ".pdf"))
