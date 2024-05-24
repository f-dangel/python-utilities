"""Test visualization of matrices with TikZ."""

from os import path

from pdf2image import convert_from_path
from torch import linspace

from fdpyutils.tikz.matshow import TikzMatrix

HEREDIR = path.dirname(path.abspath(__file__))
REPO_DIR = path.dirname(path.dirname(HEREDIR))
DOC_ASSETS_DIR = path.join(REPO_DIR, "docs", "api", "assets")


def convert_pdf_to_png(pdf_path: str):
    """Convert a PDF to a PNG.

    Args:
        pdf_path: Path to the PDF file.
    """
    (image,) = convert_from_path(pdf_path)
    image.save(pdf_path.replace(".pdf", ".png"))


def test_TikzMatrix():
    """Visualize a matrix with TikZ for the documentation."""
    mat = linspace(0, 1, 30).reshape(3, 10)
    savepath = path.join(DOC_ASSETS_DIR, "TikzMatrix.tex")
    TikzMatrix(mat).save(savepath, compile=True)
    convert_pdf_to_png(savepath.replace(".tex", ".pdf"))
