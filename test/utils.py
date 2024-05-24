"""Utility functions for testing."""

from os import getenv, path

from pdf2image import convert_from_path

# important directories
HEREDIR = path.dirname(path.abspath(__file__))
REPO_DIR = path.dirname(HEREDIR)
DOC_ASSETS_DIR = path.join(REPO_DIR, "docs", "api", "assets")

# whether we are in a Python session on Github which does not have access to `pdflatex`
RUNNING_IN_CI = getenv("GITHUB_ACTIONS")


def convert_pdf_to_png(pdf_path: str):
    """Convert a PDF to a PNG.

    Args:
        pdf_path: Path to the PDF file.
    """
    (image,) = convert_from_path(pdf_path)
    image.save(pdf_path.replace(".pdf", ".png"))
