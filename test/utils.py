"""Utility functions for testing."""

from os import getenv, path
from typing import List

from pdf2image import convert_from_path

# important directories
HEREDIR = path.dirname(path.abspath(__file__))
REPO_DIR = path.dirname(HEREDIR)
DOC_ASSETS_DIR = path.join(REPO_DIR, "docs", "api", "assets")

# whether we are in a Python session on Github which does not have access to `pdflatex`
RUNNING_IN_CI = getenv("GITHUB_ACTIONS")


def convert_pdf_to_png(pdf_path: str) -> List[str]:
    """Convert a PDF to a PNG.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of paths to the PNG files.
    """
    images = convert_from_path(pdf_path)
    filenames = (
        [pdf_path.replace(".pdf", ".png")]
        if len(images) == 1
        else [pdf_path.replace(".pdf", f"_{i}.png") for i in range(len(images))]
    )
    for image, filename in zip(images, filenames):
        image.save(filename)

    return filenames
