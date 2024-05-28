"""Utility functions for TikZ files."""

from os import makedirs, path
from subprocess import run


def write(code: str, savepath: str, compile: bool = True) -> None:
    """Write the TikZ code to a file and maybe compile it.

    Args:
        code: The TikZ code to write.
        savepath: The path to save the TikZ code to.
        compile: Whether to compile the TikZ code to a PDF.
    """
    savedir = path.dirname(savepath)
    makedirs(savedir, exist_ok=True)

    with open(savepath, "w") as f:
        f.write(code)

    if compile:
        run(["pdflatex", "-output-directory", savedir, savepath])
