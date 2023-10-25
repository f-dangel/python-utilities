"""Contains functionality to visualize matrices."""

from itertools import product
from os import path
from subprocess import run
from typing import Union

from numpy import ndarray
from torch import Tensor


class TikzMatrix:
    """Class to visualize a matrix with TikZ.

    Attributes:
        TEMPLATE: Template TikZ code containing placeholders that will be substituted
            with content when saving a figure.

    Examples:
        >>> from numpy import linspace
        >>> mat = linspace(0, 1, num=9).reshape(3, 3)
        >>> savepath = "mat.tex"
        >>> # NOTE to compile, you need `pdflatex`
        >>> TikzMatrix(mat).save(savepath, compile=False)
    """

    TEMPLATE: str = r"""
\documentclass[tikz]{standalone}

\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{colorbrewer}
\usepgfplotslibrary{colormaps}

\begin{document}

\pgfkeys{/pgfplots/canvasonly/.style={
    enlarge x limits = 0,
    enlarge y limits = 0,
    grid=PLACEHOLDER_GRID,
    xtick = {0, 1, ..., PLACEHOLDER_COLS},
    ytick = {0, 1, ..., PLACEHOLDER_ROWS},
    yticklabels={,,},
    xticklabels={,,},
    yticklabel style = {
      opacity = 0,
      scale = 0.0001 % can't use exact 0
    },
    xticklabel style = {
      opacity = 0,
      scale = 0.0001 % can't use exact 0
    },
    ylabel style = {scale = 0},
    xlabel style = {scale = 0},
    xtick style = {opacity = 0},
    ytick style = {opacity = 0},
  }}

% from https://tex.stackexchange.com/a/361606
\begin{tikzpicture}
  \begin{axis}[
    % for an overview of color maps, follow the links of
    % https://tex.stackexchange.com/a/350927
    PLACEHOLDER_COLORMAP,
    canvasonly,
    ]
    \addplot[
    matrix plot,
    mesh/cols=PLACEHOLDER_COLS,
    mesh/rows=PLACEHOLDER_ROWS,
    point meta=explicit,
    PLACEHOLDER_COLORBAR_MIN,
    PLACEHOLDER_COLORBAR_MAX,
    ] table[meta=z] {
PLACEHOLDER_CONTENT
    };
  \end{axis}
\end{tikzpicture}

\end{document}

%%% Local Variables:
%%% mode: latex
%%% TeX-master: t
%%% End:
"""

    def __init__(self, mat: Union[Tensor, ndarray]) -> None:
        """Store the matrix internally.

        Args:
            mat: The matrix that will be visualized as PyTorch tensor or NumPy array.

        Raises:
            ValueError: If the supplied array is not 2d.
        """
        if mat.ndim != 2:
            raise ValueError(f"Expected 2d array. Got {mat.ndim}d.")
        self.mat = mat

        self.grid = "major"
        self.colormap = "colormap/blackwhite"
        self.vmin: Union[float, None] = 0.0  # color bar minimum
        self.vmax: Union[float, None] = 1.0  # color bar maximum

    def save(self, savepath: str, compile: bool = False):
        """Save the matrix plot as standalone TikZ figure and maybe build to pdf.

        Args:
            savepath: Path to save the figure to (including `'.tex'`).
            compile: Whether to compile the TikZ figure to pdf. Default is `False`.
        """
        template = self.TEMPLATE
        rows, cols = self.mat.shape

        content = ["x\ty\tz"]
        content.extend(
            f"{float(col + 0.5)}\t{float(row + 0.5)}\t{self.mat[row, col]}"
            for row, col in product(range(rows), range(cols))
        )
        content = "\n".join(content)

        for placeholder, replacement in [
            ("PLACEHOLDER_CONTENT", content),
            ("PLACEHOLDER_ROWS", str(rows)),
            ("PLACEHOLDER_COLS", str(cols)),
            ("PLACEHOLDER_COLORMAP", self.colormap),
            ("PLACEHOLDER_GRID", self.grid),
            (
                "PLACEHOLDER_COLORBAR_MIN",
                f"% color bar minimum\n    point meta min={self.vmin}"
                if self.vmin is not None
                else "",
            ),
            (
                "PLACEHOLDER_COLORBAR_MAX",
                f"% color bar maximum\n    point meta max={self.vmax}"
                if self.vmax is not None
                else "",
            ),
        ]:
            template = template.replace(placeholder, replacement)

        with open(savepath, "w") as f:
            f.write(template)

        if compile:
            cmd = [
                "pdflatex",
                "-output-directory",
                path.dirname(savepath),
                path.basename(savepath),
            ]
            run(cmd, check=True)
