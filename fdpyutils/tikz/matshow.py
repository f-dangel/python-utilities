"""Contains functionality to visualize matrices."""

from itertools import product
from typing import Union

from numpy import ndarray
from torch import Tensor

from fdpyutils.tikz.utils import write


class TikzMatrix:
    """Class to visualize a matrix with TikZ.

    Attributes:
        TEMPLATE: Template TikZ code containing placeholders that will be substituted
            with content when saving a figure.

    Examples:
        >>> from numpy import linspace
        >>> mat = linspace(0, 1, num=30).reshape(3, 10)
        >>> savepath = "mat.tex"
        >>> tikz_mat = TikzMatrix(mat)
        >>> tikz_mat.highlight(1, 0, fill="blue", fill_opacity=0.5)
        >>> # NOTE to compile, you need `pdflatex`
        >>> tikz_mat.save(savepath, compile=False)

    - Example image
      ![](assets/TikzMatrix.png)
    - I used this to visualize the structured matrices in our
      [SINGD paper](https://arxiv.org/pdf/2312.05705).
    """

    TEMPLATE: str = r"""
\documentclass[tikz]{standalone}

\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{colorbrewer}
\usepgfplotslibrary{colormaps}

PLACEHOLDER_EXTRA_PREAMBLE

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
    % explicit limits required for plotting row/column vectors
    xmin=0, xmax=PLACEHOLDER_XMAX,
    ymin=0, ymax=PLACEHOLDER_YMAX,
    % explicit width and height to preserve aspect ratio of the matrix
    width=PLACEHOLDER_WIDTHcm, height=PLACEHOLDER_HEIGHTcm,
    scale only axis,
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
PLACEHOLDER_EXTRA_COMMANDS
  \end{axis}
\end{tikzpicture}

\end{document}"""

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

        self.extra_commands = []
        self.extra_preamble = []

    def highlight(
        self, column: int, row: int, fill: str = "red", fill_opacity: float = 0.5
    ) -> None:
        """Highlight a pixel in the matrix.

        Args:
            column: column index of the pixel to highlight.
            row: row index of the pixel to highlight.
            fill: colour to fill the pixel with. Default: `'red'`.
            fill_opacity: opacity of the highlighting. Default: `0.5`.
        """
        self.extra_commands.append(
            rf"\draw [line width=0, fill={fill}, fill opacity={fill_opacity}] "
            + f"({column}, {row}) rectangle ++(1, 1);"
        )

    def save(self, savepath: str, compile: bool = False):
        """Save the matrix plot as standalone TikZ figure and maybe build to pdf.

        Args:
            savepath: Path to save the figure to (including `'.tex'`).
            compile: Whether to compile the TikZ figure to pdf. Default is `False`.
        """
        template = self.TEMPLATE

        # NOTE matrix plot requires two rows and columns to figure out the cell size.
        # For single rows/columns we expand them into two and then limit their
        # visibility by setting the axis limits.
        y_max, x_max = self.mat.shape

        mat = self.mat
        if y_max == 1:
            assert isinstance(mat, Tensor)
            mat = mat.expand(2, -1)
        elif x_max == 1:
            assert isinstance(mat, Tensor)
            mat = mat.expand(-1, 2)
        rows, cols = mat.shape

        content = ["x\ty\tz"]
        content.extend(
            f"{float(col + 0.5)}\t{float(row + 0.5)}\t{mat[row, col]}"
            for row, col in product(range(rows), range(cols))
        )
        content = "\n".join(content)

        for placeholder, replacement in [
            ("PLACEHOLDER_CONTENT", content),
            ("PLACEHOLDER_XMAX", str(x_max)),
            ("PLACEHOLDER_YMAX", str(y_max)),
            ("PLACEHOLDER_WIDTH", str(2 * x_max)),
            ("PLACEHOLDER_HEIGHT", str(2 * y_max)),
            ("PLACEHOLDER_ROWS", str(rows)),
            ("PLACEHOLDER_COLS", str(cols)),
            ("PLACEHOLDER_COLORMAP", self.colormap),
            ("PLACEHOLDER_GRID", self.grid),
            ("PLACEHOLDER_EXTRA_COMMANDS", "\n".join(self.extra_commands)),
            ("PLACEHOLDER_EXTRA_PREAMBLE", "\n".join(self.extra_preamble)),
            (
                "PLACEHOLDER_COLORBAR_MIN",
                (
                    f"% color bar minimum\n    point meta min={self.vmin}"
                    if self.vmin is not None
                    else ""
                ),
            ),
            (
                "PLACEHOLDER_COLORBAR_MAX",
                (
                    f"% color bar maximum\n    point meta max={self.vmax}"
                    if self.vmax is not None
                    else ""
                ),
            ),
        ]:
            template = template.replace(placeholder, replacement)

        write(template, savepath, compile=compile)


def custom_tikz_matrix(mat: Tensor) -> TikzMatrix:
    """Create `TikzMatrix` object with custom settings for visualizing matrices.

    We specify the colour map and add colour definitions to the preamble which
    are used for highlighting pixels.

    Args:
        mat: Matrix to visualize.

    Returns:
        `TikzMatrix` object with custom settings for visualizing the matrix.
    """
    matrix = TikzMatrix(mat)
    matrix.colormap = "colormap/Greys"
    matrix.extra_preamble.extend(
        [
            r"\definecolor{VectorBlue}{RGB}{59, 69, 227}",
            r"\definecolor{VectorPink}{RGB}{253, 8, 238}",
            r"\definecolor{VectorOrange}{RGB}{250, 173, 26}"
            r"\definecolor{VectorTeal}{RGB}{82, 199, 222}",
        ]
    )
    return matrix
