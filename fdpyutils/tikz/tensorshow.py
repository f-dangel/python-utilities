"""Contains functionality to visualize tensors."""

from itertools import product
from os import path
from typing import Union

from numpy import ndarray
from torch import Tensor

from fdpyutils.tikz.matshow import custom_tikz_matrix
from fdpyutils.tikz.utils import write


class TikzTensor:
    """Class to visualize a tensor with TikZ.

    Attributes:
        TEMPLATE: Template TikZ code containing placeholders that will be replaced
            with content before compilation.

    Examples:
        >>> from torch import linspace, Sizee
        >>> shape = Size((1, 2, 3, 4, 10))
        >>> tensor = linspace(0, 1, shape.numel()).reshape(shape)
        >>> savepath = "tensor.tex"
        >>> # NOTE to compile, you need `pdflatex`
        >>> TikzTensor(tensor).save(savepath, compile=False)

    - Example image
      ![](assets/TikzTensor.png)
    """

    TEMPLATE = r"""\documentclass[tikz]{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}
PLACEHOLDER
\end{tikzpicture}
\end{document}"""

    def __init__(self, tensor: Union[Tensor, ndarray]) -> None:
        """Store the tensor internally.

        Args:
            tensor: The tensor to visualize. At most 5d.

        Raises:
            NotImplementedError: If the tensor has more than 5 dimensions.
        """
        max_dim = 5
        ndim = len(tensor.shape)
        if ndim > max_dim:
            raise NotImplementedError(
                f"(d>{max_dim})-tensors are not supported. Got {ndim}d."
            )

        # reshape the tensor into 5d if it has less dimensions
        missing = max_dim - ndim
        shape = (1,) * missing + tensor.shape
        self.tensor = tensor.reshape(shape)

    def save(self, savepath: str, compile: bool = True) -> None:
        """Save the TikZ code to visualize the tensor to a file and maybe compile it.

        The approach has two stages:

        1. Generate `.tex` and `.pdf` files for 2d fibres of the tensor.
        2. Compose the fibres into a TikZ picture displaying the entire tensor.

        Args:
            savepath: The path to save the TikZ code to.
            compile: Whether to compile the TikZ code to a PDF.
        """
        self._generate_fibres(savepath, compile=compile)
        self._combine_fibres(savepath, compile=compile)

    def _generate_fibres(self, savepath: str, compile: bool) -> None:
        """Visualize the tensor fibres.

        A fibre is a tensors slice of dimension 2. TikZ code for fibres is generated
        and compiled separately. Fibres are compiled into pdf images and saved in the
        `fibres` sub-directory.

        Args:
            savepath: The path under which the tensor will be saved.
            compile: Whether to compile the TikZ code into a pdf image.
        """
        fibresdir = path.join(path.basename(savepath), "fibres")
        D1, D2, D3, _, _ = self.tensor.shape

        for d1, d2, d3 in product(range(D1), range(D2), range(D3)):
            savepath = path.join(fibresdir, f"fibre_{d1}_{d2}_{d3}.tex")
            custom_tikz_matrix(self.tensor[d1, d2, d3]).save(savepath, compile=compile)

    def _combine_fibres(self, savepath: str, compile: bool = True) -> None:
        """Create TikZ image that lays out the tensor's fibres and maybe compile.

        Args:
            savepath: The path where the TikZ code will be stored.
            compile: Whether to compile the TikZ code into a pdf image.
        """
        fibresdir = path.join(path.basename(savepath), "fibres")
        D1, D2, D3, _, _ = self.tensor.shape

        # first dimension is laid out vertically
        vertical = []
        for d1 in range(D1):
            # second and third dimensions are laid out depthwise
            depthwise = []
            for d2, d3 in product(list(range(D2))[::-1], list(range(D3))[::-1]):
                fibre = path.join(fibresdir, f"fibre_{d1}_{d2}_{d3}.pdf")
                xshift = (1.6 + 0.6 * (D3 - 1)) * d2 + 0.6 * d3
                yshift = (2 + 0.8 * (D3 - 1)) * d2 + 0.8 * d3
                depthwise.append(
                    r"\n"
                    + f"ode [opacity=0.9, xshift={xshift}cm, yshift={yshift}cm]"
                    + r" {\includegraphics{"
                    + fibre
                    + "}};"
                )
            vertical.append(
                r"""\node [anchor=north] at (current bounding box.south) {%
\begin{tikzpicture}
TENSOR
\end{tikzpicture}
};%""".replace(
                    "TENSOR", "\n\t".join(depthwise)
                )
            )

        code = self.TEMPLATE.replace("PLACEHOLDER", "\n".join(vertical))
        write(code, savepath, compile=compile)
