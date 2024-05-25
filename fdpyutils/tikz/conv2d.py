"""Visualizing 2d convolution with TikZ."""

from itertools import product
from os import makedirs, path
from subprocess import run
from typing import Tuple

from einconv import index_pattern
from einops import einsum, rearrange
from torch import Tensor
from torch.nn.functional import pad

from fdpyutils.tikz.matshow import TikzMatrix


class TikzConv2d:
    """Class for visualizing 2d convolutions with TikZ.

    Examples:
        >>> from torch import manual_seed, rand
        >>> manual_seed(0)
        >>> # convolution hyper-parameters
        >>> N, C_in, I1, I2 = 2, 2, 4, 5
        >>> G, C_out, K1, K2 = 1, 3, 2, 3
        >>> P = (0, 1) # non-zero padding along one dimension
        >>> weight = rand(C_out, C_in // G, K1, K2)
        >>> x = rand(N, C_in, I1, I2)
        >>> # NOTE to compile, you need `pdflatex`
        >>> TikzConv2d(weight, x, "conv2d", padding=P).save(compile=False)

    - Example image (padded pixels are highlighted)
      ![](assets/TikzConv2d/example.png)
    - I used this code to create the visualizations for my
      [talk](https://pirsa.org/23120027) at Perimeter Institute.
    """

    def __init__(
        self,
        weight: Tensor,
        x: Tensor,
        savedir: str,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
    ):
        """Store convolution tensors and hyper-parameters for the animated convolution.

        Args:
            weight: Convolution kernel. Has shape `[C_out, C_in // G, K1, K2]`.
            x: Input tensor. Has shape `[N, C_in, I1, I2]`.
            savedir: Directory under which the TikZ code and pdf images are saved.
            stride: Stride of the convolution. Default: `(1, 1)`.
            padding: Padding of the convolution. Default: `(0, 0)`.
            dilation: Dilation of the convolution. Default: `(1, 1)`.

        Raises:
            ValueError: If `weight` or `x` are not 4d tensors.
        """
        if weight.ndim != 4 or x.ndim != 4:
            raise ValueError(
                f"Expected 4d `weight` and `x`, but got {weight.ndim}d and {x.ndim}d."
            )
        self.savedir = savedir
        self.fibresdir = path.join(savedir, "static_fibres")
        self.tensordir = path.join(savedir, "static_tensors")

        # store hyper-parameters
        self.N, self.C_in, self.I1, self.I2 = x.shape
        self.C_out, _, self.K1, self.K2 = weight.shape
        self.G = self.C_in // weight.shape[1]
        self.S1, self.S2 = stride
        self.P1, self.P2 = padding
        self.D1, self.D2 = dilation

        # explicitly pad x
        x = pad(x, (self.P2, self.P2, self.P1, self.P1))

        # convolution index pattern tensors and output sizes
        self.pattern1 = index_pattern(
            self.I1 + 2 * self.P1, self.K1, stride=self.S1, padding=0, dilation=self.D1
        )
        self.pattern2 = index_pattern(
            self.I2 + 2 * self.P2, self.K2, stride=self.S2, padding=0, dilation=self.D2
        )
        self.O1 = self.pattern1.shape[1]
        self.O2 = self.pattern2.shape[1]

        # store all tensors with separated channel groups
        print(x.shape)
        print(self.G)
        x = rearrange(x, "n (g c_in) i1 i2 -> n g c_in i1 i2", g=self.G)
        weight = rearrange(
            weight, "(g c_out) c_in k1 k2 -> g c_out c_in k1 k2", g=self.G
        )
        output = einsum(
            x,
            self.pattern1.float(),
            self.pattern2.float(),
            weight,
            "n g c_in i1 i2, k1 o1 i1, k2 o2 i2, g c_out c_in k1 k2 -> n g c_out o1 o2",
        )

        # normalize all tensors to use the colour map's full range
        self.x = self.normalize(x)
        self.weight = self.normalize(weight)
        self.output = self.normalize(output)

    def save(self, compile: bool = True):
        """Create the images of the convolution's input, weight, and output tensors.

        This is done in three steps:

        1.) Generate the fibres (matrix slices) of the input/weight/output tensors.
        2.) Combine the fibres into the full tensors.
        3.) Compile the full tensors into one pdf image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image. Default: `True`.
        """
        self._generate_fibres(compile=compile)
        self._generate_tensors(compile=compile)

        TEX_TEMPLATE = r"""\documentclass[tikz]{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}
  \node (input) {\includegraphics{DATAPATH/static_tensors/input}};
  \node [right=1cm of input] (star) {$\star$};
  \node [right=1cm of star] (kernel) {\includegraphics{DATAPATH/static_tensors/weight}};
  \node [right=1cm of kernel] (equal) {$=$};
  \node [right=1cm of equal] (output) {\includegraphics{DATAPATH/static_tensors/output}};
\end{tikzpicture}
\end{document}"""
        code = TEX_TEMPLATE.replace("DATAPATH", self.savedir)
        savepath = path.join(self.savedir, "example.tex")
        self.write(code, savepath, compile=compile)
        with open(savepath, "w") as f:
            f.write(code)
        if compile:
            run(["pdflatex", "-output-directory", self.savedir, savepath])

    def _generate_fibres(self, compile: bool) -> None:
        """Visualize the input/output/weight fibres.

        A fibre is a tensors slice of dimension 2. TikZ code for fibres is generated
        and compiled separately. Fibres are compiled into pdf images and saved in the
        `static_fibres` sub-directory.

        Args:
            compile: Whether to compile the TikZ code into a pdf image.
        """
        makedirs(self.fibresdir, exist_ok=True)

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))

        # plot the input fibres
        for n, g, c_in in product(N_range, G_range, C_in_range):
            savepath = path.join(self.fibresdir, f"input_n_{n}_g_{g}_c_in_{c_in}.tex")
            matrix = self.custom_tikz_matrix(self.x[n, g, c_in])
            self.highlight_padding(matrix)
            matrix.save(savepath, compile=compile)

        # plot the weight fibres
        for g, c_out, c_in in product(G_range, C_out_range, C_in_range):
            savepath = path.join(
                self.fibresdir, f"weight_g_{g}_c_out_{c_out}_c_in_{c_in}.tex"
            )
            self.custom_tikz_matrix(self.weight[g, c_out, c_in]).save(
                savepath, compile=compile
            )

        # plot the output fibres
        for n, g, c_out in product(N_range, G_range, C_out_range):
            savepath = path.join(
                self.fibresdir, f"output_n_{n}_g_{g}_c_out_{c_out}.tex"
            )
            self.custom_tikz_matrix(self.output[n, g, c_out]).save(
                savepath, compile=compile
            )

    def _generate_tensors(self, compile: bool):
        """Visualize the input/output/weight tensors.

        Requires that the fibres have been generated first by calling `self.create_fibres`.
        This function combines the fibres into a single pdf image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image.
        """
        makedirs(self.tensordir, exist_ok=True)

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))

        # generic template for compiling tensors to .pdf
        TEX_TEMPLATE = r"""\documentclass[tikz]{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}
TENSOR
\end{tikzpicture}
\end{document}"""

        # combine the output fibres into a tensor
        savepath = path.join(self.tensordir, "output.tex")

        # data points are arranged vertically,
        # each is a node containing a stack of channels
        data_points = []
        for n in N_range:
            nodes = []
            for g, c_out in product(G_range[::-1], C_out_range[::-1]):
                fibre = path.join(
                    self.fibresdir, f"output_n_{n}_g_{g}_c_out_{c_out}.pdf"
                )
                xshift = (1.6 + 0.6 * (self.C_out // self.G - 1)) * g + 0.6 * c_out
                yshift = (2 + 0.8 * (self.C_out // self.G - 1)) * g + 0.8 * c_out
                nodes.append(
                    r"\n"
                    + f"ode [opacity=0.9, xshift={xshift}cm, yshift={yshift}cm]"
                    + " {"
                    + r"\includegraphics{"
                    + fibre
                    + "}};"
                )
            node = (
                r"""\node [anchor=north] at (current bounding box.south) {%
\begin{tikzpicture}"""
                + "\n\t".join(nodes)
                + "\n"
                + r"\end{tikzpicture}"
                + "\n"
                + r"};%"
            )
            data_points.append(node)

        code = TEX_TEMPLATE.replace("TENSOR", "\n".join(data_points))
        self.write(code, savepath, compile=compile)

        # combine the input fibres into a tensor
        savepath = path.join(self.tensordir, "input.tex")

        # data points are placed vertically,
        # each is a node containing a stack of channels
        data_points = []
        for n in N_range:
            nodes = []
            for g, c_in in product(G_range[::-1], C_in_range[::-1]):
                fibre = path.join(self.fibresdir, f"input_n_{n}_g_{g}_c_in_{c_in}.pdf")
                xshift = (1.6 + 0.6 * (self.C_in // self.G - 1)) * g + 0.6 * c_in
                yshift = (2 + 0.8 * (self.C_in // self.G - 1)) * g + 0.8 * c_in
                nodes.append(
                    r"\n"
                    + f"ode [opacity=0.9, xshift={xshift}cm, yshift={yshift}cm]"
                    + " {"
                    + r"\includegraphics{"
                    + fibre
                    + "}};"
                )
            node = (
                r"""\node [anchor=north] at (current bounding box.south) {%
\begin{tikzpicture}"""
                + "\n\t".join(nodes)
                + "\n"
                + r"\end{tikzpicture}"
                + "\n"
                + r"};%"
            )
            data_points.append(node)

        code = TEX_TEMPLATE.replace("TENSOR", "\n".join(data_points))
        self.write(code, savepath, compile=compile)

        # combine the weight fibres into a tensor
        weight_savepath = path.join(self.tensordir, "weight.tex")

        # the kernel elements that produce one output channel are stacked vertically,
        # each such kernel element is a node containing a stack of fibres
        # plot the weight tensorfibres for the current frame
        commands = []
        for c_out in C_out_range:
            nodes = []
            for g, c_in in product(G_range[::-1], C_in_range[::-1]):
                fibre = path.join(
                    self.fibresdir, f"weight_g_{g}_c_out_{c_out}_c_in_{c_in}.pdf"
                )
                xshift = (1.6 + 0.6 * (self.C_in // self.G - 1)) * g + 0.6 * c_in
                yshift = (2 + 0.8 * (self.C_in // self.G - 1)) * g + 0.8 * c_in
                nodes.append(
                    r"\n"
                    + f"ode [opacity=0.9, xshift={xshift}cm, yshift={yshift}cm]"
                    + " {"
                    + r"\includegraphics{"
                    + fibre
                    + "}};"
                )

            node = (
                r"""\node [anchor=north] at (current bounding box.south) {%
\begin{tikzpicture}"""
                + "\n\t".join(nodes)
                + "\n"
                + r"\end{tikzpicture}"
                + "\n"
                + r"};%"
            )
            commands.append(node)

        code = TEX_TEMPLATE.replace("TENSOR", "\n".join(commands))
        self.write(code, weight_savepath, compile=compile)

    def highlight_padding(
        self, matrix: TikzMatrix, color: str = "VectorBlue"
    ) -> TikzMatrix:
        """Highlight the padding pixels in a TikZ matrix of a 2d slice of the input.

        Args:
            matrix: TikZ matrix of a 2d slice of the input.
            color: Colour to highlight the padding pixels with. Defaults to `"VectorBlue"`.
        """
        # for shorter notation
        I1, I2 = self.I1, self.I2
        P1, P2 = self.P1, self.P2

        highlight = []
        for i1, i2 in product(range(I1 + 2 * P1), range(I2 + 2 * P2)):
            if i1 < P1 or i1 >= P1 + I1 or i2 < P2 or i2 >= P2 + I2:
                highlight.extend(({"x": i2, "y": i1, "fill": color},))
        for kwargs in highlight:
            self.highlight(matrix, **kwargs)

        return matrix

    @staticmethod
    def normalize(tensor: Tensor) -> Tensor:
        """Normalize values of a tensor to [0, 1].

        Args:
            tensor: Input tensor.

        Returns:
            Normalized tensor.
        """
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    @staticmethod
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

    @staticmethod
    def highlight(
        matrix: TikzMatrix,
        x: int,
        y: int,
        fill: str = "VectorOrange",
        fill_opacity: float = 0.5,
    ) -> None:
        """Highlight a pixel in the matrix.

        Args:
            matrix: `TikzMatrix` object whose represented matrix's pixel is higlighted.
            x: column index of the pixel to highlight.
            y: row index of the pixel to highlight.
            fill: colour to fill the pixel with. Default: `VectorOrange`.
            fill_opacity: opacity of the highlighting. Default: `0.5`.
        """
        matrix.extra_commands.append(
            rf"\draw [line width=0, fill={fill}, fill opacity={fill_opacity}] "
            + f"({x}, {y}) rectangle ++(1, 1);"
        )

    @staticmethod
    def write(content: str, savepath: str, compile: bool = True):
        """Write content to a file and compile it using pdflatex.

        Args:
            content: Content to write.
            savepath: Path to save the content.
            compile: Whether to compile the content using pdflatex. Default: `True`.
        """
        with open(savepath, "w") as f:
            f.write(content)
        if compile:
            run(["pdflatex", "-output-directory", path.dirname(savepath), savepath])
