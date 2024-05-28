"""Visualizing 2d convolution with TikZ."""

from itertools import product
from os import makedirs, path
from typing import Dict, Optional, Tuple

from einconv import index_pattern
from einops import einsum, rearrange
from torch import Tensor
from torch.nn.functional import pad

from fdpyutils.tikz.matshow import TikzMatrix
from fdpyutils.tikz.utils import write


class TikzConv2d:
    """Class for visualizing 2d convolutions with TikZ.

    Examples:
        >>> from torch import manual_seed, rand
        >>> _ = manual_seed(0)
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

        1. Generate the fibres (matrix slices) of the input/weight/output tensors.
        2. Combine the fibres into the full tensors.
        3. Compile the full tensors into one pdf image.

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
  \node [right=1cm of star] (kernel) {%
    \includegraphics{DATAPATH/static_tensors/weight}%
  };
  \node [right=1cm of kernel] (equal) {$=$};
  \node [right=1cm of equal] (output) {%
    \includegraphics{DATAPATH/static_tensors/output}%
  };
\end{tikzpicture}
\end{document}"""
        code = TEX_TEMPLATE.replace("DATAPATH", self.savedir)
        savepath = path.join(self.savedir, "example.tex")
        write(code, savepath, compile=compile)

    def _generate_fibres(self, compile: bool) -> None:
        """Visualize the input/output/weight fibres.

        A fibre is a tensors slice of dimension 2. TikZ code for fibres is generated
        and compiled separately. Fibres are compiled into pdf images and saved in the
        `static_fibres` sub-directory.

        Args:
            compile: Whether to compile the TikZ code into a pdf image.
        """
        fibresdir = path.join(self.savedir, "static_fibres")
        makedirs(fibresdir, exist_ok=True)

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))

        # plot the input fibres
        for n, g, c_in in product(N_range, G_range, C_in_range):
            savepath = path.join(fibresdir, f"input_n_{n}_g_{g}_c_in_{c_in}.tex")
            matrix = self.custom_tikz_matrix(self.x[n, g, c_in])
            self.highlight_padding(matrix)
            matrix.save(savepath, compile=compile)

        # plot the weight fibres
        for g, c_out, c_in in product(G_range, C_out_range, C_in_range):
            savepath = path.join(
                fibresdir, f"weight_g_{g}_c_out_{c_out}_c_in_{c_in}.tex"
            )
            self.custom_tikz_matrix(self.weight[g, c_out, c_in]).save(
                savepath, compile=compile
            )

        # plot the output fibres
        for n, g, c_out in product(N_range, G_range, C_out_range):
            savepath = path.join(fibresdir, f"output_n_{n}_g_{g}_c_out_{c_out}.tex")
            self.custom_tikz_matrix(self.output[n, g, c_out]).save(
                savepath, compile=compile
            )

    @staticmethod
    def _combine_fibres_into_tensor(
        dims: Tuple[int, int, int], filenames: Dict[Tuple[int, int, int], str]
    ) -> str:
        """Create TikZ image that lays out the fibres of a 5d tensor into a single one.

        Args:
            dims: The dimensions of the tensor's three leading axes.
            filenames: Mapping from the three leading indices to the fiber's file name.

        Returns:
            The TikZ code for the tensor.
        """
        # generic template for compiling tensors to .pdf
        TEX_TEMPLATE = r"""\documentclass[tikz]{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}
TENSOR
\end{tikzpicture}
\end{document}"""

        D1, D2, D3 = dims

        # first dimension is laid out vertically
        vertical = []
        for d1 in range(D1):
            # second and third dimensions are laid out depthwise
            depthwise = []
            for d2, d3 in product(list(range(D2))[::-1], list(range(D3))[::-1]):
                fibre = filenames[d1, d2, d3]
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

        return TEX_TEMPLATE.replace("TENSOR", "\n".join(vertical))

    def _generate_tensors(self, compile: bool):
        """Visualize the input/output/weight tensors.

        Requires that the fibres have been generated first by calling
        `self.create_fibres`. This function combines the fibres into a single pdf image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image.
        """
        fibresdir = path.join(self.savedir, "static_fibres")
        tensordir = path.join(self.savedir, "static_tensors")
        makedirs(tensordir, exist_ok=True)

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))

        # combine the output fibres into a tensor
        dims = (self.N, self.G, self.C_out // self.G)
        filenames = {
            (n, g, c): path.join(fibresdir, f"output_n_{n}_g_{g}_c_out_{c}.pdf")
            for n, g, c in product(N_range, G_range, C_out_range)
        }
        code = self._combine_fibres_into_tensor(dims, filenames)
        savepath = path.join(tensordir, "output.tex")
        write(code, savepath, compile=compile)

        # combine the input fibres into a tensor
        dims = (self.N, self.G, self.C_in // self.G)
        filenames = {
            (n, g, c): path.join(fibresdir, f"input_n_{n}_g_{g}_c_in_{c}.pdf")
            for n, g, c in product(N_range, G_range, C_in_range)
        }
        code = self._combine_fibres_into_tensor(dims, filenames)
        savepath = path.join(tensordir, "input.tex")
        write(code, savepath, compile=compile)

        # combine the weight fibres into a tensor
        dims = (self.C_out // self.G, self.G, self.C_in // self.G)
        filenames = {
            (c_out, g, c_in): path.join(
                fibresdir, f"weight_g_{g}_c_out_{c_out}_c_in_{c_in}.pdf"
            )
            for c_out, g, c_in in product(C_out_range, G_range, C_in_range)
        }
        code = self._combine_fibres_into_tensor(dims, filenames)
        savepath = path.join(tensordir, "weight.tex")
        write(code, savepath, compile=compile)

    def highlight_padding(self, matrix: TikzMatrix, color: str = "VectorBlue"):
        """Highlight the padding pixels in a TikZ matrix of a 2d slice of the input.

        Args:
            matrix: TikZ matrix of a 2d slice of the input.
            color: Colour to highlight the padding pixels with. Defaults to
                `"VectorBlue"`.
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


class TikzConv2dAnimated(TikzConv2d):
    """Class for visualizing animated 2d convolutions with TikZ.

    Examples:
        >>> from torch import manual_seed, rand
        >>> _ = manual_seed(0)
        >>> N, C_in, I1, I2 = 2, 2, 3, 4
        >>> G, C_out, K1, K2 = 1, 3, 2, 3
        >>> P = (0, 1) # non-zero padding along one dimension
        >>> weight = rand(C_out, C_in // G, K1, K2)
        >>> x = rand(N, C_in, I1, I2)
        >>> # NOTE to compile, you need `pdflatex`
        >>> TikzConv2dAnimated(weight, x, "conv2d", padding=P).save(
        ...     compile=False, max_frames=10
        ... )

    - Example animation (padding pixels are highlighted)
      ![](assets/TikzConv2dAnimated/example.gif)
      If you set `compile=True` above, there will be an `example.pdf` file in the
      supplied directory. You can compile it to a `.gif` using the command
      ```bash
      convert -verbose -delay 100 -loop 0 -density 300 example.pdf example.gif`
      ```
    - I used this code to create the visualizations for my
      [talk](https://pirsa.org/23120027) at Perimeter Institute.

    Attributes:
        GROUP_COLORS: Colors used to highlight different channel groups.
    """

    GROUP_COLORS = ["VectorOrange", "VectorTeal", "VectorPink"]

    def save(self, compile: bool = True, max_frames: Optional[int] = None):
        """Create the frames of the convolution's input, weight, and output tensors.

        This is done in three steps. For each frame:

        1. Generate the fibres (matrix slices) of the input/weight/output tensors.
        2. Combine the fibres into the full tensors.
        3. Compile the full tensors into one pdf image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image. Default: `True`.
            max_frames: Maximum number of frames to generate. Default: `None`.
        """
        self._generate_fibres(compile=compile, max_frames=max_frames)
        self._generate_tensors(compile=compile, max_frames=max_frames)

        num_frames = self.output.numel() if max_frames is None else max_frames
        frames = [
            r"""\begin{tikzpicture}
  \node (input) {\includegraphics{DATAPATH/animated_tensors/input_frame_FRAME}};
  \node [right=1cm of input] (star) {$\star$};
  \node [right=1cm of star] (kernel) {%
    \includegraphics{DATAPATH/animated_tensors/weight_frame_FRAME}%
  };
  \node [right=1cm of kernel] (equal) {$=$};
  \node [right=1cm of equal] (output) {%
    \includegraphics{DATAPATH/animated_tensors/output_frame_FRAME}%
  };
\end{tikzpicture}""".replace(
                "DATAPATH", self.savedir
            ).replace(
                "FRAME", str(n)
            )
            for n in range(num_frames)
        ]
        code = r"""\documentclass[tikz]{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
CONTENT
\end{document}""".replace(
            "CONTENT", "\n".join(frames)
        )
        savepath = path.join(self.savedir, "example.tex")
        write(code, savepath, compile=compile)

    def _generate_fibres(  # noqa: C901
        self, compile: bool = True, max_frames: Optional[int] = None
    ) -> None:
        """Visualize the input/output/weight fibres during the convolution.

        This generates frames that can be arranged into an animation.

        Args:
            compile: Whether to compile the generated frames to pdf. Default: `True`.
            max_frames: Maximum number of frames to generate. If `None`, all frames are
                generated.

        Raises:
            ValueError: If there are not enough colours to distinguish all groups.
        """
        if self.G > len(self.GROUP_COLORS):
            raise ValueError(
                f"Not enough colours available to distinguish groups ({self.G}). "
                + f"Please add more to `GROUP_COLORS` (has {len(self.GROUP_COLORS)})."
            )

        fibresdir = path.join(self.savedir, "animated_fibres")
        makedirs(fibresdir, exist_ok=True)

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))
        O1_range = list(range(self.O1))
        O2_range = list(range(self.O2))

        # frames of output, parameterized by which entries have been computed
        for frame, (o2_done, o1_done, g_done, c_out_done, n_done) in enumerate(
            product(O2_range, O1_range, G_range, C_out_range, N_range)
        ):
            if max_frames is not None and frame + 1 > max_frames:
                break

            # zero out the entries that have not yet been computed
            output = rearrange(
                self.output, "n g c_out o1 o2 -> (o2 o1 g c_out n)"
            ).clone()
            if frame != output.numel() - 1:
                output[frame + 1 :] = 0
            output = rearrange(
                output,
                "(o2 o1 g c_out n) -> n g c_out o1 o2",
                n=self.N,
                g=self.G,
                c_out=self.C_out // self.G,
                o1=self.O1,
                o2=self.O2,
            )

            # plot the output fibres for the current frame
            for n, g, c_out in product(N_range, G_range, C_out_range):
                savepath = path.join(
                    fibresdir, f"output_n_{n}_g_{g}_c_out_{c_out}_frame_{frame}.tex"
                )
                matrix = self.custom_tikz_matrix(output[n, g, c_out])

                # highlight currently computed entry
                highlighted = (
                    [{"x": o2_done, "y": o1_done, "fill": self.GROUP_COLORS[g]}]
                    if c_out == c_out_done and g == g_done and n == n_done
                    else []
                )
                for kwargs in highlighted:
                    self.highlight(matrix, **kwargs)

                matrix.save(savepath, compile=compile)

            # plot the input fibres for the current frame
            for n, g, c_in in product(N_range, G_range, C_in_range):
                savepath = path.join(
                    fibresdir, f"input_n_{n}_g_{g}_c_in_{c_in}_frame_{frame}.tex"
                )
                matrix = self.custom_tikz_matrix(self.x[n, g, c_in])
                self.highlight_padding(matrix)

                # highlight active entries
                highlighted = (
                    [
                        {"x": i2, "y": i1, "fill": self.GROUP_COLORS[g]}
                        for i1, i2 in product(
                            range(self.I1 + 2 * self.P1), range(self.I2 + 2 * self.P2)
                        )
                        if self.pattern1[:, o1_done, i1].any()
                        and self.pattern2[:, o2_done, i2].any()
                    ]
                    if g == g_done and n == n_done
                    else []
                )
                for kwargs in highlighted:
                    self.highlight(matrix, **kwargs)

                matrix.save(savepath, compile=compile)

            # plot the weight fibres for the current frame
            for g, c_out, c_in in product(G_range, C_out_range, C_in_range):
                savepath = path.join(
                    fibresdir,
                    f"weight_g_{g}_c_out_{c_out}_c_in_{c_in}_frame_{frame}.tex",
                )
                matrix = self.custom_tikz_matrix(self.weight[g, c_out, c_in])
                # highlight active entries
                highlighted = (
                    [
                        {"x": k2, "y": k1, "fill": self.GROUP_COLORS[g]}
                        for k1, k2 in product(range(self.K1), range(self.K2))
                    ]
                    if g == g_done and c_out == c_out_done
                    else []
                )
                for kwargs in highlighted:
                    self.highlight(matrix, **kwargs)

                matrix.save(savepath, compile=compile)

    def _generate_tensors(self, compile: bool = True, max_frames: Optional[int] = None):
        """Visualize the input/output/weight tensors during the convolution.

        Args:
            compile: Whether to compile the generated LaTeX files to pdf. Defaults to
                `True`.
            max_frames: Maximum number of frames to generate. Defaults to `None` (all).
        """
        fibresdir = path.join(self.savedir, "animated_fibres")
        tensordir = path.join(self.savedir, "animated_tensors")
        makedirs(tensordir, exist_ok=True)

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))
        O1_range = list(range(self.O1))
        O2_range = list(range(self.O2))

        # frames of output, parameterized by which entries have been computed
        for frame, (_, _, _, _, _) in enumerate(
            product(O1_range, O2_range, G_range, C_out_range, N_range)
        ):
            if max_frames is not None and frame + 1 > max_frames:
                break

            # plot the output fibres for the current frame
            dims = (self.N, self.G, self.C_out // self.G)
            filenames = {
                (n, g, c): path.join(
                    fibresdir,
                    f"output_n_{n}_g_{g}_c_out_{c}_frame_{frame}.pdf",
                )
                for n, g, c in product(N_range, G_range, C_out_range)
            }
            code = self._combine_fibres_into_tensor(dims, filenames)
            savepath = path.join(tensordir, f"output_frame_{frame}.tex")
            write(code, savepath, compile=compile)

            # plot the input fibres for the current frame
            dims = (self.N, self.G, self.C_in // self.G)
            filenames = {
                (n, g, c): path.join(
                    fibresdir,
                    f"input_n_{n}_g_{g}_c_in_{c}_frame_{frame}.pdf",
                )
                for n, g, c in product(N_range, G_range, C_in_range)
            }
            code = self._combine_fibres_into_tensor(dims, filenames)
            savepath = path.join(tensordir, f"input_frame_{frame}.tex")
            write(code, savepath, compile=compile)

            # plot the weight fibres for the current frame
            dims = (self.C_out // self.G, self.G, self.C_in // self.G)
            filenames = {
                (c_out, g, c_in): path.join(
                    fibresdir,
                    f"weight_g_{g}_c_out_{c_out}_c_in_{c_in}_frame_{frame}.pdf",
                )
                for c_out, g, c_in in product(C_out_range, G_range, C_in_range)
            }
            code = self._combine_fibres_into_tensor(dims, filenames)
            savepath = path.join(tensordir, f"weight_frame_{frame}.tex")
            write(code, savepath, compile=compile)
