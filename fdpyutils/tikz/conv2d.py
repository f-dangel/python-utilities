"""Visualizing 2d convolution with TikZ."""

from itertools import product
from os import path
from typing import Optional, Tuple

from einconv import index_pattern
from einops import einsum, rearrange
from torch import Tensor
from torch.nn.functional import pad

from fdpyutils.tikz.tensorshow import TikzTensor
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

        This is done in two steps:

        1. Compile the input, weight, and output tensors.
        2. Combine them into one image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image. Default: `True`.
        """
        self._generate_tensors(compile=compile)

        TEX_TEMPLATE = r"""\documentclass[tikz]{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}
  \node (input) {\includegraphics{DATAPATH/tensors/input}};
  \node [right=1cm of input] (star) {$\star$};
  \node [right=1cm of star] (kernel) {%
    \includegraphics{DATAPATH/tensors/weight}%
  };
  \node [right=1cm of kernel] (equal) {$=$};
  \node [right=1cm of equal] (output) {%
    \includegraphics{DATAPATH/tensors/output}%
  };
\end{tikzpicture}
\end{document}"""
        code = TEX_TEMPLATE.replace("DATAPATH", self.savedir)
        savepath = path.join(self.savedir, "example.tex")
        write(code, savepath, compile=compile)

    def _generate_tensors(self, compile: bool):
        """Visualize the input/output/weight tensors.

        Args:
            compile: Whether to compile the TikZ code into a pdf image.
        """
        tensordir = path.join(self.savedir, "tensors")

        x_tikz = TikzTensor(self.x)
        self.highlight_padding(x_tikz)
        x_tikz.save(path.join(tensordir, "input.tex"), compile=compile)

        output_tikz = TikzTensor(self.output)
        output_tikz.save(path.join(tensordir, "output.tex"), compile=compile)

        weight_tikz = TikzTensor(self.weight)
        weight_tikz.save(path.join(tensordir, "weight.tex"), compile=compile)

    def highlight_padding(self, tensor: TikzTensor, color: str = "VectorBlue"):
        """Highlight the padding pixels of a TikZ tensor.

        Args:
            tensor: The TikZ tensor whose padded pixels will be highlighted.
            color: Colour to highlight the padding pixels with. Defaults to
                `"VectorBlue"`.
        """
        # for shorter notation
        I1, I2 = self.I1, self.I2
        P1, P2 = self.P1, self.P2
        N, G, C_in = self.N, self.G, self.C_in

        for i1, i2 in product(range(I1 + 2 * P1), range(I2 + 2 * P2)):
            if i1 < P1 or i1 >= P1 + I1 or i2 < P2 or i2 >= P2 + I2:
                for n, g, c_in in product(range(N), range(G), range(C_in // G)):
                    tensor.highlight((n, g, c_in, i1, i2), fill=color)

    @staticmethod
    def normalize(tensor: Tensor) -> Tensor:
        """Normalize values of a tensor to [0, 1].

        Args:
            tensor: Input tensor.

        Returns:
            Normalized tensor.
        """
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


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
      convert -verbose -delay 100 -loop 0 -density 300 example.pdf example.gif
      ```
      which requires the `imagemagick` library.
    - I used this code to create the visualizations for my
      [talk](https://pirsa.org/23120027) at Perimeter Institute.

    Attributes:
        GROUP_COLORS: Colors used to highlight different channel groups.
    """

    GROUP_COLORS = ["VectorOrange", "VectorTeal", "VectorPink"]

    def save(self, compile: bool = True, max_frames: Optional[int] = None):
        """Create the frames of the convolution's input, weight, and output tensors.

        This is done in two steps. For each frame:

        1. Generate the input, weight, and output tensors.
        2. Compile the full tensors into one pdf image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image. Default: `True`.
            max_frames: Maximum number of frames to generate. Default: `None`.
        """
        self._generate_tensors(compile=compile, max_frames=max_frames)

        num_frames = self.output.numel() if max_frames is None else max_frames
        frames = [
            r"""\begin{tikzpicture}
  \node (input) {\includegraphics{DATAPATH/tensors/input_frame_FRAME}};
  \node [right=1cm of input] (star) {$\star$};
  \node [right=1cm of star] (kernel) {%
    \includegraphics{DATAPATH/tensors/weight_frame_FRAME}%
  };
  \node [right=1cm of kernel] (equal) {$=$};
  \node [right=1cm of equal] (output) {%
    \includegraphics{DATAPATH/tensors/output_frame_FRAME}%
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

    def _generate_tensors(
        self, compile: bool = True, max_frames: Optional[int] = None
    ) -> None:
        """Visualize the input/output/weight tensors during the convolution.

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

        tensordir = path.join(self.savedir, "tensors")

        N_range = list(range(self.N))
        G_range = list(range(self.G))
        C_in_range = list(range(self.C_in // self.G))
        C_out_range = list(range(self.C_out // self.G))
        O1_range = list(range(self.O1))
        O2_range = list(range(self.O2))

        # frames of output, parameterized by which entries have been computed
        for frame, (o2, o1, g, c_out, n) in enumerate(
            product(O2_range, O1_range, G_range, C_out_range, N_range)
        ):
            if max_frames is not None and frame + 1 > max_frames:
                break

            channel_color = self.GROUP_COLORS[g]

            # zero out the entries that have not yet been computed
            output = rearrange(
                self.output, "n g c_out o1 o2 -> (o2 o1 g c_out n)"
            ).clone()
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

            # plot the output
            output_tikz = TikzTensor(output)
            # highlight the current output pixel
            output_tikz.highlight((n, g, c_out, o1, o2), fill=channel_color)
            output_tikz.save(
                path.join(tensordir, f"output_frame_{frame}.tex"), compile=compile
            )

            # plot the input
            x_tikz = TikzTensor(self.x)
            self.highlight_padding(x_tikz)
            # highlight the active entries
            for i1, i2 in product(
                range(self.I1 + 2 * self.P1), range(self.I2 + 2 * self.P2)
            ):
                if self.pattern1[:, o1, i1].any() and self.pattern2[:, o2, i2].any():
                    for c_in in C_in_range:
                        x_tikz.highlight((n, g, c_in, i1, i2), fill=channel_color)
            x_tikz.save(
                path.join(tensordir, f"input_frame_{frame}.tex"), compile=compile
            )

            # plot the weight
            w_tikz = TikzTensor(self.weight)
            for c_in, k1, k2 in product(C_in_range, range(self.K1), range(self.K2)):
                w_tikz.highlight((g, c_out, c_in, k1, k2), fill=channel_color)
            w_tikz.save(
                path.join(tensordir, f"weight_frame_{frame}.tex"), compile=compile
            )
