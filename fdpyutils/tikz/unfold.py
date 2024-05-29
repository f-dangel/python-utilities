"""Visualize input unfolding with TikZ."""

from itertools import product
from os import path
from typing import Optional, Tuple

from einops import rearrange
from torch import Tensor

from fdpyutils.einops.einsum import einsum
from fdpyutils.tikz.conv2d import TikzConv2dAnimated
from fdpyutils.tikz.tensorshow import TikzTensor
from fdpyutils.tikz.utils import write


class TikzUnfoldAnimated(TikzConv2dAnimated):
    """Class for visualizing input unfolding of a 2d convolution with TikZ.

    Examples:
        >>> from torch import manual_seed, rand
        >>> _ = manual_seed(0)
        >>> N, C_in, I1, I2 = 1, 2, 5, 1
        >>> G, C_out, K1, K2 = 1, 4, 3, 1
        >>> P = (1, 0) # non-zero padding along one dimension
        >>> weight = rand(C_out, C_in // G, K1, K2)
        >>> x = rand(N, C_in, I1, I2)
        >>> # NOTE to compile, you need `pdflatex`
        >>> TikzUnfoldAnimated(weight, x, "unfold", padding=P).save(
        ...     compile=False, max_frames=10
        ... )

    - Example animation (left is matricized output, middle is matricized kernel, right
      is unfolded input)
      ![](assets/TikzUnfoldAnimated/example.gif)
      If you set `compile=True` above, there will be an `example.pdf` file in the
      supplied directory. You can compile it to a `.gif` using the command
      ```bash
      convert -verbose -delay 100 -loop 0 -density 300 example.pdf example.gif
      ```
      which requires the `imagemagick` library.
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
        """Store convolution hyper-parameters for the animated input unfolding.

        Args:
            weight: Convolution kernel. Has shape `[C_out, C_in // G, K1, K2]`.
            x: Input tensor. Has shape `[N, C_in, I1, I2]`.
            savedir: Directory under which the TikZ code and pdf images are saved.
            stride: Stride of the convolution. Default: `(1, 1)`.
            padding: Padding of the convolution. Default: `(0, 0)`.
            dilation: Dilation of the convolution. Default: `(1, 1)`.
        """
        super().__init__(
            weight, x, savedir, stride=stride, padding=padding, dilation=dilation
        )
        self.x_unfolded = einsum(
            self.x,
            self.pattern1.float(),
            self.pattern2.float(),
            "n g c i1 i2, k1 o1 i1, k2 o2 i2 -> n g (c k1 k2) (o1 o2)",
        )
        self.weight_as_mat = rearrange(
            self.weight, "g c_out c_in k1 k2 -> g c_out (c_in k1 k2)"
        )
        self.output_as_mat = rearrange(
            self.output, "n g c_out o1 o2 -> n g c_out (o1 o2)"
        )

    def save(self, compile: bool = True, max_frames: Optional[int] = None) -> None:
        """Create frames of the unfolded input, matricized kernel and matricized output.

        This is done in two steps. For each frame:

        1. Generate the individual tensors.
        2. Compile the full tensors into one pdf image.

        Args:
            compile: Whether to compile the TikZ code into a pdf image. Default: `True`.
            max_frames: Maximum number of frames to generate. Default: `None`.
        """
        self._generate_tensors(compile=compile, max_frames=max_frames)

        num_frames = self.output.numel() if max_frames is None else max_frames
        frames = [
            r"""\begin{tikzpicture}
  \node (output) {\includegraphics{DATAPATH/tensors/output_frame_FRAME}};
  \node [right=1cm of output] (equal) {$=$};
  \node [right=1cm of equal] (kernel) {%
    \includegraphics{DATAPATH/tensors/weight_frame_FRAME}%
  };
  \node [right=1cm of kernel] (cdot) {$\cdot$};
  \node [right=1cm of cdot] (input) {%
    \includegraphics{DATAPATH/tensors/input_frame_FRAME}%
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
        """Visualize matricized output+weight and unfolded input during convolution.

        This generates frames that can be arranged into an animation.

        Args:
            compile: Whether to compile the generated frames to pdf. Default: `True`.
            max_frames: Maximum number of frames to generate. If `None, all frames are
                generated. Default: `None`.

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
            o1_o2 = o1 * self.O2 + o2

            # zero out the entries that have not yet been computed
            output_as_mat = rearrange(
                self.output_as_mat,
                "n g c_out (o1 o2) -> (o2 o1 g c_out n)",
                o1=self.O1,
                o2=self.O2,
            ).clone()
            output_as_mat[frame + 1 :] = 0
            output_as_mat = rearrange(
                output_as_mat,
                "(o2 o1 g c_out n) -> n g c_out (o1 o2)",
                n=self.N,
                g=self.G,
                c_out=self.C_out // self.G,
                o1=self.O1,
                o2=self.O2,
            )

            # plot the matricized output
            output_tikz = TikzTensor(output_as_mat)
            # highlight the current output pixel
            output_tikz.highlight((n, g, c_out, o1_o2), fill=channel_color)
            output_tikz.save(
                path.join(tensordir, f"output_frame_{frame}.tex"), compile=compile
            )

            # plot the unfolded input
            x_tikz = TikzTensor(self.x_unfolded)
            self.highlight_padding(x_tikz)
            # highlight column o1_o2
            for c_in_k1_k2 in range(self.C_in // self.G * self.K1 * self.K2):
                x_tikz.highlight((n, g, c_in_k1_k2, o1_o2), fill=channel_color)
            x_tikz.save(
                path.join(tensordir, f"input_frame_{frame}.tex"), compile=compile
            )

            # plot the matricized weight
            w_tikz = TikzTensor(self.weight_as_mat)
            # highlight row c_out
            for c_in_k1_k2 in range(self.C_in // self.G * self.K1 * self.K2):
                w_tikz.highlight((g, c_out, c_in_k1_k2), fill=channel_color)
            w_tikz.save(
                path.join(tensordir, f"weight_frame_{frame}.tex"), compile=compile
            )

    def highlight_padding(self, tensor: TikzTensor, color: str = "VectorBlue"):
        """Highlight the padding pixels in the unfolded input.

        Args:
            tensor: The unfolded input TikZ tensor whose padded pixels are highlighted.
            color: Colour to highlight the padding pixels with. Defaults to
                `"VectorBlue"`.
        """
        # for shorter notation
        I1, I2 = self.I1, self.I2
        P1, P2 = self.P1, self.P2
        O1, O2 = self.O1, self.O2
        K1, K2 = self.K1, self.K2
        N, G, C_in = self.N, self.G, self.C_in

        i1_i2 = {
            (i1, i2)
            for i1, i2 in product(range(I1 + 2 * P1), range(I2 + 2 * P2))
            if i1 < P1 or i1 >= P1 + I1 or i2 < P2 or i2 >= P2 + I2
        }

        for (i1, i2), k1, k2, o1, o2 in product(
            i1_i2, range(K1), range(K2), range(O1), range(O2)
        ):
            o1_o2 = o1 * O2 + o2
            if self.pattern1[k1, o1, i1] and self.pattern2[k2, o2, i2]:
                for n, g, c in product(range(N), range(G), range(C_in // G)):
                    c_in_k1_k2 = c * K1 * K2 + k1 * k2 + k2
                    tensor.highlight((n, g, c_in_k1_k2, o1_o2), fill=color)


class TikzUnfoldWeightAnimated(TikzUnfoldAnimated):
    """Class for visualizing kernel unfolding of a 2d convolution with TikZ.

    Examples:
        >>> from torch import manual_seed, rand
        >>> _ = manual_seed(0)
        >>> N, C_in, I1, I2 = 1, 2, 5, 1
        >>> G, C_out, K1, K2 = 1, 3, 4, 1
        >>> P = (1, 0)
        >>> weight = rand(C_out, C_in // G, K1, K2)
        >>> x = rand(N, C_in, I1, I2)
        >>> savedir = path.join(DOC_ASSETS_DIR, "TikzUnfoldWeightAnimated")
        >>> # NOTE to compile, you need `pdflatex`
        >>> TikzUnfoldWeightAnimated(weight, x, savedir, padding=P).save(compile=False)

    - Example animation (left is vectorized output, middle is unfolded kernel, right
      is vectorized input)
      ![](assets/TikzUnfoldWeightAnimated/example.gif)
      If you set `compile=True` above, there will be an `example.pdf` file in the
      supplied directory. You can compile it to a `.gif` using the command
      ```bash
      convert -verbose -delay 100 -loop 0 -density 300 example.pdf example.gif
      ```
      which requires the `imagemagick` library.
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
        """Store convolution hyper-parameters for the animated input unfolding.

        Args:
            weight: Convolution kernel. Has shape `[C_out, C_in // G, K1, K2]`.
            x: Input tensor. Has shape `[N, C_in, I1, I2]`.
            savedir: Directory under which the TikZ code and pdf images are saved.
            stride: Stride of the convolution. Default: `(1, 1)`.
            padding: Padding of the convolution. Default: `(0, 0)`.
            dilation: Dilation of the convolution. Default: `(1, 1)`.
        """
        super().__init__(
            weight, x, savedir, stride=stride, padding=padding, dilation=dilation
        )
        self.x_as_vec = rearrange(self.x, "n g c i1 i2 -> n g (c i1 i2)")
        self.weight_unfolded = einsum(
            self.pattern1.float(),
            self.pattern2.float(),
            self.weight,
            "k1 o1 i1, k2 o2 i2, g c_out c_in k1 k2 -> g (c_out o1 o2) (c_in i1 i2)",
        )
        self.output_as_vec = rearrange(
            self.output, "n g c_out o1 o2 -> n g (c_out o1 o2)"
        )

    def _generate_tensors(
        self, compile: bool = True, max_frames: Optional[int] = None
    ) -> None:
        """Visualize vectorized output+input and unfolded weight during convolution.

        This generates frames that can be arranged into an animation.

        Args:
            compile: Whether to compile the generated frames to pdf. Default: `True`.
            max_frames: Maximum number of frames to generate. If `None, all frames are
                generated. Default: `None`.

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
            c_out_o1_o2 = c_out * self.O1 * self.O2 + o1 * self.O2 + o2

            # zero out the entries that have not yet been computed
            output_as_vec = rearrange(
                self.output_as_vec,
                "n g (c_out o1 o2) -> (o2 o1 g c_out n)",
                c_out=self.C_out // self.G,
                o1=self.O1,
                o2=self.O2,
            ).clone()
            output_as_vec[frame + 1 :] = 0
            output_as_vec = rearrange(
                output_as_vec,
                "(o2 o1 g c_out n) -> n g (c_out o1 o2)",
                n=self.N,
                g=self.G,
                c_out=self.C_out // self.G,
                o1=self.O1,
                o2=self.O2,
            )

            # plot the vectorized output
            output_tikz = TikzTensor(output_as_vec.unsqueeze(-1))
            # highlight the current output pixel
            output_tikz.highlight((n, g, c_out_o1_o2, 0), fill=channel_color)
            output_tikz.save(
                path.join(tensordir, f"output_frame_{frame}.tex"), compile=compile
            )

            # plot the vectorized input
            x_tikz = TikzTensor(self.x_as_vec.unsqueeze(-1))
            self.highlight_padding(x_tikz)
            # highlight entire vector
            for c_in_i1_i2 in range(
                self.C_in // self.G * (self.I1 + 2 * self.P1) * (self.I2 + 2 * self.P2)
            ):
                x_tikz.highlight((n, g, c_in_i1_i2, 0), fill=channel_color)
            x_tikz.save(
                path.join(tensordir, f"input_frame_{frame}.tex"), compile=compile
            )

            # plot the matricized weight
            w_tikz = TikzTensor(self.weight_unfolded)
            # highlight row c_out_o1_o2
            for c_in_i1_i2 in range(
                self.C_in // self.G * (self.I1 + 2 * self.P1) * (self.I2 + 2 * self.P2)
            ):
                w_tikz.highlight((g, c_out_o1_o2, c_in_i1_i2), fill=channel_color)
            w_tikz.save(
                path.join(tensordir, f"weight_frame_{frame}.tex"), compile=compile
            )

    def highlight_padding(self, tensor: TikzTensor, color: str = "VectorBlue"):
        """Highlight the padding pixels in the unfolded input.

        Args:
            tensor: The vectorized input TikZ tensor whose padded pixels are
                highlighted.
            color: Colour to highlight the padding pixels with. Defaults to
                `"VectorBlue"`.
        """
        # for shorter notation
        I1, I2 = self.I1, self.I2
        P1, P2 = self.P1, self.P2
        N, G, C_in = self.N, self.G, self.C_in
        I1_eff, I2_eff = I1 + 2 * P1, I2 + 2 * P2

        i1_i2 = {
            (i1, i2)
            for i1, i2 in product(range(I1_eff), range(I2_eff))
            if i1 < P1 or i1 >= P1 + I1 or i2 < P2 or i2 >= P2 + I2
        }

        for i1, i2 in i1_i2:
            for n, g, c in product(range(N), range(G), range(C_in // G)):
                c_in_i1_i2 = c * I1_eff * I2_eff + i1 * I2_eff + i2
                tensor.highlight((n, g, c_in_i1_i2, 0), fill=color)
