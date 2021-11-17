import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import AxesImage
from torch.nn.modules.utils import _pair
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, List, Optional, Union

plt.ion()

import math
import torch
import numpy as np

from typing import Tuple, Union
from torch.nn.modules.utils import _pair





def reshape_locally_connected_weights(
    w: torch.Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    input_sqrt: Union[int, Tuple[int, int]],
) -> torch.Tensor:

    """
    Get the weights from a locally connected layer slice (channel) and reshape them to be two-dimensional and square.
    :param w: Weights from a locally connected layer.
    :param n_filters: No. of neuron filters.
    :param kernel_size: Side length(s) of convolutional kernel.
    :param conv_size: Side length(s) of convolution population.
    :param input_sqrt: Sides length(s) of input neurons.
    :return: Locally connected weights reshaped as a collection of spatially ordered square grids.
    """

    kernel_size = _pair(kernel_size)
    conv_size = _pair(conv_size)
    input_sqrt = _pair(input_sqrt)

    k1, k2 = kernel_size
    c1, c2 = conv_size
    i1, i2 = input_sqrt
    fs = int(math.ceil(math.sqrt(n_filters)))

    w_ = torch.zeros((n_filters * k1, k2 * c1 * c2))

    for n1 in range(c1):
        for n2 in range(c2):
            for feature in range(n_filters):
                n = n1 * c2 + n2
                filter_ = w[feature, n1, n2, :, :
                ].view(k1, k2)
                w_[feature * k1 : (feature + 1) * k1, n * k2 : (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[
                (n // fs) * i1 : ((n // fs) + 1) * i2,
                (n % fs) * i2 : ((n % fs) + 1) * i2,
            ] = w_[n * i1 : (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[
                                k1 * (n1 * fs + f1) : k1 * (n1 * fs + f1 + 1),
                                k2 * (n2 * fs + f2) : k2 * (n2 * fs + f2 + 1),
                            ] = w_[
                                (f1 * fs + f2) * k1 : (f1 * fs + f2 + 1) * k1,
                                (n1 * c2 + n2) * k2 : (n1 * c2 + n2 + 1) * k2,
                            ]

        return square




def plot_convergence_and_histogram(
    weights: torch.Tensor,
    convergences: List,
    figsize: Tuple[int, int] = (7, 7),

) -> AxesImage:

    fig, axs = plt.subplots(2,figsize=figsize)
    axs[0].hist(weights.flatten().cpu())
    axs[0].set_title('Histogram')
    
    axs[1].plot(convergences)
    axs[1].set_title('Convergence')
    axs[1].set(xlabel='trials', ylabel='Convergence rate')
    


def plot_fully_connected_weights(
    weights: torch.Tensor,
    wmin: Optional[float] = 0,
    wmax: Optional[float] = 1,
    im: Optional[AxesImage] = None,
    figsize: Tuple[int, int] = (5, 5),
    cmap: str = "hot_r",
    save: Optional[str] = None,
) -> AxesImage:
    # language=rst
    """
    Plot a fully-connected weight matrix.

    :param weights: Weight matrix of ``Connection`` object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :param save: file name to save fig, if None = not saving fig.
    :return: ``AxesImage`` for re-drawing the weights plot.
    """

    local_weights = weights.detach().clone().cpu().numpy()
    if save is not None:
        plt.ioff()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()

        a = save.split(".")
        if len(a) == 2:
            save = a[0] + ".1." + a[1]
        else:
            a[1] = "." + str(1 + int(a[1])) + ".png"
            save = a[0] + a[1]

        plt.savefig(save, bbox_inches="tight")
        plt.savefig(a[0] + ".png", bbox_inches="tight")

        plt.close(fig)
        plt.ion()
        return im, save
    else:
        if not im:
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)

            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect("auto")

            plt.colorbar(im, cax=cax)
            fig.tight_layout()
        else:
            im.set_data(local_weights)

        return im



def plot_locally_connected_feature_maps(weights: torch.Tensor,
    n_filters: int,
    in_chans: int,
    slice_to_plot: int,
    input_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    wmin: float = 0.0,
    wmax: float = 1.0,
    im: Optional[AxesImage] = None,
    lines: bool = True,
    figsize: Tuple[int, int] = (5, 5),
    cmap: str = "hot_r",
    color: str='r'
    ) -> AxesImage:

    """
    Plot a 2D local connection slice feature map
    :param weights: Weight matrix of Conv2dConnection object.
    :param n_filters: No. of convolution kernels in use.
    :param in_channels: No. of input channels
    :param slice_to_plot: The 2D slice to plot
    :param input_size: The square input size
    :param kernel_size: Side length(s) of 2D convolution kernels.
    :param conv_size: Side length(s) of 2D convolution population.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param lines: Whether or not to draw horizontal and vertical lines separating input
        regions.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """

    n_sqrt = int(np.ceil(np.sqrt(n_filters)))

    sel_slice = weights.view(in_chans, n_filters, conv_size, conv_size, kernel_size, kernel_size).cpu()
    sel_slice = sel_slice[slice_to_plot, ...]
    
    reshaped = reshape_locally_connected_weights(
        sel_slice, n_filters, kernel_size, conv_size, input_size
    )

    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(reshaped.cpu(), cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        kernel_size = _pair(kernel_size)
        conv_size = _pair(conv_size)
        
        if lines:
            for i in range(
                n_sqrt * kernel_size[0],
                n_sqrt * conv_size[0] * kernel_size[0],
                n_sqrt * kernel_size[0],
            ):
                ax.axhline(i - 0.5, color=color, linestyle="--")

            for i in range(
                n_sqrt * kernel_size[1],
                n_sqrt * conv_size[1] * kernel_size[1],
                n_sqrt * kernel_size[1],
            ):
                ax.axvline(i - 0.5, color=color, linestyle="--")

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im
    
def plot_locally_connected_activations(
    weights: torch.Tensor,
    spikes : torch.Tensor,
    n_filters: int,
    in_chans: int,
    slice_to_plot: int,
    input_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    wmin: float = 0.0,
    wmax: float = 1.0,
    scale_factor : float = 1.0,
    im: Optional[AxesImage] = None,
    lines: bool = True,
    figsize: Tuple[int, int] = (5, 5),
    cmap: str = "hot_r",
    color: str='r'
) -> AxesImage:
    """
    Plot the activations of a 2D locally connected slice
    :param weights: Weight matrix of Conv2dConnection object.
    :param n_filters: No. of convolution kernels in use.
    :param in_channels: No. of input channels
    :param slice_to_plot: The 2D slice to plot
    :param input_size: The square input size
    :param kernel_size: Side length(s) of 2D convolution kernels.
    :param conv_size: Side length(s) of 2D convolution population.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param lines: Whether or not to draw horizontal and vertical lines separating input
        regions.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """
    x = scale_factor * spikes / torch.max(spikes)
    x = x.clip(wmin,wmax).repeat_interleave(kernel_size, dim=0).repeat_interleave(kernel_size, dim=1).cpu()
    n_sqrt = int(np.ceil(np.sqrt(n_filters)))

    sel_slice = weights.view(in_chans, n_filters, conv_size, conv_size, kernel_size, kernel_size).cpu()
    sel_slice = sel_slice[slice_to_plot, ...]

    reshaped = reshape_locally_connected_weights(
        sel_slice, n_filters, kernel_size, conv_size, input_size
    )

    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(reshaped.cpu()*x, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        kernel_size = _pair(kernel_size)
        conv_size = _pair(conv_size)
        
        if lines:
            for i in range(
                n_sqrt * kernel_size[0],
                n_sqrt * conv_size[0] * kernel_size[0],
                n_sqrt * kernel_size[0],
            ):
                ax.axhline(i - 0.5, color=color, linestyle="--")

            for i in range(
                n_sqrt * kernel_size[1],
                n_sqrt * conv_size[1] * kernel_size[1],
                n_sqrt * kernel_size[1],
            ):
                ax.axvline(i - 0.5, color=color, linestyle="--")

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im
   
def plot_locally_connected_output_weights(
    weights: torch.Tensor,
    weights_fc: torch.Tensor,
    ind_neuron_in_group: int,
    label: int,
    n_per_class : int,
    n_filters: int,
    in_chans: int,
    slice_to_plot: int,
    input_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    wmin: float = 0.0,
    wmax: float = 1.0,
    im: Optional[AxesImage] = None,
    lines: bool = True,
    figsize: Tuple[int, int] = (5, 5),
    cmap: str = "hot_r",
    color: str='r'
) -> AxesImage:
    """
    Plot the weights multiplication of LC and output weights, 
    for a given neuron in one of the output neuron groups in BioLCNet
    :param weights: Weight matrix of LocalConnection2D object.
    :param weights_fc: Weight matrix of Connection object.
    :param label: The neuron group class label
    :param n_per_class: The number of neurons per class
    :param n_filters: No. of convolution kernels in use.
    :param in_channels: No. of input channels
    :param slice_to_plot: The 2D slice to plot
    :param inp_size: The square input size
    :param kernel_size: Side length(s) of 2D convolution kernels.
    :param conv_size: Side length(s) of 2D convolution population.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param lines: Whether or not to draw horizontal and vertical lines separating input
        regions.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """
    n_sqrt = int(np.ceil(np.sqrt(n_filters)))

    sel_slice = weights.view(in_chans, n_filters, conv_size, conv_size, kernel_size, kernel_size).cpu()
    sel_slice = sel_slice[slice_to_plot, ...]

    reshaped = reshape_locally_connected_weights(
        sel_slice, n_filters, kernel_size, conv_size, input_size
    )

    ind_neuron = label * n_per_class + ind_neuron_in_group
    w = weights_fc[:,ind_neuron].view(reshaped.shape[0]//kernel_size,reshaped.shape[1]//kernel_size)
    w = w.clip(wmin,wmax).repeat_interleave(kernel_size, dim=0).repeat_interleave(kernel_size, dim=1).cpu()

    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(reshaped.cpu()*w, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        kernel_size = _pair(kernel_size)
        conv_size = _pair(conv_size)
        
        if lines:
            for i in range(
                n_sqrt * kernel_size[0],
                n_sqrt * conv_size[0] * kernel_size[0],
                n_sqrt * kernel_size[0],
            ):
                ax.axhline(i - 0.5, color=color, linestyle="--")

            for i in range(
                n_sqrt * kernel_size[1],
                n_sqrt * conv_size[1] * kernel_size[1],
                n_sqrt * kernel_size[1],
            ):
                ax.axvline(i - 0.5, color=color, linestyle="--")

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im
