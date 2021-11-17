from typing import Optional, Union, Tuple, Sequence

import numpy as np
import torch
from torch.nn.modules.utils import _pair

from bindsnet.network.nodes import Nodes
from torch.nn.parameter import Parameter
from bindsnet.network.topology import AbstractConnection


class LocalConnection2D(AbstractConnection):
    """
    2D Local connection between one or two population of neurons supporting multi-channel 3D inputs;
    the logic is different from the BindsNet implementaion, but some lines are unchanged
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        input_shape: Tuple,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        """
        Instantiates a 'LocalConnection` object. Source population can be multi-channel

        Neurons in the post-synaptic population are ordered by receptive field; that is,
        if there are `n_conv` neurons in each post-synaptic patch, then the first
        `n_conv` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param input_shape: The 2D shape of each input channel
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule. For now, only PostPre has been implemented for the multi-channel-input implementation
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kwargs.get('padding', 0)

        shape = input_shape

        if kernel_size == shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                (shape[0] - kernel_size[0]) // stride[0] + 1,
                (shape[1] - kernel_size[1]) // stride[1] + 1,
            )

        self.conv_size = conv_size
        self.conv_prod = int(np.prod(conv_size))
        self.kernel_prod = int(np.prod(kernel_size))
         
        assert (
            target.n == out_channels * self.conv_prod
        ), "Target layer size must be n_filters * (kernel_size ** 2)."

        w = kwargs.get("w", None)

        if w is None:
            w = torch.rand(
                self.in_channels, 
                self.out_channels * self.conv_prod,
                self.kernel_prod
            )
        else:
            assert w.shape == (
                self.in_channels, 
                self.out_channels * self.conv_prod,
                self.kernel_prod
                )
        if self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", None), requires_grad=False)


    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights
        # s: batch, ch_in, w_in, h_in => s_unfold: batch, ch_in, ch_out * w_out * h_out, k ** 2
        # w: ch_in, ch_out * w_out * h_out, k ** 2
        # a_post: batch, ch_in, ch_out * w_out * h_out, k ** 2 => batch, ch_out * w_out * h_out (= target.n)

        self.s_unfold = s.unfold(
            -2,self.kernel_size[0],self.stride[0]
        ).unfold(
            -2,self.kernel_size[1],self.stride[1]
        ).reshape(
            s.shape[0], 
            self.in_channels,
            self.conv_prod,
            self.kernel_prod,
        ).repeat(
            1,
            1,
            self.out_channels,
            1,
        )

        a_post = self.s_unfold.to(self.w.device) * self.w.unsqueeze(0)
        
        return a_post.sum(-1).sum(1).view(
            a_post.shape[0], self.out_channels, *self.conv_size,
            )

    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            # get a view and modify in-place
            # w: ch_in, ch_out * w_out * h_out, k ** 2
            w = self.w.view(
                self.w.shape[0]*self.w.shape[1], self.w.shape[2]
            )

            for fltr in range(w.shape[0]):
                w[fltr,:] *= self.norm / w[fltr,:].sum(0)


    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.target.reset_state_variables()
