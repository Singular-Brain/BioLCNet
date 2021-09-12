from typing import Union, Optional, Sequence

import torch

from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
)

from utils.locally_connected_multi_chan import LocalConnection2D
from bindsnet.learning import LearningRule

class PostPre(LearningRule):
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive. This BindsNet class is modified
    to be adaptable with multi-channel local connection
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )


    def _connection_update(self, **kwargs) -> None:
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float().to(self.connection.w.device)
            target_x = (self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]).to(self.connection.w.device)
            self.connection.w -= (self.reduction(torch.bmm(source_s, target_x), dim=0))*self.soft_bound_decay()
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1]:
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            ).to(self.connection.w.device)
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2).to(self.connection.w.device)
            self.connection.w += (self.reduction(torch.bmm(source_x, target_s), dim=0))*self.soft_bound_decay()
            del source_x, target_s

        super().update()
        
    def _local_connection_update(self, **kwargs) -> None:
        """
        Post-pre learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size
        kernel_width = self.connection.kernel_size[0]
        kernel_height = self.connection.kernel_size[1]
        in_channels = self.connection.in_channels
        out_channels = self.connection.out_channels
        width_out = self.connection.conv_size[0]
        height_out = self.connection.conv_size[1]


        ## target_x (s) ch_o, w_o, h_o  
        target_x = self.target.x.reshape(batch_size, out_channels * width_out * height_out, 1) 
        target_x = target_x * torch.eye(out_channels * width_out * height_out).to(self.connection.w.device)
        source_s = self.source.s.type(torch.float).unfold(-2, kernel_width,stride[0]).unfold(-2, kernel_height, stride[1]).reshape(
            batch_size, 
            width_out * height_out,
            in_channels *  kernel_width *  kernel_height,
        ).repeat(
            1,
            out_channels,
            1,
        )
        
        target_s = self.target.s.type(torch.float).reshape(batch_size, out_channels * width_out*height_out,1)
        target_s = target_s * torch.eye(out_channels * width_out * height_out).to(self.connection.w.device)
        source_x = self.source.x.unfold(-2, kernel_width,stride[0]).unfold(-2, kernel_height, stride[1]).reshape(
            batch_size, 
            width_out * height_out,
            in_channels *  kernel_width *  kernel_height,
        ).repeat(
            1,
            out_channels,
            1,
        )

        # Pre-synaptic update.
        if self.nu[0]:
            pre = self.reduction(torch.bmm(target_x,source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(torch.bmm(target_s, source_x),dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

