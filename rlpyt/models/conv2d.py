
import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import conv2d_output_shape

from symmetrizer.nn.modules import GlobalAveragePool, GlobalMaxPool

import matplotlib.pyplot as plt
import numpy as np


class Conv2dModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            use_avgpool=False,
            use_gmaxpool=False,
            head_sizes=None,  # Put an MLP head on top.
            ):
        super().__init__()
        self.n = 0
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.global_pool = False
        if use_avgpool:
            self.global_pool = True
            sequence.append(GlobalAveragePool())
        if use_gmaxpool:
            self.global_pool = True
            sequence.append(GlobalMaxPool())
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """
        if self.n % 5000 == 0:
            x = input
            print(x.min(), x.max(), x.mean())
            fig, ax = plt.subplots(1, 1)
            if len(x.shape) == 3:
                ax.imshow(x.detach().cpu().numpy())
            else:
                ax.imshow(np.swapaxes(x[0].detach().cpu().numpy(), 0, 2))
            plt.savefig(f"plots/layer_0.png")
            plt.close()
            for l, layer in enumerate(self.conv):
                 x = layer(x)
                 fig, ax = plt.subplots(4, 4)
                 for i, axi in enumerate(fig.axes):
                     axi.imshow(x[0][i].detach().cpu().numpy())
                 plt.savefig(f"plots/layer_{l+1}.png")
                 plt.close()
        self.n += 1
        """
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        if self.global_pool:
            return c
        return h * w * c


class Conv2dHeadModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            output_size=None,  # if None: nonlinearity applied to output.
            paddings=None,
            nonlinearity=torch.nn.ReLU,
            use_maxpool=False,
            use_avgpool=False,
            use_gmaxpool=False
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            use_avgpool=use_avgpool,
            use_gmaxpool=use_gmaxpool
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size, hidden_sizes,
                output_size=output_size, nonlinearity=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input):
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_size(self):
        return self._output_size
