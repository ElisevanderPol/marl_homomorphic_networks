import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from symmetrizer.nn.traffic_networks import StandardDecentralizedModel, \
    BasisDecentralizedModel
from symmetrizer.ops.traffic_ops import get_locs


class TrafficGraphModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=[512],
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            basis=None,
            ):
        super().__init__()
        self.conv = StandardDecentralizedModel(3, n_agents=4,
                                               channels=channels,
                                               filters=kernel_sizes,
                                               strides=strides,
                                               paddings=paddings,
                                               hidden_sizes=fc_sizes)

    def forward(self, inputs, prev_action, prev_reward, imshow=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        image = inputs[:, :-1]
        loc_grid = inputs[:, -1]
        locs = get_locs(loc_grid, 4)
        img = image.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        fc_out = self.conv(locs, img.view(T * B, *img_shape))
        pi = F.softmax(fc_out[0], dim=-1)
        v = fc_out[1]

        pi = pi.squeeze(0)
        v = v.squeeze()

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


class TrafficBasisGraphModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=[512],
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            basis=None,
            ):
        super().__init__()
        self.conv = BasisDecentralizedModel(3, n_agents=4, channels=channels,
                                            filters=kernel_sizes,
                                            strides=strides, paddings=paddings,
                                            hidden_sizes=fc_sizes,
                                            basis=basis)

    def forward(self, inputs, prev_action, prev_reward, imshow=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        image = inputs[:, :-1]
        loc_grid = inputs[:, -1]
        locs = get_locs(loc_grid, 4)
        img = image.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        fc_out = self.conv(locs, img.view(T * B, *img_shape))
        pi = F.softmax(fc_out[0], dim=-1).squeeze(0)
        v = fc_out[1].squeeze()

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v
