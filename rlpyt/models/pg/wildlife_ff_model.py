import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from symmetrizer.nn.wildlife_networks import StandardDecentralizedModel, \
    BasisDecentralizedModel
from symmetrizer.ops.wildlife_ops import get_locs


class WildlifeGraphModel(torch.nn.Module):

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
            n_agents=2,
            ):
        super().__init__()

        self.conv = StandardDecentralizedModel(1, n_agents, channels=channels,
                                               filters=kernel_sizes,
                                               strides=strides,
                                               paddings=paddings,
                                               hidden_sizes=fc_sizes)
        self.n_agents = n_agents

    def forward(self, inputs, prev_action, prev_reward, imshow=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        image = inputs[:, :-1]
        loc_grid = inputs[:, -1]
        n_agents = image.shape[1]
        locs = get_locs(loc_grid, n_agents)
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


class WildlifeBasisGraphModel(torch.nn.Module):

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
            n_agents=2,
            ):
        super().__init__()
        self.conv = BasisDecentralizedModel(1, n_agents, channels=channels,
                                            filters=kernel_sizes,
                                            strides=strides, paddings=paddings,
                                            hidden_sizes=fc_sizes,
                                            basis=basis)
        self.n_agents = n_agents

    def forward(self, inputs, prev_action, prev_reward, imshow=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        image = inputs[:, :-1]
        loc_grid = inputs[:, -1]
        n_agents = image.shape[1]
        locs = get_locs(loc_grid, n_agents)
        img = image.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        fc_out = self.conv(locs, img.view(T * B, *img_shape))
        pi = F.softmax(fc_out[0], dim=-1).squeeze(0)
        v = fc_out[1].squeeze()

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v
