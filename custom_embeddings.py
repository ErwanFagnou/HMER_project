import collections.abc
import math
from typing import Optional

import torch
from torch import nn
from transformers import ViTConfig


class CustomViTEmbeddings(nn.Module):
    """
    Modified from ViTEmbeddings, to support images of different sizes.

    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(1, 1, config.hidden_size, dtype=torch.float32), mean=0.0, std=config.initializer_range
            )
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None

        # Replaced ViTPatchEmbeddings with just a CNN (+ flatten and transpose in forward)
        self.patch_embeddings = CustomViTPatchEmbeddings(config)
        # self.patch_embeddings = self  # needed for the model to be able to find the projection layer
        # num_channels, hidden_size, patch_size = config.num_channels, config.hidden_size, config.patch_size
        # self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        grid_rows = math.ceil(config.max_image_height / config.patch_size)
        grid_cols = math.ceil(config.max_image_width / config.patch_size)
        self.use_gabor_position_embeddings = config.use_gabor_position_embeddings
        if self.use_gabor_position_embeddings:
            self.position_embeddings = GaborPositionEmbeddings(grid_rows, grid_cols, config.hidden_size)
        else:
            self.position_embeddings = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.zeros(1, math.ceil(config.max_image_height / config.patch_size),
                                math.ceil(config.max_image_width / config.patch_size), config.hidden_size, dtype=torch.float32),
                    mean=0.0,
                    std=config.initializer_range,
                )
            )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def maybe_pad(self, pixel_values, height, width):
        pad_h, pad_w = self.patch_embeddings.patch_size
        if width % pad_w != 0:
            pad_values = (0, pad_w - width % pad_w)
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % pad_h != 0:
            pad_values = (0, 0, 0, pad_h - height % pad_h)
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def get_position_embeddings(self):
        if self.use_gabor_position_embeddings:
            return self.position_embeddings()
        return self.position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        **_,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.patch_embeddings(pixel_values)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask


        # add positional encoding to each token
        patch_rows = math.ceil(height / self.config.patch_size)
        patch_cols = math.ceil(width / self.config.patch_size)
        position_embeddings = self.get_position_embeddings()[:, :patch_rows, :patch_cols, :].flatten(1, 2)
        used_patches_mask = (embeddings != self.patch_embeddings.projection.bias).any(dim=2).any(dim=0)
        embeddings = (embeddings + position_embeddings)[:, used_patches_mask]

        # (this was previously before adding the positional embeddings, which is stupid)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings


class CustomViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        patch_size, max_h, max_w = config.patch_size, config.max_image_height, config.max_image_width
        num_channels, hidden_size = config.num_channels, config.hidden_size

        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = math.ceil(max_h / patch_size[1]) * math.ceil(max_w / patch_size[0])
        self.image_size = (max_h, max_w)
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, **_) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class GaborPositionEmbeddings(nn.Module):
    initializer_std = 0.2

    def __init__(self, grid_height, grid_width, embedding_dim, projection=False):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.embedding_dim = embedding_dim
        self.projection = projection

        # power (1/sigma), freq (1/lambda), phase (psi), weight (w)
        init_mean = torch.tensor([2, 5 * (2*torch.pi), 0, 0], dtype=torch.float32).view(4, 1, 1)
        init_radius = torch.tensor([2, 5 * (2*torch.pi), torch.pi, 1], dtype=torch.float32).view(4, 1, 1)
        self.gabor_x = nn.Parameter(
            nn.init.uniform_(torch.empty(4, 1, embedding_dim, dtype=torch.float32), -1, 1)
            * init_mean + init_radius
        )
        self.gabor_y = nn.Parameter(
            nn.init.uniform_(torch.empty(4, 1, embedding_dim, dtype=torch.float32), -1, 1)
            * init_mean + init_radius
        )
        # self.edge_weights = nn.Parameter(torch.nn(embedding_dim, 4, 1))  # 4 edges of the grid
        self.edge_weights = nn.Parameter(
            nn.init.uniform_(torch.empty(4, embedding_dim, dtype=torch.float32), -1, 1)
        )  # 4 edges of the grid

        # precompute some values
        self.x = torch.linspace(0, 1, grid_width).unsqueeze(1)  # from 0 to 1 for numerical stability, rescaled later
        self.y = torch.linspace(0, 1, grid_height).unsqueeze(1)
        self.exp_x2 = torch.exp(-self.x**2)
        self.exp_y2 = torch.exp(-self.y**2)

        self.projection_layer = nn.Linear(embedding_dim, embedding_dim, bias=True) if projection else None

    def forward(self) -> torch.Tensor:
        device = self.gabor_x.device
        if self.x.device != device:
            self.x = self.x.to(self.gabor_x.device)
            self.y = self.y.to(self.gabor_x.device)
            self.exp_x2 = self.exp_x2.to(self.gabor_x.device)
            self.exp_y2 = self.exp_y2.to(self.gabor_x.device)

        gabor_x = self.exp_x2 ** self.gabor_x[0] \
                  * torch.cos(self.gabor_x[1] * self.x + self.gabor_x[2]) \
                  * self.gabor_x[3]
        gabor_y = self.exp_y2 ** self.gabor_y[0] \
                    * torch.cos(self.gabor_y[1] * self.y + self.gabor_y[2]) \
                    * self.gabor_y[3]

        position_embeddings = gabor_x.unsqueeze(0) + gabor_y.unsqueeze(1)

        position_embeddings[0, :] += self.edge_weights[0]
        position_embeddings[-1, :] += self.edge_weights[1]
        position_embeddings[:, 0] += self.edge_weights[2]
        position_embeddings[:, -1] += self.edge_weights[3]

        if self.projection:
            position_embeddings = self.projection_layer(position_embeddings)

        position_embeddings = position_embeddings.unsqueeze(0)
        return position_embeddings


if __name__ == '__main__':
    pe = GaborPositionEmbeddings(16, 16, 512)
    position_embeddings = pe()

    print(position_embeddings)
    print(position_embeddings.shape)

    import matplotlib.pyplot as plt
    for i in range(position_embeddings.shape[3]):
        plt.imshow(position_embeddings[0, :, :, i].detach().numpy())
        plt.show()
