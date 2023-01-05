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

        self.position_embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(1, config.max_image_height // config.patch_size,
                            config.max_image_width // config.patch_size, config.hidden_size, dtype=torch.float32),
                mean=0.0,
                std=config.initializer_range,
            )
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        **_,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        used_patches_mask = (embeddings != self.patch_embeddings.projection.bias).any(dim=2).any(dim=0)

        # add positional encoding to each token
        patch_rows = height // self.config.patch_size
        patch_cols = width // self.config.patch_size
        position_embeddings = self.position_embeddings[:, :patch_rows, :patch_cols, :].flatten(1, 2)
        # print("before", embeddings.shape, position_embeddings.shape, used_patches_mask.shape)
        embeddings = (embeddings + position_embeddings)[:, used_patches_mask]
        # print("after", embeddings.shape)

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
        num_patches = (max_h / patch_size[1]) * (max_w / patch_size[0])
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


if __name__ == '__main__':
    position_embeddings = nn.Parameter(
        nn.init.trunc_normal_(
            torch.zeros(1, 1080, 1920, 100, dtype=torch.float32),
            mean=0.0,
            std=0.2,
        )
    )

    print(position_embeddings.shape)
    print(position_embeddings.flatten(1, 2).shape)
