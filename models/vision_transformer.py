import torch
from torch import nn
from transformers import VisionEncoderDecoderModel, ViTModel, TrOCRForCausalLM, ViTConfig, TrOCRConfig

from train import HMERModel


class ViT(HMERModel):

    def __init__(self):
        super().__init__()

        self.max_seq_len = 200
        self.fc1 = nn.Linear(1, 112 * self.max_seq_len)
        nn.init.zeros_(self.fc1.weight)

    def forward(self, x, true_out):
        x = torch.mean(x, dim=(1, 2))
        out = self.fc1(x)
        out = out.view(-1, self.max_seq_len, 112)
        out = out[:, :true_out.shape[1], :]
        return out


class TrOCR(HMERModel):
    def __init__(self, label2id: dict):
        super().__init__()
        encoder_config = ViTConfig(
            image_size=224,
        )
        encoder = ViTModel(encoder_config)
        decoder = TrOCRForCausalLM(TrOCRConfig())
        model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    def forward(self, x, true_out):
        x = torch.mean(x, dim=(1, 2))
        out = self.fc1(x)
        out = out.view(-1, self.max_seq_len, 112)
        out = out[:, :true_out.shape[1], :]
        return out
