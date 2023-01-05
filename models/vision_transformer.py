import torch
from torch import nn
from transformers import VisionEncoderDecoderModel, ViTModel, TrOCRForCausalLM, ViTConfig, TrOCRConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTPatchEmbeddings

from custom_embeddings import CustomViTEmbeddings
from datasets import DatasetManager
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
    def __init__(self, dataset: DatasetManager):
        super().__init__()
        encoder_config = ViTConfig(
            hidden_size=100,
            intermediate_size=400,
            num_hidden_layers=3,
            num_attention_heads=10,
            max_image_height=dataset.max_img_h,
            max_image_width=dataset.max_img_w,
            num_channels=1,
            patch_size=16,
        )
        decoder_config = TrOCRConfig(
            d_model=100,
            decoder_ffn_dim=200,
            decoder_layers=1,
            decoder_attention_heads=10,
            max_position_embeddings=512,
            use_learned_position_embeddings=True,
            vocab_size=len(dataset.label2id),
            decoder_start_token_id=dataset.label2id['<sos>'],
            bos_token_id=dataset.label2id['<sos>'],
            eos_token_id=dataset.label2id['<eos>'],
            pad_token_id=dataset.label2id['<pad>'],
        )
        self.encoder = ViTModel(encoder_config)
        self.encoder.embeddings = CustomViTEmbeddings(encoder_config)

        self.decoder = TrOCRForCausalLM(decoder_config)

        self.encoder_decoder = VisionEncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        self.encoder_decoder.config.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.encoder_decoder.config.pad_token_id = decoder_config.pad_token_id
        self.encoder_decoder.config.eos_token_id = decoder_config.eos_token_id
        self.encoder_decoder.config.bos_token_id = decoder_config.bos_token_id

        self.result = None

    def forward(self, pixel_values, true_out):
        # print(pixel_values.shape)
        # remove <sos> and <eos> tokens
        if true_out[0, 0] == self.decoder.config.bos_token_id and true_out[0, 1] == self.decoder.config.eos_token_id:
            true_out = true_out[:, 1:-1]
        x = pixel_values.unsqueeze(1)  # add channel dim
        self.result = self.encoder_decoder(pixel_values=x, labels=true_out)
        return self.result.loss.unsqueeze(0)

    def generate(self, pixel_values):
        x = pixel_values.unsqueeze(1)  # add channel dim
        # # inputs_ids = torch.full((pixel_values.shape[0], 1), self.encoder_decoder.config.eos_token_id, dtype=torch.long, device=pixel_values.device)
        # inputs_ids = torch.tensor([[self.encoder_decoder.config.bos_token_id, 10]], dtype=torch.long, device=pixel_values.device)

        encoded = self.encoder_decoder.encoder(pixel_values=x)
        decoded = self.encoder_decoder.generate(encoder_outputs=encoded)

        # decoded = self.encoder_decoder.decoder.generate(encoder_outputs=encoded)

        # validate_model_kwargs_fn = self.encoder_decoder._validate_model_kwargs
        # self.encoder_decoder._validate_model_kwargs = lambda _: True
        #
        # decoded = self.encoder_decoder.generate(pixel_values=x, interpolate_pos_encoding=True)
        #
        # self.encoder_decoder._validate_model_kwargs = validate_model_kwargs_fn
        return decoded
