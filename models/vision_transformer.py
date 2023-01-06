import torch
from torch import nn
from transformers import VisionEncoderDecoderModel, ViTModel, TrOCRForCausalLM, ViTConfig, TrOCRConfig
from transformers.utils import ModelOutput

from custom_embeddings import CustomViTEmbeddings
from datasets import DatasetManager
from train import HMERModel
import config


def get_ViT_encoder(dataset: DatasetManager):
    encoder_config = ViTConfig(
        hidden_size=50,
        intermediate_size=100,
        num_hidden_layers=3,
        num_attention_heads=10,
        max_image_height=dataset.max_img_h,
        max_image_width=dataset.max_img_w,
        num_channels=1,
        patch_size=16,
        hidden_dropout_prob=0.2,
        use_gabor_position_embeddings=config.use_gabor_position_embeddings,
    )
    encoder = ViTModel(encoder_config)
    encoder.embeddings = CustomViTEmbeddings(encoder_config)

    # initialize weights
    # for layer in encoder.encoder.layer:
    #     att_layer: ViTSelfAttention = layer.attention.attention
    #     # nn.init.orthogonal_(att_layer.value.weight)
    #     # nn.init.orthogonal_(att_layer.query.weight)
    #     att_layer.value.weight.data = torch.eye(att_layer.value.weight.shape[0])
    #     att_layer.query.weight.data = torch.eye(att_layer.query.weight.shape[0])
    #     att_layer.key.load_state_dict(att_layer.query.state_dict())  # queries and keys are equal
    #
    #     fc_layer: ViTIntermediate = layer.intermediate
    #     fc_layer.dense.weight.data = torch.eye(fc_layer.dense.weight.shape[0], fc_layer.dense.weight.shape[1])
    #
    #     output_layer: ViTOutput = layer.output
    #     output_layer.dense.weight.data = torch.eye(output_layer.dense.weight.shape[0], output_layer.dense.weight.shape[1])

    return encoder


def get_CNN_encoder(dataset: DatasetManager):

    class CNNEncoder(nn.Module):
        output_size = 64
        num_channels = [1, 16, 32, 48, 64]
        kernel_sizes = [3, 3, 3, 3]
        cnn_activation_cls = nn.ELU
        fc_activation_cls = nn.ELU
        dropout_rate = 0.2

        config = ViTConfig(
            hidden_size=output_size,
        )
        main_input_name = 'pixel_values'

        def __init__(self):
            super().__init__()
            self.cnn_activation = self.cnn_activation_cls()
            self.fc_activation = self.fc_activation_cls()

            layers = []
            for i in range(len(self.num_channels) - 1):
                layers.append(nn.Conv2d(self.num_channels[i], self.num_channels[i + 1], self.kernel_sizes[i], padding='same'))
                layers.append(nn.MaxPool2d(2))
                layers.append(self.cnn_activation)
            self.cnn = nn.Sequential(*layers)

            self.dropout = nn.Dropout(self.dropout_rate)

            # TODO: layer norm?

            # self.fc1 = nn.Linear(self.num_channels[-1], self.output_size)
            # self.fc2 = nn.Linear(self.output_size, self.output_size)

        def get_output_embeddings(self):
            return None

        def forward(self, pixel_values: torch.Tensor, **_):
            x = self.cnn(pixel_values)
            x = self.dropout(x)
            # shape: (batch_size, channels, height, width)

            x = x.flatten(2).transpose(1, 2)
            # shape: (batch_size, height * width, channels)

            # x = self.fc1(x)
            # x = self.fc_activation(x)
            # x = self.fc2(x)
            # x = self.fc_activation(x)

            return ModelOutput(last_hidden_state=x, hidden_states=None, attentions=None)

    return CNNEncoder()


def get_decoder(dataset: DatasetManager):
    decoder_config = TrOCRConfig(
        d_model=25,
        decoder_ffn_dim=50,
        decoder_layers=1,
        decoder_attention_heads=5,
        max_position_embeddings=512,
        use_learned_position_embeddings=True,
        vocab_size=len(dataset.label2id),
        decoder_start_token_id=dataset.label2id['<sos>'],
        bos_token_id=dataset.label2id['<sos>'],
        eos_token_id=dataset.label2id['<eos>'],
        pad_token_id=dataset.label2id['<pad>'],
        dropout=0.2,
        use_cache=False,  # to use less memory
    )
    decoder = TrOCRForCausalLM(decoder_config)

    # initialize weights
    # for layer in decoder.model.decoder.layers:
    #     att_layer = layer.encoder_attn
    #     att_layer.v_proj.weight.data = torch.eye(att_layer.v_proj.weight.shape[0])
    #     att_layer.q_proj.weight.data = torch.eye(att_layer.q_proj.weight.shape[0])
    #     att_layer.k_proj.load_state_dict(att_layer.q_proj.state_dict())  # queries and keys are equal
    #
    #     layer.fc1.weight.data = torch.eye(layer.fc1.weight.shape[0], layer.fc1.weight.shape[1])
    #     layer.fc2.weight.data = torch.eye(layer.fc2.weight.shape[0], layer.fc2.weight.shape[1])

    return decoder


class TrOCR(HMERModel):
    def __init__(self, dataset: DatasetManager):
        super().__init__()
        # self.encoder = get_ViT_encoder(dataset)
        self.encoder = get_CNN_encoder(dataset)

        self.decoder = get_decoder(dataset)

        self.encoder_decoder = VisionEncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        self.encoder_decoder.config.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        self.encoder_decoder.config.pad_token_id = self.decoder.config.pad_token_id
        self.encoder_decoder.config.eos_token_id = self.decoder.config.eos_token_id
        self.encoder_decoder.config.bos_token_id = self.decoder.config.bos_token_id

        self.result = None

    def forward(self, pixel_values, true_out):
        # print(pixel_values.shape)
        # print(true_out)
        # remove <sos> and <eos> tokens
        # if true_out[0, 0] == self.decoder.config.bos_token_id and true_out[0, 1] == self.decoder.config.eos_token_id:
        #     true_out = true_out[:, 1:-1]
        x = pixel_values.unsqueeze(1)  # add channel dim
        self.result = self.encoder_decoder(pixel_values=x, labels=true_out)
        return self.result.loss.unsqueeze(0)

    # def configure_optimizers(self):
    #     params = self.decoder.output_projection.parameters()
    #     return torch.optim.Adam(params, **config.opt_kwargs)

    def generate(self, pixel_values):
        x = pixel_values.unsqueeze(1)  # add channel dim
        # # inputs_ids = torch.full((pixel_values.shape[0], 1), self.encoder_decoder.config.eos_token_id, dtype=torch.long, device=pixel_values.device)
        # inputs_ids = torch.tensor([[self.encoder_decoder.config.bos_token_id, 10]], dtype=torch.long, device=pixel_values.device)

        # encoded = self.encoder_decoder.encoder(pixel_values=x)
        # decoded = self.encoder_decoder.generate(encoder_outputs=encoded)

        # decoded = self.encoder_decoder.decoder.generate(encoder_outputs=encoded)

        decoded = self.encoder_decoder.generate(pixel_values=x)

        return decoded
