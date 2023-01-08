import math
from itertools import chain

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import VisionEncoderDecoderModel, ViTModel, TrOCRForCausalLM, ViTConfig, TrOCRConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput

from custom_embeddings import CustomViTEmbeddings, GaborPositionEmbeddings
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


class CNNEncoder(nn.Module):
    output_size = 64
    num_channels = [1, 16, 16, 32, 48, output_size]
    kernel_sizes = [3, 3, 3, 3, 3]
    cnn_activation_cls = nn.ELU
    fc_activation_cls = nn.ELU
    dropout_rate = 0.2
    use_gabor_position_embeddings = config.use_gabor_position_embeddings

    config = ViTConfig(
        hidden_size=output_size,
    )
    main_input_name = 'pixel_values'

    def __init__(self, dataset: DatasetManager):
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

        grid_rows = math.ceil(dataset.max_img_h / (2 ** len(self.kernel_sizes)))
        grid_cols = math.ceil(dataset.max_img_w / (2 ** len(self.kernel_sizes)))
        if self.use_gabor_position_embeddings:
            emb_size = config.gabor_embeddings_size if config.project_position_embeddings else self.output_size
            self.pos_embeddings = GaborPositionEmbeddings(grid_rows, grid_cols, emb_size,
                                                          projection=config.project_position_embeddings,
                                                          projection_size=self.output_size)
        else:
            self.pos_embeddings = nn.Parameter(torch.zeros(1, grid_rows, grid_cols, self.output_size))
            nn.init.trunc_normal_(self.pos_embeddings, std=0.02)

        # layer norm?
        # self.layer_norm = nn.LayerNorm(self.output_size)

        # self.fc1 = nn.Linear(self.num_channels[-1], self.output_size)
        # self.fc2 = nn.Linear(self.output_size, self.output_size)

    def get_output_embeddings(self):
        return None

    def get_position_embeddings(self):
        if self.use_gabor_position_embeddings:
            return self.pos_embeddings()
        else:
            return self.pos_embeddings

    def forward(self, pixel_values: torch.Tensor, **_):
        x = self.cnn(pixel_values)
        # shape: (batch_size, channels, height, width)

        x = x.permute(0, 2, 3, 1)
        # shape: (batch_size, height,  width, channels)

        # dropout
        x = self.dropout(x)

        # add position embeddings
        x = x + self.get_position_embeddings()[:, :x.shape[1], :x.shape[2]]


        x = x.flatten(start_dim=1, end_dim=2)
        # shape: (batch_size, height * width, channels)

        # x = self.fc1(x)
        # x = self.fc_activation(x)
        # x = self.fc2(x)
        # x = self.fc_activation(x)

        # x = self.layer_norm(x)

        return ModelOutput(last_hidden_state=x, hidden_states=None, attentions=None)


def get_decoder(dataset: DatasetManager):
    decoder_config = TrOCRConfig(
        d_model=50,
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
        # layernorm_embedding=False,
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


class CustomDecoder(nn.Module):
    dropout_rate = 0.2

    # Self-attention
    self_attention_num_heads = 4
    hidden_state_dim = 64  # = encoder output dim

    skip_connections = config.skip_connection

    def __init__(self, dataset: DatasetManager):
        super().__init__()
        nb_classes = len(dataset.label2id)

        self.layernorm_encoder_outputs = nn.LayerNorm(self.hidden_state_dim)

        if config.use_past_true_outputs:
            self.token_embeddings = nn.Embedding(nb_classes, self.hidden_state_dim)
            nn.init.kaiming_normal_(self.token_embeddings.weight, nonlinearity='relu')

        self.sequence_pos_embedding = nn.Parameter(
                nn.init.kaiming_normal_(torch.zeros(1, dataset.max_label_len, self.hidden_state_dim), nonlinearity='relu'))
        self.self_attn = nn.MultiheadAttention(self.hidden_state_dim, self.self_attention_num_heads, dropout=self.dropout_rate, batch_first=True)

        # Encoder output (image patch encodings)
        self.image_attn = nn.MultiheadAttention(self.hidden_state_dim, 1, dropout=self.dropout_rate, batch_first=True)

        self.initial_hidden_state = nn.Parameter(torch.zeros(1, self.hidden_state_dim))

        self.fc1 = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)
        self.fc2 = nn.Linear(self.hidden_state_dim, nb_classes)
        self.activation = nn.ELU()

        # Recursive part
        self.fc_rnn = nn.Linear(self.hidden_state_dim, self.hidden_state_dim)
        self.layernorm_hidden_state = nn.LayerNorm(self.hidden_state_dim)

    def recursive_forward(self, prev_hidden_states, encoder_output):
        # get last hidden state
        hidden_state = prev_hidden_states[:, -1:, :]
        # shape: (batch_size, 1, hidden_state_dim)

        # add relative position embeddings  TODO: is relative better?
        prev_hidden_states = prev_hidden_states + self.sequence_pos_embedding[:, -prev_hidden_states.shape[1]:]
        # shape: (batch_size, prev_seq_len, hidden_state_dim)

        # get query to make: given the current state, where is the next symbol in the image?
        query = self.self_attn(hidden_state, prev_hidden_states, prev_hidden_states)[0]
        # shape: (batch_size, 1, self_attention_dim)

        # use the query on the image
        x = self.image_attn(query, encoder_output, encoder_output)[0][:, 0]  # todo: optimize
        # shape: (batch_size, hidden_state_dim)

        x = self.activation(x)
        x = self.fc_rnn(x)
        if self.skip_connections:
            x = x + hidden_state[:, 0]
        next_hidden_state = self.layernorm_hidden_state(x)
        return next_hidden_state

    def forward(self, inputs_embeds, input_ids, stop_at_id=None):
        # inputs_embeds: (batch_size, height * width, d_model)
        # inputs_ids: (batch_size, seq_len)
        batch_size = inputs_embeds.shape[0]

        inputs_embeds = self.layernorm_encoder_outputs(inputs_embeds)

        if config.use_past_true_outputs:
            input_ids_embeds = self.token_embeddings(input_ids)

        hidden_states = []
        hidden_states_with_decision = self.initial_hidden_state.repeat(batch_size, 1).unsqueeze(1)
        for t in range(input_ids.shape[1]):
            hidden_state = self.recursive_forward(hidden_states_with_decision, inputs_embeds)
            hidden_states.append(hidden_state)

            if config.use_past_true_outputs:
                hidden_state_with_decision = hidden_state + input_ids_embeds[:, t]
            else:
                hidden_state_with_decision = hidden_state
            hidden_states_with_decision = torch.cat([hidden_states_with_decision,
                                                     hidden_state_with_decision.unsqueeze(1)], dim=1)
            # if stop_at_id is not None and (hidden_states[:, -1] == stop_at_id).all():
            #     break

        hidden_states = torch.stack(hidden_states, dim=1)

        x = self.fc1(hidden_states)
        x = self.activation(x)
        logits = self.fc2(x)

        return ModelOutput(logits=logits, hidden_states=None, attentions=None)


class TrOCR(HMERModel):
    def __init__(self, dataset: DatasetManager):
        super().__init__(mask_token_id=dataset.label2id['<pad>'])
        # self.encoder = get_ViT_encoder(dataset)
        self.encoder = CNNEncoder(dataset)

        # self.decoder = get_decoder(dataset)

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


class CustomEncoderDecoder(HMERModel):
    def __init__(self, dataset: DatasetManager):
        super().__init__()
        self.encoder = CNNEncoder(dataset)
        self.decoder = CustomDecoder(dataset)

        self.vocab_size = len(dataset.label2id)
        self.eos_token_id = dataset.label2id['<eos>']
        self.result = None

    def configure_optimizers(self):
        if config.use_pretrained_encoder:
            params = self.decoder.parameters()
            if config.pretrain_learn_encoder_positional_embeddings:
                pos_embeddings = self.encoder.pos_embeddings
                new_params = [pos_embeddings] if isinstance(pos_embeddings, nn.Parameter) else pos_embeddings.parameters()
                params = chain(params, new_params)
        else:
            params = None
        return super().configure_optimizers(params)

    def forward(self, pixel_values, labels=None, sequence_length=0):
        sequence_length = labels.shape[1] if labels is not None else sequence_length
        x = pixel_values.unsqueeze(1)  # add channel dim

        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs.last_hidden_state, labels)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(decoder_outputs.logits.reshape(-1, self.vocab_size), labels.view(-1))

        self.result = Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
        )

        return loss

    @torch.no_grad()
    def generate(self, pixel_values, max_length=50):
        x = pixel_values.unsqueeze(1)  # add channel dim

        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs.last_hidden_state, max_length, stop_at_id=self.eos_token_id)
        logits = decoder_outputs.logits
        return logits.argmax(dim=-1)
