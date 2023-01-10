import gc

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, TrOCRConfig
from transformers.generation_utils import GenerationMixin
from transformers.utils import ModelOutput

from datasets import DatasetManager


class WAPDecoder(PreTrainedModel):
    dropout_rate = 0.4
    noise_std = 0.5

    encoder_dim = 256
    embedding_dim = 256
    hidden_dim = 256
    attention_dim = 256

    def __init__(self, dataset: DatasetManager):
        nb_classes = len(dataset.label2id)
        self.config = TrOCRConfig(
            vocab_size=nb_classes,
            decoder_start_token_id=dataset.label2id['<sos>'],
            bos_token_id=dataset.label2id['<sos>'],
            eos_token_id=dataset.label2id['<eos>'],
            pad_token_id=dataset.label2id['<pad>'],
            # is_decoder=True,
            # is_encoder_decoder=True,
        )
        super().__init__(self.config)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.embedding = nn.Embedding(nb_classes, self.embedding_dim)

        self.rnn_cell = nn.GRUCell(input_size=self.embedding_dim + self.encoder_dim,
                                   hidden_size=self.hidden_dim, bias=False)
        self.h_layernorm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        # self.h_layernorm = nn.LayerNorm(self.hidden_dim, elementwise_affine=True)
        self.context_layernorm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False)
        # self.context_layernorm = nn.LayerNorm(self.encoder_dim, elementwise_affine=True)

        # context vector
        self.attention_encoder_proj = nn.Linear(self.encoder_dim, self.attention_dim, bias=True)
        self.attention_hidden_state_proj = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.attention_coverage_proj = nn.Linear(1, self.attention_dim, bias=False)
        self.attention_activation = nn.Tanh()
        self.attention_proj = nn.Linear(self.attention_dim, 1)

        # final layers
        self.fc = nn.Linear(self.embedding_dim + self.hidden_dim + self.encoder_dim, nb_classes, bias=True)

    def forward(self, encoder_outputs, input_ids, stop_at_id=None, **kwargs):
        # print('input_ids', input_ids.shape)
        # print('encoder_outputs', encoder_outputs.shape)
        embedded_seq = self.embedding(input_ids)
        embedded_seq = self.dropout(embedded_seq)

        # encoder_outputs = self.input_dropout(encoder_outputs)
        attention_logits_1 = self.attention_encoder_proj(encoder_outputs)

        att_weights_cumsum = torch.zeros(encoder_outputs.shape[0], encoder_outputs.shape[1], 1, device=encoder_outputs.device)

        all_vectors = []
        h_prev = torch.zeros(encoder_outputs.shape[0], self.hidden_dim, device=encoder_outputs.device)
        for t in range(embedded_seq.shape[1]):
            # context vector
            att_weights_cumsum_dropout = self.dropout(att_weights_cumsum)
            x = attention_logits_1 + self.attention_hidden_state_proj(h_prev).unsqueeze(1) \
                + self.attention_coverage_proj(att_weights_cumsum_dropout)
            x = self.attention_activation(x)
            x = self.dropout(x)
            x = self.attention_proj(x)
            att_weights = torch.softmax(x, dim=1)
            gc.collect()  # clean memory
            context = torch.sum(att_weights * encoder_outputs, dim=1)
            context = self.context_layernorm(context)
            if self.training:
                context = context + torch.randn_like(context) * self.noise_std

            gc.collect()  # clean memory

            att_weights_cumsum = att_weights_cumsum + att_weights

            prev_word = embedded_seq[:, t-1, :] if t > 0 else torch.zeros_like(embedded_seq[:, 0, :])
            rnn_input = torch.cat((prev_word, context), dim=1)
            h_t = self.rnn_cell(input=rnn_input, hx=h_prev)
            h_t = self.h_layernorm(h_t)
            if self.training:
                h_t = h_t + torch.randn_like(h_t) * self.noise_std

            all_vectors.append(torch.cat([rnn_input, h_t], dim=1))

            h_prev = h_t

        logits = self.fc(torch.stack(all_vectors, dim=1))

        return ModelOutput(logits=logits, hidden_states=None, attentions=None)

    def prepare_inputs_for_generation(self, input_ids, encoder_outputs, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # if past:
        #     input_ids = input_ids[:, -1:]

        # remove bos token
        # if input_ids.shape[1] > 1:
        #     input_ids = input_ids[:, 1:]
        #
        # # add a new token for generation (because len(pred) == len(input_ids))
        # input_ids = torch.cat([input_ids, input_ids.new_zeros(input_ids.shape[0], 1)], dim=1)

        # shifts the input ids for generation (the last token is ignored)
        input_ids = input_ids.roll(shifts=-1, dims=1)

        # if multiple beams
        if input_ids.shape[0] > encoder_outputs.shape[0]:
            assert input_ids.shape[0] % encoder_outputs.shape[0] == 0
            encoder_outputs = encoder_outputs.repeat(input_ids.shape[0] // encoder_outputs.shape[0], 1, 1)

        return {
            "input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
        }
