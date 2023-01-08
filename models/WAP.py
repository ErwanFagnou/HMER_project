import torch
from torch import nn
from transformers.utils import ModelOutput

from datasets import DatasetManager


class WAPDecoder(nn.Module):
    input_dropout_rate = 0.2
    noise_std = 0.5

    encoder_dim = 32
    embedding_dim = 256
    hidden_dim = 256
    attention_dim = 50

    def __init__(self, dataset: DatasetManager):
        super().__init__()
        nb_classes = len(dataset.label2id)

        self.input_dropout = nn.Dropout(self.input_dropout_rate)

        self.embedding = nn.Embedding(nb_classes, self.embedding_dim)

        self.rnn_cell = nn.GRUCell(input_size=self.embedding_dim + self.encoder_dim,
                                   hidden_size=self.hidden_dim, bias=False)
        self.h_layernorm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.context_layernorm = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)

        # context vector
        self.attention_logits_1 = nn.Linear(self.encoder_dim, self.attention_dim, bias=True)
        self.attention_logits_2 = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.attention_activation = nn.Tanh()
        self.attention_proj = nn.Linear(self.attention_dim, 1)

        # final layers
        self.fc = nn.Linear(self.embedding_dim + self.hidden_dim + self.encoder_dim, nb_classes, bias=True)

    def forward(self, inputs_embeds, input_ids, stop_at_id=None):
        embedded_seq = self.embedding(input_ids)
        embedded_seq = self.input_dropout(embedded_seq)

        attention_logits_1 = self.attention_logits_1(inputs_embeds)

        all_vectors = []
        h_prev = torch.zeros(inputs_embeds.shape[0], self.hidden_dim, device=inputs_embeds.device)
        for t in range(embedded_seq.shape[1]):
            # context vector
            x = attention_logits_1 + self.attention_logits_2(h_prev).unsqueeze(1)
            x = self.attention_activation(x)
            x = self.attention_proj(x)
            x = torch.softmax(x, dim=1)
            context = torch.sum(x * inputs_embeds, dim=1)
            context = self.context_layernorm(context)
            if self.training:
                context = context + torch.randn_like(context) * self.noise_std

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
