import torch
from torch import nn
from transformers.utils import ModelOutput

from datasets import DatasetManager


class WAPDecoder(nn.Module):
    dropout_rate = 0.2

    encoder_dim = 64
    embedding_dim = 30
    hidden_dim = 50
    attention_dim = 50

    def __init__(self, dataset: DatasetManager):
        super().__init__()
        nb_classes = len(dataset.label2id)

        self.embedding = nn.Embedding(nb_classes, self.embedding_dim)

        self.rnn_cell = nn.GRUCell(input_size=self.embedding_dim + self.encoder_dim,
                                   hidden_size=self.hidden_dim, bias=False)

        # context vector
        self.attention_logits_1 = nn.Linear(self.encoder_dim, self.attention_dim, bias=False)
        self.attention_logits_2 = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.attention_activation = nn.Tanh()
        self.attention_proj = nn.Linear(self.attention_dim, 1)

        # final layers
        self.fc = nn.Linear(self.embedding_dim + self.hidden_dim + self.encoder_dim, nb_classes, bias=True)

    def forward(self, inputs_embeds, input_ids, stop_at_id=None):
        embedded_seq = self.embedding(input_ids)

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

            h_t = self.rnn_cell(input=torch.cat([embedded_seq[:, t, :], context], dim=1), hx=h_prev)
            all_vectors.append(torch.cat([embedded_seq[:, t, :], context, h_t], dim=1))

            h_prev = h_t

        logits = self.fc(torch.stack(all_vectors, dim=1))

        return ModelOutput(logits=logits, hidden_states=None, attentions=None)
