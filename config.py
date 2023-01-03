import torch.optim.optimizer
from torch import nn

# Dataset parameters
additional_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
padding = False

# Training parameters
epochs = 100
batch_size = 32
optimizer = torch.optim.Adam
opt_kwargs = dict(lr=1e-3)

loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=additional_tokens['<pad>'])