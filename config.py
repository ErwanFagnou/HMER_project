import torch.optim.optimizer
from torch import nn

# Dataset parameters
downscale = 1
additional_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
precompute_padding = False
batch_padding = True

# Training parameters
epochs = 100
effective_batch_size = 64
batch_size = 8
accumulate_grad_batches = effective_batch_size // batch_size
optimizer = torch.optim.Adam
opt_kwargs = dict(lr=1e-3)

# loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=additional_tokens['<pad>'])
loss_fn = lambda loss, _: loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
