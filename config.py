import torch.optim.optimizer
from torch import nn

# Reload
reload_from_checkpoint = False
checkpoint_path = "lightning_logs/version_36/checkpoints/epoch=28-step=4031.ckpt"

# Dataset parameters
downscale = 1
additional_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
precompute_padding = False
batch_padding = True
include_sos_and_eos = False  # taken care of by transformer

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
