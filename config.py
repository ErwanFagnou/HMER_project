import torch.optim.optimizer
from torch import nn

# Reload
reload_from_checkpoint = False
checkpoint_path = "lightning_logs/version_54/checkpoints/epoch=56-step=7923.ckpt"

# Model
use_gabor_position_embeddings = True

# Dataset parameters
downscale = 2
additional_tokens = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
precompute_padding = False
batch_padding = True
include_sos_and_eos = False  # taken care of by transformer

# Training parameters
epochs = 1000
effective_batch_size = 64
batch_size = 16
accumulate_grad_batches = effective_batch_size // batch_size
optimizer = torch.optim.Adam
opt_kwargs = dict(lr=1e-3)

# loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=additional_tokens['<pad>'])
loss_fn = lambda loss, _: loss

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
trainer_accelerator = 'gpu' if use_gpu else 'cpu'
