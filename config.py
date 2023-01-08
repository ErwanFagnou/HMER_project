import torch.optim.optimizer
from torch import nn

# Reload
reload_from_checkpoint = False
checkpoint_path = "lightning_logs/version_54/checkpoints/epoch=56-step=7923.ckpt"

# Model
name = "Model_V4"
use_gabor_position_embeddings = False
gabor_embeddings_size = 32
project_position_embeddings = True

# Dataset parameters
downscale = 1
additional_tokens = {"<eos>": 0, "<pad>": 0, "<sos>": 1}
precompute_padding = False
batch_padding = True
include_sos_and_eos = False  # taken care of by transformer

# Training parameters
epochs = 500
effective_batch_size = 64
batch_size = 64
accumulate_grad_batches = effective_batch_size // batch_size
optimizer = torch.optim.Adam
opt_kwargs = dict(lr=1e-3, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
lr_scheduler_kwargs = dict(T_max=epochs, eta_min=1e-6)

# optimizer = torch.optim.SGD
# opt_kwargs = dict(lr=1e-2, momentum=0.9, weight_decay=0)

# loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=additional_tokens['<pad>'])

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
trainer_accelerator = 'gpu' if use_gpu else 'cpu'

config_dict = dict(
    reload_from_checkpoint=reload_from_checkpoint,
    checkpoint_path=checkpoint_path,
    use_gabor_position_embeddings=use_gabor_position_embeddings,
    project_position_embeddings=project_position_embeddings,
    downscale=downscale,
    additional_tokens=additional_tokens,
    precompute_padding=precompute_padding,
    batch_padding=batch_padding,
    include_sos_and_eos=include_sos_and_eos,
    epochs=epochs,
    effective_batch_size=effective_batch_size,
    batch_size=batch_size,
    accumulate_grad_batches=accumulate_grad_batches,
    optimizer=optimizer,
    opt_kwargs=opt_kwargs,
    **opt_kwargs,
)
