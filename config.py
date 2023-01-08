import torch.optim.optimizer
from torch import nn

# Reload
reload_from_checkpoint = False
checkpoint_path = "checkpoints/Model_V4-1j2620gu/epoch=205-step=28634-last.ckpt"
weights_only = True

# Model
name = "Model_V4"

use_pretrained_encoder = True
pretrained_path = "final_models/CNN-V2.pt"
pretrain_learn_encoder_positional_embeddings = True

use_gabor_position_embeddings = False
gabor_embeddings_size = 32
project_position_embeddings = True

use_past_true_outputs = True
skip_connection = False

# Dataset parameters
downscale = 1
additional_tokens = {"<eos>": 0, "<pad>": 0, "<sos>": 1}
precompute_padding = False
batch_padding = True
include_sos_and_eos = False  # taken care of by transformer

# Training parameters
epochs = 500
effective_batch_size = 64
batch_size = 16
accumulate_grad_batches = effective_batch_size // batch_size
optimizer = torch.optim.Adam
opt_kwargs = dict(lr=1e-3, weight_decay=1e-5)
# optimizer = torch.optim.SGD
# opt_kwargs = dict(lr=1e-4, momentum=0.9, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
lr_scheduler_kwargs = dict(T_max=epochs, eta_min=1e-6)


# loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=additional_tokens['<pad>'])

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
trainer_accelerator = 'gpu' if use_gpu else 'cpu'

config_dict = dict(
    name=name,
    use_pretrained_encoder=use_pretrained_encoder,
    pretrained_path=pretrained_path,
    reload_from_checkpoint=reload_from_checkpoint,
    checkpoint_path=checkpoint_path,
    weights_only=weights_only,
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
