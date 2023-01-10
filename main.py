import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import config
import datasets
from models.vision_transformer import TrOCR, CustomEncoderDecoder

if __name__ == '__main__':
    crohme = datasets.DatasetManager(
        datasets.CROHME,
        batch_size=config.batch_size,
        precompute_padding=config.precompute_padding,
        batch_padding=config.batch_padding,
        downscale=config.downscale,
        include_sos_and_eos=config.include_sos_and_eos,
    )
    print(crohme.max_img_h, crohme.max_img_w, crohme.max_label_len)

    model = CustomEncoderDecoder(crohme)
    # model = TrOCR(crohme)

    if config.reload_from_checkpoint and config.weights_only:
        print(f'Reloading weights from checkpoint {config.reload_from_checkpoint}')
        model = CustomEncoderDecoder.load_from_checkpoint(config.checkpoint_path, dataset=crohme)

    if config.use_pretrained_encoder:
        print(f'Loading pretrained encoder from {config.pretrained_path}')
        if config.pretrained_path.endswith('/CNN-V2.pt'):
            pretrained: TrOCR = torch.load(config.pretrained_path)
            model.encoder.load_state_dict(pretrained.encoder.state_dict())
        elif config.pretrained_path.endswith('/CNN-V3.pt'):
            pretrained: TrOCR = torch.load(config.pretrained_path)
            model.encoder.load_state_dict(pretrained.encoder.state_dict())
        else:
            raise ValueError("Invalid pretrained path")

    model = model.to(config.device)

    wandb_logger = WandbLogger(project="HMER", entity="efagnou", name=config.name)
    wandb_logger.log_hyperparams(config.config_dict)

    save_dir = f"checkpoints/{config.name}-{wandb_logger.version}"
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_loss",
        filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch:02d}-{step:05d}-last",
    )

    scheduler_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    nb_devices = torch.cuda.device_count()
    devices = [max(range(nb_devices), key=lambda i: torch.cuda.get_device_properties(i).total_memory)]
    # devices = [0]

    trainer_kwargs = {}
    if config.reload_from_checkpoint and not config.weights_only:
        trainer_kwargs['resume_from_checkpoint'] = config.checkpoint_path
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[val_checkpoint_callback, last_checkpoint_callback, scheduler_callback],
        accelerator=config.trainer_accelerator,
        devices=devices,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        **trainer_kwargs,
        # precision=16,
    )
    trainer.fit(model=model, train_dataloaders=crohme.train_loader, val_dataloaders=crohme.test_loaders['TEST14'])
