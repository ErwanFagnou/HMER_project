import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import config
import datasets
from models.vision_transformer import TrOCR


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

    model = TrOCR(crohme).to(config.device)

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

    trainer_kwargs = {}
    if config.reload_from_checkpoint:
        trainer_kwargs['resume_from_checkpoint'] = config.checkpoint_path
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[val_checkpoint_callback, last_checkpoint_callback, scheduler_callback],
        accelerator=config.trainer_accelerator,
        devices=1,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        **trainer_kwargs,
    )
    trainer.fit(model=model, train_dataloaders=crohme.train_loader, val_dataloaders=crohme.test_loaders['TEST14'])
