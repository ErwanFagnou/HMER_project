import pytorch_lightning as pl
import torch

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

    # imgs, true_labels = next(iter(crohme.train_loader))
    # imgs = imgs[:1].to(config.device)
    # true_labels = true_labels[:1].type(torch.LongTensor).to(config.device)
    # print(imgs.shape, true_labels.shape)
    # model(imgs, true_labels)

    # img = crohme.train_loader.dataset[0][0].unsqueeze(0).to(config.device)
    # # img = torch.randn(1, 224, 224)
    # true_labels = crohme.train_loader.dataset[0][1].unsqueeze(0).type(torch.LongTensor).to(config.device)
    # print(img.shape, true_labels.shape)
    # model(img, true_labels)

    # result = model.result
    # print(result.logits.shape)
    # print(config.loss_fn(result, true_labels))

    # print(crohme.test_loaders)

    trainer_kwargs = {}
    if config.reload_from_checkpoint:
        trainer_kwargs['resume_from_checkpoint'] = config.checkpoint_path
    trainer = pl.Trainer(
        accelerator=config.trainer_accelerator,
        devices=1,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        **trainer_kwargs,
    )
    trainer.fit(model=model, train_dataloaders=crohme.train_loader, val_dataloaders=crohme.test_loaders['TEST14'])
