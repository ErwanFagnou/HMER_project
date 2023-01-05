import pytorch_lightning as pl
import torch

import config
import datasets
from models.vision_transformer import TrOCR


if __name__ == '__main__':
    crohme = datasets.DatasetManager(datasets.CROHME, batch_size=config.batch_size, precompute_padding=config.precompute_padding, batch_padding=config.batch_padding, downscale=config.downscale)
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

    pl_device = 'gpu' if config.device == 'cuda' else 'cpu'
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.epochs, accumulate_grad_batches=config.accumulate_grad_batches)
    trainer.fit(model=model, train_dataloaders=crohme.train_loader)  #, val_dataloaders=list(crohme.test_loaders.values()))
