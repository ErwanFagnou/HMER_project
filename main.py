import pytorch_lightning as pl
import torch

import config
import datasets
from models.vision_transformer import TrOCR


if __name__ == '__main__':
    crohme = datasets.DatasetManager(datasets.CROHME, batch_size=config.batch_size, precompute_padding=config.precompute_padding, batch_padding=config.batch_padding, downscale=config.downscale)
    print(crohme.max_img_h, crohme.max_img_w, crohme.max_label_len)

    model = TrOCR(crohme).to(config.device)

    # img = crohme.train_loader.dataset[0][0].unsqueeze(0).unsqueeze(0)
    # # img = torch.randn(1, 1, 224, 224)
    # true_labels = crohme.train_loader.dataset[0][1].unsqueeze(0).type(torch.LongTensor)
    # print(img.shape, true_labels.shape)
    #
    # result = model(img, true_labels)
    # print(result.logits.shape)
    # print(config.loss_fn(result, true_labels))

    print(crohme.test_loaders)

    pl_device = 'gpu' if config.device == 'cuda' else 'cpu'
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.epochs)
    trainer.fit(model=model, train_dataloaders=crohme.train_loader)  #, val_dataloaders=list(crohme.test_loaders.values()))
