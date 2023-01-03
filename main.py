import pytorch_lightning as pl

import config
import datasets
from models.vision_transformer import ViT

crohme = datasets.DatasetManager(datasets.CROHME, batch_size=config.batch_size, padding=config.padding)

model = ViT()

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.epochs)
trainer.fit(model, crohme.train_loader)
