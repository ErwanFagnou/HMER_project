from abc import ABC

import pytorch_lightning as pl
import torch

import config


class HMERModel(pl.LightningModule, ABC):
    def __init__(self, mask_token_id=config.additional_tokens['<pad>']):
        super(HMERModel, self).__init__()
        self.mask_token_id = mask_token_id

    def configure_optimizers(self):
        optimizer = config.optimizer(self.parameters(), **config.opt_kwargs)
        lr_scheduler = config.lr_scheduler(optimizer, **config.lr_scheduler_kwargs)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx, log_prefix='train_'):
        inputs, outputs = batch

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.type(torch.FloatTensor).to(self.device)
            outputs = outputs.type(torch.LongTensor).to(self.device)
            loss = self(inputs, outputs)

            pred_ids = self.result.logits.argmax(dim=-1)
            mask = outputs != self.mask_token_id
            acc = (pred_ids == outputs)[mask].float().mean()
            self.log(log_prefix + 'acc', acc, on_epoch=True)
            self.log(log_prefix + 'error', 1 - acc, on_epoch=True)

        else:
            loss = 0
            for sample_input, sample_output in zip(inputs, outputs):
                sample_input = sample_input[None, :].type(torch.FloatTensor).to(self.device)
                sample_output = sample_output[None, :].type(torch.LongTensor).to(self.device)
                loss = self(sample_input, sample_output)
            loss /= len(inputs)

        self.log(log_prefix + 'loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.training_step(batch, batch_idx, log_prefix='val_')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log_prefix='test_')
        return loss

    # def train_dataloader(self):
    #     pass
    #
    # def val_dataloader(self):
    #     pass
    #
    # def test_dataloader(self):
    #     pass
