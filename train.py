from abc import ABC

import pytorch_lightning as pl
import torch

import config


class HMERModel(pl.LightningModule, ABC):
    def __init__(self):
        super(HMERModel, self).__init__()

    def configure_optimizers(self):
        return config.optimizer(self.parameters(), **config.opt_kwargs)

    def training_step(self, batch, batch_idx):
        inputs, outputs = batch

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.type(torch.FloatTensor).to(self.device)
            outputs = outputs.to(self.device)
            pred = self(inputs, outputs)
            pred_flat = pred.view(-1, pred.shape[-1])
            outputs_flat = outputs.view(-1)
            loss = config.loss_fn(pred_flat, outputs_flat) / inputs.shape[0]
        else:
            loss = 0
            for sample_input, sample_output in zip(inputs, outputs):
                sample_input = sample_input[None, :].type(torch.FloatTensor).to(self.device)
                sample_output = sample_output[None, :].type(torch.LongTensor).to(self.device)
                pred = self(sample_input, sample_output)
                pred_flat = pred.view(-1, pred.shape[-1])
                outputs_flat = sample_output.view(-1)
                # print(sample_input, pred, sample_output)
                loss += config.loss_fn(pred_flat, outputs_flat)
            loss /= len(inputs)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    # def train_dataloader(self):
    #     pass
    #
    # def val_dataloader(self):
    #     pass
    #
    # def test_dataloader(self):
    #     pass
