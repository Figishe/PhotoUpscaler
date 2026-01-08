import lightning as L
import torch
from model.upscaler import SuperResNet
from torch import nn
import torch.nn.functional as F
from model.loss import gradient_loss, laplacian_loss, ycbcr_mae_split
from functools import partial

class LitSuperResNet(L.LightningModule):
    def __init__(self, lr=1e-6, start_channels=64, depth=2, downscale_lowres=2, lambda_y=1.0, lambda_cbcr=0.5, lambda_grad=0.1, lambda_laplasian=0.02):
        super().__init__()
        self.model = SuperResNet(start_channels=start_channels, depth=depth)

        self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=False
        )

        self.lr = lr
        self.lambda_y = lambda_y
        self.lambda_cbcr = lambda_cbcr
        self.lambda_grad = lambda_grad
        self.lambda_laplasian = lambda_laplasian

        self.downscale_lowres = downscale_lowres
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def batch_preprocess(self, batch):
        y = batch
        y = y.to(self.device, non_blocking=True)

        x = F.interpolate(
            y,
            scale_factor=1 / self.downscale_lowres,
            mode="bicubic",
            align_corners=False
        )

        return x, y

    def loss(self, pred, target, mode):
        mae_y, mae_cbcr = ycbcr_mae_split(pred, target)
        grad = gradient_loss(pred, target)
        laplasian = laplacian_loss(pred, target)

        total_loss = (self.lambda_y * mae_y +
                      self.lambda_cbcr * mae_cbcr +
                      self.lambda_grad * grad +
                      self.lambda_laplasian * laplasian)

        self.log(f"{mode}/loss_y", mae_y, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss_cbcr", mae_cbcr, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss_grad", grad, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss_laplasian", laplasian, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{mode}/loss", total_loss, on_epoch=True, on_step=False, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)
        pred = self(x)
        return self.loss(pred, y, mode='train')

    def validation_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)
        pred = self(x)
        return self.loss(pred, y, mode='val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=600,
            eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
