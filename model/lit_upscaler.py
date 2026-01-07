import lightning as L
import torch
from model.upscaler import SuperResNet
from torch import nn
import torch.nn.functional as F
from model.loss import ycbcr_weighted_mse
from functools import partial

class LitSuperResNet(L.LightningModule):
    def __init__(self, lr=1e-6, start_channels=64, depth=2, downscale_lowres=2, lambda_y=1.0, lambda_cbcr=0.5):
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
        self.loss = partial(ycbcr_weighted_mse, weights=(lambda_y, lambda_cbcr, lambda_cbcr))
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


    def training_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)

        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.batch_preprocess(batch)

        pred = self(x)
        loss = self.loss(pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)