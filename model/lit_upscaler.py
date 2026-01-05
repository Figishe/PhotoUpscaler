import lightning as L
import torch
from upscaler import SuperResNet
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid

class LitSuperResNet(L.LightningModule):
    def __init__(self, lr=1e-6, start_channels=64, depth=2, downscale_hires=2, downscale_lowres=2):
        super().__init__()
        self.model = SuperResNet(start_channels=start_channels, depth=depth)

        self.model = torch.compile(
            self.model,
            mode="max-autotune",
            fullgraph=False
        )

        self.lr = lr
        self.loss = F.mse_loss
        self.downscale_hires = downscale_hires
        self.downscale_lowres = downscale_lowres
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def batch_preprocess(self, batch):
        y_hires = batch
        y_hires = y_hires.to(self.device, non_blocking=True)

        y = F.interpolate(
            y_hires,
            scale_factor=1 / self.downscale_hires, # for denoising of original photo
            mode="bicubic",
            align_corners=False
        )

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


class ImageLoggerCallback(L.Callback):
    def __init__(self, val_samples, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs
        self.fixed_batch = next(iter(val_samples))  # HR batch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        pl_module.eval()

        with torch.no_grad():
            x, y = pl_module.batch_preprocess(self.fixed_batch)
            pred = pl_module(x)

        imgs = torch.cat([y.cpu(), pred.cpu()], dim=2)
        grid = make_grid(imgs, nrow=4)

        pl_module.logger.experiment.add_image(
            "GT_vs_Pred", grid, global_step=epoch
        )

