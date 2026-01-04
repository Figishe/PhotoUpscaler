import lightning as L
import torch
from upscaler import SuperResNet
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid

class LitSuperResNet(L.LightningModule):
    def __init__(self, lr=1e-6, start_channels=64, depth=2):
        super().__init__()
        self.model = SuperResNet(start_channels=start_channels, depth=depth)
        self.lr = lr
        self.loss = F.mse_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ImageLoggerCallback(L.Callback):
    def __init__(self, val_samples, log_every_n_epochs=1):
        self.val_samples = val_samples
        self.log_every_n_epochs = log_every_n_epochs

        x, y = next(iter(val_samples))
        self.fixed_x = x
        self.fixed_y = y

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        x = self.fixed_x.to(pl_module.device)
        y = self.fixed_y.to(pl_module.device)

        with torch.no_grad():
            pred = pl_module(x)

        # берём первые 4 картинки из батча
        img_gt   = y[:4].cpu()
        img_pred = pred[:4].cpu()

        # объединяем по каналу C для отображения GT сверху, предсказание снизу
        imgs = torch.cat([img_gt, img_pred], dim=2)  # вертикально
        grid = make_grid(imgs, nrow=4)  # 4 картинки в ряд

        pl_module.logger.experiment.add_image("GT_vs_Pred", grid, global_step=epoch)
